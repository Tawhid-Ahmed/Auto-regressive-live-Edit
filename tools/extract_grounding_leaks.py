"""List samples where edited answer leaks on wrong-image probes (image_swap / hard_negative)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


def flatten_results(results_path: str):
    with open(results_path, encoding="utf-8") as f:
        batches = json.load(f)
    if batches and isinstance(batches[0], list):
        return [s for batch in batches for s in batch]
    return batches


def img_key(path: str) -> str:
    if not path:
        return ""
    path = path.replace("\\", "/")
    m = re.search(r"mmkb_images/(.+)$", path)
    return m.group(1) if m else os.path.basename(path)


def sample_key(res: dict) -> tuple:
    rel = res["reliability"][0]
    return (rel.get("prompt", "").strip(), rel.get("target", "").strip(), img_key(rel.get("image")))


def row_key(row: dict) -> tuple:
    return (row["src"].strip(), row["alt"].strip(), row["image"].strip())


def analyze(results_path: str, data_path: str, label: str, n: int, seed: int | None):
    flat = flatten_results(results_path)
    all_rows = load_rows(data_path)
    rows = list(all_rows[:n])
    if seed is not None:
        np.random.default_rng(seed).shuffle(rows)
    if len(flat) != len(rows):
        raise ValueError("%s: %d results vs %d rows" % (label, len(flat), len(rows)))

    out = {
        "label": label,
        "results_path": results_path,
        "data_path": data_path,
        "eval_seed_shuffle": seed,
        "n": len(flat),
        "full_leak_swap": [],
        "full_leak_hard_neg": [],
        "partial_leak_swap": [],
        "partial_leak_hard_neg": [],
    }
    for i, (res, row) in enumerate(zip(flat, rows)):
        rel = res["reliability"][0]
        swap = res["generality"]["image_swap"][0]
        hn = res["generality"]["hard_negative"][0]
        gp = row.get("grounding_probes") or {}

        swap_partner = None
        hn_partner = None
        if gp.get("image_swap_partner_index") is not None and 0 <= gp["image_swap_partner_index"] < len(all_rows):
            pr = all_rows[gp["image_swap_partner_index"]]
            swap_partner = {"row": gp["image_swap_partner_index"], "alt": pr["alt"], "pred": pr.get("pred"), "image": pr["image"]}
        if gp.get("hard_neg_partner_index") is not None and 0 <= gp["hard_neg_partner_index"] < len(all_rows):
            pr = all_rows[gp["hard_neg_partner_index"]]
            hn_partner = {"row": gp["hard_neg_partner_index"], "alt": pr["alt"], "pred": pr.get("pred"), "image": pr["image"]}

        entry = {
            "eval_index": i,
            "source_index": row.get("source_index", i),
            "prompt": rel.get("prompt", row["src"]),
            "target": rel.get("target", row["alt"]),
            "pred_prior": row.get("pred"),
            "correct_image": img_key(rel.get("image")) or row["image"],
            "reliability_acc": rel["acc"],
            "reliability_pred": (rel.get("predict_after_edit") or "").strip(),
            "swap_partner": swap_partner,
            "swap_image": gp.get("image_swap"),
            "swap_acc": swap["acc"],
            "swap_pred": (swap.get("predict_after_edit") or "").strip(),
            "hard_neg_partner": hn_partner,
            "hard_neg_image": gp.get("hard_neg_image"),
            "hard_neg_acc": hn["acc"],
            "hard_neg_pred": (hn.get("predict_after_edit") or "").strip(),
            "leak_type_swap": _leak_type(rel, swap),
            "leak_type_hard_neg": _leak_type(rel, hn),
        }
        if swap["acc"] == 1.0:
            out["full_leak_swap"].append(entry)
        elif swap["acc"] > 0:
            out["partial_leak_swap"].append(entry)
        if hn["acc"] == 1.0:
            out["full_leak_hard_neg"].append(entry)
        elif hn["acc"] > 0:
            out["partial_leak_hard_neg"].append(entry)
    return out


def _leak_type(rel: dict, probe: dict) -> str:
    """Classify whether wrong-image output matches this edit's target or something else."""
    target = (rel.get("target") or "").strip().lower()
    pred = (probe.get("predict_after_edit") or "").strip().lower()
    if probe.get("acc", 0) == 0:
        return "no_leak"
    if pred == target or target in pred:
        return "own_target_on_wrong_image"
    return "other_answer_on_wrong_image"


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract wrong-image answer leakage samples")
    ap.add_argument("--results", required=True, help="seed_*_results.json from grounding eval")
    ap.add_argument("--data", required=True, help="grounding_eval_*.json")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("-seed", type=int, default=42, help="Eval shuffle seed (same as test_grounding_eval -seed)")
    ap.add_argument("--label", default="run")
    ap.add_argument("-out", default="doc/results/meeting_validation/step1/grounding_wrong_image_leaks.json")
    args = ap.parse_args()

    out = analyze(args.results, args.data, args.label, args.n, args.seed)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    own_swap = sum(1 for e in out["full_leak_swap"] if e["leak_type_swap"] == "own_target_on_wrong_image")
    own_hn = sum(1 for e in out["full_leak_hard_neg"] if e["leak_type_hard_neg"] == "own_target_on_wrong_image")
    print("Wrote %s" % out_path)
    print("full leak image_swap: %d (own target on wrong img: %d)" % (len(out["full_leak_swap"]), own_swap))
    print("full leak hard_neg:   %d (own target on wrong img: %d)" % (len(out["full_leak_hard_neg"]), own_hn))
    print("partial leak swap:    %d" % len(out["partial_leak_swap"]))
    print("partial leak hard_neg:%d" % len(out["partial_leak_hard_neg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
