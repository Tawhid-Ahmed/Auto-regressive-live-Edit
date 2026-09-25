"""Analyze Step 2 resolution sweep mean_results.json files.

Compares grounding probe metrics across image_size values (224 / 336 / 448).
Reuses Step 1 probe parsing from analyze_grounding_results.py.

Usage:
  python tools/meeting_validation/step2/analyze_resolution_results.py \\
    eval_results/meeting_validation/step2/.../seed_42_mean_results.json \\
    --labels res224 res336 res448 \\
    -out doc/results/meeting_validation/step2/step2_resolution_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.analyze_grounding_results import GROUNDING_PROBES, analyze_mean


def _image_size_from_path(path: str, data: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if data is not None:
        if "image_size" in data:
            return int(data["image_size"])
        meta = data.get("total_mean", data)
        if isinstance(meta, dict) and "image_size" in meta:
            return int(meta["image_size"])
    m = re.search(r"res(\d{3})", path.replace("\\", "/"), re.I)
    if m:
        return int(m.group(1))
    return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_verdict(rows: List[Dict[str, Any]]) -> str:
    """Step 2 gate: flat curves → Q-Former bottleneck; rising → resolution helps pre-patch."""
    if len(rows) < 2:
        return "INSUFFICIENT_DATA"

    primary = ["swap_gap_pp", "hard_negative_false_activation_rate"]
    rel_vals = [r.get("reliability_acc") for r in rows if r.get("reliability_acc") is not None]
    swap_vals = [r.get("probe_acc", {}).get("image_swap") for r in rows if r.get("probe_acc", {}).get("image_swap") is not None]

    def _spread(key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            return None
        return max(vals) - min(vals)

    swap_gap_spread = _spread("swap_gap_pp")
    hn_spread_pp = None
    hn = [r.get("hard_negative_false_activation_rate") for r in rows if r.get("hard_negative_false_activation_rate") is not None]
    if len(hn) >= 2:
        hn_spread_pp = (max(hn) - min(hn)) * 100

    rel_spread_pp = None
    if len(rel_vals) >= 2:
        rel_spread_pp = (max(rel_vals) - min(rel_vals)) * 100

    flat_threshold_pp = 2.0
    if swap_gap_spread is not None and swap_gap_spread < flat_threshold_pp:
        if hn_spread_pp is not None and hn_spread_pp < flat_threshold_pp:
            return (
                "PASS (flat curves - resolution irrelevant until patch layer; "
                "supports Q-Former bottleneck)"
            )
    if rel_spread_pp is not None and rel_spread_pp >= 3.0:
        return "REFRAME (resolution moves reliability ≥3 pp — Q-Former may already benefit)"
    if swap_gap_spread is not None and swap_gap_spread >= 3.0:
        return "REFRAME (swap-gap varies ≥3 pp across resolution — partial pre-Q-Former signal)"
    return "PASS (weak resolution effect — proceed to Step 3 encoder probe)"


def discover_mean_results(eval_dir: str) -> List[str]:
    pattern = os.path.join(eval_dir, "**", "seed_*_mean_results.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        pattern = os.path.join(eval_dir, "**", "mean_results.json")
        paths = sorted(glob.glob(pattern, recursive=True))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Step 2 resolution sweep results")
    ap.add_argument("paths", nargs="*", help="mean_results.json paths (or omit with --eval-dir)")
    ap.add_argument("--eval-dir", type=str, default=None, help="Auto-discover runs under step2 eval folder")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("-out", "--output_json", type=str, default=None)
    args = ap.parse_args()

    paths = list(args.paths or [])
    if args.eval_dir:
        eval_dir = args.eval_dir if os.path.isabs(args.eval_dir) else os.path.join(ROOT, args.eval_dir)
        paths.extend(discover_mean_results(eval_dir))
    paths = sorted(set(paths))
    if not paths:
        ap.error("No mean_results.json files found")

    labels = args.labels
    if labels is None:
        labels = []
        for p in paths:
            raw = _load_json(p)
            sz = _image_size_from_path(p, raw)
            labels.append(f"res{sz}" if sz else os.path.basename(os.path.dirname(os.path.dirname(p))))

    rows = []
    for p, label in zip(paths, labels):
        row = analyze_mean(p, label)
        raw = _load_json(p)
        sz = _image_size_from_path(p, raw)
        row["image_size"] = sz
        rows.append(row)

    rows.sort(key=lambda r: (r.get("image_size") is None, r.get("image_size") or 0))
    verdict = gate_verdict(rows)
    print("=" * 60)
    print(f"Step 2 gate: {verdict}")

    summary = {
        "runs": rows,
        "gate": verdict,
        "resolution_spread_pp": {
            "reliability": None,
            "swap_gap": None,
            "hard_negative": None,
        },
    }
    rel = [r["reliability_acc"] for r in rows if r.get("reliability_acc") is not None]
    sg = [r["swap_gap_pp"] for r in rows if r.get("swap_gap_pp") is not None]
    hn = [r["hard_negative_false_activation_rate"] for r in rows if r.get("hard_negative_false_activation_rate") is not None]
    if len(rel) >= 2:
        summary["resolution_spread_pp"]["reliability"] = round((max(rel) - min(rel)) * 100, 2)
    if len(sg) >= 2:
        summary["resolution_spread_pp"]["swap_gap"] = round(max(sg) - min(sg), 2)
    if len(hn) >= 2:
        summary["resolution_spread_pp"]["hard_negative"] = round((max(hn) - min(hn)) * 100, 2)

    out_path = args.output_json or "doc/results/meeting_validation/step2/step2_resolution_summary.json"
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
