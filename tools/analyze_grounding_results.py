"""Analyze Step 1 grounding eval mean_results.json from test_vllm_edit runs.

Computes swap-gap, hard-negative false activation, modality consistency,
and prompt-conflict accuracy. Compare baseline vs AR when two paths given.

Usage:
  python tools/analyze_grounding_results.py \\
    eval_results/.../seed_42_mean_results.json \\
    --label baseline

  python tools/analyze_grounding_results.py baseline.json ar.json --labels baseline ar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GROUNDING_PROBES = (
    "image_swap",
    "hard_negative",
    "modality_consistency",
    "prompt_conflict",
)


def _load_mean(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "total_mean" in data:
        return data["total_mean"]
    return data


def _acc(section: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    cur = section
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return float(cur) if cur is not None else default


def analyze_mean(path: str, label: str) -> Dict[str, Any]:
    mean = _load_mean(path)
    rel_acc = _acc(mean, "reliability", "acc")
    gen = mean.get("generality", {})

    probe_acc = {}
    for probe in GROUNDING_PROBES:
        probe_acc[probe] = _acc(gen, probe, "acc")

    swap_gap = None
    if rel_acc is not None and probe_acc.get("image_swap") is not None:
        swap_gap = rel_acc - probe_acc["image_swap"]

    consistency_gap = None
    if rel_acc is not None and probe_acc.get("modality_consistency") is not None:
        consistency_gap = rel_acc - probe_acc["modality_consistency"]

    out = {
        "label": label,
        "path": path,
        "reliability_acc": rel_acc,
        "probe_acc": probe_acc,
        "swap_gap_pp": swap_gap * 100 if swap_gap is not None else None,
        "consistency_gap_pp": consistency_gap * 100 if consistency_gap is not None else None,
        "hard_negative_false_activation_rate": probe_acc.get("hard_negative"),
        "conflict_accuracy": probe_acc.get("prompt_conflict"),
        "modality_consistency_acc": probe_acc.get("modality_consistency"),
    }

    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  file: {path}")
    if rel_acc is not None:
        print(f"  reliability acc: {rel_acc * 100:.2f}%")
    for probe in GROUNDING_PROBES:
        v = probe_acc.get(probe)
        if v is not None:
            print(f"  {probe} acc: {v * 100:.2f}%")
    if swap_gap is not None:
        print(f"  swap-gap (rel - swap): {swap_gap * 100:.2f} pp")
    if consistency_gap is not None:
        print(f"  consistency-gap (rel - consistency): {consistency_gap * 100:.2f} pp")
    if probe_acc.get("hard_negative") is not None:
        print(f"  hard-negative false activation proxy: {probe_acc['hard_negative'] * 100:.2f}%")
    if probe_acc.get("prompt_conflict") is not None:
        print(f"  prompt-conflict accuracy: {probe_acc['prompt_conflict'] * 100:.2f}%")
    return out


def gate_verdict(rows: List[Dict[str, Any]]) -> str:
    """Step 1 gate: strong GO if baseline fails >=2 probes."""
    baseline = rows[0]
    rel = baseline.get("reliability_acc")
    if rel is None:
        return "INSUFFICIENT_DATA"

    failures = 0
    swap_gap = baseline.get("swap_gap_pp")
    if swap_gap is not None and swap_gap < 5:
        failures += 1
    hn = baseline.get("hard_negative_false_activation_rate")
    if hn is not None and hn > 0.5:
        failures += 1
    mc = baseline.get("modality_consistency_acc")
    if mc is not None and (rel - mc) > 0.03:
        failures += 1
    conflict = baseline.get("conflict_accuracy")
    if conflict is not None and conflict < rel - 0.05:
        failures += 1

    if failures >= 2:
        return "PASS (strong GO — baseline fails >=2 probes)"
    if failures == 1:
        return "REFRAME (weak signal — 1 probe failure; proceed Steps 2–3)"
    return "REFRAME (probes pass — resolution/encoder may explain; still run Steps 2–3)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Step 1 grounding eval results")
    ap.add_argument("paths", nargs="+", help="mean_results.json paths")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("-out", "--output_json", type=str, default=None)
    args = ap.parse_args()

    labels = args.labels or [
        os.path.basename(os.path.dirname(os.path.dirname(p))) for p in args.paths
    ]
    rows = [analyze_mean(p, label) for p, label in zip(args.paths, labels)]
    verdict = gate_verdict(rows)
    print("=" * 60)
    print(f"Gate: {verdict}")

    if args.output_json:
        out_path = args.output_json
        if not os.path.isabs(out_path):
            out_path = os.path.join(ROOT, out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"runs": rows, "gate": verdict}, f, indent=2)
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
