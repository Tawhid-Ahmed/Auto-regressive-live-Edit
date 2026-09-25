"""Analyze Step 3 image-alteration eval mean_results.json.

Usage:
  python tools/meeting_validation/step3/analyze_alteration_results.py \\
    eval_results/meeting_validation/step3/.../seed_42_mean_results.json \\
    --label baseline -out doc/results/meeting_validation/step3/step3_alteration_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from dataset.image_alterations import ALTERATION_LABELS, ALTERATION_NAMES

ALTERATION_PROBES = tuple("alter_%s" % k for k in ALTERATION_NAMES)


def _load_mean(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "total_mean" in data:
        return data["total_mean"]
    return data


def _acc(section: Dict[str, Any], *keys: str) -> Optional[float]:
    cur = section
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return float(cur) if cur is not None else None


def analyze_mean(path: str, label: str) -> Dict[str, Any]:
    mean = _load_mean(path)
    rel_acc = _acc(mean, "reliability", "acc")
    gen = mean.get("generality", {})

    probe_acc = {}
    for probe in ALTERATION_PROBES:
        probe_acc[probe] = _acc(gen, probe, "acc")

    image_rephrase = _acc(gen, "image_rephrase", "acc")
    gaps_pp = {}
    if rel_acc is not None:
        for probe in ALTERATION_PROBES:
            v = probe_acc.get(probe)
            if v is not None:
                gaps_pp[probe] = (rel_acc - v) * 100

    out = {
        "label": label,
        "path": path,
        "reliability_acc": rel_acc,
        "image_rephrase_acc": image_rephrase,
        "probe_acc": probe_acc,
        "alteration_gap_pp": gaps_pp,
        "mean_alteration_gap_pp": (
            sum(gaps_pp.values()) / len(gaps_pp) if gaps_pp else None
        ),
    }

    print("=" * 60)
    print(label)
    print("=" * 60)
    print("  file: %s" % path)
    if rel_acc is not None:
        print("  reliability acc: %.2f%%" % (rel_acc * 100))
    if image_rephrase is not None:
        print("  image rephrase acc: %.2f%%" % (image_rephrase * 100))
    for probe in ALTERATION_PROBES:
        v = probe_acc.get(probe)
        alt_key = probe.replace("alter_", "", 1)
        name = ALTERATION_LABELS.get(alt_key, probe)
        if v is not None:
            gap = gaps_pp.get(probe)
            gap_s = " (gap %.2f pp)" % gap if gap is not None else ""
            print("  %s: %.2f%%%s" % (name, v * 100, gap_s))
    return out


def gate_verdict(row: Dict[str, Any]) -> str:
    rel = row.get("reliability_acc")
    gaps = row.get("alteration_gap_pp") or {}
    if rel is None or not gaps:
        return "INSUFFICIENT_DATA"

    vals = list(gaps.values())
    mean_gap = sum(vals) / len(vals)
    max_gap = max(vals)
    min_gap = min(vals)

    if mean_gap >= 5.0 and max_gap >= 8.0:
        return (
            "PASS (edits fragile to appearance change - mean gap %.1f pp, max %.1f pp; "
            "supports pose/view-invariant routing need)"
            % (mean_gap, max_gap)
        )
    if mean_gap >= 3.0:
        return "REFRAME (moderate alteration sensitivity - mean gap %.1f pp)" % mean_gap
    if max_gap - min_gap < 2.0:
        return "REFRAME (flat across alteration types - similar to Step 2 resolution invariance)"
    return "REFRAME (mixed alteration profile - inspect per-probe before Step 4 encoder)"


def discover_mean_results(eval_dir: str) -> List[str]:
    pattern = os.path.join(eval_dir, "**", "seed_*_mean_results.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        pattern = os.path.join(eval_dir, "**", "mean_results.json")
        paths = sorted(glob.glob(pattern, recursive=True))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Step 3 alteration eval results")
    ap.add_argument("paths", nargs="*", help="mean_results.json paths")
    ap.add_argument("--eval-dir", type=str, default=None)
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

    labels = args.labels or [
        os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths
    ]
    rows = [analyze_mean(p, label) for p, label in zip(paths, labels)]
    verdict = gate_verdict(rows[0]) if len(rows) == 1 else gate_verdict(rows[0])
    print("=" * 60)
    print("Step 3 gate: %s" % verdict)

    summary = {"runs": rows, "gate": verdict, "alteration_probes": list(ALTERATION_PROBES)}
    out_path = args.output_json or "doc/results/meeting_validation/step3/step3_alteration_summary.json"
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
