"""Summarize Step 5 patch spike comparison JSON.

Usage:
  python tools/meeting_validation/step5/analyze_patch_spikes.py \\
    doc/results/meeting_validation/step5/patch_spike_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

SPIKE_LABELS = {
    "baseline": "Baseline (mean-pool)",
    "4a_max": "4a — max pool",
    "4a_attn": "4a — attn pool",
    "4b_rerank": "4b — patch rerank",
    "4c_vision_gate": "4c — vision-gated chunks",
}


def print_summary(data: Dict[str, Any]) -> None:
    meta = data.get("meta", {})
    print("=" * 60)
    print("Step 5 patch spike summary")
    print("=" * 60)
    print("  samples: %s  seed: %s" % (meta.get("n_samples"), meta.get("seed")))
    print("  baseline ckpt: %s" % meta.get("baseline_ckpt"))
    if meta.get("ar_ckpt"):
        print("  AR ckpt (4c): %s" % meta.get("ar_ckpt"))
    print("")
    print("| Spike | Hard-neg false-fire | Rel false-cross | Rel miss |")
    print("|-------|--------------------:|----------------:|---------:|")
    for spike, block in data.get("summaries", {}).items():
        s = block.get("summary", {})
        ff = s.get("hard_negative_false_activation_rate")
        fc = s.get("reliability_false_cross_rate")
        miss = s.get("reliability_miss_rate")
        label = SPIKE_LABELS.get(spike, spike)
        print(
            "| %s | %s | %s | %s |"
            % (
                label[:28],
                "%.1f%%" % (ff * 100) if ff is not None else "—",
                "%.1f%%" % (fc * 100) if fc is not None else "—",
                "%.1f%%" % (miss * 100) if miss is not None else "—",
            )
        )
    print("=" * 60)
    print("Gate: %s" % data.get("gate", "N/A"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Step 5 patch spike JSON")
    ap.add_argument(
        "comparison_json",
        nargs="?",
        default="doc/results/meeting_validation/step5/patch_spike_comparison.json",
    )
    args = ap.parse_args()

    path = args.comparison_json if os.path.isabs(args.comparison_json) else os.path.join(ROOT, args.comparison_json)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print_summary(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
