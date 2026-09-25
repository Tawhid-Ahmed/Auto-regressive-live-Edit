"""Summarize Step 4 encoder routing scores JSON.

Usage:
  python tools/meeting_validation/step4/analyze_encoder_routing.py \\
    doc/results/meeting_validation/step4/encoder_routing_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def print_summary(data: Dict[str, Any]) -> None:
    meta = data.get("meta", {})
    print("=" * 60)
    print("Step 4 encoder routing summary")
    print("=" * 60)
    print("  samples: %s  pool entries: %s  seed: %s" % (
        meta.get("n_samples"), meta.get("n_pool_entries"), meta.get("seed")))
    print("  ckpt: %s" % meta.get("ckpt"))
    print("")
    for enc, summ in data.get("summaries", {}).items():
        ff = summ.get("hard_neg_false_fire_rate")
        auc = summ.get("hard_neg_auc_margin") or summ.get("hard_neg_auc_own_expert")
        miss = summ.get("reliability_miss_rate")
        print("  [%s]" % enc)
        if ff is not None:
            print("    hard-neg false-fire: %.2f%%" % (ff * 100))
        if auc is not None:
            print("    hard-neg AUC: %.4f" % auc)
        if miss is not None:
            print("    reliability miss: %.2f%%" % (miss * 100))
    print("=" * 60)
    print("Gate: %s" % data.get("gate", "N/A"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Step 4 encoder routing JSON")
    ap.add_argument("scores_json", nargs="?", default="doc/results/meeting_validation/step4/encoder_routing_scores.json")
    args = ap.parse_args()

    path = args.scores_json if os.path.isabs(args.scores_json) else os.path.join(ROOT, args.scores_json)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print_summary(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
