"""Analyze routing diagnostic JSON from Step 0."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List


def analyze(path: str, label: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s = d["summary"]
    probes = d["probes"]
    pool_ids: List[int] = d["pool_sample_ids"]

    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  n_samples: {s['n_samples']}, n_pool_entries: {s['n_pool_entries']}, ar_mode: {s.get('ar_mode')}")
    for k, v in s.items():
        if k.endswith("_rate"):
            print(f"  {k}: {v * 100:.2f}%")
    print(f"  mean_n_retrieved: {s['mean_n_retrieved']}")

    buckets = {}
    for probe_name in ("reliability", "text_rephrase", "image_rephrase", "image_loc"):
        rows = [p for p in probes if p["probe"] == probe_name]
        only_correct = correct_and_wrong = only_wrong = none = 0
        for p in rows:
            sid = p["sample_id"]
            expected = {i for i, x in enumerate(pool_ids) if x == sid}
            got = set(p["retrieved_indices"])
            if not got:
                none += 1
            elif got <= expected:
                only_correct += 1
            elif got & expected:
                correct_and_wrong += 1
            else:
                only_wrong += 1
        n = len(rows)
        buckets[probe_name] = {
            "only_correct_pct": 100 * only_correct / n,
            "correct_and_wrong_pct": 100 * correct_and_wrong / n,
            "only_wrong_pct": 100 * only_wrong / n,
            "none_pct": 100 * none / n,
        }
        print(f"  [{probe_name}] only_correct={only_correct}/{n} ({100*only_correct/n:.1f}%)")
        print(f"  [{probe_name}] correct+wrong={correct_and_wrong}/{n} ({100*correct_and_wrong/n:.1f}%)")
        print(f"  [{probe_name}] only_wrong={only_wrong}/{n} ({100*only_wrong/n:.1f}%)")
        print(f"  [{probe_name}] none_retrieved={none}/{n} ({100*none/n:.1f}%)")
    return {"summary": s, "buckets": buckets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSON paths")
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()
    labels = args.labels or [p.split("/")[-1].replace(".json", "") for p in args.paths]
    for path, label in zip(args.paths, labels):
        analyze(path, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
