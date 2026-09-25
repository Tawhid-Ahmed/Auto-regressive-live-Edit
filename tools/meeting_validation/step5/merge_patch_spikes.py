"""Merge per-spike JSON files into patch_spike_comparison.json.

Usage:
  python tools/meeting_validation/step5/merge_patch_spikes.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.meeting_validation.step5.patch_spike_routing import gate_verdict

SPIKE_FILES = {
    "baseline": "patch_spike_baseline.json",
    "4a_max": "patch_spike_4a.json",
    "4a_attn": "patch_spike_4a_attn.json",
    "4b_rerank": "patch_spike_4b.json",
    "4c_vision_gate": "patch_spike_4c.json",
}


def merge_spike_dir(out_dir: str) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    output_files: Dict[str, str] = {}
    meta: Dict[str, Any] = {"timestamp": datetime.now().isoformat(timespec="seconds")}

    for spike_name, fname in SPIKE_FILES.items():
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        summaries[spike_name] = {
            "summary": data.get("summary", {}),
            "config": data.get("config", {}),
        }
        output_files[spike_name] = fname
        if "meta" in data:
            meta.setdefault("n_samples", data["meta"].get("n_samples"))
            meta.setdefault("seed", 42)
            if spike_name == "baseline":
                meta["baseline_ckpt"] = data["meta"].get("ckpt")
            if spike_name == "4c_vision_gate":
                meta["ar_ckpt"] = data["meta"].get("ckpt")

    meta["spikes_run"] = list(summaries.keys())
    result = {
        "meta": meta,
        "summaries": summaries,
        "gate": gate_verdict(summaries),
        "output_files": output_files,
    }
    out_path = os.path.join(out_dir, "patch_spike_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s (%d spikes)" % (out_path, len(summaries)))
    print("Gate: %s" % result["gate"])
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge Step 5 per-spike JSON files")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="doc/results/meeting_validation/step5",
    )
    args = ap.parse_args()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    merge_spike_dir(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
