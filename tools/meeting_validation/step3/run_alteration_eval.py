"""Run Step 3 image-alteration eval + analyze + plot.

Usage:
  python tools/meeting_validation/step3/run_alteration_eval.py -dvc cuda:0 -dsn 500 -seed 42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

DEFAULT_CKPT = (
    "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/"
    "checkpoints/epoch-20-i-27800-ema_loss-0.7936"
)


def _latest_mean(eval_root: str, tag: str) -> str | None:
    pattern = os.path.join(
        eval_root, "liveedit", "blip2-opt-2.7b", f"VLKEB_ALTERATION-{tag}-*",
        "sequential_edit_*", "seed_*_mean_results.json",
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Step 3 image alteration eval")
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=500)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("-sen", "--sequential_edit_n", type=int, default=50)
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    eval_root = os.path.join(ROOT, "eval_results", "meeting_validation", "step3")
    tag = "step3_alteration_baseline"
    manifest_path = os.path.join(
        ROOT, "doc", "results", "meeting_validation", "step3", "step3_run_manifest.json"
    )
    manifest = {"started": datetime.now().isoformat(), "tag": tag}

    if not args.analyze_only:
        if args.skip_existing and _latest_mean(eval_root, tag):
            print("[skip] existing results for %s" % tag)
        else:
            cmd = [
                sys.executable, "-W", "ignore", "test_alteration_eval.py",
                "-en", "liveedit", "-mn", "blip2",
                "-sen", str(args.sequential_edit_n),
                "-dvc", args.device,
                "-dsn", str(args.data_sample_n),
                "-seed", str(args.eval_seed),
                "-ckpt", args.editor_ckpt_path,
                "-enp", tag,
            ]
            print("Running:", " ".join(cmd))
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                return rc
        manifest["mean_results"] = _latest_mean(eval_root, tag)

    manifest["finished"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    mean_path = manifest.get("mean_results") or _latest_mean(eval_root, tag)
    if not mean_path:
        print("ERROR: no mean_results found")
        return 1

    for script, extra in (
        ("analyze_alteration_results.py", [mean_path, "--label", "baseline"]),
        ("plot_alteration_results.py", []),
    ):
        cmd = [sys.executable, os.path.join("tools", "meeting_validation", "step3", script)]
        cmd.extend(extra)
        if script.startswith("plot"):
            cmd.extend([
                "--analysis-json",
                "doc/results/meeting_validation/step3/step3_alteration_summary.json",
            ])
        elif script.startswith("analyze"):
            cmd.extend([
                "-out", "doc/results/meeting_validation/step3/step3_alteration_summary.json",
            ])
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
