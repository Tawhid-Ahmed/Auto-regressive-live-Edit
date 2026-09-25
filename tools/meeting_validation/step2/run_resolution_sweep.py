"""Run Step 2 resolution sweep (224 / 336 / 448) via test_grounding_eval.py.

Usage:
  python tools/meeting_validation/step2/run_resolution_sweep.py -dvc cuda:0 -dsn 500 -seed 42

After all runs finish:
  python tools/meeting_validation/step2/analyze_resolution_results.py --eval-dir eval_results/meeting_validation/step2
  python tools/meeting_validation/step2/plot_resolution_results.py --analysis-json doc/results/meeting_validation/step2/step2_resolution_summary.json
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
RESOLUTIONS = [224, 336, 448]


def _latest_mean(eval_root: str, tag: str) -> str | None:
    pattern = os.path.join(
        eval_root, "liveedit", "blip2-opt-2.7b", f"VLKEB_GROUNDING-{tag}-*",
        "sequential_edit_*", "seed_*_mean_results.json",
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Step 2 image resolution sweep")
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=500)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("-sen", "--sequential_edit_n", type=int, default=50)
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=DEFAULT_CKPT)
    ap.add_argument(
        "--resolutions", type=int, nargs="+", default=RESOLUTIONS,
        help="Image sizes to sweep (default: 224 336 448)",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip resolution if mean_results exists")
    ap.add_argument("--analyze-only", action="store_true", help="Skip GPU runs; analyze/plot existing results")
    args = ap.parse_args()

    eval_root = os.path.join(ROOT, "eval_results", "meeting_validation", "step2")
    os.makedirs(eval_root, exist_ok=True)
    manifest_path = os.path.join(ROOT, "doc", "results", "meeting_validation", "step2", "step2_run_manifest.json")
    manifest = {"started": datetime.now().isoformat(), "runs": []}

    if not args.analyze_only:
        for size in args.resolutions:
            tag = f"step2_res{size}"
            if args.skip_existing and _latest_mean(eval_root, tag):
                print(f"[skip] {tag} — existing results found")
                manifest["runs"].append({"image_size": size, "tag": tag, "skipped": True})
                continue
            cmd = [
                sys.executable, "-W", "ignore", "test_grounding_eval.py",
                "-en", "liveedit", "-mn", "blip2",
                "-sen", str(args.sequential_edit_n),
                "-dvc", args.device,
                "-dsn", str(args.data_sample_n),
                "-seed", str(args.eval_seed),
                "-ckpt", args.editor_ckpt_path,
                "-enp", tag,
                "--meeting_step", "2",
                "--image_size", str(size),
            ]
            print("=" * 60)
            print("Running:", " ".join(cmd))
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                print(f"ERROR: run failed for image_size={size} (exit {rc})")
                return rc
            mean_path = _latest_mean(eval_root, tag)
            manifest["runs"].append({
                "image_size": size,
                "tag": tag,
                "mean_results": mean_path,
                "skipped": False,
            })

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest["finished"] = datetime.now().isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest {manifest_path}")

    analyze_cmd = [
        sys.executable,
        "tools/meeting_validation/step2/analyze_resolution_results.py",
        "--eval-dir", "eval_results/meeting_validation/step2",
        "-out", "doc/results/meeting_validation/step2/step2_resolution_summary.json",
    ]
    plot_cmd = [
        sys.executable,
        "tools/meeting_validation/step2/plot_resolution_results.py",
        "--analysis-json", "doc/results/meeting_validation/step2/step2_resolution_summary.json",
    ]
    for cmd in (analyze_cmd, plot_cmd):
        print("=" * 60)
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
