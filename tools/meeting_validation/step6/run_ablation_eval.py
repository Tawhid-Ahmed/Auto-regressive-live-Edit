"""Run Step 6 per-layer ablation evals on grounding probe suite.

Usage:
  python tools/meeting_validation/step6/run_ablation_eval.py \\
    -ckpt records/meeting_validation/step6/.../epoch-20-... \\
    -dvc cuda:0 -dsn 500 -seed 42

  # L0 only on legacy baseline ckpt:
  python tools/meeting_validation/step6/run_ablation_eval.py \\
    --layers L0 -ckpt records/liveedit/.../epoch-20-...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.meeting_validation.step6.layer_ablation import LAYER_NAMES, AblationConfig

DEFAULT_BASELINE_CKPT = (
    "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/"
    "checkpoints/epoch-20-i-27800-ema_loss-0.7936"
)


def _latest_mean(eval_root: str, tag: str) -> Optional[str]:
    pattern = os.path.join(
        eval_root, "liveedit", "blip2-opt-2.7b", f"VLKEB_GROUNDING-{tag}-*",
        "sequential_edit_*", "seed_*_mean_results.json",
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6 layer ablation grounding eval")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=500)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("-sen", "--sequential_edit_n", type=int, default=50)
    ap.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=list(LAYER_NAMES),
        help="Ablation layers to eval (default: L0..L5)",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    eval_root = os.path.join(ROOT, "eval_results", "meeting_validation", "step6")
    manifest = {
        "started": datetime.now().isoformat(),
        "ckpt": args.editor_ckpt_path,
        "layers": args.layers,
        "runs": {},
    }

    if not args.analyze_only:
        for layer in args.layers:
            cfg = AblationConfig.for_layer(layer)
            tag = cfg.eval_name_postfix()
            if args.skip_existing and _latest_mean(eval_root, tag):
                print("[skip] %s" % tag)
                manifest["runs"][layer] = {"tag": tag, "mean_results": _latest_mean(eval_root, tag)}
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
                "--meeting_step", "6",
                "--ablation_layer", layer,
            ]
            if cfg.ar_mode:
                cmd.append("--ar_mode")
                cmd.extend(["--chunk_size", str(cfg.chunk_size)])
            print("Running:", " ".join(cmd))
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                return rc
            manifest["runs"][layer] = {
                "tag": tag,
                "mean_results": _latest_mean(eval_root, tag),
                "config": cfg.to_dict(),
            }

    manifest["finished"] = datetime.now().isoformat()
    manifest_path = os.path.join(
        ROOT, "doc", "results", "meeting_validation", "step6", "step6_run_manifest.json"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    cmd = [
        sys.executable,
        os.path.join("tools", "meeting_validation", "step6", "analyze_ablation_results.py"),
        "-manifest", manifest_path,
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc == 0:
        plot_cmd = [
            sys.executable,
            os.path.join("tools", "meeting_validation", "step6", "plot_ablation_table.py"),
        ]
        subprocess.call(plot_cmd, cwd=ROOT)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
