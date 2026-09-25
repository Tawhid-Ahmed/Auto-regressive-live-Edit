"""Train PatchGrounding LiveEdit on VLKEB-long (Step 6).

Checkpoints use the same layout as baseline / AR long-form runs:
  records/liveedit/blip2-opt-2.7b/VLKEB_long_patch-YYYY.MM.DD-HH.MM.SS/checkpoints/

Usage:
  python tools/meeting_validation/step6/run_train_patch.py -dvc cuda:0 -eps 20
  python tools/meeting_validation/step6/run_train_patch.py -dvc cuda:0 -eps 20 --ar_mode
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import runpy
import sys
import warnings
from datetime import datetime
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

DEFAULT_TRAIN_JSON = os.path.join(
    ROOT, "..", "vlkeb_long_ollama", "out", "train_longform_ollama.json"
)
RECORDS_ROOT = "records"
MODEL_SUBDIR = os.path.join("liveedit", "blip2-opt-2.7b")
TRAIN_SCRIPT = os.path.join(ROOT, "train_vllm_editor.py")


def _resolve_train_prefix(ar_mode: bool, override: Optional[str]) -> str:
    if override:
        return override
    return "VLKEB_long_patch_ar" if ar_mode else "VLKEB_long_patch"


def _find_latest_run_dir(prefix: str) -> Optional[str]:
    parent = os.path.join(ROOT, RECORDS_ROOT, MODEL_SUBDIR)
    if not os.path.isdir(parent):
        return None
    matches = sorted(
        d for d in glob.glob(os.path.join(parent, prefix + "-*"))
        if os.path.isdir(d)
    )
    return matches[-1] if matches else None


def _build_train_argv(args: argparse.Namespace, train_json: str, train_prefix: str) -> list[str]:
    cmd = [
        TRAIN_SCRIPT,
        "-en", "liveedit", "-mn", "blip2", "-dna", "VLKEB",
        "-bs", str(args.batch_size),
        "-dvc", args.device,
        "-edvc", str(args.extra_device),
        "-lkpt", args.load_ckpt_path,
        "-eps", str(args.epochs),
        "-tnp", train_prefix,
        "-sci", str(args.save_ckpt_per_i),
        "-lpi", str(args.log_per_i),
        "-dpath", train_json,
        "--patch_grounding",
        "--patch_encoder", args.patch_encoder,
    ]
    if args.ar_mode:
        cmd.extend(["--ar_mode", "--chunk_size", str(args.chunk_size)])
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6 patch-grounding training launcher")
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-edvc", "--extra_device", type=int, default=0)
    ap.add_argument("-bs", "--batch_size", type=int, default=4)
    ap.add_argument("-eps", "--epochs", type=int, default=20)
    ap.add_argument("-lkpt", "--load_ckpt_path", type=str, default="none")
    ap.add_argument("-tnp", "--train_name_prefix", type=str, default=None,
                    help="Default: VLKEB_long_patch or VLKEB_long_patch_ar with --ar_mode")
    ap.add_argument("-dpath", "--train_json", type=str, default=DEFAULT_TRAIN_JSON)
    ap.add_argument("--patch_encoder", type=str, default="dinov2", choices=["dinov2", "eva_patches"])
    ap.add_argument("--ar_mode", action="store_true", help="Train with AR chunks (L5 stack)")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("-sci", "--save_ckpt_per_i", type=int, default=500,
                    help="Checkpoint interval (matches VLKEB long baseline / AR runs)")
    ap.add_argument("-lpi", "--log_per_i", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    train_json = args.train_json
    if not os.path.isabs(train_json):
        train_json = os.path.normpath(os.path.join(ROOT, train_json))

    train_prefix = _resolve_train_prefix(args.ar_mode, args.train_name_prefix)
    started_at = datetime.now()
    timeline = started_at.strftime("%Y.%m.%d-%H.%M.%S")
    expected_run_name = "%s-%s" % (train_prefix, timeline)

    manifest = {
        "started": started_at.isoformat(),
        "train_json": train_json,
        "patch_encoder": args.patch_encoder,
        "ar_mode": args.ar_mode,
        "chunk_size": args.chunk_size if args.ar_mode else None,
        "records_root": RECORDS_ROOT,
        "train_name_prefix": train_prefix,
        "expected_run_name": expected_run_name,
        "expected_ckpt_dir": os.path.join(RECORDS_ROOT, MODEL_SUBDIR, expected_run_name, "checkpoints"),
    }
    manifest_path = os.path.join(
        ROOT, "doc", "results", "meeting_validation", "step6", "step6_train_manifest.json"
    )

    cmd = _build_train_argv(args, train_json, train_prefix)
    print("Command:", " ".join('"%s"' % a if " " in a else a for a in cmd))
    print("Expected run dir:", os.path.join(RECORDS_ROOT, MODEL_SUBDIR, "<%s>-<timestamp>" % train_prefix))
    print(
        "\nNOTE: BLIP2 + DINOv2 load can sit at 'Loading checkpoint shards: 0%%' for ~30-60s.\n"
        "      Do not press Ctrl+C during that phase; training starts once loading finishes.\n"
    )
    if args.dry_run:
        return 0

    old_argv = sys.argv
    sys.argv = cmd
    warnings.filterwarnings("ignore")
    rc = 0
    try:
        runpy.run_path(TRAIN_SCRIPT, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        rc = code if isinstance(code, int) else (0 if code is None else 1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Training stopped by user (Ctrl+C).")
        rc = 130
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        sys.argv = old_argv

    if rc != 0:
        print("\n[ERROR] Training failed with exit code %s." % rc)

    manifest["finished"] = datetime.now().isoformat()
    manifest["exit_code"] = rc
    latest_run = _find_latest_run_dir(train_prefix)
    if latest_run:
        manifest["run_dir"] = os.path.relpath(latest_run, ROOT).replace("\\", "/")
        manifest["checkpoints_dir"] = os.path.join(manifest["run_dir"], "checkpoints")
        runtime_path = os.path.join(latest_run, "runtime.json")
        if os.path.isfile(runtime_path):
            with open(runtime_path, encoding="utf-8") as f:
                manifest["runtime"] = json.load(f)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
