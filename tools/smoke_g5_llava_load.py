"""G5 smoke: load LLaVA-1.5-7B + LiveEdit L0 ckpt, one edit + probe_routing.

Usage (repo root):
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/smoke_g5_llava_load.py
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/smoke_g5_llava_load.py --no_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CKPT = (
    ROOT / "records/liveedit/llava-v1.5-7b/"
    "VLKEB_llava_L0_dn500-2026.07.25-19.25.39/checkpoints/epoch-3-i-1900-ema_loss-8.1575"
)
OUT_DEFAULT = ROOT / "doc/results/cvpr2027/g5/smoke_llava_load.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="G5 LLaVA load smoke")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--no_ckpt", action="store_true", help="Load backbone only (no LiveEdit ckpt)")
    ap.add_argument("-out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    import torch
    from utils import get_full_model_name, get_model_path, load_vllm_editor
    from dataset.vllm_grounding import GroundingVLKEB

    mn = get_full_model_name("llava")
    mpath = get_model_path("llava")
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": mn,
        "model_path": str(mpath).replace("\\", "/"),
        "model_path_exists": os.path.isdir(mpath),
        "device": args.dvc,
        "cuda_available": torch.cuda.is_available(),
        "ckpt": None if args.no_ckpt else args.ckpt,
        "steps": [],
    }
    if not rec["model_path_exists"]:
        rec["error"] = "LLaVA weights missing"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print("FAIL: weights missing at", mpath)
        return 1

    print("Loading LLaVA LiveEdit…", flush=True)
    t0 = time.time()
    ckpt = None if args.no_ckpt else args.ckpt
    editor = load_vllm_editor("liveedit", mn, args.dvc, None, ckpt, False)
    rec["steps"].append({"load_editor_sec": round(time.time() - t0, 1)})
    if torch.cuda.is_available():
        rec["steps"].append({
            "cuda_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        })
        print("CUDA alloc/reserved GB:", rec["steps"][-1], flush=True)

    dpath = ROOT / "data/VLKEB/meeting_validation/step1/grounding/pair_closed/grounding_pairclosed_n20_s42.json"
    img = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
    data = GroundingVLKEB(str(dpath), str(img), 2, sample_seed=42)
    sample = data.data_with_img[0]
    t1 = time.time()
    with torch.no_grad():
        editor.restore_to_original_model()
        editor.edit_one_piece(sample["requests"][0])
        routing = editor.probe_routing(
            editor.vllm,
            sample["requests"][0]["prompt"],
            sample["requests"][0]["image"],
            target=sample["requests"][0]["target_new"],
        )
    rec["steps"].append({
        "edit_probe_sec": round(time.time() - t1, 1),
        "n_retrieved": int(routing.get("n_retrieved", len(routing.get("retrieved_indices") or []))),
        "retrieved_indices": list(routing.get("retrieved_indices") or []),
    })
    if torch.cuda.is_available():
        rec["steps"].append({
            "cuda_alloc_gb_after": round(torch.cuda.memory_allocated() / 1e9, 3),
            "cuda_reserved_gb_after": round(torch.cuda.memory_reserved() / 1e9, 3),
        })
    rec["ok"] = True
    rec["total_sec"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("OK wrote", out, "total_sec=", rec["total_sec"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
