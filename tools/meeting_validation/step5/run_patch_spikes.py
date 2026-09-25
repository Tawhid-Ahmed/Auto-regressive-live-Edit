"""Step 5: run patch + cross-attention inference spikes (4a / 4b / 4c).

Builds a sequential edit pool once per spike config, then probes grounding
routing (hard-negative false activation, reliability false-cross).

Usage (from repo root):
  python tools/meeting_validation/step5/run_patch_spikes.py \\
    -ckpt "records/liveedit/blip2-opt-2.7b/.../epoch-20-..." \\
    -dvc cuda:0 -dsn 50 -seed 42

  # AR ckpt for 4c vision-gated chunks:
  python tools/meeting_validation/step5/run_patch_spikes.py \\
    -ckpt "records/.../VLKEB_long_ar-.../epoch-20-..." --ar_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

from tools.meeting_validation.step5.patch_spike_routing import (
    SPIKE_NAMES,
    SpikeConfig,
    apply_spike_config,
    build_edit_pool,
    gate_verdict,
    run_spike_diagnostics,
)

DEFAULT_BASELINE_CKPT = (
    "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/"
    "checkpoints/epoch-20-i-27800-ema_loss-0.7936"
)
DEFAULT_AR_CKPT = (
    "records/liveedit/blip2-opt-2.7b/VLKEB_long_ar-2026.06.04-13.00.34/"
    "checkpoints/epoch-20-i-26600-ema_loss-1.5124"
)


def _resolve(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT_PATH, path)


def _spike_output_name(cfg: SpikeConfig) -> str:
    if cfg.name == "baseline":
        return "patch_spike_baseline.json"
    if cfg.name == "4a_max":
        return "patch_spike_4a.json"
    if cfg.name == "4a_attn":
        return "patch_spike_4a_attn.json"
    if cfg.name == "4b_rerank":
        return "patch_spike_4b.json"
    if cfg.name == "4c_vision_gate":
        return "patch_spike_4c.json"
    return "patch_spike_%s.json" % cfg.name


def _run_spike_group(
    editor,
    samples: List[Dict[str, Any]],
    spikes: List[str],
    chunk_size: int,
    ckpt: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    out_files: Dict[str, str] = {}
    for spike in spikes:
        cfg = SpikeConfig.for_spike(spike, chunk_size=chunk_size)
        pool_sample_ids = build_edit_pool(editor, samples, cfg)
        result = run_spike_diagnostics(editor, samples, cfg, pool_sample_ids)
        result["meta"] = {
            "ckpt": ckpt,
            "n_samples": len(samples),
            "n_pool_entries": len(pool_sample_ids),
        }
        results[cfg.name] = result
        out_files[cfg.name] = _spike_output_name(cfg)
    return {"results": results, "output_files": out_files}


def run_all_spikes(
    ckpt: str,
    ar_ckpt: str,
    device: str,
    sample_n: int,
    seed: int,
    data_path: str,
    img_root: str,
    spikes: List[str],
    chunk_size: int,
    preload_images: bool,
) -> Dict[str, Any]:
    import gc

    import torch
    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    data_path = _resolve(data_path)
    img_root = _resolve(img_root)

    eval_data = GroundingVLKEB(data_path, img_root, sample_n, preload_images=preload_images)
    samples = deepcopy(eval_data.data_with_img)
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    rng.shuffle(order)
    samples = [samples[i] for i in order]

    baseline_spikes = []
    ar_spikes = []
    for spike in spikes:
        cfg = SpikeConfig.for_spike(spike, chunk_size=chunk_size)
        if cfg.ar_mode:
            ar_spikes.append(spike)
        else:
            baseline_spikes.append(spike)

    spike_results: Dict[str, Any] = {}
    per_spike_json: Dict[str, str] = {}

    if baseline_spikes:
        editor = load_vllm_editor(
            "liveedit", get_full_model_name("blip2"), device, None, ckpt, False
        )
        group = _run_spike_group(editor, samples, baseline_spikes, chunk_size, ckpt)
        spike_results.update(group["results"])
        per_spike_json.update(group["output_files"])
        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ar_spikes:
        editor = load_vllm_editor(
            "liveedit", get_full_model_name("blip2"), device, None, ar_ckpt, False
        )
        group = _run_spike_group(editor, samples, ar_spikes, chunk_size, ar_ckpt)
        spike_results.update(group["results"])
        per_spike_json.update(group["output_files"])
        del editor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = {k: {"summary": v["summary"], "config": v["config"]} for k, v in spike_results.items()}

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "baseline_ckpt": ckpt,
            "ar_ckpt": ar_ckpt,
            "data_path": data_path,
            "img_root": img_root,
            "seed": seed,
            "device": device,
            "n_samples": len(samples),
            "spikes_run": list(spike_results.keys()),
        },
        "summaries": summaries,
        "gate": gate_verdict(summaries),
        "spikes": spike_results,
        "output_files": per_spike_json,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 5 patch inference spikes")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--ar_ckpt", type=str, default=DEFAULT_AR_CKPT, help="Checkpoint for 4c AR spike")
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=50)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument(
        "-dpath",
        "--grounding_json",
        type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", type=str, default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument(
        "--spikes",
        type=str,
        nargs="+",
        default=list(SPIKE_NAMES),
        help="Subset of: %s" % ", ".join(SPIKE_NAMES),
    )
    ap.add_argument("--no_preload_images", action="store_true")
    ap.add_argument(
        "-out_dir",
        "--output_dir",
        type=str,
        default="doc/results/meeting_validation/step5",
    )
    args = ap.parse_args()

    result = run_all_spikes(
        ckpt=args.editor_ckpt_path,
        ar_ckpt=args.ar_ckpt,
        device=args.device,
        sample_n=args.data_sample_n,
        seed=args.eval_seed,
        data_path=args.grounding_json,
        img_root=args.img_root,
        spikes=args.spikes,
        chunk_size=args.chunk_size,
        preload_images=not args.no_preload_images,
    )

    out_dir = _resolve(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for spike_name, fname in result["output_files"].items():
        spike_data = result["spikes"][spike_name]
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(spike_data, f, indent=2)
        print("Wrote %s" % path)

    combined_path = os.path.join(out_dir, "patch_spike_comparison.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": result["meta"],
                "summaries": result["summaries"],
                "gate": result["gate"],
                "output_files": result["output_files"],
            },
            f,
            indent=2,
        )
    print("Wrote %s" % combined_path)
    print("Gate: %s" % result["gate"])
    for name, summ in result["summaries"].items():
        ff = summ["summary"].get("hard_negative_false_activation_rate")
        if ff is not None:
            print("  %s: hard-neg false-fire %.1f%%" % (name, ff * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
