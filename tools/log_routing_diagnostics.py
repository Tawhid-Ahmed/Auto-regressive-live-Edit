"""Step 0: log LiveEdit MoE routing diagnostics on existing checkpoints.

Runs sequential edits on N VLKEB samples, then probes retrieve_moes scores
(vis_sim, retrieval_map, false fires) on reliability / generality / locality queries.

Usage (from repo root):
  conda run -n liveedit python tools/log_routing_diagnostics.py \\
    -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-.../checkpoints/epoch-20-....pt" \\
    -dvc cuda:0 -dsn 150 -seed 42 \\
    -dpath data/VLKEB_long/eval_longform.json \\
    -out doc/results/routing_diagnostics_baseline.json

Architecture-only summary (no GPU / checkpoint):
  conda run -n liveedit python tools/log_routing_diagnostics.py --architecture-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH


def _architecture_summary() -> Dict[str, Any]:
    """Static bottleneck notes for LiveEdit wiring (optional; BLIP-2 weights not required).

    Metadata only — taxonomy / routing probe scores do not depend on this block.
    When blip2-opt-2.7b weights are absent (e.g. LLaVA-only lab tree), return a
    stub so ``--log_routing`` evals on other backbones still complete.
    """
    from utils.GLOBAL import model_path_map

    mp = model_path_map.get("blip2-opt-2.7b")
    cfg_json = os.path.join(mp, "config.json") if mp else ""
    if not cfg_json or not os.path.isfile(cfg_json):
        return {
            "available": False,
            "note": "BLIP-2 config.json not on disk; architecture summary skipped",
            "routing_key_collapse": "mean(2) over eqe_n cross-attn outputs → scalar per pool entry",
        }

    from transformers import Blip2Config

    cfg = Blip2Config.from_json_file(cfg_json)
    nq = int(cfg.num_query_tokens)
    # EVA-ViT patch grid for default 224 input: (224/16)^2 + 1 cls ≈ 257
    vit_patches = 257
    eqe_n = 32  # LiveEdit config default (editor/vllm_editors/liveedit/config/blip2.yaml)
    cfg_yaml = os.path.join(ROOT, "editor", "vllm_editors", "liveedit", "config", "blip2.yaml")
    if os.path.isfile(cfg_yaml):
        import yaml
        with open(cfg_yaml, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        eqe_n = int(y.get("retrieval_editor", {}).get("eqe_n", eqe_n))

    return {
        "available": True,
        "vit_patch_tokens": vit_patches,
        "qformer_query_tokens": nq,
        "editor_routing_tokens": eqe_n,
        "compression_ratio_patches_to_routing": round(vit_patches / eqe_n, 2),
        "routing_key_collapse": "mean(2) over eqe_n cross-attn outputs → scalar per pool entry",
        "patch_path_to_editor": "discarded — image_embeds not passed past Q-Former in blip2.py",
        "bottleneck_layers": [
            "EVA-ViT (~257 patches) → Q-Former cross-attn → 32 LLM tokens",
            "32 vision tokens → QVExtractor → eqe_n=32 routing keys → mean-pool → scalar vis_sim",
        ],
    }


def _expected_pool_indices(sample_id: int, pool_sample_ids: List[int]) -> Set[int]:
    return {i for i, sid in enumerate(pool_sample_ids) if sid == sample_id}


def _classify_routing(
    probe: str,
    sample_id: int,
    retrieved_indices: List[int],
    pool_sample_ids: List[int],
) -> Dict[str, Any]:
    expected = _expected_pool_indices(sample_id, pool_sample_ids)
    retrieved = set(retrieved_indices)
    retrieved_samples = {pool_sample_ids[i] for i in retrieved_indices if i < len(pool_sample_ids)}
    out: Dict[str, Any] = {
        "probe": probe,
        "sample_id": sample_id,
        "retrieved_indices": retrieved_indices,
        "n_retrieved": len(retrieved_indices),
        "expected_indices": sorted(expected),
    }
    if probe == "image_loc":
        out["false_fire"] = len(retrieved) > 0
        out["routing_ok"] = len(retrieved) == 0
    elif probe in ("reliability", "text_rephrase", "image_rephrase"):
        out["miss"] = len(expected & retrieved) == 0 and len(expected) > 0
        out["false_cross_edit"] = any(
            pool_sample_ids[i] != sample_id for i in retrieved_indices if i < len(pool_sample_ids)
        )
        out["routing_ok"] = expected.issubset(retrieved) and not out["false_cross_edit"]
    return out


def run_diagnostics(
    ckpt: str,
    device: str,
    sample_n: int,
    seed: int,
    data_path: str,
    img_root: str,
    ar_mode: bool,
    chunk_size: int,
    preload_images: bool,
) -> Dict[str, Any]:
    import torch
    from dataset.vllm import VLKEB
    from utils import get_full_model_name, load_vllm_editor

    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = ar_mode
        editor.chunk_size = chunk_size

    eval_data = VLKEB(data_path, img_root, sample_n, preload_images=preload_images)
    samples = deepcopy(eval_data.data_with_img)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    editor.restore_to_original_model()
    pool_sample_ids: List[int] = []
    for sid, sample in enumerate(samples):
        req = sample["requests"][0]
        editor.edit_one_piece(req)
        n_new = len(editor.requests_pool) - len(pool_sample_ids)
        pool_sample_ids.extend([sid] * n_new)

    probes_log: List[Dict[str, Any]] = []
    aggregate: Dict[str, List[bool]] = defaultdict(list)

    for sid, sample in enumerate(samples):
        req = sample["requests"][0]
        probe_specs = [
            ("reliability", req["prompt"], req["image"], req["target_new"]),
            ("text_rephrase", sample["generality"]["text_rephrase"][0]["prompt"],
             sample["generality"]["text_rephrase"][0]["image"],
             sample["generality"]["text_rephrase"][0]["target"]),
            ("image_rephrase", sample["generality"]["image_rephrase"][0]["prompt"],
             sample["generality"]["image_rephrase"][0]["image"],
             sample["generality"]["image_rephrase"][0]["target"]),
            ("image_loc", sample["locality"]["image_loc"][0]["prompt"],
             sample["locality"]["image_loc"][0]["image"],
             sample["locality"]["image_loc"][0]["target"]),
        ]
        for probe_name, prompt, image, target in probe_specs:
            routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
            classification = _classify_routing(
                probe_name, sid, routing["retrieved_indices"], pool_sample_ids)
            entry = {**routing, **classification}
            probes_log.append(entry)
            if probe_name == "image_loc":
                aggregate["locality_false_fire"].append(classification["false_fire"])
            if probe_name == "image_rephrase":
                aggregate["image_rephrase_false_cross"].append(classification["false_cross_edit"])
                aggregate["image_rephrase_miss"].append(classification["miss"])
            if probe_name == "reliability":
                aggregate["reliability_miss"].append(classification["miss"])
                aggregate["reliability_false_cross"].append(classification["false_cross_edit"])

    def _rate(key: str) -> Optional[float]:
        vals = aggregate.get(key, [])
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_samples": len(samples),
        "n_pool_entries": len(pool_sample_ids),
        "ar_mode": ar_mode,
        "chunk_size": chunk_size if ar_mode else None,
        "reliability_miss_rate": _rate("reliability_miss"),
        "reliability_false_cross_rate": _rate("reliability_false_cross"),
        "image_rephrase_miss_rate": _rate("image_rephrase_miss"),
        "image_rephrase_false_cross_rate": _rate("image_rephrase_false_cross"),
        "locality_false_fire_rate": _rate("locality_false_fire"),
        "mean_n_retrieved": {
            probe: float(np.mean([e["n_retrieved"] for e in probes_log if e["probe"] == probe]))
            for probe in ("reliability", "text_rephrase", "image_rephrase", "image_loc")
        },
        "mean_ivr_eqe_std": {
            probe: float(np.mean([e["ivr_eqe_std"] for e in probes_log if e["probe"] == probe]))
            for probe in ("reliability", "text_rephrase", "image_rephrase")
        },
    }

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": ckpt,
            "data_path": data_path,
            "seed": seed,
            "device": device,
        },
        "architecture": _architecture_summary(),
        "summary": summary,
        "probes": probes_log,
        "pool_sample_ids": pool_sample_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 0 LiveEdit routing diagnostics")
    ap.add_argument("--architecture-only", action="store_true",
                    help="Print static vision-bottleneck analysis (no GPU)")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=None)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=150)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("-dpath", "--vlkeb_eval_json", type=str,
                    default="data/VLKEB_long/eval_longform.json")
    ap.add_argument("--img_root", type=str,
                    default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("--ar_mode", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--no_preload_images", action="store_true",
                    help="Load images lazily from disk (lower RAM, slower)")
    ap.add_argument("-out", "--output_json", type=str,
                    default="doc/results/routing_diagnostics.json")
    args = ap.parse_args()

    if args.architecture_only:
        arch = _architecture_summary()
        print(json.dumps(arch, indent=2))
        return 0

    if not args.editor_ckpt_path:
        ap.error("-ckpt is required unless --architecture-only is set")

    result = run_diagnostics(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        sample_n=args.data_sample_n,
        seed=args.eval_seed,
        data_path=args.vlkeb_eval_json,
        img_root=args.img_root,
        ar_mode=args.ar_mode,
        chunk_size=args.chunk_size,
        preload_images=not args.no_preload_images,
    )

    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
    print("Summary:", json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
