"""Step 4: compare offline encoder routing signals (Q-Former vs EVA patches vs DINOv2).

Runs sequential edits on GroundingVLKEB samples, then scores reliability and
hard-negative probes under three encoder conditions without retraining.

Usage (from repo root):
  conda run -n liveedit python tools/meeting_validation/step4/compare_encoder_routing.py \\
    -ckpt "records/liveedit/blip2-opt-2.7b/.../epoch-20-..." \\
    -dvc cuda:0 -dsn 150 -seed 42 \\
    -out doc/results/meeting_validation/step4/encoder_routing_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

from tools.meeting_validation.step4.encoder_routing import (
    ENCODER_NAMES,
    EvaPatchEncoder,
    ExternalPatchEncoder,
    patch_route_outcome,
    qformer_route_from_probe,
    summarize_encoder,
)


def _resolve(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT_PATH, path)


def _outcome_to_dict(o) -> Dict[str, Any]:
    return {
        "encoder": o.encoder,
        "probe": o.probe,
        "sample_id": o.sample_id,
        "sim_prot": o.sim_prot,
        "score_own_expert": o.score_own_expert,
        "score_best_other": o.score_best_other,
        "margin_own_vs_other": o.margin_own_vs_other,
        "retrieved_indices": o.retrieved_indices,
        "n_retrieved": len(o.retrieved_indices),
        "false_fire_on_target": o.false_fire_on_target,
        "miss": o.miss,
        "false_cross_edit": o.false_cross_edit,
        "routing_ok": o.routing_ok,
    }


def gate_verdict(summaries: Dict[str, Dict[str, Any]]) -> str:
    a = summaries.get("qformer", {})
    b = summaries.get("eva_patches", {})
    c = summaries.get("dinov2", {})
    a_ff = a.get("hard_neg_false_fire_rate")
    b_ff = b.get("hard_neg_false_fire_rate")
    c_ff = c.get("hard_neg_false_fire_rate")
    b_auc = b.get("hard_neg_auc_margin") or b.get("hard_neg_auc_own_expert")
    c_auc = c.get("hard_neg_auc_margin") or c.get("hard_neg_auc_own_expert")
    a_auc = a.get("hard_neg_auc_margin") or a.get("hard_neg_auc_own_expert")

    if a_ff is None:
        return "INSUFFICIENT_DATA"

    winners = []
    for name, ff, auc in (
        ("eva_patches", b_ff, b_auc),
        ("dinov2", c_ff, c_auc),
    ):
        if ff is None:
            continue
        better_ff = ff < a_ff - 0.02
        better_auc = auc is not None and a_auc is not None and auc > a_auc + 0.02
        if better_ff or better_auc:
            winners.append(name)

    if winners:
        return (
            "PASS → Proceed to Step 5 (patch spikes). "
            "Offline encoders beat Q-Former on hard negatives: %s"
            % ", ".join(winners)
        )
    if b_ff is not None and c_ff is not None and b_ff >= a_ff and c_ff >= a_ff:
        return (
            "REFRAME → Q-Former routing not clearly worse than patch encoders offline; "
            "Step 5 may need routing-pool redesign not encoder swap alone"
        )
    return "REFRAME → Mixed encoder profile — inspect per-metric before Step 5"


def run_compare(
    ckpt: str,
    device: str,
    sample_n: int,
    seed: int,
    data_path: str,
    img_root: str,
    external_encoder: str,
    skip_external: bool,
    preload_images: bool,
) -> Dict[str, Any]:
    import torch
    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    data_path = _resolve(data_path)
    img_root = _resolve(img_root)

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    eval_data = GroundingVLKEB(data_path, img_root, sample_n, preload_images=preload_images)
    samples = deepcopy(eval_data.data_with_img)

    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    rng.shuffle(order)
    samples = [samples[i] for i in order]

    eva_enc = EvaPatchEncoder(editor.vllm)
    ext_enc: Optional[ExternalPatchEncoder] = None
    if not skip_external:
        ext_enc = ExternalPatchEncoder(external_encoder, device)

    editor.restore_to_original_model()
    pool_sample_ids: List[int] = []
    eva_keys: List[torch.Tensor] = []
    ext_keys: List[torch.Tensor] = []

    image_key_cache: Dict[str, torch.Tensor] = {}

    def _eva_key(image) -> torch.Tensor:
        path = getattr(image, "filename", None) or str(id(image))
        if path not in image_key_cache:
            image_key_cache[path] = eva_enc.key(image)
        return image_key_cache[path]

    def _ext_key(image) -> torch.Tensor:
        path = ("ext:" + str(getattr(image, "filename", None) or id(image)))
        if path not in image_key_cache:
            image_key_cache[path] = ext_enc.key(image)
        return image_key_cache[path]

    for sid, sample in enumerate(samples):
        req = sample["requests"][0]
        prev_len = len(pool_sample_ids)
        editor.edit_one_piece(req)
        n_new = len(editor.requests_pool) - prev_len
        edit_image = req["image"]
        for _ in range(n_new):
            pool_sample_ids.append(sid)
            eva_keys.append(eva_enc.key(edit_image))
            if ext_enc is not None:
                ext_keys.append(ext_enc.key(edit_image))

    eva_pool = torch.stack(eva_keys, dim=0)
    ext_pool = torch.stack(ext_keys, dim=0) if ext_keys else None
    eva_scale = 1.0 / (eva_pool.shape[1] ** 0.5)
    ext_scale = 1.0 / (ext_pool.shape[1] ** 0.5) if ext_pool is not None else None

    module_dim = int(editor.cfg.retrieval_editor.module_dim)
    qformer_scale = editor.sim_scale

    all_outcomes: Dict[str, list] = {k: [] for k in ENCODER_NAMES}

    for sid, sample in enumerate(samples):
        req = sample["requests"][0]
        probes = [
            ("reliability", req["prompt"], req["image"], req["target_new"]),
        ]
        if "hard_negative" in sample.get("generality", {}):
            hn = sample["generality"]["hard_negative"][0]
            probes.append(("hard_negative", hn["prompt"], hn["image"], hn["target"]))

        for probe_name, prompt, image, target in probes:
            routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
            q_out = qformer_route_from_probe(routing, sid, pool_sample_ids, probe_name)
            all_outcomes["qformer"].append(q_out)

            eva_q = _eva_key(image)
            eva_out = patch_route_outcome(
                "eva_patches", probe_name, sid, eva_q, eva_pool, pool_sample_ids, eva_scale
            )
            all_outcomes["eva_patches"].append(eva_out)

            if ext_pool is not None:
                ext_q = _ext_key(image)
                enc_name = external_encoder if external_encoder in ENCODER_NAMES else "dinov2"
                ext_out = patch_route_outcome(
                    enc_name, probe_name, sid, ext_q, ext_pool, pool_sample_ids, ext_scale
                )
                all_outcomes[enc_name].append(ext_out)

    summaries = {
        enc: summarize_encoder(rows)
        for enc, rows in all_outcomes.items()
        if rows
    }

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": ckpt,
            "data_path": data_path,
            "img_root": img_root,
            "seed": seed,
            "device": device,
            "n_samples": len(samples),
            "n_pool_entries": len(pool_sample_ids),
            "external_encoder": external_encoder if not skip_external else None,
            "module_dim": module_dim,
            "qformer_scale": float(qformer_scale),
            "eva_hidden_dim": eva_enc.hidden_dim,
            "external_hidden_dim": ext_enc.hidden_dim if ext_enc else None,
        },
        "summaries": summaries,
        "gate": gate_verdict(summaries),
        "probes": {
            enc: [_outcome_to_dict(o) for o in rows]
            for enc, rows in all_outcomes.items()
            if rows
        },
        "pool_sample_ids": pool_sample_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 4 encoder routing comparison")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, required=True)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=150)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument(
        "-dpath",
        "--grounding_json",
        type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", type=str, default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument(
        "--external_encoder",
        type=str,
        default="dinov2",
        choices=("dinov2", "siglip"),
        help="Frozen external spatial encoder (condition C)",
    )
    ap.add_argument("--skip_external", action="store_true", help="Only run qformer + eva_patches")
    ap.add_argument("--no_preload_images", action="store_true")
    ap.add_argument(
        "-out",
        "--output_json",
        type=str,
        default="doc/results/meeting_validation/step4/encoder_routing_scores.json",
    )
    args = ap.parse_args()

    result = run_compare(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        sample_n=args.data_sample_n,
        seed=args.eval_seed,
        data_path=args.grounding_json,
        img_root=args.img_root,
        external_encoder=args.external_encoder,
        skip_external=args.skip_external,
        preload_images=not args.no_preload_images,
    )

    out_path = _resolve(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s" % out_path)
    print("Gate: %s" % result["gate"])
    for enc, summ in result["summaries"].items():
        print(
            "  %s: hard-neg false-fire=%.1f%% AUC=%.3f"
            % (
                enc,
                (summ.get("hard_neg_false_fire_rate") or 0) * 100,
                summ.get("hard_neg_auc_margin") or summ.get("hard_neg_auc_own_expert") or float("nan"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
