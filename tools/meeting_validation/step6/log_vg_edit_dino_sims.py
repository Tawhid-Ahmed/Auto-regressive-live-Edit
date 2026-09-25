"""Log per-probe DINO sim(query_img, expert_edit_img) for VG-Edit gate calibration.

Builds a sequential edit pool (with DINO keys stored at edit time), then probes
reliability / image_swap / hard_negative with Q-Former routing only
(skip_decode_vision_gate=True) while recording DINO similarities for each
Q-Former candidate.

Usage:
  python tools/meeting_validation/step6/log_vg_edit_dino_sims.py \\
    -ckpt "records/liveedit/.../epoch-20-..." \\
    -dvc cuda:0 -dsn 100 -seed 42 -sen 10 \\
    -out doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

PROBE_NAMES = ("reliability", "image_swap", "hard_negative", "modality_consistency")


def _probe_specs(sample: Dict[str, Any]) -> List[tuple]:
    req = sample["requests"][0]
    specs = [("reliability", req["prompt"], req["image"], req["target_new"])]
    gen = sample.get("generality", {})
    for name in ("image_swap", "hard_negative", "modality_consistency"):
        if name in gen and gen[name]:
            g = gen[name][0]
            specs.append((name, g["prompt"], g["image"], g["target"]))
    return specs


def _annotate_candidates(
    candidates: List[Dict[str, Any]],
    sample_id: int,
    pool_sample_ids: List[int],
) -> None:
    for c in candidates:
        pi = int(c["pool_index"])
        c["pool_sample_id"] = pool_sample_ids[pi] if pi < len(pool_sample_ids) else -1
        c["is_own_expert"] = c["pool_sample_id"] == sample_id


def _summarize_probe(
    probe: str,
    sample_id: int,
    pool_sample_ids: List[int],
    routing: Dict[str, Any],
    image=None,
    editor=None,
) -> Dict[str, Any]:
    qformer_idx = routing.get("qformer_retrieved_indices", routing.get("retrieved_indices", []))
    false_fire_own = any(
        pool_sample_ids[i] == sample_id for i in qformer_idx if i < len(pool_sample_ids)
    )
    false_cross = any(
        pool_sample_ids[i] != sample_id for i in qformer_idx if i < len(pool_sample_ids)
    )
    out: Dict[str, Any] = {
        "probe": probe,
        "sample_id": sample_id,
        "n_qformer_retrieved": len(qformer_idx),
        "qformer_retrieved_indices": qformer_idx,
        "false_fire_own_expert": false_fire_own if probe in ("image_swap", "hard_negative") else False,
        "false_cross_edit": false_cross,
        "reliability_miss": False,
    }
    if probe == "reliability":
        expected = {i for i, sid in enumerate(pool_sample_ids) if sid == sample_id}
        retr = set(qformer_idx)
        out["reliability_miss"] = len(expected & retr) == 0 and len(expected) > 0

    pool_sims: List[float] = []
    if editor is not None and image is not None:
        pool_sims = editor.dino_sims_vs_pool(image)
    if pool_sims:
        own_pool_idx = [i for i, sid in enumerate(pool_sample_ids) if sid == sample_id]
        own_pool_sims = [pool_sims[i] for i in own_pool_idx if i < len(pool_sims)]
        if own_pool_sims:
            out["dino_sim_query_vs_own_edit_img"] = max(own_pool_sims)
        wrong_pool_sims = [
            pool_sims[i] for i, sid in enumerate(pool_sample_ids)
            if sid != sample_id and i < len(pool_sims)
        ]
        if wrong_pool_sims:
            out["dino_sim_query_vs_best_wrong_edit_img"] = max(wrong_pool_sims)
            out["dino_sim_query_vs_mean_wrong_edit_img"] = float(np.mean(wrong_pool_sims))
        if own_pool_sims and wrong_pool_sims:
            out["dino_margin_own_vs_best_wrong"] = max(own_pool_sims) - max(wrong_pool_sims)

    dino = routing.get("dino_gate")
    if not dino:
        out["dino_available"] = bool(pool_sims)
        return out

    out["dino_available"] = True
    candidates = [dict(c) for c in dino.get("candidates", [])]
    _annotate_candidates(candidates, sample_id, pool_sample_ids)
    out["dino_candidates"] = candidates

    own_sims = [c["dino_sim"] for c in candidates if c.get("is_own_expert")]
    wrong_sims = [c["dino_sim"] for c in candidates if not c.get("is_own_expert")]
    out["dino_sim_own_max"] = max(own_sims) if own_sims else None
    out["dino_sim_own_min"] = min(own_sims) if own_sims else None
    out["dino_sim_wrong_max"] = max(wrong_sims) if wrong_sims else None
    out["dino_sim_wrong_min"] = min(wrong_sims) if wrong_sims else None
    out["n_own_candidates"] = len(own_sims)
    out["n_wrong_candidates"] = len(wrong_sims)

    if own_sims:
        out["own_gate_pass_any"] = any(c.get("gate_pass") for c in candidates if c.get("is_own_expert"))
        out["own_abs_pass_any"] = any(c.get("abs_pass") for c in candidates if c.get("is_own_expert"))
    else:
        out["own_gate_pass_any"] = False
        out["own_abs_pass_any"] = False

    if wrong_sims and probe in ("image_swap", "hard_negative"):
        out["retrieved_wrong_dino_max"] = max(wrong_sims)
    return out


def run_dino_sim_logging(
    ckpt: str,
    device: str,
    sample_n: int,
    seed: int,
    sequential_n: int,
    data_path: str,
    img_root: str,
    abs_threshold: float,
    margin: float,
) -> Dict[str, Any]:
    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    editor.enable_decode_vision_gate(margin=margin, abs_threshold=abs_threshold)

    eval_data = GroundingVLKEB(data_path, img_root, sample_n)
    samples = deepcopy(eval_data.data_with_img)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    samples = samples[:sample_n]

    editor.restore_to_original_model()
    pool_sample_ids: List[int] = []
    for sid, sample in enumerate(samples):
        editor.edit_one_piece(sample["requests"][0])
        n_new = len(editor.requests_pool) - len(pool_sample_ids)
        pool_sample_ids.extend([sid] * n_new)

    probes_log: List[Dict[str, Any]] = []
    for sid, sample in enumerate(samples):
        for probe_name, prompt, image, target in _probe_specs(sample):
            routing = editor.probe_routing(
                editor.vllm, prompt, image, target=target, skip_decode_vision_gate=True,
            )
            entry = _summarize_probe(
                probe_name, sid, pool_sample_ids, routing, image=image, editor=editor,
            )
            entry["vis_sim_prot"] = routing.get("vis_sim_prot")
            probes_log.append(entry)

    def _collect(key: str, probe_filter: str) -> List[float]:
        vals = []
        for e in probes_log:
            if e["probe"] != probe_filter:
                continue
            v = e.get(key)
            if v is not None:
                vals.append(float(v))
        return vals

    summary: Dict[str, Any] = {
        "n_samples": len(samples),
        "n_pool_entries": len(pool_sample_ids),
        "n_probes": len(probes_log),
        "dino_available_rate": float(np.mean([e.get("dino_available", False) for e in probes_log])),
        "swap_false_fire_rate": _rate(probes_log, "image_swap", "false_fire_own_expert"),
        "swap_false_cross_rate": _rate(probes_log, "image_swap", "false_cross_edit"),
        "hard_neg_false_cross_rate": _rate(probes_log, "hard_negative", "false_cross_edit"),
        "dino_margin_own_vs_best_wrong": {
            "reliability": _stats(_collect("dino_margin_own_vs_best_wrong", "reliability")),
            "image_swap": _stats(_collect("dino_margin_own_vs_best_wrong", "image_swap")),
            "hard_negative": _stats(_collect("dino_margin_own_vs_best_wrong", "hard_negative")),
        },
        "dino_sim_query_vs_best_wrong_edit_img": {
            "image_swap": _stats(_collect("dino_sim_query_vs_best_wrong_edit_img", "image_swap")),
            "hard_negative": _stats(_collect("dino_sim_query_vs_best_wrong_edit_img", "hard_negative")),
        },
        "hard_neg_false_fire_rate": _rate(probes_log, "hard_negative", "false_fire_own_expert"),
        "reliability_miss_rate": _rate(probes_log, "reliability", "reliability_miss"),
        "dino_sim_own_max": {
            "reliability": _stats(_collect("dino_sim_own_max", "reliability")),
            "image_swap": _stats(_collect("dino_sim_own_max", "image_swap")),
            "hard_negative": _stats(_collect("dino_sim_own_max", "hard_negative")),
        },
        "dino_sim_query_vs_own_edit_img": {
            "reliability": _stats(_collect("dino_sim_query_vs_own_edit_img", "reliability")),
            "image_swap": _stats(_collect("dino_sim_query_vs_own_edit_img", "image_swap")),
            "hard_negative": _stats(_collect("dino_sim_query_vs_own_edit_img", "hard_negative")),
        },
        "retrieved_wrong_dino_max": {
            "image_swap": _stats(_collect("retrieved_wrong_dino_max", "image_swap")),
            "hard_negative": _stats(_collect("retrieved_wrong_dino_max", "hard_negative")),
        },
    }

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": ckpt,
            "data_path": data_path,
            "seed": seed,
            "device": device,
            "sequential_n": sequential_n,
            "sample_n": sample_n,
            "decode_vision_abs_threshold": abs_threshold,
            "decode_vision_margin": margin,
            "skip_decode_vision_gate_at_probe": True,
        },
        "summary": summary,
        "probes": probes_log,
        "pool_sample_ids": pool_sample_ids,
    }


def _rate(probes: List[Dict[str, Any]], probe_name: str, key: str) -> Optional[float]:
    vals = [bool(e.get(key, False)) for e in probes if e["probe"] == probe_name]
    return float(np.mean(vals)) if vals else None


def _stats(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="VG-Edit DINO sim logging for gate calibration")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, required=True)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=100)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("-sen", "--sequential_edit_n", type=int, default=10)
    ap.add_argument(
        "-dpath", "--grounding_eval_json", type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", type=str, default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("--abs-threshold", type=float, default=0.45, dest="abs_threshold")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument(
        "-out", "--output_json", type=str,
        default="doc/results/meeting_validation/step6/vg_edit_dino_sims.json",
    )
    args = ap.parse_args()

    result = run_dino_sim_logging(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        sample_n=args.data_sample_n,
        seed=args.eval_seed,
        sequential_n=args.sequential_edit_n,
        data_path=args.grounding_eval_json,
        img_root=args.img_root,
        abs_threshold=args.abs_threshold,
        margin=args.margin,
    )

    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s" % out_path)
    print("Summary:", json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
