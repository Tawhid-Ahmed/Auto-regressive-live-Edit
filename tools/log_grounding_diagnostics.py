"""Step 1: routing diagnostics on grounding probes (hard-negative false activation).

Complements token-accuracy eval from test_vllm_edit.py by logging retrieve_moes
on image_swap / hard_negative / modality_consistency / prompt_conflict probes.

Usage:
  python tools/log_grounding_diagnostics.py \\
    -ckpt "records/liveedit/.../epoch-20-..." \\
    -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json \\
    -dvc cuda:0 -dsn 150 -seed 42 \\
    -out doc/results/grounding_diagnostics_baseline.json
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

from tools.log_routing_diagnostics import (
    _architecture_summary,
    _classify_routing,
    _expected_pool_indices,
)
from utils.GLOBAL import ROOT_PATH


GROUNDING_PROBE_NAMES = (
    "reliability",
    "image_swap",
    "hard_negative",
    "modality_consistency",
    "prompt_conflict",
    "image_rephrase",
)


def _probe_specs(sample: Dict[str, Any]) -> List[tuple]:
    req = sample["requests"][0]
    specs = [("reliability", req["prompt"], req["image"], req["target_new"])]
    gen = sample.get("generality", {})
    for name in ("image_swap", "hard_negative", "modality_consistency", "prompt_conflict"):
        if name in gen and gen[name]:
            g = gen[name][0]
            specs.append((name, g["prompt"], g["image"], g["target"]))
    if "image_rephrase" in gen and gen["image_rephrase"]:
        g = gen["image_rephrase"][0]
        specs.append(("image_rephrase", g["prompt"], g["image"], g["target"]))
    return specs


def collect_grounding_routing_diagnostics(
    editor,
    samples: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Probe routing false_cross / false_activation on a built edit pool (no model reload)."""
    editor.restore_to_original_model()
    pool_sample_ids: List[int] = []
    import torch
    for sid, sample in enumerate(samples):
        # Eval-time edits must not retain autograd graphs — without no_grad,
        # DEK+patch_grounding OOMs around a few hundred lifelong edits on 16GB.
        with torch.no_grad():
            editor.edit_one_piece(sample["requests"][0])
        n_new = len(editor.requests_pool) - len(pool_sample_ids)
        pool_sample_ids.extend([sid] * n_new)
        if (sid + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    probes_log: List[Dict[str, Any]] = []
    aggregate: Dict[str, List[bool]] = defaultdict(list)

    for sid, sample in enumerate(samples):
        for probe_name, prompt, image, target in _probe_specs(sample):
            with torch.no_grad():
                routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
            classification = _classify_routing(
                probe_name if probe_name != "image_swap" else "image_rephrase",
                sid,
                routing["retrieved_indices"],
                pool_sample_ids,
            )
            if probe_name in ("image_swap", "hard_negative"):
                classification["false_cross_edit"] = any(
                    pool_sample_ids[i] != sid
                    for i in routing["retrieved_indices"]
                    if i < len(pool_sample_ids)
                )
                classification["false_activation"] = classification["false_cross_edit"]
            # probe_name must win: _classify_routing embeds its own "probe" key
            # (image_swap is classified via the image_rephrase branch), which used
            # to clobber image_swap -> image_rephrase in stored logs (M4 / G0).
            entry = {**routing, **classification, "probe": probe_name}
            probes_log.append(entry)
            if probe_name in ("hard_negative", "image_swap"):
                aggregate[f"{probe_name}_false_activation"].append(
                    classification.get("false_activation", False))
            if probe_name == "reliability":
                aggregate["reliability_false_cross"].append(
                    classification.get("false_cross_edit", False))

    def _rate(key: str) -> Optional[float]:
        vals = aggregate.get(key, [])
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_samples": len(samples),
        "n_pool_entries": len(pool_sample_ids),
        "hard_negative_false_activation_rate": _rate("hard_negative_false_activation"),
        "image_swap_false_activation_rate": _rate("image_swap_false_activation"),
        "hard_negative_false_cross_rate": _rate("hard_negative_false_activation"),
        "image_swap_false_cross_rate": _rate("image_swap_false_activation"),
        "reliability_false_cross_rate": _rate("reliability_false_cross"),
        "mean_n_retrieved": {
            probe: float(np.mean([e["n_retrieved"] for e in probes_log if e["probe"] == probe]))
            for probe in GROUNDING_PROBE_NAMES
            if any(e["probe"] == probe for e in probes_log)
        },
    }

    return {
        "meta": meta or {},
        "architecture": _architecture_summary(),
        "summary": summary,
        "probes": probes_log,
        "pool_sample_ids": pool_sample_ids,
    }


def run_grounding_diagnostics(
    ckpt: str,
    device: str,
    sample_n: int,
    seed: int,
    data_path: str,
    img_root: str,
    ar_mode: bool,
    chunk_size: int,
) -> Dict[str, Any]:
    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = ar_mode
        editor.chunk_size = chunk_size

    eval_data = GroundingVLKEB(data_path, img_root, sample_n)
    samples = deepcopy(eval_data.data_with_img)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": ckpt,
        "data_path": data_path,
        "seed": seed,
        "device": device,
        "ar_mode": ar_mode,
    }
    return collect_grounding_routing_diagnostics(editor, samples, meta=meta)


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 1 grounding routing diagnostics")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, required=True)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=150)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument(
        "-dpath", "--vlkeb_eval_json", type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument(
        "--img_root", type=str,
        default="data/VLKEB/VLKEB_images/mmkb_images",
    )
    ap.add_argument("--ar_mode", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("-out", "--output_json", type=str,
                      default="doc/results/meeting_validation/step1/grounding_diagnostics_baseline.json")
    args = ap.parse_args()

    result = run_grounding_diagnostics(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        sample_n=args.data_sample_n,
        seed=args.eval_seed,
        data_path=args.vlkeb_eval_json,
        img_root=args.img_root,
        ar_mode=args.ar_mode,
        chunk_size=args.chunk_size,
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
