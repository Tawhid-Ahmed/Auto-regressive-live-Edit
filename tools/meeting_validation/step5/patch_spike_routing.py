"""Step 5 patch + cross-attention inference spike helpers.

Spikes (inference-only, opt-in flags on LiveEdit / BLIP2):
  - 4a: no mean-pool routing (max / attn-weighted pool over eqe_n)
  - 4b: top-k coarse experts reranked with ViT patch cross-attention
  - 4c: AR chunk gate from Q-Former image cross-attention peaks
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SPIKE_NAMES = ("baseline", "4a_max", "4a_attn", "4b_rerank", "4c_vision_gate")


@dataclass
class SpikeConfig:
    name: str
    routing_pool_mode: str = "mean"
    patch_rerank_enabled: bool = False
    expose_vit_grid: bool = False
    ar_vision_gated_chunks: bool = False
    ar_mode: bool = False
    chunk_size: int = 16

    @classmethod
    def for_spike(cls, spike: str, chunk_size: int = 16) -> "SpikeConfig":
        spike = spike.lower().replace("-", "_")
        if spike in ("baseline", "base", "mean"):
            return cls("baseline")
        if spike in ("4a_max", "4a", "max"):
            return cls("4a_max", routing_pool_mode="max")
        if spike in ("4a_attn", "attn"):
            return cls("4a_attn", routing_pool_mode="attn")
        if spike in ("4b", "4b_rerank", "rerank"):
            return cls(
                "4b_rerank",
                routing_pool_mode="max",
                patch_rerank_enabled=True,
                expose_vit_grid=True,
            )
        if spike in ("4c", "4c_vision_gate", "vision_gate"):
            return cls(
                "4c_vision_gate",
                ar_vision_gated_chunks=True,
                ar_mode=True,
                expose_vit_grid=True,
                chunk_size=chunk_size,
            )
        raise ValueError("Unknown spike %r; expected one of %s" % (spike, SPIKE_NAMES))


def apply_spike_config(editor, cfg: SpikeConfig) -> None:
    """Set opt-in Step 5 flags on a loaded LiveEdit editor."""
    editor.routing_pool_mode = cfg.routing_pool_mode
    editor.patch_rerank_enabled = cfg.patch_rerank_enabled
    editor.ar_vision_gated_chunks = cfg.ar_vision_gated_chunks
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = cfg.ar_mode
    if cfg.ar_mode:
        editor.chunk_size = cfg.chunk_size
    if hasattr(editor.vllm, "set_expose_vit_grid"):
        editor.vllm.set_expose_vit_grid(cfg.expose_vit_grid)
    elif hasattr(editor.vllm, "expose_vit_grid"):
        editor.vllm.expose_vit_grid = cfg.expose_vit_grid


def _probe_specs(sample: Dict[str, Any]) -> List[Tuple[str, str, Any, str]]:
    req = sample["requests"][0]
    specs = [("reliability", req["prompt"], req["image"], req["target_new"])]
    gen = sample.get("generality", {})
    for name in (
        "image_swap",
        "hard_negative",
        "modality_consistency",
        "prompt_conflict",
        "image_rephrase",
    ):
        if name in gen and gen[name]:
            g = gen[name][0]
            specs.append((name, g["prompt"], g["image"], g["target"]))
    return specs


def _classify_probe(
    probe: str,
    sample_id: int,
    retrieved_indices: List[int],
    pool_sample_ids: List[int],
) -> Dict[str, Any]:
    expected = {i for i, sid in enumerate(pool_sample_ids) if sid == sample_id}
    retrieved = set(retrieved_indices)
    out: Dict[str, Any] = {"routing_ok": True}
    if probe in ("hard_negative", "image_swap"):
        out["false_activation"] = any(
            pool_sample_ids[i] != sample_id
            for i in retrieved_indices
            if i < len(pool_sample_ids)
        )
        out["routing_ok"] = not out["false_activation"]
    elif probe == "reliability":
        out["miss"] = len(expected & retrieved) == 0 and len(expected) > 0
        out["false_cross_edit"] = any(
            pool_sample_ids[i] != sample_id
            for i in retrieved_indices
            if i < len(pool_sample_ids)
        )
        out["routing_ok"] = expected.issubset(retrieved) and not out["false_cross_edit"]
    return out


def run_spike_diagnostics(
    editor,
    samples: List[Dict[str, Any]],
    spike_cfg: SpikeConfig,
    pool_sample_ids: List[int],
) -> Dict[str, Any]:
    """Probe routing under one spike configuration (pool must already be built)."""
    from editor.vllm_editors.liveedit.chunk_utils import split_target_into_chunks

    probes_log: List[Dict[str, Any]] = []
    false_activation: List[bool] = []
    false_cross: List[bool] = []
    miss: List[bool] = []

    tokenizer = editor.vllm.get_llm_tokenizer()

    for sid, sample in enumerate(samples):
        for probe_name, prompt, image, target in _probe_specs(sample):
            chunk_index = None
            if spike_cfg.ar_mode and spike_cfg.ar_vision_gated_chunks:
                chunks = split_target_into_chunks(
                    tokenizer, target, spike_cfg.chunk_size, None
                )
                offset = max(sum(len(c) for c in chunks) - 1, 0) if chunks else 0
                chunk_index = editor.infer_ar_chunk_index(
                    offset, image=image, chunk_size=spike_cfg.chunk_size, max_chunks=len(chunks) or None
                )
            elif spike_cfg.ar_mode:
                chunks = split_target_into_chunks(
                    tokenizer, target, spike_cfg.chunk_size, None
                )
                chunk_index = len(chunks) - 1 if chunks else None

            routing = editor.probe_routing(
                editor.vllm, prompt, image, target=target, chunk_index=chunk_index
            )
            cls_probe = (
                "image_rephrase" if probe_name == "image_swap" else probe_name
            )
            classification = _classify_probe(
                cls_probe, sid, routing["retrieved_indices"], pool_sample_ids
            )
            entry = {
                "probe": probe_name,
                "sample_id": sid,
                "chunk_index": chunk_index,
                **routing,
                **classification,
            }
            probes_log.append(entry)
            if probe_name in ("hard_negative", "image_swap"):
                false_activation.append(classification.get("false_activation", False))
            if probe_name == "reliability":
                false_cross.append(classification.get("false_cross_edit", False))
                miss.append(classification.get("miss", False))

    def _rate(vals: List[bool]) -> Optional[float]:
        return float(np.mean(vals)) if vals else None

    return {
        "spike": spike_cfg.name,
        "config": {
            "routing_pool_mode": spike_cfg.routing_pool_mode,
            "patch_rerank_enabled": spike_cfg.patch_rerank_enabled,
            "ar_vision_gated_chunks": spike_cfg.ar_vision_gated_chunks,
            "ar_mode": spike_cfg.ar_mode,
        },
        "summary": {
            "hard_negative_false_activation_rate": _rate(false_activation),
            "image_swap_false_activation_rate": _rate(
                [e.get("false_activation", False) for e in probes_log if e["probe"] == "image_swap"]
            ),
            "reliability_false_cross_rate": _rate(false_cross),
            "reliability_miss_rate": _rate(miss),
            "mean_n_retrieved_hard_negative": float(
                np.mean([e["n_retrieved"] for e in probes_log if e["probe"] == "hard_negative"])
            )
            if any(e["probe"] == "hard_negative" for e in probes_log)
            else None,
        },
        "probes": probes_log,
    }


def build_edit_pool(
    editor,
    samples: List[Dict[str, Any]],
    spike_cfg: SpikeConfig,
) -> List[int]:
    """Sequential edits; returns pool_sample_ids aligned with evr_pool."""
    apply_spike_config(editor, spike_cfg)
    editor.restore_to_original_model()
    pool_sample_ids: List[int] = []
    for sid, sample in enumerate(samples):
        prev = len(pool_sample_ids)
        editor.edit_one_piece(sample["requests"][0])
        n_new = len(editor.requests_pool) - prev
        pool_sample_ids.extend([sid] * n_new)
    return pool_sample_ids


def gate_verdict(summaries: Dict[str, Dict[str, Any]]) -> str:
    base = summaries.get("baseline", {}).get("summary", {})
    base_ff = base.get("hard_negative_false_activation_rate")
    if base_ff is None:
        return "INSUFFICIENT_DATA"

    improvements = []
    for spike in ("4a_max", "4a_attn", "4b_rerank", "4c_vision_gate"):
        s = summaries.get(spike, {}).get("summary", {})
        ff = s.get("hard_negative_false_activation_rate")
        if ff is not None and ff < base_ff - 0.02:
            improvements.append(spike)

    if improvements:
        return (
            "PASS → Inference spikes improve hard-negative routing: %s. "
            "Pick winner + DINOv2 (Step 4) for Step 6 retrain."
            % ", ".join(improvements)
        )
    if any(summaries.get(k) for k in ("4b_rerank", "4c_vision_gate")):
        return (
            "REFRAME → 4a alone insufficient; evaluate 4b/4c jointly with full "
            "grounding eval before Step 6"
        )
    return "REFRAME → Routing-pool redesign may need trained patch layer (Step 6)"
