"""Step 6 layer ablation configs (motivation → impact table).

Layers map to meeting-validation ablations for PatchGroundingModule retrain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


LAYER_NAMES = ("L0", "L1", "L2", "L3", "L4", "L5", "VG")

LAYER_TABLE: List[Dict[str, str]] = [
    {
        "layer": "L0",
        "component": "Baseline LiveEdit",
        "motivation": "Reference — Q-Former routing only",
        "turn_off": "—",
        "primary_metric": "Reliability, locality",
    },
    {
        "layer": "L1",
        "component": "PatchGroundingModule",
        "motivation": "Q-Former bottleneck; hard negatives",
        "turn_off": "patch_grounding_enabled=False",
        "primary_metric": "Hard-negative false activation",
    },
    {
        "layer": "L2",
        "component": "Patch routing (no mean-pool)",
        "motivation": "Scalar key collapse",
        "turn_off": "routing_pool_mode=mean",
        "primary_metric": "Hard-negative AUC / false-fire",
    },
    {
        "layer": "L3",
        "component": "Cross-attn expert gen",
        "motivation": "Expert should depend on region",
        "turn_off": "patch_expert_gen_enabled=False",
        "primary_metric": "Image generality",
    },
    {
        "layer": "L4",
        "component": "Verification gate",
        "motivation": "MC-MKE consistency — dual Q-Former + patch threshold",
        "turn_off": "verification_gate_enabled=False",
        "primary_metric": "Modality consistency probe",
    },
    {
        "layer": "L5",
        "component": "AR + vision-gated chunks",
        "motivation": "Long-form visual segment alignment (Step 5 4c winner)",
        "turn_off": "ar_vision_gated_chunks=False; fixed chunks",
        "primary_metric": "Mid-trough + chunk-EM",
    },
    {
        "layer": "VG",
        "component": "VG-Edit decode vision gate",
        "motivation": "DINO competitive margin blocks wrong-image MoE injection (no retrain)",
        "turn_off": "decode_vision_gate_enabled=False",
        "primary_metric": "Swap-gap, hard-negative acc",
    },
]


@dataclass
class AblationConfig:
    name: str
    patch_grounding: bool = False
    patch_encoder: str = "dinov2"
    patch_routing_only: bool = False
    patch_expert_gen: bool = True
    verification_gate: bool = False
    routing_pool_mode: str = "mean"
    ar_mode: bool = False
    ar_vision_gated_chunks: bool = False
    chunk_size: int = 16
    expose_vit_grid: bool = False
    decode_vision_gate: bool = False
    decode_vision_margin: float = 0.0
    decode_vision_abs_threshold: float = 0.45

    @classmethod
    def for_layer(cls, layer: str, chunk_size: int = 16) -> "AblationConfig":
        layer = layer.upper()
        if layer == "L0":
            return cls("L0")
        if layer in ("VG", "VGEDIT", "VG_EDIT"):
            return cls(
                "VG",
                decode_vision_gate=True,
                decode_vision_margin=0.0,
                decode_vision_abs_threshold=0.45,
            )
        if layer == "L1":
            return cls("L1", patch_grounding=True, patch_expert_gen=False)
        if layer == "L2":
            return cls("L2", patch_grounding=True, routing_pool_mode="max", patch_expert_gen=False)
        if layer == "L3":
            return cls("L3", patch_grounding=True, patch_expert_gen=True)
        if layer in ("AVBL", "AVBT"):
            return cls(
                "AVBL",
                patch_grounding=True,
                patch_expert_gen=True,
                patch_routing_only=True,
            )
        if layer == "L4":
            return cls(
                "L4",
                patch_grounding=True,
                patch_expert_gen=True,
                verification_gate=True,
            )
        if layer == "L5":
            return cls(
                "L5",
                patch_grounding=True,
                patch_expert_gen=True,
                verification_gate=True,
                ar_mode=True,
                ar_vision_gated_chunks=True,
                expose_vit_grid=True,
                chunk_size=chunk_size,
            )
        raise ValueError("Unknown layer %r; expected one of %s" % (layer, LAYER_NAMES))

    def eval_name_postfix(self) -> str:
        return "step6_%s" % self.name.lower()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_ablation_config(editor, cfg: AblationConfig) -> None:
    """Apply Step 6 ablation flags to a loaded LiveEdit editor."""
    editor.patch_grounding_enabled = False
    editor.patch_routing_only = cfg.patch_routing_only
    editor.patch_expert_gen_enabled = False
    editor.verification_gate_enabled = cfg.verification_gate
    editor.routing_pool_mode = cfg.routing_pool_mode
    editor.ar_vision_gated_chunks = cfg.ar_vision_gated_chunks
    editor.ar_mode = cfg.ar_mode
    if cfg.ar_mode:
        editor.chunk_size = cfg.chunk_size
    if cfg.patch_grounding:
        if editor.patch_grounding_module is None:
            editor.enable_patch_grounding(
                encoder=cfg.patch_encoder,
                expert_gen=cfg.patch_expert_gen,
            )
        else:
            editor.patch_grounding_enabled = True
            editor.patch_encoder_name = cfg.patch_encoder
            editor.patch_expert_gen_enabled = cfg.patch_expert_gen
            editor._install_patch_image_cache_hook()
    if hasattr(editor.vllm, "set_expose_vit_grid"):
        editor.vllm.set_expose_vit_grid(cfg.expose_vit_grid)
    elif hasattr(editor.vllm, "expose_vit_grid"):
        editor.vllm.expose_vit_grid = cfg.expose_vit_grid
    editor.decode_vision_gate_enabled = cfg.decode_vision_gate
    editor.decode_vision_margin = float(cfg.decode_vision_margin)
    editor.decode_vision_abs_threshold = float(cfg.decode_vision_abs_threshold)
    if cfg.decode_vision_gate:
        editor.enable_decode_vision_gate(
            margin=cfg.decode_vision_margin,
            abs_threshold=cfg.decode_vision_abs_threshold,
        )
