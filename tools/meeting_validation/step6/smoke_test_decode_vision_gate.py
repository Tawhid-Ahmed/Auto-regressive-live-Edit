"""Unit tests for VG-Edit decode vision gate logic (no GPU / model load).

Run:
  python tools/meeting_validation/step6/smoke_test_decode_vision_gate.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from editor.vllm_editors.liveedit.patch_encoder import patch_mean_key
from tools.meeting_validation.step6.layer_ablation import AblationConfig, apply_ablation_config


def _decode_vision_gate_filter(
    retrieval_map_gated: torch.Tensor,
    pool_indices: torch.Tensor,
    query_patches: torch.Tensor,
    dino_pool: list,
    device: str,
    margin: float,
    abs_threshold: float | None = None,
) -> torch.Tensor:
    if not dino_pool or not retrieval_map_gated.any():
        return retrieval_map_gated
    query_key = patch_mean_key(query_patches).to(device)
    gated_pos = retrieval_map_gated.nonzero(as_tuple=True)[0]
    full_indices = [int(pool_indices[i].item()) for i in gated_pos.tolist()]
    sims = torch.stack([dino_pool[i].to(device) for i in full_indices], dim=0)
    sims = torch.mv(sims, query_key)
    out = torch.zeros_like(retrieval_map_gated)
    for p, sim_val in enumerate(sims.tolist()):
        if abs_threshold is not None and sim_val < float(abs_threshold):
            continue
        other = [s for j, s in enumerate(sims.tolist()) if j != p]
        other_max = max(other) if other else -1e9
        if sim_val >= other_max + margin:
            out[gated_pos[p]] = True
    return out


class _GateStub:
    device = "cpu"
    decode_vision_margin = 0.0
    decode_vision_abs_threshold = None
    dino_key_pool: list = []

    def _decode_vision_gate_filter(self, retrieval_map_gated, pool_indices, query_patches):
        return _decode_vision_gate_filter(
            retrieval_map_gated,
            pool_indices,
            query_patches,
            self.dino_key_pool,
            self.device,
            self.decode_vision_margin,
            self.decode_vision_abs_threshold,
        )


def test_patch_mean_key_unit() -> None:
    p = torch.randn(1, 10, 8)
    k = patch_mean_key(p)
    assert k.shape == (8,)
    assert abs(float(k.norm()) - 1.0) < 1e-5
    print("[PASS] patch_mean_key")


def test_gate_keeps_best_matching_expert() -> None:
    stub = _GateStub()
    keys = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    stub.dino_key_pool = [F.normalize(k, dim=0) for k in keys]
    query = torch.tensor([[1.0, 0.0]])
    query_patches = query.unsqueeze(0)
    pool_indices = torch.arange(2)
    retrieval = torch.tensor([True, True])
    out = stub._decode_vision_gate_filter(retrieval, pool_indices, query_patches)
    assert out.tolist() == [True, False], out.tolist()
    print("[PASS] gate keeps best-matching expert only")


def test_gate_margin_blocks_tie() -> None:
    stub = _GateStub()
    stub.decode_vision_margin = 0.1
    k = torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0)
    stub.dino_key_pool = [k.clone(), k.clone()]
    query_patches = k.unsqueeze(0).unsqueeze(0)
    pool_indices = torch.arange(2)
    retrieval = torch.tensor([True, True])
    out = stub._decode_vision_gate_filter(retrieval, pool_indices, query_patches)
    assert not out.any(), "positive margin should block tied experts"
    print("[PASS] gate margin blocks ties")


def test_abs_threshold_blocks_single_expert() -> None:
    stub = _GateStub()
    k_expert = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
    stub.dino_key_pool = [k_expert]
    stub.decode_vision_margin = 0.0
    stub.decode_vision_abs_threshold = 0.45
    q_wrong = F.normalize(torch.tensor([0.0, 1.0]), dim=0).unsqueeze(0).unsqueeze(0)
    out = stub._decode_vision_gate_filter(
        torch.tensor([True]), torch.arange(1), q_wrong
    )
    assert not out.any(), "wrong image should block lone expert"
    q_match = F.normalize(torch.tensor([1.0, 0.0]), dim=0).unsqueeze(0).unsqueeze(0)
    out = stub._decode_vision_gate_filter(
        torch.tensor([True]), torch.arange(1), q_match
    )
    assert out.any(), "matching image should allow lone expert"
    print("[PASS] abs threshold blocks single expert (v2)")


def test_vg_ablation_config() -> None:
    cfg = AblationConfig.for_layer("VG")
    assert cfg.decode_vision_gate
    assert cfg.name == "VG"
    print("[PASS] VG ablation config")


def test_apply_vg_sets_flags() -> None:
    class Ed:
        def __init__(self):
            self.patch_grounding_enabled = False
            self.patch_grounding_module = None
            self.decode_vision_gate_enabled = False
            self.decode_vision_margin = 0.0
            self.patch_encoder_name = "dinov2"
            self.vllm = type("V", (), {})()
            self._patch_cache_wrapped = False

        def enable_patch_grounding(self, **kw):
            pass

        def enable_decode_vision_gate(self, margin=0.0, encoder="dinov2", abs_threshold=0.45):
            self.decode_vision_gate_enabled = True
            self.decode_vision_margin = margin
            self.decode_vision_abs_threshold = abs_threshold

        def _install_patch_image_cache_hook(self):
            pass

    ed = Ed()
    apply_ablation_config(ed, AblationConfig.for_layer("VG"))
    assert ed.decode_vision_gate_enabled
    print("[PASS] apply_ablation VG")


if __name__ == "__main__":
    test_patch_mean_key_unit()
    test_gate_keeps_best_matching_expert()
    test_gate_margin_blocks_tie()
    test_abs_threshold_blocks_single_expert()
    test_vg_ablation_config()
    test_apply_vg_sets_flags()
    print("\nAll VG-Edit decode gate smoke tests passed.")
