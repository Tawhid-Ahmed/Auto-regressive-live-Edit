"""Smoke test PatchGroundingModule forward + ablation flag wiring (no GPU model load).

Run:
  python tools/meeting_validation/step6/smoke_test_patch_grounding.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from editor.vllm_editors.liveedit.modules import PatchGroundingModule
from tools.meeting_validation.step6.layer_ablation import (
    LAYER_NAMES,
    AblationConfig,
    LAYER_TABLE,
    apply_ablation_config,
)


def test_patch_module_shapes() -> None:
    mod = PatchGroundingModule(
        patch_dim=768,
        llm_mid_dim=2560,
        module_dim=1024,
        cross_att_head_n=8,
        eqe_n=4,
    )
    q = torch.randn(1, 12, 2560)
    p = torch.randn(1, 257, 768)
    evr = mod(q, p)
    assert evr.shape == (1, 4, 1024), evr.shape
    prot = mod.extract_from_prototype(q)
    assert prot.shape == (1, 4, 1024), prot.shape
    print("[PASS] PatchGroundingModule shapes")


def test_ablation_configs() -> None:
    for layer in LAYER_NAMES:
        cfg = AblationConfig.for_layer(layer)
        assert cfg.name == layer
    assert len(LAYER_TABLE) == len(LAYER_NAMES)
    print("[PASS] ablation configs for", LAYER_NAMES)


class _StubEditor:
  def __init__(self):
      self.patch_grounding_module = None
      self.moegen_c_patch = None
      self.vllm = type("V", (), {"set_expose_vit_grid": lambda *a, **k: None})()

  def enable_patch_grounding(self, encoder="dinov2", expert_gen=True, patch_dim=None):
      self.patch_grounding_enabled = True
      self.patch_encoder_name = encoder
      self.patch_expert_gen_enabled = expert_gen
      self.patch_grounding_module = object()
      self._install_patch_image_cache_hook = lambda: None


def test_apply_ablation() -> None:
    ed = _StubEditor()
    apply_ablation_config(ed, AblationConfig.for_layer("L3"))
    assert ed.patch_grounding_enabled
    assert ed.patch_expert_gen_enabled
    apply_ablation_config(ed, AblationConfig.for_layer("L0"))
    assert not ed.patch_grounding_enabled
    print("[PASS] apply_ablation_config")


if __name__ == "__main__":
    test_patch_module_shapes()
    test_ablation_configs()
    test_apply_ablation()
    print("\nAll Step 6 smoke tests passed.")
