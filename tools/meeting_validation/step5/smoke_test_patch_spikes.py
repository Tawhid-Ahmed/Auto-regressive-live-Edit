"""Smoke test Step 5 patch spike flags (no GPU checkpoint required for chunk utils).

  python tools/meeting_validation/step5/smoke_test_patch_spikes.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import torch

from editor.vllm_editors.liveedit.chunk_utils import (
    infer_ar_chunk_index,
    positional_chunk_index,
    vision_gated_chunk_index,
)
from tools.meeting_validation.step5.patch_spike_routing import SpikeConfig, SPIKE_NAMES


def test_chunk_utils() -> None:
    attn = torch.rand(32, 257)
    pos = positional_chunk_index(20, chunk_size=16, max_chunks=8)
    assert pos == 1
    vis = vision_gated_chunk_index(attn, 20, 16, 8)
    assert 0 <= vis < 8
    blended = infer_ar_chunk_index(attn, 20, 16, 8, vision_gated=True)
    assert blended == vis
    legacy = infer_ar_chunk_index(attn, 20, 16, 8, vision_gated=False)
    assert legacy == pos
    print("  [PASS] chunk_utils vision-gated + positional indices")


def test_spike_configs() -> None:
    for name in SPIKE_NAMES:
        cfg = SpikeConfig.for_spike(name)
        assert cfg.name
    b = SpikeConfig.for_spike("4b")
    assert b.patch_rerank_enabled and b.expose_vit_grid
    c = SpikeConfig.for_spike("4c")
    assert c.ar_vision_gated_chunks and c.ar_mode
    print("  [PASS] SpikeConfig for all spikes")


def test_pool_vis_sim_shapes() -> None:
    """Shape check for routing pool modes (no full editor load)."""
    per_eqe = torch.randn(1, 5, 32)
    for mode in ("mean", "max", "attn"):
        if mode == "mean":
            out = per_eqe.mean(2)
        elif mode == "max":
            out = per_eqe.max(dim=2).values
        else:
            weights = torch.softmax(per_eqe[0], dim=1)
            out = (weights * per_eqe[0]).sum(dim=1).unsqueeze(0)
        assert out.shape == (1, 5), mode
    print("  [PASS] routing pool mean/max/attn output shapes")


def main() -> int:
    print("Step 5 smoke tests")
    test_chunk_utils()
    test_spike_configs()
    test_pool_vis_sim_shapes()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
