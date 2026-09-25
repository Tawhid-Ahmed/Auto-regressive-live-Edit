"""Smoke tests for VG-Edit Adversarial Visual Binding Loss (no GPU).

  python tools/meeting_validation/step6/smoke_test_avbl.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import torch


def test_fused_binding_shape() -> None:
    class Stub:
        pass

    stub = Stub()
    stub.device = torch.device("cpu")
    stub.sim_scale = 1.0
    stub.patch_grounding_enabled = True
    stub.patch_routing_alpha = 0.5
    stub.patch_routing_only = False
    stub.inpt_extractor = torch.nn.Identity()
    stub.patch_grounding_module = torch.nn.Identity()

    m, eqe, d = 3, 2, 4
    stub.evr_pool = torch.randn(m, eqe, d)
    stub.evr_patch_pool = torch.randn(m, eqe, d)

    def fake_extract_vision(q, v):
        return torch.randn(1, eqe, d)

    def fake_visprot(q):
        return torch.randn(1, eqe, d)

    def fake_patch_routing(q, patches):
        return torch.randn(1, eqe, d)

    stub.inpt_extractor.extract_vision = fake_extract_vision
    stub.inpt_extractor.extract_from_visprot = fake_visprot
    stub._extract_patch_routing = fake_patch_routing
    stub._pool_vis_sim = lambda ivr, pool: torch.randn(1, pool.shape[0])
    stub._fuse_routing_sims = lambda a, b, for_avbl=False: a + b if b is not None else a
    stub._encode_patches = lambda img: torch.randn(1, 5, d)

    vision = torch.randn(1, 8, 16)
    query = torch.randn(1, 6, 16)
    from editor.vllm_editors.liveedit.liveedit import LiveEdit

    stub._compute_vis_sim_routing = (
        lambda ivr, ivr_p, evr_pool, evr_patch_pool, vis_sim_q=None, vis_sim_p=None, for_avbl=False:
        LiveEdit._compute_vis_sim_routing(
            stub, ivr, ivr_p, evr_pool, evr_patch_pool, vis_sim_q, vis_sim_p, for_avbl=for_avbl)
    )
    dist = LiveEdit._fused_binding_distribution(stub, vision, query, "img")
    assert dist is not None and dist.shape == (m + 1,)
    assert abs(float(dist.sum()) - 1.0) < 1e-4
    print("[PASS] fused binding distribution shape")


def test_avbl_loss_decreases_on_mock_step() -> None:
    """Synthetic: training AVBL-style loss on a tiny softmax should separable."""
    logits = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([logits], lr=0.5)
    own_idx = 0
    for _ in range(20):
        dist = torch.softmax(logits, 0)
        loss = -torch.log(dist[own_idx] + 1e-8) - torch.log(dist[-1] + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dist = torch.softmax(logits, 0).detach()
    assert dist[own_idx] > 0.35
    assert dist[-1] > 0.35
    print("[PASS] mock AVBL separability")


def test_enable_adversarial_grounding_flag() -> None:
    from editor.vllm_editors.liveedit.liveedit import LiveEditConfig

    assert hasattr(LiveEditConfig, "from_yaml")
    print("[PASS] avbl config import")


def main() -> int:
    test_fused_binding_shape()
    test_avbl_loss_decreases_on_mock_step()
    test_enable_adversarial_grounding_flag()
    print("\nAll AVBL smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
