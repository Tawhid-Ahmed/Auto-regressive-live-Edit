"""Frozen patch encoders for PatchGroundingModule (DINOv2 default).

Paper: DINOv2 supplies spatial patch tokens for Dual-Expert Keys vision scoring
and AVBT binding on adversarial images. Encoder weights stay frozen; only the
trainable PatchGroundingModule / MoE generators are optimized.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def patch_mean_key(patches: torch.Tensor, skip_cls: bool = True) -> torch.Tensor:
    """Mean-pool patch tokens and L2-normalize → [D] (pool dedup / decode gate)."""
    if patches.dim() == 3:
        patches = patches[0]
    if skip_cls and patches.shape[0] > 1:
        patches = patches[1:]
    return F.normalize(patches.float().mean(dim=0), dim=0)


class FrozenPatchEncoder:
    """Encode images to a patch grid with a frozen vision backbone (paper DINOv2)."""

    DINOV2_DIM = 768
    EVA_DIM = 1408

    def __init__(self, name: str, device: str) -> None:
        self.name = name.lower()
        self.device = device
        self._encoder = None
        self.hidden_dim = self.DINOV2_DIM if self.name == "dinov2" else self.EVA_DIM

    def _lazy_init(self) -> None:
        if self._encoder is not None:
            return
        if self.name == "dinov2":
            from transformers import AutoImageProcessor, AutoModel

            model_id = "facebook/dinov2-base"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id).to(self.device).eval()
            self.hidden_dim = int(self._model.config.hidden_size)
        elif self.name == "eva_patches":
            self._vllm = None
        else:
            raise ValueError("Unknown patch encoder %r (use dinov2 or eva_patches)" % self.name)
        self._encoder = True

    @torch.no_grad()
    def encode_patches(self, image, vllm=None) -> torch.Tensor:
        """Return patch grid [1, P, D] on CPU for storage efficiency."""
        from dataset.vllm import pil_image_from_path_or_pil

        image = pil_image_from_path_or_pil(image)
        if image is None:
            raise ValueError("encode_patches received None image (skip text-only samples upstream)")
        self._lazy_init()
        if self.name == "dinov2":
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self._model(**inputs)
            return out.last_hidden_state.detach().cpu()
        assert vllm is not None, "eva_patches encoder requires BLIP2 vllm"
        return vllm.get_vit_patch_embeds(image).detach().cpu()

    def patch_dim(self) -> int:
        self._lazy_init()
        return self.hidden_dim
