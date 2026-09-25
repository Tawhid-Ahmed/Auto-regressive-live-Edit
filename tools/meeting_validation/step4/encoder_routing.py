"""Offline encoder routing helpers for Step 4 (EVA patches vs Q-Former vs DINOv2).

Inference-only probes — no editor retrain. Patch encoders use mean-pooled L2-normalized
patch grids with competitive margin routing (own-expert sim vs best other).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ENCODER_NAMES = ("qformer", "eva_patches", "dinov2", "siglip")


@dataclass
class RoutingOutcome:
    encoder: str
    probe: str
    sample_id: int
    sims: List[float]
    sim_prot: float
    retrieved_indices: List[int]
    false_fire_on_target: bool
    miss: bool
    false_cross_edit: bool
    routing_ok: bool
    score_own_expert: float
    score_best_other: float
    margin_own_vs_other: float


def expected_pool_indices(sample_id: int, pool_sample_ids: List[int]) -> Set[int]:
    return {i for i, sid in enumerate(pool_sample_ids) if sid == sample_id}


def classify_retrieval(
    probe: str,
    sample_id: int,
    retrieved_indices: List[int],
    pool_sample_ids: List[int],
) -> Dict[str, bool]:
    expected = expected_pool_indices(sample_id, pool_sample_ids)
    retrieved = set(retrieved_indices)
    out: Dict[str, bool] = {}
    if probe == "hard_negative":
        out["false_fire_on_target"] = any(
            pool_sample_ids[i] == sample_id for i in retrieved_indices if i < len(pool_sample_ids)
        )
        out["miss"] = False
        out["false_cross_edit"] = out["false_fire_on_target"]
        out["routing_ok"] = not out["false_fire_on_target"]
    elif probe in ("reliability", "text_rephrase", "image_rephrase"):
        out["miss"] = len(expected & retrieved) == 0 and len(expected) > 0
        out["false_cross_edit"] = any(
            pool_sample_ids[i] != sample_id for i in retrieved_indices if i < len(pool_sample_ids)
        )
        out["false_fire_on_target"] = False
        out["routing_ok"] = expected.issubset(retrieved) and not out["false_cross_edit"]
    else:
        out["miss"] = False
        out["false_cross_edit"] = False
        out["false_fire_on_target"] = False
        out["routing_ok"] = True
    return out


def patch_mean_key(patches: torch.Tensor, skip_cls: bool = True) -> torch.Tensor:
    """Mean-pool patch tokens and L2-normalize → [D]."""
    if patches.dim() == 3:
        patches = patches[0]
    if skip_cls and patches.shape[0] > 1:
        patches = patches[1:]
    key = patches.float().mean(dim=0)
    return F.normalize(key, dim=0)


def _own_other_indices(sample_id: int, pool_sample_ids: List[int]) -> Tuple[List[int], List[int]]:
    own = [i for i, sid in enumerate(pool_sample_ids) if sid == sample_id]
    other = [i for i, sid in enumerate(pool_sample_ids) if sid != sample_id]
    return own, other


def score_own_expert(sims: torch.Tensor, sample_id: int, pool_sample_ids: List[int]) -> float:
    own, _ = _own_other_indices(sample_id, pool_sample_ids)
    if not own:
        return float(sims.max().item()) if sims.numel() else 0.0
    return float(sims[own].max().item())


def score_best_other(sims: torch.Tensor, sample_id: int, pool_sample_ids: List[int]) -> float:
    _, other = _own_other_indices(sample_id, pool_sample_ids)
    if not other:
        return 0.0
    return float(sims[other].max().item())


def margin_scores(sims: torch.Tensor, sample_id: int, pool_sample_ids: List[int]) -> Tuple[float, float, float]:
    own_s = score_own_expert(sims, sample_id, pool_sample_ids)
    other_s = score_best_other(sims, sample_id, pool_sample_ids)
    return own_s, other_s, own_s - other_s


def offline_patch_route(
    query_key: torch.Tensor,
    pool_keys: torch.Tensor,
    scale: float,
    sample_id: int,
    pool_sample_ids: List[int],
) -> Tuple[torch.Tensor, float, List[int]]:
    """Competitive patch routing: retrieve own experts that beat the best other expert."""
    q = query_key.float()
    pool = pool_keys.float()
    sims = (pool @ q) * scale
    own, other = _own_other_indices(sample_id, pool_sample_ids)
    other_max = float(sims[other].max().item()) if other else -1e9
    retrieved = [i for i in own if float(sims[i].item()) >= other_max]
    _, _, margin = margin_scores(sims, sample_id, pool_sample_ids)
    return sims, margin, retrieved


def qformer_route_from_probe(routing: Dict[str, Any], sample_id: int, pool_sample_ids: List[int], probe: str):
    sims = torch.tensor(routing["vis_sim"], dtype=torch.float32)
    prot = float(routing["vis_sim_prot"])
    retrieved = routing["retrieved_indices"]
    cls = classify_retrieval(probe, sample_id, retrieved, pool_sample_ids)
    own_s, other_s, margin = margin_scores(sims, sample_id, pool_sample_ids)
    return RoutingOutcome(
        encoder="qformer",
        probe=probe,
        sample_id=sample_id,
        sims=routing["vis_sim"],
        sim_prot=prot,
        retrieved_indices=retrieved,
        score_own_expert=own_s,
        score_best_other=other_s,
        margin_own_vs_other=margin,
        **cls,
    )


def patch_route_outcome(
    encoder: str,
    probe: str,
    sample_id: int,
    query_key: torch.Tensor,
    pool_keys: torch.Tensor,
    pool_sample_ids: List[int],
    scale: float,
) -> RoutingOutcome:
    sims, margin, retrieved = offline_patch_route(
        query_key, pool_keys, scale, sample_id, pool_sample_ids
    )
    cls = classify_retrieval(probe, sample_id, retrieved, pool_sample_ids)
    own_s, other_s, _ = margin_scores(sims, sample_id, pool_sample_ids)
    return RoutingOutcome(
        encoder=encoder,
        probe=probe,
        sample_id=sample_id,
        sims=sims.cpu().tolist(),
        sim_prot=float(margin),
        retrieved_indices=retrieved,
        score_own_expert=own_s,
        score_best_other=other_s,
        margin_own_vs_other=float(margin),
        **cls,
    )


def binary_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0:
        return None
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")[::-1]
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapz(tpr, fpr))


class EvaPatchEncoder:
  """Pre-Q-Former EVA-ViT patch grid from BLIP2."""

  def __init__(self, vllm) -> None:
      self.vllm = vllm
      self.hidden_dim: Optional[int] = None

  @torch.no_grad()
  def encode_patches(self, image) -> torch.Tensor:
      proc = self.vllm.processor
      model = self.vllm.model
      inpt = proc(images=image, return_tensors="pt")
      pixel_values = inpt["pixel_values"].to(self.vllm.device)
      model_dtype = next(model.parameters()).dtype
      pixel_values = pixel_values.to(dtype=model_dtype)
      interpolate_pos = (
          getattr(self.vllm, "eval_image_size", None) is not None
          and self.vllm.eval_image_size != getattr(self.vllm, "native_image_size", 224)
      )
      vision_outputs = model.vision_model(
          pixel_values=pixel_values,
          interpolate_pos_encoding=interpolate_pos,
          return_dict=True,
      )
      patches = vision_outputs[0]
      self.hidden_dim = int(patches.shape[-1])
      return patches

  def key(self, image) -> torch.Tensor:
      return patch_mean_key(self.encode_patches(image))


class ExternalPatchEncoder:
  """Frozen DINOv2 or SigLIP patch encoder with fixed mean-pool projector."""

  def __init__(self, name: str, device: str) -> None:
      from transformers import AutoImageProcessor, AutoModel

      self.name = name
      self.device = device
      if name == "dinov2":
          model_id = "facebook/dinov2-base"
      elif name == "siglip":
          model_id = "google/siglip-base-patch16-224"
      else:
          raise ValueError("Unknown external encoder: %s" % name)
      self.processor = AutoImageProcessor.from_pretrained(model_id)
      self.model = AutoModel.from_pretrained(model_id).to(device).eval()
      self.hidden_dim = int(self.model.config.hidden_size)

  @torch.no_grad()
  def encode_patches(self, image) -> torch.Tensor:
      inputs = self.processor(images=image, return_tensors="pt")
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      if self.name == "siglip":
          out = self.model.vision_model(**inputs)
          patches = out.last_hidden_state
      else:
          out = self.model(**inputs)
          patches = out.last_hidden_state
      return patches

  def key(self, image) -> torch.Tensor:
      return patch_mean_key(self.encode_patches(image), skip_cls=self.name != "siglip")


def summarize_encoder(outcomes: List[RoutingOutcome]) -> Dict[str, Any]:
    by_probe: Dict[str, List[RoutingOutcome]] = {}
    for o in outcomes:
        by_probe.setdefault(o.probe, []).append(o)

    def _rate(probe: str, attr: str) -> Optional[float]:
        rows = by_probe.get(probe, [])
        if not rows:
            return None
        return float(np.mean([getattr(r, attr) for r in rows]))

    hard = by_probe.get("hard_negative", [])
    auc_labels = [int(o.false_fire_on_target) for o in hard]
    auc_scores_margin = [o.margin_own_vs_other for o in hard]
    auc_scores_own = [o.score_own_expert for o in hard]

    return {
        "n_probes": len(outcomes),
        "hard_neg_false_fire_rate": _rate("hard_negative", "false_fire_on_target"),
        "hard_neg_auc_margin": binary_auc(auc_labels, auc_scores_margin),
        "hard_neg_auc_own_expert": binary_auc(auc_labels, auc_scores_own),
        "reliability_miss_rate": _rate("reliability", "miss"),
        "reliability_false_cross_rate": _rate("reliability", "false_cross_edit"),
        "mean_score_own_expert": {
            probe: float(np.mean([o.score_own_expert for o in rows])) if rows else None
            for probe, rows in by_probe.items()
        },
    }
