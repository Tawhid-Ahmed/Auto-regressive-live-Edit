from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditorWithTraining
from utils import find_module, move_to_device
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple, Optional
from dataset.vllm import BaseVLLMEditData
from .modules import QVExtractor, LowRankGenerator, PatchGroundingModule
from .patch_encoder import FrozenPatchEncoder, patch_mean_key
from .chunk_utils import split_target_into_chunks, reconstruct_text_from_chunks
from dataclasses import dataclass
from ...base import BaseConfig
from torch.optim import Adam
from collections import deque
import torch, os, yaml
from torch import nn
import numpy as np
import threading
import time

@dataclass
class LiveEditConfig(BaseConfig):
    @dataclass
    class TrainConfig:
        lr:float
        lr_cut_it:List[int]
        lr_cut_rate:float
        rel_lambda: float
        gen_lambda: float
        loc_lambda: float
        soft_routing_lambda: float
        hard_routing_lambda: float
    @dataclass
    class RetrievalEditor:
        module_dim: int
        cross_att_head_n: int
        lora_rank: int
        lora_scale: float
        eqe_n: int # editing query representation
    edit_model_name: str
    retrieval_editor: RetrievalEditor
    train_cfg: TrainConfig
    llm_mid_dim: int
    llm_layer_tmp: str
    edit_layer_i: int

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_cfg'] = self.TrainConfig(**data['train_cfg'])
        data['retrieval_editor'] = self.RetrievalEditor(**data['retrieval_editor'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise

class LiveEdit(VLLMBaseEditorWithTraining):
    '''Lifelong Vision Language Model Editor'''

    def __init__(self, vllm: BaseVLLMForEdit, config:LiveEditConfig, device='cuda:0', 
            vllm_data_proc: BaseVLLMForEdit = None, data_proc_device = None, verbose = False):
        super().__init__(vllm, config, device)
        self.cfg = config
        self.verbose = verbose
        if vllm_data_proc != None:
            self.vllm_data_proc = vllm_data_proc
            self.vllm_data_proc.model.eval()
            self.vllm_data_proc.model.requires_grad_(False)
            self.data_proc_device = data_proc_device
            self.vllm_data_proc.set_device(data_proc_device)
        # initialize training parameters
        self.sim_scale = 1 / self.cfg.retrieval_editor.module_dim**0.5
        (eqe_n, llm_mid_dim, module_dim, cross_att_head_n, lora_rank, lora_scale) = (self.cfg.retrieval_editor.eqe_n, 
            self.cfg.llm_mid_dim, self.cfg.retrieval_editor.module_dim, self.cfg.retrieval_editor.cross_att_head_n, 
            self.cfg.retrieval_editor.lora_rank, self.cfg.retrieval_editor.lora_scale)
        self.edit_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vllm.get_img_token_n(), False).to(self.device)
        self.inpt_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vllm.get_img_token_n(), True).to(self.device)
        self.moegen_c = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.moegen_r = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.instant_reps_norm = nn.LayerNorm(self.cfg.llm_mid_dim).to(self.device)
        # Step 5 inference spikes (opt-in; legacy defaults preserve mean-pool routing)
        self.routing_pool_mode = "mean"  # mean | max | attn
        self.patch_rerank_enabled = False
        self.patch_rerank_top_k = 8
        self.ar_vision_gated_chunks = False
        self.vit_patch_pool: List[torch.Tensor] = []
        # Step 6 patch-grounding layer (opt-in; legacy path when disabled)
        self.patch_grounding_enabled = False
        self.patch_encoder_name = "dinov2"
        self.patch_routing_only = False
        self.patch_expert_gen_enabled = False
        self.verification_gate_enabled = False
        self.patch_routing_alpha = 0.5
        # VG-Edit: decode-time DINO gate (margin + optional absolute match, v2)
        self.decode_vision_gate_enabled = False
        self.decode_vision_margin = 0.0
        self.decode_vision_abs_threshold: Optional[float] = None
        self.dino_key_pool: List[torch.Tensor] = []
        # DEK: dual-gate (vision ∧ text) + optional legacy embed-fuse + pool dedup
        self.fused_expert_key_enabled = False  # legacy Q+patch embed fuse — collapses rel; prefer dual_gate
        self.dek_dual_gate_enabled = False
        self.dek_text_margin = 0.0
        self.dek_text_alpha = 1.0  # AVBL: add alpha * txt_sim into binding logits
        # G2 inference routing rule (glossary / --routing_rule). When dual_gate is on and
        # rule is still the default, retrieve_moes treats it as ``and``.
        self.routing_rule = "liveedit"
        self.carml_scope_tau = 0.05  # min fused prob for CARML top-1 to fire
        self.carml_top_k = 15
        self.pool_dedup_enabled = False
        self.pool_dedup_threshold = 0.95
        self.pool_dedup_mode = "merge"  # merge | skip
        # VG-Edit AVBL: adversarial visual binding loss (training-time)
        self.adversarial_grounding_lambda: float = 0.0
        self.adversarial_grounding_neg_weight: float = 1.0
        self.avbl_hinge_tau: float = 0.0
        self.avbl_use_contrastive: bool = False
        self.avbl_contrastive_margin: float = 0.1
        self.avbl_targeted_false_cross: bool = True
        self.avbl_targeted_false_cross_weight: float = 1.0
        self.avbl_rel_anchor_min: float = 0.5
        # Retrieval-margin: hinge on pre-softmax vis_sim (0=off)
        self.avbl_retrieval_margin: float = 0.0
        # Hard DINO false-cross neighbor mining + token refuse on adv images
        self.avbl_dino_neighbor_mine: bool = False
        self.avbl_neighbor_weight: float = 2.0
        self.avbl_refuse_lambda: float = 0.0
        self.avbl_refuse_tau: float = -1.0  # penalize if mean logP(target|adv) > tau
        # Phase D: cross-batch buffer of recent (dino_key, moe_c, moe_r, eqr)
        self.avbl_pool_buffer_size: int = 0
        self._avbl_pool_buffer: deque = deque()
        self._patch_encoder: Optional[FrozenPatchEncoder] = None
        self._last_query_patches: Optional[torch.Tensor] = None
        self.patch_grounding_module: Optional[PatchGroundingModule] = None
        self.moegen_c_patch: Optional[LowRankGenerator] = None
        # for inference & other hyper-parameters
        self.edit_layer_path = self.cfg.llm_layer_tmp.format(self.cfg.edit_layer_i)
        self.train_edit_residual = None  # Set in train_a_batch; hook must not assume it exists during data prep
        self._model_forward_lock = threading.Lock()  # avoid data-loader and train_a_batch running model concurrently
        self.wrap_and_hook()
        self.restore_to_original_model()
        self.set_train(False)
        self.is_editing = False

    def enable_patch_grounding(
        self,
        encoder: str = "dinov2",
        expert_gen: bool = True,
        patch_dim: Optional[int] = None,
    ) -> None:
        """Construct trainable patch-grounding modules (Step 6 L1/L3)."""
        self.patch_grounding_enabled = True
        self.patch_encoder_name = encoder
        self.patch_expert_gen_enabled = expert_gen
        if patch_dim is not None:
            dim = patch_dim
        elif encoder.lower() == "dinov2":
            dim = FrozenPatchEncoder.DINOV2_DIM
        else:
            dim = FrozenPatchEncoder.EVA_DIM
        eqe_n = self.cfg.retrieval_editor.eqe_n
        module_dim = self.cfg.retrieval_editor.module_dim
        cross_att_head_n = self.cfg.retrieval_editor.cross_att_head_n
        lora_rank = self.cfg.retrieval_editor.lora_rank
        lora_scale = self.cfg.retrieval_editor.lora_scale
        llm_mid_dim = self.cfg.llm_mid_dim
        self.patch_grounding_module = PatchGroundingModule(
            dim, llm_mid_dim, module_dim, cross_att_head_n, eqe_n
        ).to(self.device)
        if expert_gen:
            self.moegen_c_patch = LowRankGenerator(
                llm_mid_dim, lora_rank, lora_scale, module_dim, module_dim, cross_att_head_n
            ).to(self.device)
        self._install_patch_image_cache_hook()

    def enable_decode_vision_gate(
        self,
        margin: float = 0.0,
        encoder: str = "dinov2",
        abs_threshold: Optional[float] = 0.45,
    ) -> None:
        """VG-Edit v2: block MoE unless DINO sim(query, expert_img) passes abs + margin gates."""
        self.decode_vision_gate_enabled = True
        self.decode_vision_margin = float(margin)
        self.decode_vision_abs_threshold = abs_threshold
        if not self.patch_grounding_enabled:
            self.patch_encoder_name = encoder
        self._install_patch_image_cache_hook()

    def enable_dek(
        self,
        fused_expert_key: bool = False,
        pool_dedup: bool = True,
        pool_dedup_threshold: float = 0.95,
        dual_gate: bool = True,
        text_margin: float = 0.0,
        text_alpha: float = 1.0,
        pool_dedup_mode: str = "merge",
        encoder: str = "dinov2",
    ) -> None:
        """DEK: vision∧text dual-gate retrieval + DINO pool hygiene.

        Prefer ``dual_gate`` (patch/Q vision score AND prompt-key agreement).
        Legacy ``fused_expert_key`` (QFormer+patch embedding blend) collapses
        reliability and should stay off unless recalibrated.
        """
        self.fused_expert_key_enabled = bool(fused_expert_key)
        self.dek_dual_gate_enabled = bool(dual_gate)
        self.dek_text_margin = float(text_margin)
        self.dek_text_alpha = float(text_alpha)
        if dual_gate and getattr(self, "routing_rule", "liveedit") in ("liveedit", "vision_only"):
            self.routing_rule = "and"
        self.pool_dedup_enabled = bool(pool_dedup)
        self.pool_dedup_threshold = float(pool_dedup_threshold)
        mode = str(pool_dedup_mode).lower().strip()
        self.pool_dedup_mode = mode if mode in ("merge", "skip") else "merge"
        if not self.patch_grounding_enabled:
            self.enable_patch_grounding(encoder=encoder, expert_gen=True)
        self._install_patch_image_cache_hook()

    ROUTING_RULES = (
        "liveedit",
        "vision_only",
        "text_only",
        "and",
        "or",
        "sum",
        "carml",
        "always_reject",
    )

    def set_routing_rule(self, rule: str, text_margin: Optional[float] = None,
                         carml_scope_tau: Optional[float] = None) -> None:
        """G2: select inference routing rule (see CVPR2027 glossary ``routing_rule``)."""
        r = str(rule).lower().strip()
        if r not in self.ROUTING_RULES:
            raise ValueError("routing_rule must be one of %s, got %r" % (self.ROUTING_RULES, rule))
        self.routing_rule = r
        if text_margin is not None:
            self.dek_text_margin = float(text_margin)
        if carml_scope_tau is not None:
            self.carml_scope_tau = float(carml_scope_tau)
        # Keep legacy flag in sync so older call sites stay consistent.
        self.dek_dual_gate_enabled = (r == "and")

    def enable_adversarial_grounding(
        self,
        lambda_adv: float = 1.0,
        neg_weight: float = 1.0,
        hinge_tau: float = 0.0,
        use_contrastive: bool = False,
        contrastive_margin: float = 0.1,
        targeted_false_cross: bool = True,
        rel_anchor_min: float = 0.5,
        retrieval_margin: float = 0.0,
        dino_neighbor_mine: bool = False,
        neighbor_weight: float = 2.0,
        refuse_lambda: float = 0.0,
        refuse_tau: float = -1.0,
        pool_buffer_size: int = 0,
    ) -> None:
        """VG-Edit AVBL: train with swap/hard-neg visual binding negatives."""
        self.adversarial_grounding_lambda = float(lambda_adv)
        self.adversarial_grounding_neg_weight = float(neg_weight)
        self.avbl_hinge_tau = float(hinge_tau)
        self.avbl_use_contrastive = bool(use_contrastive)
        self.avbl_contrastive_margin = float(contrastive_margin)
        self.avbl_targeted_false_cross = bool(targeted_false_cross)
        self.avbl_rel_anchor_min = float(rel_anchor_min)
        self.avbl_retrieval_margin = float(retrieval_margin)
        self.avbl_dino_neighbor_mine = bool(dino_neighbor_mine)
        self.avbl_neighbor_weight = float(neighbor_weight)
        self.avbl_refuse_lambda = float(refuse_lambda)
        self.avbl_refuse_tau = float(refuse_tau)
        self.avbl_pool_buffer_size = max(0, int(pool_buffer_size))
        self._avbl_pool_buffer = (
            deque(maxlen=self.avbl_pool_buffer_size)
            if self.avbl_pool_buffer_size > 0
            else deque()
        )
        if not self.patch_grounding_enabled:
            self.enable_patch_grounding(encoder="dinov2", expert_gen=True)

    def _forward_query_reps(
        self, vllm: BaseVLLMForEdit, prompt: str, image, target: str = "."
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mid-layer query/vision reps for a probe image (differentiable)."""
        (input_embeds, vt_range), _, label_masks = vllm.prompts_imgs_target_to_xym(
            [prompt], [image], [target])
        query_end = input_embeds['inputs_embeds'].shape[1] - label_masks.shape[1] + 1
        was_editing = self.is_editing
        self.is_editing = True
        try:
            edit_signal = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path)
        finally:
            self.is_editing = was_editing
        reps_f = edit_signal.float()
        return reps_f[:, vt_range[0]:vt_range[1]], reps_f[:, vt_range[1]:query_end]

    def _fused_binding_logits(
        self, vision_reps: torch.Tensor, query_reps: torch.Tensor, query_image,
        for_avbl: bool = False,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-softmax routing logits: (vis_sim [1,m], vis_sim_prot [1,1])."""
        m = int(self.evr_pool.shape[0])
        if m == 0:
            return None
        evr_pool = self.evr_pool.detach()
        evr_patch_pool = self.evr_patch_pool.detach()
        ivr = self.inpt_extractor.extract_vision(query_reps, vision_reps)
        vis_sim_q = self._pool_vis_sim(ivr, evr_pool)
        vis_sim_p = None
        query_patches = None
        ivr_p = None
        if (
            self.patch_grounding_enabled
            and self.patch_grounding_module is not None
            and evr_patch_pool.shape[0] == m
        ):
            query_patches = self._encode_patches(query_image)
            ivr_p = self._extract_patch_routing(query_reps, query_patches)
            vis_sim_p = self._pool_vis_sim(ivr_p, evr_patch_pool)
        vis_sim = self._compute_vis_sim_routing(
            ivr, ivr_p, evr_pool, evr_patch_pool, vis_sim_q, vis_sim_p, for_avbl=for_avbl)
        # DEK dual-gate training: fold prompt-key sims into binding logits so
        # AVBL optimizes the same vision∧text signal used at retrieve_moes.
        if getattr(self, "dek_dual_gate_enabled", False) and self.eqr_pool.shape[0] == m:
            txt_sim, _ = self._text_routing_sim(query_reps, self.eqr_pool.detach())
            alpha = float(getattr(self, "dek_text_alpha", 1.0))
            vis_sim = vis_sim + alpha * txt_sim
        patch_only = self.patch_routing_only and not for_avbl
        if (
            patch_only
            and self.patch_grounding_enabled
            and self.patch_grounding_module is not None
            and query_patches is not None
        ):
            ivr_prot = self._extract_patch_prototype_routing(query_reps)
            ivr_base = ivr_p if vis_sim_p is not None else ivr
        else:
            ivr_prot = self.inpt_extractor.extract_from_visprot(query_reps)
            ivr_base = ivr_p if (for_avbl and ivr_p is not None and vis_sim_p is not None) else ivr
        vis_sim_prot = (
            torch.einsum("bed,bed->be", ivr_base, ivr_prot).mean(1, True) * self.sim_scale
        )
        return vis_sim, vis_sim_prot

    def _fused_binding_distribution(
        self, vision_reps: torch.Tensor, query_reps: torch.Tensor, query_image,
        for_avbl: bool = False,
    ) -> Optional[torch.Tensor]:
        """Softmax routing distribution over pool experts + prototype [m+1]."""
        logits = self._fused_binding_logits(vision_reps, query_reps, query_image, for_avbl=for_avbl)
        if logits is None:
            return None
        vis_sim, vis_sim_prot = logits
        return torch.softmax(torch.cat([vis_sim, vis_sim_prot], 1), 1)[0]

    def compute_avbl_positive_loss(
        self,
        vllm: BaseVLLMForEdit,
        prompt: str,
        rel_image,
        own_pool_indices: List[int],
        target: str = ".",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-8
        log: Dict[str, float] = {}
        vision_reps, query_reps = self._forward_query_reps(vllm, prompt, rel_image, target)
        vision_reps = vision_reps.detach()
        query_reps = query_reps.detach()
        dist_pos = self._fused_binding_distribution(
            vision_reps, query_reps, rel_image, for_avbl=True)
        if dist_pos is None or not own_pool_indices:
            return torch.tensor(0.0, device=self.device), log
        own_idx = torch.tensor(own_pool_indices, device=self.device, dtype=torch.long)
        own_prob = dist_pos[own_idx].sum().clamp(eps, 1.0 - eps)
        loss_pos = -torch.log(own_prob)
        log['avbl_rel_loss'] = float(loss_pos.detach())
        log['avbl_rel_own_prob'] = float(own_prob.detach())
        # Retrieval margin: own logit should beat best-other by margin
        ret_m = float(getattr(self, "avbl_retrieval_margin", 0.0))
        if ret_m > 0:
            logits = self._fused_binding_logits(
                vision_reps, query_reps, rel_image, for_avbl=True)
            if logits is not None:
                vis_sim, _ = logits
                own_logit = vis_sim[0, own_idx].max()
                other_mask = torch.ones(vis_sim.shape[1], dtype=torch.bool, device=self.device)
                other_mask[own_idx] = False
                if other_mask.any():
                    other_logit = vis_sim[0, other_mask].max()
                    loss_rm = torch.relu(ret_m - (own_logit - other_logit))
                    loss_pos = loss_pos + loss_rm
                    log['avbl_ret_margin_pos'] = float(loss_rm.detach())
                    log['avbl_ret_margin_gap'] = float((own_logit - other_logit).detach())
        return loss_pos, log

    def compute_avbl_negative_loss(
        self,
        vllm: BaseVLLMForEdit,
        prompt: str,
        adv_image,
        own_pool_indices: Optional[List[int]] = None,
        target: str = ".",
        hard_neighbor_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-8
        log: Dict[str, float] = {}
        v_adv, q_adv = self._forward_query_reps(vllm, prompt, adv_image, target)
        v_adv = v_adv.detach()
        q_adv = q_adv.detach()
        dist_neg = self._fused_binding_distribution(v_adv, q_adv, adv_image, for_avbl=True)
        if dist_neg is None:
            return torch.tensor(0.0, device=self.device), log
        prot_prob = dist_neg[-1].clamp(eps, 1.0 - eps)
        expert_mass = dist_neg[:-1].sum()
        hinge_tau = float(getattr(self, "avbl_hinge_tau", 0.0))
        hinge = torch.relu(expert_mass - hinge_tau) if hinge_tau > 0 else torch.tensor(1.0, device=self.device)
        loss_neg = hinge * (-torch.log(prot_prob))
        wrong_prob = None
        own_set = set(own_pool_indices) if own_pool_indices is not None else set()
        if own_pool_indices is not None:
            wrong_idx = [i for i in range(dist_neg.shape[0] - 1) if i not in own_set]
            if wrong_idx:
                wrong_idx_t = torch.tensor(wrong_idx, device=self.device, dtype=torch.long)
                wrong_prob = dist_neg[wrong_idx_t].max()
                if getattr(self, "avbl_use_contrastive", False):
                    margin = float(getattr(self, "avbl_contrastive_margin", 0.1))
                    loss_contrast = torch.relu(wrong_prob - margin)
                    loss_neg = loss_neg + loss_contrast
                    log['avbl_contrast_loss'] = float(loss_contrast.detach())
                if getattr(self, "avbl_targeted_false_cross", True):
                    wp = wrong_prob.clamp(eps, 1.0 - eps)
                    w = float(getattr(self, "avbl_targeted_false_cross_weight", 1.0))
                    loss_targeted = -torch.log(1.0 - wp) * w
                    loss_neg = loss_neg + loss_targeted
                    log['avbl_targeted_loss'] = float(loss_targeted.detach())
                log['avbl_wrong_prob'] = float(wrong_prob.detach())
                log['avbl_top_wrong_prob'] = float(wrong_prob.detach())
            # Hard DINO-mined neighbor: suppress that specific expert under adv image
            if (
                hard_neighbor_idx is not None
                and 0 <= int(hard_neighbor_idx) < dist_neg.shape[0] - 1
                and int(hard_neighbor_idx) not in own_set
            ):
                nb = int(hard_neighbor_idx)
                nb_prob = dist_neg[nb].clamp(eps, 1.0 - eps)
                nw = float(getattr(self, "avbl_neighbor_weight", 2.0))
                loss_nb = -torch.log(1.0 - nb_prob) * nw
                loss_neg = loss_neg + loss_nb
                log['avbl_dino_neighbor_idx'] = float(nb)
                log['avbl_dino_neighbor_prob'] = float(nb_prob.detach())
                log['avbl_dino_neighbor_loss'] = float(loss_nb.detach())
        # Retrieval margin on adversarial: own expert logit should fall below prototype by margin
        ret_m = float(getattr(self, "avbl_retrieval_margin", 0.0))
        if ret_m > 0 and own_pool_indices:
            logits = self._fused_binding_logits(v_adv, q_adv, adv_image, for_avbl=True)
            if logits is not None:
                vis_sim, vis_sim_prot = logits
                own_idx = torch.tensor(own_pool_indices, device=self.device, dtype=torch.long)
                own_logit = vis_sim[0, own_idx].max()
                prot_logit = vis_sim_prot[0, 0]
                loss_rm = torch.relu(own_logit - prot_logit + ret_m)
                loss_neg = loss_neg + loss_rm
                log['avbl_ret_margin_neg'] = float(loss_rm.detach())
                log['avbl_ret_margin_own_vs_prot'] = float((own_logit - prot_logit).detach())
        log['avbl_prot_loss'] = float((-torch.log(prot_prob)).detach())
        log['avbl_expert_mass'] = float(expert_mass.detach())
        log['avbl_hinge'] = float(hinge.detach()) if hinge_tau > 0 else 1.0
        return loss_neg, log

    @torch.no_grad()
    def _mine_dino_false_cross_neighbor(
        self,
        adv_image,
        batch_edit_images: List,
        own_idx: int,
    ) -> Optional[int]:
        """Batch index of edit image most DINO-similar to adv (≠ own) — false-cross culprit."""
        hit = self._mine_dino_false_cross_neighbor_ext(adv_image, batch_edit_images, own_idx)
        if hit is None or hit[0] != "batch":
            return None
        return int(hit[1])

    @torch.no_grad()
    def _mine_dino_false_cross_neighbor_ext(
        self,
        adv_image,
        batch_edit_images: List,
        own_idx: int,
    ) -> Optional[Tuple[str, int]]:
        """Mine DINO-nearest wrong expert from batch and optional pool buffer.

        Returns ``('batch', i)`` or ``('buffer', i)`` or None.
        Buffer hits are used for refuse loss only (not in current routing dist).
        """
        adv_key = self._dino_key_for_image(adv_image)
        best_src: Optional[str] = None
        best_i: Optional[int] = None
        best_sim = -1e9
        if batch_edit_images and len(batch_edit_images) >= 2:
            for i, img in enumerate(batch_edit_images):
                if i == own_idx or img is None:
                    continue
                sim = float(torch.dot(adv_key, self._dino_key_for_image(img).to(self.device)))
                if sim > best_sim:
                    best_sim = sim
                    best_src, best_i = "batch", i
        buf = getattr(self, "_avbl_pool_buffer", None)
        if buf:
            for i, entry in enumerate(buf):
                key = entry.get("dino_key")
                if key is None:
                    continue
                sim = float(torch.dot(adv_key, key.to(self.device)))
                if sim > best_sim:
                    best_sim = sim
                    best_src, best_i = "buffer", i
        if best_src is None or best_i is None:
            return None
        return best_src, int(best_i)

    @torch.no_grad()
    def _push_avbl_pool_buffer(
        self,
        batch_edit_images: List,
        batch_moe_cs: torch.Tensor,
        batch_moe_rs: torch.Tensor,
        batch_eqrs: torch.Tensor,
    ) -> None:
        """Append detached batch experts into the cross-batch AVBL mining buffer."""
        n = int(getattr(self, "avbl_pool_buffer_size", 0) or 0)
        if n <= 0 or not batch_edit_images:
            return
        buf = getattr(self, "_avbl_pool_buffer", None)
        if buf is None:
            self._avbl_pool_buffer = deque(maxlen=n)
            buf = self._avbl_pool_buffer
        bsz = min(len(batch_edit_images), batch_moe_cs.shape[0], batch_moe_rs.shape[0], batch_eqrs.shape[0])
        for i in range(bsz):
            img = batch_edit_images[i]
            if img is None:
                continue
            buf.append({
                "dino_key": self._dino_key_for_image(img).detach().to(self.device),
                "moe_c": batch_moe_cs[i].detach(),
                "moe_r": batch_moe_rs[i].detach(),
                "eqr": batch_eqrs[i].detach(),
            })

    def compute_avbl_refuse_loss(
        self,
        vllm: BaseVLLMForEdit,
        prompt: str,
        adv_image,
        target: str,
        moe_c: torch.Tensor,
        moe_r: torch.Tensor,
        eqr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Push down P(edit target | adv image) when a (wrong) expert residual is applied."""
        log: Dict[str, float] = {}
        (input_embeds, vt_range), label_ids, label_masks = vllm.prompts_imgs_target_to_xym(
            [prompt], [adv_image], [target])
        query_end = input_embeds['inputs_embeds'].shape[1] - label_masks.shape[1] + 1
        was_editing = self.is_editing
        self.is_editing = True
        try:
            mid_inpt = vllm.get_mid_module_inpt(input_embeds, vt_range, self.edit_layer_path)
            mid_out = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path)
        finally:
            self.is_editing = was_editing
        reps_f = mid_out.float()
        query_reps = reps_f[:, vt_range[1]:query_end]
        iqr = self.inpt_extractor.extract_query(query_reps)
        if moe_c.dim() == 2:
            moe_c = moe_c.unsqueeze(0)
        if moe_r.dim() == 2:
            moe_r = moe_r.unsqueeze(0)
        if eqr.dim() == 2:
            eqr = eqr.unsqueeze(0)
        fuse_coe = self.get_moe_fuse_coe(iqr, eqr)
        # Residual is applied on edit-layer output in the train hook; build it from mid_out.
        self.train_edit_residual = self.get_edit_residual(reps_f, moe_c, moe_r, fuse_coe)
        try:
            logits = vllm.forward_from_mid_layer(
                input_embeds, vt_range, mid_inpt, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i
            ).logits
            nll = vllm.label_loss(logits, label_ids, label_masks, True)
        finally:
            self.train_edit_residual = None
        log_prob = -nll
        tau = float(getattr(self, "avbl_refuse_tau", -1.0))
        loss_refuse = torch.relu(log_prob - tau)
        log['avbl_refuse_logprob'] = float(log_prob.detach())
        log['avbl_refuse_nll'] = float(nll.detach())
        log['avbl_refuse_loss'] = float(loss_refuse.detach())
        return loss_refuse, log

    def compute_avbl_losses(
        self,
        vllm: BaseVLLMForEdit,
        prompt: str,
        rel_image,
        adversarial_images: List,
        own_pool_indices: List[int],
        target: str = ".",
        neg_weight: Optional[float] = None,
        batch_edit_images: Optional[List] = None,
        batch_moe_cs: Optional[torch.Tensor] = None,
        batch_moe_rs: Optional[torch.Tensor] = None,
        batch_eqrs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """AVBL: pull reliability binding; push swap/hard-neg toward prototype (no expert)."""
        if neg_weight is None:
            neg_weight = float(getattr(self, "adversarial_grounding_neg_weight", 1.0))
        log: Dict[str, float] = {}
        loss_pos, lpos = self.compute_avbl_positive_loss(
            vllm, prompt, rel_image, own_pool_indices, target)
        log.update(lpos)
        loss = loss_pos
        anchor_min = float(getattr(self, "avbl_rel_anchor_min", 0.0))
        if anchor_min > 0 and "avbl_rel_own_prob" in lpos:
            neg_scale = min(1.0, float(lpos["avbl_rel_own_prob"]) / anchor_min)
        else:
            neg_scale = 1.0
        log["avbl_neg_anchor_scale"] = neg_scale
        own_idx = int(own_pool_indices[0]) if own_pool_indices else 0
        refuse_lambda = float(getattr(self, "avbl_refuse_lambda", 0.0))
        for i, adv_img in enumerate(adversarial_images):
            if adv_img is None:
                continue
            hard_nb: Optional[int] = None
            mined: Optional[Tuple[str, int]] = None
            if getattr(self, "avbl_dino_neighbor_mine", False) and batch_edit_images:
                mined = self._mine_dino_false_cross_neighbor_ext(
                    adv_img, batch_edit_images, own_idx)
                if mined is not None and mined[0] == "batch":
                    hard_nb = int(mined[1])
            loss_neg, lneg = self.compute_avbl_negative_loss(
                vllm, prompt, adv_img, own_pool_indices, target,
                hard_neighbor_idx=hard_nb)
            loss = loss + neg_weight * neg_scale * loss_neg
            for k, v in lneg.items():
                log['adv%d_%s' % (i, k)] = v
            if (
                refuse_lambda > 0
                and batch_moe_cs is not None
                and batch_moe_rs is not None
                and batch_eqrs is not None
            ):
                moe_c = moe_r = eqr = None
                expert_tag = -1.0
                if mined is not None and mined[0] == "buffer":
                    buf = getattr(self, "_avbl_pool_buffer", None)
                    bi = int(mined[1])
                    if buf is not None and 0 <= bi < len(buf):
                        entry = buf[bi]
                        moe_c, moe_r, eqr = entry["moe_c"], entry["moe_r"], entry["eqr"]
                        expert_tag = float(bi)
                        log['adv%d_refuse_src' % i] = 1.0  # buffer
                else:
                    expert_i = hard_nb if hard_nb is not None else own_idx
                    if 0 <= expert_i < batch_moe_cs.shape[0]:
                        moe_c = batch_moe_cs[expert_i]
                        moe_r = batch_moe_rs[expert_i]
                        eqr = batch_eqrs[expert_i]
                        expert_tag = float(expert_i)
                        log['adv%d_refuse_src' % i] = 0.0  # batch
                if moe_c is not None:
                    loss_ref, lref = self.compute_avbl_refuse_loss(
                        vllm, prompt, adv_img, target, moe_c, moe_r, eqr,
                    )
                    loss = loss + refuse_lambda * neg_scale * loss_ref
                    for k, v in lref.items():
                        log['adv%d_%s' % (i, k)] = v
                    log['adv%d_refuse_expert' % i] = expert_tag
        return loss, log

    def _get_patch_encoder(self) -> FrozenPatchEncoder:
        if self._patch_encoder is None:
            self._patch_encoder = FrozenPatchEncoder(self.patch_encoder_name, self.device)
        return self._patch_encoder

    @torch.no_grad()
    def _encode_patches(self, image) -> torch.Tensor:
        enc = self._get_patch_encoder()
        vllm = self.vllm if self.patch_encoder_name == "eva_patches" else None
        patches = enc.encode_patches(image, vllm=vllm)
        return patches.to(self.device)

    def _install_patch_image_cache_hook(self) -> None:
        if getattr(self.vllm, "_patch_cache_wrapped", False):
            return
        orig = self.vllm.prompts_imgs_target_to_xym

        def wrapped(prompts, imgs, targets):
            # Text locality can pass image=None; skip patch encode for those samples.
            if (
                (self.patch_grounding_enabled or self.decode_vision_gate_enabled)
                and imgs
                and imgs[0] is not None
            ):
                self._last_query_patches = self._encode_patches(imgs[0])
            else:
                self._last_query_patches = None
            return orig(prompts, imgs, targets)

        self.vllm.prompts_imgs_target_to_xym = wrapped
        self.vllm._patch_cache_wrapped = True

    def _current_query_patches(self, image=None) -> Optional[torch.Tensor]:
        if image is not None:
            return self._encode_patches(image)
        if self._last_query_patches is not None:
            return self._last_query_patches
        return None

    def _clear_query_patch_cache(self) -> None:
        self._last_query_patches = None

    def _extract_patch_routing(
        self, query_reps: torch.Tensor, patch_reps: torch.Tensor
    ) -> torch.Tensor:
        assert self.patch_grounding_module is not None
        return self.patch_grounding_module(query_reps, patch_reps)

    def _extract_patch_prototype_routing(self, query_reps: torch.Tensor) -> torch.Tensor:
        assert self.patch_grounding_module is not None
        return self.patch_grounding_module.extract_from_prototype(query_reps)

    def _fuse_routing_sims(
        self, vis_sim_q: torch.Tensor, vis_sim_p: Optional[torch.Tensor],
        for_avbl: bool = False,
    ) -> torch.Tensor:
        if for_avbl and vis_sim_p is not None and vis_sim_q is not None:
            alpha = float(getattr(self, "patch_routing_alpha", 0.5))
            return alpha * vis_sim_p + (1.0 - alpha) * vis_sim_q
        if vis_sim_p is None or self.patch_routing_only:
            return vis_sim_p if self.patch_routing_only and vis_sim_p is not None else vis_sim_q
        alpha = float(getattr(self, "patch_routing_alpha", 0.5))
        return alpha * vis_sim_q + (1.0 - alpha) * vis_sim_p

    def _compute_vis_sim_routing(
        self,
        ivr: torch.Tensor,
        ivr_p: Optional[torch.Tensor],
        evr_pool: torch.Tensor,
        evr_patch_pool: torch.Tensor,
        vis_sim_q: Optional[torch.Tensor] = None,
        vis_sim_p: Optional[torch.Tensor] = None,
        for_avbl: bool = False,
    ) -> torch.Tensor:
        """Shared QFormer/patch routing score for retrieve_moes and AVBL binding loss."""
        if vis_sim_q is None:
            vis_sim_q = self._pool_vis_sim(ivr, evr_pool)
        if (
            ivr_p is not None
            and evr_patch_pool.shape[0] == evr_pool.shape[0]
            and evr_patch_pool.shape[0] > 0
        ):
            if vis_sim_p is None:
                vis_sim_p = self._pool_vis_sim(ivr_p, evr_patch_pool)
        else:
            vis_sim_p = None
        if (
            getattr(self, "fused_expert_key_enabled", False)
            and vis_sim_p is not None
            and evr_patch_pool.shape[0] == evr_pool.shape[0]
        ):
            alpha = float(getattr(self, "patch_routing_alpha", 0.5))
            evr_fused = alpha * evr_pool + (1.0 - alpha) * evr_patch_pool
            ivr_fused = alpha * ivr + (1.0 - alpha) * ivr_p
            return self._pool_vis_sim(ivr_fused, evr_fused)
        return self._fuse_routing_sims(vis_sim_q, vis_sim_p, for_avbl=for_avbl)

    def _pool_vis_sim(self, ivr: torch.Tensor, evr_pool: torch.Tensor) -> torch.Tensor:
        """Aggregate eqe_n routing similarities → [1, m] (Step 5 spike 4a)."""
        per_eqe = torch.einsum("bed,med->bme", ivr, evr_pool) * self.sim_scale
        mode = getattr(self, "routing_pool_mode", "mean")
        if mode == "max":
            return per_eqe.max(dim=2).values
        if mode == "attn":
            weights = torch.softmax(per_eqe[0], dim=1)
            return (weights * per_eqe[0]).sum(dim=1).unsqueeze(0)
        return per_eqe.mean(2)

    def _text_routing_sim(
        self, query_reps: torch.Tensor, eqr_pool: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prompt-key similarities vs pool eqr → (txt_sim [1,m], iqr [1,eqe,d])."""
        iqr = self.inpt_extractor.extract_query(query_reps)
        return self._pool_vis_sim(iqr, eqr_pool), iqr

    def _apply_dek_dual_gate(
        self,
        retrieval_map: torch.Tensor,
        txt_sim: torch.Tensor,
    ) -> torch.Tensor:
        """Keep visual survivors only if they agree with global prompt-key ranking.

        text_ok: txt_sim[i] >= max_j(txt_sim[j]) - text_margin
        Combined with prior visual gate → vision ∧ text agreement.
        """
        if txt_sim is None or txt_sim.numel() == 0:
            return torch.zeros_like(retrieval_map)
        if not retrieval_map.any():
            return retrieval_map
        margin = float(getattr(self, "dek_text_margin", 0.0))
        ref = txt_sim[0].max()
        text_ok = txt_sim[0] >= (ref - margin)
        return retrieval_map & text_ok

    def _text_ok_mask(self, txt_sim: Optional[torch.Tensor], m_gate: int) -> torch.Tensor:
        """Prompt-key survivors: txt_sim[i] >= max - text_margin."""
        device = self.evr_pool_gated.device
        if txt_sim is None or txt_sim.numel() == 0:
            return torch.zeros(m_gate, dtype=torch.bool, device=device)
        margin = float(getattr(self, "dek_text_margin", 0.0))
        ref = txt_sim[0].max()
        return txt_sim[0] >= (ref - margin)

    def _dino_channel_sims(self, m_gate: int, pool_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Cosine sim of query DINO key vs each gated pool expert (1, m_gate), or None."""
        dino_pool = getattr(self, "dino_key_pool", [])
        query_patches = self._current_query_patches()
        m_full = int(self.evr_pool.shape[0])
        if query_patches is None or len(dino_pool) != m_full or m_gate == 0:
            return None
        q = patch_mean_key(query_patches).to(self.device)
        sims = []
        for gi in pool_indices.tolist():
            k = dino_pool[int(gi)].to(self.device)
            sims.append(torch.dot(q, k))
        return torch.stack(sims, dim=0).unsqueeze(0)  # [1, m_gate]

    def _carml_retrieval_map(
        self,
        vis_sim: torch.Tensor,
        txt_sim: Optional[torch.Tensor],
        pool_indices: torch.Tensor,
        m_gate: int,
    ) -> torch.Tensor:
        """CARML-style multi-channel fusion (Section 3.3-ish): per-channel softmax,
        inverse-entropy weights, top-K overlap conflict dampening, top-1 + scope tau.

        Channels: Q-Former vision, prompt key (if available), DINOv2 (if available).
        Scope threshold ``carml_scope_tau`` is a fixed prior until train-split tuning.
        """
        device = vis_sim.device
        channels = [vis_sim[0]]
        if txt_sim is not None and txt_sim.numel() == m_gate:
            channels.append(txt_sim[0])
        dino = self._dino_channel_sims(m_gate, pool_indices)
        if dino is not None:
            channels.append(dino[0])
        probs, entropies = [], []
        top_sets = []
        k = min(int(getattr(self, "carml_top_k", 15)), m_gate)
        for ch in channels:
            p = torch.softmax(ch, dim=0)
            probs.append(p)
            # entropy in nats; higher => less confident
            ent = -(p * (p.clamp_min(1e-12).log())).sum()
            entropies.append(ent)
            top_sets.append(set(torch.topk(p, k=k).indices.tolist()))
        # Inverse-entropy channel weights
        inv = torch.stack([1.0 / (e + 1e-6) for e in entropies])
        w = inv / inv.sum()
        fused = torch.zeros(m_gate, device=device, dtype=probs[0].dtype)
        for wi, p in zip(w, probs):
            fused = fused + wi * p
        # Conflict: mean pairwise Jaccard of top-K; low overlap => dampen peak
        if len(top_sets) >= 2:
            jaccs = []
            for a in range(len(top_sets)):
                for b in range(a + 1, len(top_sets)):
                    inter = len(top_sets[a] & top_sets[b])
                    union = len(top_sets[a] | top_sets[b]) or 1
                    jaccs.append(inter / union)
            conflict = 1.0 - float(sum(jaccs) / len(jaccs))
            fused = fused * max(0.05, 1.0 - conflict)
        fused = fused / fused.sum().clamp_min(1e-12)
        top1 = int(fused.argmax().item())
        tau = float(getattr(self, "carml_scope_tau", 0.05))
        out = torch.zeros(m_gate, dtype=torch.bool, device=device)
        if float(fused[top1]) >= tau:
            out[top1] = True
        # stash for probe_routing
        self._last_carml_fused = fused.detach()
        self._last_carml_channel_weights = w.detach()
        return out

    def _apply_routing_rule(
        self,
        rule: str,
        vis_ok: torch.Tensor,
        vis_sim: torch.Tensor,
        vis_sim_prot: torch.Tensor,
        txt_sim: Optional[torch.Tensor],
        pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Map G2 ``routing_rule`` + channel scores to a boolean retrieval map [m_gate]."""
        m_gate = int(vis_ok.numel())
        device = vis_ok.device
        text_ok = self._text_ok_mask(txt_sim, m_gate)
        if rule in ("liveedit", "vision_only"):
            return vis_ok
        if rule == "text_only":
            return text_ok
        if rule == "and":
            return vis_ok & text_ok
        if rule == "or":
            return vis_ok | text_ok
        if rule == "sum":
            if txt_sim is None or txt_sim.numel() == 0:
                return vis_ok
            alpha = float(getattr(self, "dek_text_alpha", 1.0))
            fused = vis_sim[0] + alpha * txt_sim[0]
            return fused > vis_sim_prot[0, 0]
        if rule == "carml":
            return self._carml_retrieval_map(vis_sim, txt_sim, pool_indices, m_gate)
        if rule == "always_reject":
            return torch.zeros(m_gate, dtype=torch.bool, device=device)
        raise ValueError("Unknown routing_rule %r" % rule)

    def _dino_key_for_image(self, image) -> torch.Tensor:
        dino_patches = self._encode_patches(image)
        return patch_mean_key(dino_patches).to(self.device)

    def _pool_dedup_merge_index(self, new_key: torch.Tensor) -> Optional[int]:
        if not getattr(self, "pool_dedup_enabled", False):
            return None
        dino_pool = getattr(self, "dino_key_pool", [])
        if not dino_pool:
            return None
        thr = float(getattr(self, "pool_dedup_threshold", 0.95))
        sims = [float(torch.dot(k.to(self.device), new_key)) for k in dino_pool]
        best_i = int(max(range(len(sims)), key=lambda i: sims[i]))
        return best_i if sims[best_i] >= thr else None

    def _decode_vision_gate_filter(
        self,
        retrieval_map_gated: torch.Tensor,
        pool_indices: torch.Tensor,
        query_patches: torch.Tensor,
    ) -> torch.Tensor:
        """v2: abs DINO match + competitive margin among Q-Former-passing candidates."""
        margin = float(getattr(self, "decode_vision_margin", 0.0))
        abs_thr = getattr(self, "decode_vision_abs_threshold", None)
        dino_pool = getattr(self, "dino_key_pool", [])
        if not dino_pool or not retrieval_map_gated.any():
            return retrieval_map_gated
        query_key = patch_mean_key(query_patches).to(self.device)
        gated_pos = retrieval_map_gated.nonzero(as_tuple=True)[0]
        if len(gated_pos) == 0:
            return retrieval_map_gated
        full_indices = [int(pool_indices[i].item()) for i in gated_pos.tolist()]
        sims = torch.stack(
            [dino_pool[i].to(self.device) for i in full_indices],
            dim=0,
        )
        sims = torch.mv(sims, query_key)
        sim_list = sims.tolist()
        out = torch.zeros_like(retrieval_map_gated)
        for p, sim_val in enumerate(sim_list):
            if abs_thr is not None and sim_val < float(abs_thr):
                continue
            other = [s for j, s in enumerate(sim_list) if j != p]
            other_max = max(other) if other else -1e9
            if sim_val >= other_max + margin:
                out[gated_pos[p]] = True
        return out

    def _build_dino_gate_diagnostics(
        self,
        retrieval_map_qformer: torch.Tensor,
        pool_indices: torch.Tensor,
        query_patches: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """Per Q-Former candidate: DINO sim(query, expert_edit_img) and gate pass flags."""
        dino_pool = getattr(self, "dino_key_pool", [])
        m = self.evr_pool.shape[0]
        if not dino_pool or len(dino_pool) != m or not retrieval_map_qformer.any():
            return None
        margin = float(getattr(self, "decode_vision_margin", 0.0))
        abs_thr = getattr(self, "decode_vision_abs_threshold", None)
        query_key = patch_mean_key(query_patches).to(self.device)
        gated_pos = retrieval_map_qformer.nonzero(as_tuple=True)[0]
        candidates: List[Dict[str, Any]] = []
        sim_list: List[float] = []
        for p in gated_pos.tolist():
            full_i = int(pool_indices[p].item())
            sim_val = float(torch.dot(dino_pool[full_i].to(self.device), query_key).item())
            sim_list.append(sim_val)
            candidates.append({"pool_index": full_i, "dino_sim": sim_val})
        for i, c in enumerate(candidates):
            other = [s for j, s in enumerate(sim_list) if j != i]
            other_max = max(other) if other else -1e9
            abs_ok = abs_thr is None or c["dino_sim"] >= float(abs_thr)
            margin_ok = c["dino_sim"] >= other_max + margin
            c["abs_pass"] = bool(abs_ok)
            c["margin_pass"] = bool(margin_ok)
            c["gate_pass"] = bool(abs_ok and margin_ok)
            c["margin_vs_best_other"] = float(c["dino_sim"] - other_max)
        return {
            "n_qformer_candidates": len(candidates),
            "candidates": candidates,
            "decode_vision_margin": margin,
            "decode_vision_abs_threshold": abs_thr,
        }

    def _patch_cross_attn_score(
        self, query_patches: torch.Tensor, expert_patches: torch.Tensor
    ) -> float:
        """Patch-grid cross-attention rerank score (Step 5 spike 4b)."""
        import torch.nn.functional as F

        q = F.normalize(query_patches.float(), dim=-1)
        k = F.normalize(expert_patches.float(), dim=-1)
        sim = torch.matmul(q, k.transpose(0, 1))
        scale = q.shape[-1] ** 0.5
        attn = torch.softmax(sim / scale, dim=1)
        ctx = torch.matmul(attn, k)
        return float((ctx * q).sum(dim=-1).mean().item())

    def _rerank_with_patch_cross_attn(
        self,
        candidate_full_indices: torch.Tensor,
        vis_sim: torch.Tensor,
        vision_reps: torch.Tensor,
        gated_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Rerank coarse candidates using ViT patch cross-attention (4b)."""
        vit_pool = getattr(self, "vit_patch_pool", None)
        if not vit_pool or len(vit_pool) == 0:
            return candidate_full_indices
        query_patches = getattr(self.vllm, "_last_vit_patches", None)
        if query_patches is None:
            return candidate_full_indices
        query_patches = query_patches[0].to(vision_reps.device)

        top_k = int(getattr(self, "patch_rerank_top_k", 8))
        if len(candidate_full_indices) > top_k:
            coarse_scores = vis_sim[0, gated_positions]
            _, order = coarse_scores.sort(descending=True)
            keep = order[:top_k]
            candidate_full_indices = candidate_full_indices[keep]
            gated_positions = gated_positions[keep]

        scored = []
        for full_idx, gated_idx in zip(
            candidate_full_indices.tolist(), gated_positions.tolist()
        ):
            if full_idx >= len(vit_pool):
                continue
            expert_patches = vit_pool[full_idx].to(vision_reps.device)
            patch_score = self._patch_cross_attn_score(query_patches, expert_patches)
            coarse = float(vis_sim[0, gated_idx].item())
            scored.append((full_idx, 0.5 * coarse + 0.5 * patch_score))
        if not scored:
            return candidate_full_indices
        scored.sort(key=lambda x: x[1], reverse=True)
        return torch.tensor([i for i, _ in scored], device=vision_reps.device, dtype=torch.long)

    def _append_vit_patch_for_edit(self, image) -> None:
        if not getattr(self, "patch_rerank_enabled", False):
            return
        if not hasattr(self.vllm, "get_vit_patch_embeds"):
            return
        patches = self.vllm.get_vit_patch_embeds(image)[0].detach().cpu()
        self.vit_patch_pool.append(patches)

    def infer_ar_chunk_index(
        self,
        query_token_offset: int,
        image=None,
        chunk_size: Optional[int] = None,
        max_chunks: Optional[int] = None,
    ) -> int:
        """AR chunk index for routing gate (positional or vision-gated 4c)."""
        from .chunk_utils import infer_ar_chunk_index as _infer

        chunk_size = chunk_size if chunk_size is not None else getattr(self, "chunk_size", 16)
        max_chunks = max_chunks if max_chunks is not None else getattr(self, "max_chunks", None)
        cross_attn = getattr(self.vllm, "_last_qformer_cross_attn", None)
        if image is not None and cross_attn is None and hasattr(self.vllm, "get_qformer_cross_attention"):
            cross_attn = self.vllm.get_qformer_cross_attention(image)
        return _infer(
            cross_attn,
            query_token_offset,
            chunk_size,
            max_chunks,
            vision_gated=getattr(self, "ar_vision_gated_chunks", False),
            n_qformer_queries=self.vllm.get_img_token_n(),
        )

    def wrap_and_hook(self):
        '''Only for inference stage.'''
        self.now_infer_vt_range = None # [vt_begin, vt_end], The vision token range of current VLLM inference.
        self.now_infer_inpt_embd_shape = None # The shape of current input embedding 
        self.now_infer_query_pos_end = None # The range of query of current input embedding 
        # modify VLLM function `get_llm_outpt` for retrieval editor
        def get_llm_outpt_wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(llm_inpt, vt_range):
                if not self.is_train and not self.is_editing and len(self.eqr_pool) > 0:
                    self.now_infer_vt_range = vt_range
                    self.now_infer_inpt_embd_shape = llm_inpt['inputs_embeds'].shape
                    if 'query_range' in llm_inpt:
                        self.now_infer_query_pos_end = llm_inpt['query_range'][1]
                    elif self.verbose:
                        print("Without query range, will use all reps except for visual reps as query reps.")
                return get_llm_outpt(llm_inpt, vt_range)
            return wrapped_get_llm_outpt 
        if not hasattr(self, 'original_get_llm_outpt'):
            self.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = get_llm_outpt_wrap(self.original_get_llm_outpt)
        # initialize for retrieval editor hooked into the VLLM
        def apply_edit_residual(outpt, edit_residual):
            if isinstance(outpt, (tuple, list)):
                outpt = list(outpt)
                er = edit_residual.to(dtype=outpt[0].dtype)
                outpt[0] = outpt[0] + er
            else:
                outpt = outpt + edit_residual.to(dtype=outpt.dtype)
            return outpt
        def edit_with_moes(module, args, output):
            if self.is_train and getattr(self, 'train_edit_residual', None) is not None:
                output = apply_edit_residual(output, self.train_edit_residual)
                self.train_edit_residual = None
            elif (not self.is_editing and len(self.eqr_pool) > 0 and self.now_infer_vt_range != None):
                # inpt_reps = output[0] if isinstance(output, tuple) else output
                reps = output[0] if isinstance(output, (tuple, list)) else output
                assert reps.shape == self.now_infer_inpt_embd_shape 
                reps_f = reps.float()
                vision_reps = reps_f[:, self.now_infer_vt_range[0]:self.now_infer_vt_range[1]]
                query_reps = reps_f[:, self.now_infer_vt_range[1]:self.now_infer_query_pos_end]
                self.now_infer_inpt_embd_shape = self.now_infer_vt_range = self.now_infer_query_pos_end = None
                chunk_idx = getattr(self, 'current_infer_chunk_index', None)
                if not getattr(self, 'ar_use_routing_gate', True):
                    chunk_idx = None  # ablation: routing gate off = full pool
                moe_cs, moe_rs, fuse_coe = self.retrieve_moes(vision_reps, query_reps, chunk_index=chunk_idx)
                edit_residual = self.get_edit_residual(reps_f, moe_cs, moe_rs, fuse_coe)
                output = apply_edit_residual(output, edit_residual)
            return output
        edit_layer = find_module(self.vllm.model, self.edit_layer_path)
        if hasattr(edit_layer, 'edit_layer_hooked'): 
            edit_layer._forward_hooks.clear()
        edit_layer.edit_layer_hooked = True
        edit_layer.register_forward_hook(edit_with_moes)

    def retrieve_moes(self, vision_reps:torch.Tensor, query_reps:torch.Tensor,
                      chunk_index: Optional[int] = None, return_retr_details: bool = False,
                      skip_decode_vision_gate: bool = False):
        '''Only for inference stage. When chunk_index is set (AR mode), gate retrieval to pool entries with that chunk index.'''
        t0 = time.perf_counter()
        # inpt_vision_reps: [b, l, hidden_size]
        assert len(vision_reps) == len(query_reps) == 1
        m = self.evr_pool.shape[0]
        pool_indices = torch.arange(m, device=vision_reps.device)
        # Chunk-aware gate: restrict to experts with matching chunk index when provided
        chunk_index_pool = getattr(self, 'chunk_index_pool', None)
        if chunk_index is not None and chunk_index_pool and len(chunk_index_pool) == m:
            gate = torch.tensor([ci == chunk_index for ci in chunk_index_pool], device=vision_reps.device, dtype=torch.bool)
            if gate.any():
                pool_indices = pool_indices[gate]
                self.evr_pool_gated = self.evr_pool[gate]
                self.eqr_pool_gated = self.eqr_pool[gate]
                self.moe_cs_pool_gated = self.moe_cs_pool[gate]
                self.moe_rs_pool_gated = self.moe_rs_pool[gate]
                if hasattr(self, "evr_patch_pool") and self.evr_patch_pool.shape[0] == m:
                    self.evr_patch_pool_gated = self.evr_patch_pool[gate]
                else:
                    self.evr_patch_pool_gated = self.evr_patch_pool
            else:
                pool_indices = torch.arange(m, device=vision_reps.device)
                self.evr_pool_gated = self.evr_pool
                self.eqr_pool_gated = self.eqr_pool
                self.moe_cs_pool_gated = self.moe_cs_pool
                self.moe_rs_pool_gated = self.moe_rs_pool
                self.evr_patch_pool_gated = getattr(self, "evr_patch_pool", torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=vision_reps.device))
        else:
            self.evr_pool_gated = self.evr_pool
            self.eqr_pool_gated = self.eqr_pool
            self.moe_cs_pool_gated = self.moe_cs_pool
            self.moe_rs_pool_gated = self.moe_rs_pool
            self.evr_patch_pool_gated = getattr(self, "evr_patch_pool", torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=vision_reps.device))
        # Get lora weights (same logic on gated or full pool)
        ivr = self.inpt_extractor.extract_vision(query_reps, vision_reps) # [1, eqe_n, module_dim]
        vis_sim_q = self._pool_vis_sim(ivr, self.evr_pool_gated) # [1, m_gate]
        vis_sim_p = None
        if (
            self.patch_grounding_enabled
            and self.patch_grounding_module is not None
            and hasattr(self, "evr_patch_pool_gated")
            and self.evr_patch_pool_gated.shape[0] > 0
        ):
            query_patches = self._current_query_patches()
            if query_patches is not None:
                ivr_p = self._extract_patch_routing(query_reps, query_patches)
                vis_sim_p = self._pool_vis_sim(ivr_p, self.evr_patch_pool_gated)
        vis_sim = self._compute_vis_sim_routing(
            ivr,
            ivr_p if vis_sim_p is not None else None,
            self.evr_pool_gated,
            self.evr_patch_pool_gated,
            vis_sim_q,
            vis_sim_p,
        )
        ivr_prot = self.inpt_extractor.extract_from_visprot(query_reps) # [1, eqe_n, module_dim]
        vis_sim_prot = torch.einsum('bed,bed->be', ivr, ivr_prot).mean(1, True) * self.sim_scale # [1, 1]
        if self.verification_gate_enabled and vis_sim_p is not None:
            ivr_prot_p = self._extract_patch_prototype_routing(query_reps)
            vis_sim_prot_p = torch.einsum("bed,bed->be", ivr_p, ivr_prot_p).mean(1, True) * self.sim_scale
            vis_ok = (
                (vis_sim_q > vis_sim_prot) & (vis_sim_p > vis_sim_prot_p)
            ).to(torch.bool)[0]
        else:
            vis_ok = (vis_sim > vis_sim_prot).to(torch.bool)[0]  # [m_gate]
        # G2 routing_rule (liveedit / vision_only / text_only / and / or / sum / carml / always_reject).
        # Legacy: dek_dual_gate_enabled with default rule ⇒ ``and``.
        rule = str(getattr(self, "routing_rule", "liveedit") or "liveedit").lower()
        if getattr(self, "dek_dual_gate_enabled", False) and rule in ("liveedit", "vision_only"):
            rule = "and"
        txt_sim_gated = None
        iqr_for_fuse = None
        need_text = rule in ("text_only", "and", "or", "sum", "carml") or return_retr_details
        if need_text and self.eqr_pool_gated.shape[0] > 0:
            txt_sim_gated, iqr_for_fuse = self._text_routing_sim(
                query_reps, self.eqr_pool_gated)
        self._last_vis_ok = vis_ok.detach()
        self._last_txt_sim = None if txt_sim_gated is None else txt_sim_gated.detach()
        self._last_text_ok = self._text_ok_mask(
            txt_sim_gated, int(vis_ok.numel())).detach()
        self._last_routing_rule = rule
        retrieval_map_gated = self._apply_routing_rule(
            rule, vis_ok, vis_sim, vis_sim_prot, txt_sim_gated, pool_indices)
        if getattr(self, "patch_rerank_enabled", False) and retrieval_map_gated.any():
            gated_positions = retrieval_map_gated.nonzero(as_tuple=True)[0]
            full_indices = pool_indices[gated_positions]
            reranked = self._rerank_with_patch_cross_attn(
                full_indices, vis_sim, vision_reps, gated_positions
            )
            retrieval_map_gated = torch.zeros_like(retrieval_map_gated)
            for gi in reranked.tolist():
                local_i = (pool_indices == gi).nonzero(as_tuple=True)[0]
                if len(local_i) > 0:
                    retrieval_map_gated[local_i[0]] = True
        dino_gate_diag = None
        query_patches = self._current_query_patches()
        qformer_local_map = retrieval_map_gated.clone()
        qformer_full_indices = pool_indices[qformer_local_map].cpu().tolist()
        if (
            return_retr_details
            and query_patches is not None
            and len(getattr(self, "dino_key_pool", [])) == m
        ):
            dino_gate_diag = self._build_dino_gate_diagnostics(
                qformer_local_map, pool_indices, query_patches
            )
        if (
            not skip_decode_vision_gate
            and getattr(self, "decode_vision_gate_enabled", False)
            and retrieval_map_gated.any()
            and len(getattr(self, "dino_key_pool", [])) == m
        ):
            if query_patches is not None:
                retrieval_map_gated = self._decode_vision_gate_filter(
                    retrieval_map_gated, pool_indices, query_patches
                )
        # Map back to full pool indices for indexing moe_cs_pool / moe_rs_pool / eqr_pool
        selected_gated = pool_indices[retrieval_map_gated]
        retrieval_map = torch.zeros(m, dtype=torch.bool, device=vision_reps.device)
        if len(selected_gated) > 0:
            retrieval_map[selected_gated] = True
        moe_cs = self.moe_cs_pool[retrieval_map]
        moe_rs = self.moe_rs_pool[retrieval_map]
        # Get fuse coefficient
        if iqr_for_fuse is None:
            iqr_for_fuse = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
        eqr = self.eqr_pool[retrieval_map] # [m, eqe_n, module_dim]
        fuse_coe = self.get_moe_fuse_coe(iqr_for_fuse, eqr)
        elapsed = time.perf_counter() - t0
        self._last_retrieval_time_ms = elapsed * 1000.0  # for eval/overhead logging
        if getattr(self, 'verbose', False):
            n_retr = int(retrieval_map.sum().item())
            gate_used = chunk_index is not None and chunk_index_pool and len(chunk_index_pool) == m
            print(f"[LiveEdit retrieval] chunk_index={chunk_index} gate_used={gate_used} n_retrieved={n_retr} time_ms={self._last_retrieval_time_ms:.2f}")
        if not return_retr_details:
            self._clear_query_patch_cache()
            return moe_cs, moe_rs, fuse_coe
        else:
            retrieved_indices = retrieval_map.nonzero(as_tuple=True)[0].cpu().tolist()
            retrieved_requests = [r for r, m in zip(self.requests_pool, retrieval_map) if m]
            return (
                vis_sim, vis_sim_prot, retrieved_indices, retrieved_requests,
                moe_cs, moe_rs, fuse_coe, dino_gate_diag, qformer_full_indices,
            )

    @torch.no_grad()
    def probe_routing(self, vllm: BaseVLLMForEdit, prompt: str, image, target: str = ".",
                      chunk_index: Optional[int] = None,
                      skip_decode_vision_gate: bool = False) -> Dict[str, Any]:
        """Score MoE retrieval for a query without applying the edit residual (Step 0 diagnostics)."""
        (input_embeds, vt_range), _, label_masks = vllm.prompts_imgs_target_to_xym(
            [prompt], [image], [target])
        query_end = input_embeds['inputs_embeds'].shape[1] - label_masks.shape[1] + 1
        was_editing = self.is_editing
        self.is_editing = True
        try:
            edit_signal = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path)
        finally:
            self.is_editing = was_editing
        reps_f = edit_signal.float()
        vision_reps = reps_f[:, vt_range[0]:vt_range[1]]
        query_reps = reps_f[:, vt_range[1]:query_end]
        ivr = self.inpt_extractor.extract_vision(query_reps, vision_reps)
        (
            vis_sim, vis_sim_prot, retrieved_indices, _, _, _, fuse_coe,
            dino_gate_diag, qformer_full_indices,
        ) = self.retrieve_moes(
            vision_reps, query_reps, chunk_index=chunk_index,
            return_retr_details=True, skip_decode_vision_gate=skip_decode_vision_gate,
        )
        # M3: actual residual fuse coefficients (get_moe_fuse_coe = softmax*sigmoid over prompt keys)
        fuse_list = []
        fuse_by_index: Dict[str, float] = {}
        if fuse_coe is not None and len(retrieved_indices) > 0:
            fuse_list = [float(x) for x in fuse_coe[0].detach().cpu().tolist()]
            fuse_by_index = {str(i): w for i, w in zip(retrieved_indices, fuse_list)}
        # Soft weights over the full pool (same formula on all experts) for weight-aware taxonomy.
        pool_fuse: List[float] = []
        if self.eqr_pool.shape[0] > 0:
            iqr_all = self.inpt_extractor.extract_query(query_reps)
            pool_fuse_t = self.get_moe_fuse_coe(iqr_all, self.eqr_pool)
            pool_fuse = [float(x) for x in pool_fuse_t[0].detach().cpu().tolist()]
        txt_sim = getattr(self, "_last_txt_sim", None)
        vis_ok = getattr(self, "_last_vis_ok", None)
        text_ok = getattr(self, "_last_text_ok", None)
        out: Dict[str, Any] = {
            'vis_sim': vis_sim[0].cpu().tolist(),
            'vis_sim_prot': float(vis_sim_prot[0, 0].cpu()),
            'retrieved_indices': retrieved_indices,
            'qformer_retrieved_indices': qformer_full_indices,
            'n_retrieved': len(retrieved_indices),
            'n_qformer_retrieved': len(qformer_full_indices),
            'n_pool': int(self.evr_pool.shape[0]),
            'ivr_eqe_std': float(ivr.std(dim=1).mean().cpu()),
            'vis_token_n': int(vision_reps.shape[1]),
            'query_token_n': int(query_reps.shape[1]),
            'chunk_index': chunk_index,
            'routing_pool_mode': getattr(self, 'routing_pool_mode', 'mean'),
            'routing_rule': getattr(self, '_last_routing_rule', getattr(self, 'routing_rule', 'liveedit')),
            'patch_rerank_enabled': bool(getattr(self, 'patch_rerank_enabled', False)),
            'ar_vision_gated_chunks': bool(getattr(self, 'ar_vision_gated_chunks', False)),
            'decode_vision_gate_enabled': bool(getattr(self, 'decode_vision_gate_enabled', False)),
            'decode_vision_margin': float(getattr(self, 'decode_vision_margin', 0.0)),
            'decode_vision_abs_threshold': getattr(self, 'decode_vision_abs_threshold', None),
            'skip_decode_vision_gate': bool(skip_decode_vision_gate),
            # M3 / G2 soft weights
            'fuse_coe': fuse_list,
            'fuse_coe_by_index': fuse_by_index,
            'pool_fuse_coe': pool_fuse,
            'dek_text_margin': float(getattr(self, 'dek_text_margin', 0.0)),
        }
        if txt_sim is not None:
            out['txt_sim'] = txt_sim[0].cpu().tolist()
        if vis_ok is not None:
            out['vision_ok_indices'] = vis_ok.nonzero(as_tuple=True)[0].cpu().tolist()
        if text_ok is not None:
            out['text_ok_indices'] = text_ok.nonzero(as_tuple=True)[0].cpu().tolist()
        if getattr(self, '_last_carml_fused', None) is not None:
            out['carml_fused'] = self._last_carml_fused.cpu().tolist()
            out['carml_channel_weights'] = self._last_carml_channel_weights.cpu().tolist()
            out['carml_scope_tau'] = float(getattr(self, 'carml_scope_tau', 0.05))
        if dino_gate_diag is not None:
            out['dino_gate'] = dino_gate_diag
        return out

    @torch.no_grad()
    def dino_sims_vs_pool(self, image) -> List[float]:
        """DINO cosine sim between query image and each stored expert edit image."""
        dino_pool = getattr(self, "dino_key_pool", [])
        if not dino_pool:
            return []
        patches = self._encode_patches(image)
        query_key = patch_mean_key(patches).to(self.device)
        return [float(torch.dot(k.to(self.device), query_key).item()) for k in dino_pool]
    
    ############################################################################
    ######################### LiveEdit Basic Functions ############################
    ############################################################################
    def _pack_edit_forward(self, vllm: BaseVLLMForEdit, prompt: str, image, target: str):
        '''One mid-layer forward → (xym, mid_reps, edit_rep_slices). Avoids duplicate LLaVA passes.'''
        xym = vllm.prompts_imgs_target_to_xym([prompt], [image], [target])
        (input_embeds, vt_range), label_ids, label_masks = xym
        mid = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path)
        reps = (
            mid[:, :vt_range[0]].float(),
            mid[:, vt_range[0]:vt_range[1]].float(),
            mid[:, vt_range[1]:1 - label_masks.shape[1]].float(),
            mid[:, 1 - label_masks.shape[1]:].float(),
        )
        return xym, mid, reps

    def get_reps_for_edit(self, vllm:BaseVLLMForEdit, request: Dict):
        '''request: {'prompt': str, 'image': PILImage, 'target': str}'''
        return self._pack_edit_forward(vllm, request['prompt'], request['image'], request['target'])[2]

    def get_new_edit(self, vision_reps:torch.Tensor, query_reps:torch.Tensor, ans_reps:torch.Tensor,
                     patch_reps:Optional[torch.Tensor]=None)->Tuple[torch.Tensor]:
        # vision_reps: [1, l1, d]; query_reps: [1, l2, d]; ans_reps: [1, l3, d]
        assert len(vision_reps) == len(query_reps) == len(ans_reps) == 1
        evr = self.edit_extractor.extract_vision(query_reps, vision_reps)
        eqr = self.edit_extractor.extract_query(query_reps)
        edit_reps = torch.cat([vision_reps, query_reps, ans_reps], 1)
        moe_c = self.moegen_c(edit_reps)
        if (
            self.patch_grounding_enabled
            and self.patch_expert_gen_enabled
            and self.moegen_c_patch is not None
            and patch_reps is not None
        ):
            patch_ctx = self._extract_patch_routing(query_reps, patch_reps)
            moe_c = moe_c + self.moegen_c_patch(patch_ctx)
        moe_r = self.moegen_r(edit_reps)
        evr_patch = None
        if self.patch_grounding_enabled and self.patch_grounding_module is not None and patch_reps is not None:
            evr_patch = self._extract_patch_routing(query_reps, patch_reps)
        if evr_patch is not None:
            return eqr, evr, moe_c, moe_r, evr_patch
        return eqr, evr, moe_c, moe_r
    
    def get_edit_residual(self, inpt_reps:torch.Tensor, moe_cs:torch.Tensor, 
                          moe_rs:torch.Tensor, fuse_coe:torch.Tensor)->torch.Tensor:
        '''inpt_reps: [1, l, hidden_size]; moe_in_cs: [m, lora_rank, hidden_size (ffn_in)]; 
            moe_in_rs: [m, lora_rank, ffn_out]; moe_out_cs: [m, lora_rank, ffn_out]; 
            moe_out_rs: [m, lora_rank, ffn_in]; fuse_coe: [1, m]'''
        assert len(inpt_reps) == len(fuse_coe) == 1
        x = self.instant_reps_norm(inpt_reps)[0] # [1, l, hidden_size] -> [l, hidden_size]
        # x = torch.einsum('ld,mrd,mrD,m->lD', x, moe_cs, moe_rs, fuse_coe[0])
        x = torch.relu(torch.einsum('ld,mrd->lmr', x, moe_cs))
        x = torch.einsum('lmr,mrd,m->ld', x, moe_rs, fuse_coe[0])
        return x.unsqueeze(0) # [1, l, hidden_size]
    
    def get_moe_fuse_coe(self, iqrs:torch.Tensor, eqrs:torch.Tensor, split = False):
        sim = torch.einsum('ned,med->nme', iqrs, eqrs).mean(2) * self.sim_scale # [n, m]
        rela_sim = torch.softmax(sim, 1) # [n, m]
        abs_sim = torch.sigmoid(sim) # [n, m] rescale ???
        if split: 
            return rela_sim, abs_sim
        return rela_sim * abs_sim # [n, m]

    ############################################################################
    ########################### Editor Functions ###############################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'liveedit', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def restore_to_original_model(self):
        self.requests_pool = []
        self.eqr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device) # editing query representation pool
        self.evr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device) # editing-relevant vision representation pool
        self.evr_patch_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device)
        self.moe_cs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device) # lora 1 column weights pool
        self.moe_rs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device) # lora 1 row weights pool
        # Chunk index per pool entry for AR routing gate (-1 = non-AR / single-pass)
        self.chunk_index_pool: List[int] = []
        self.vit_patch_pool = []
        self.dino_key_pool = []

    def _append_edit_to_pool(
        self,
        request: Dict,
        eqr: torch.Tensor,
        evr: torch.Tensor,
        moe_c: torch.Tensor,
        moe_r: torch.Tensor,
        chunk_index: int = -1,
        evr_patch: Optional[torch.Tensor] = None,
    ) -> None:
        need_dino = (
            getattr(self, "decode_vision_gate_enabled", False)
            or getattr(self, "pool_dedup_enabled", False)
        )
        new_key = self._dino_key_for_image(request["image"]) if need_dino else None
        merge_i = self._pool_dedup_merge_index(new_key) if new_key is not None else None
        # skip mode: drop near-duplicate visual keys instead of overwriting a different fact
        if (
            merge_i is not None
            and getattr(self, "pool_dedup_mode", "merge") == "skip"
        ):
            return
        if merge_i is not None:
            self.requests_pool[merge_i] = request
            self.eqr_pool[merge_i] = eqr[0] if eqr.shape[0] == 1 else eqr
            self.evr_pool[merge_i] = evr[0] if evr.shape[0] == 1 else evr
            self.moe_cs_pool[merge_i] = moe_c[0] if moe_c.shape[0] == 1 else moe_c
            self.moe_rs_pool[merge_i] = moe_r[0] if moe_r.shape[0] == 1 else moe_r
            if getattr(self, "chunk_index_pool", None) is not None:
                self.chunk_index_pool[merge_i] = chunk_index
            if evr_patch is not None and self.evr_patch_pool.shape[0] > merge_i:
                self.evr_patch_pool[merge_i] = evr_patch[0] if evr_patch.shape[0] == 1 else evr_patch
            if new_key is not None:
                self.dino_key_pool[merge_i] = new_key
            if getattr(self, "patch_rerank_enabled", False) and hasattr(self.vllm, "get_vit_patch_embeds"):
                self.vit_patch_pool[merge_i] = self.vllm.get_vit_patch_embeds(request["image"])[0].detach().cpu()
            return
        self.requests_pool.append(request)
        self.eqr_pool = torch.cat([self.eqr_pool, eqr], 0)
        self.evr_pool = torch.cat([self.evr_pool, evr], 0)
        self.moe_cs_pool = torch.cat([self.moe_cs_pool, moe_c], 0)
        self.moe_rs_pool = torch.cat([self.moe_rs_pool, moe_r], 0)
        if getattr(self, "chunk_index_pool", None) is not None:
            self.chunk_index_pool.append(chunk_index)
        if evr_patch is not None:
            self.evr_patch_pool = torch.cat([self.evr_patch_pool, evr_patch], 0)
        if new_key is not None:
            self.dino_key_pool.append(new_key)
        self._append_vit_patch_for_edit(request["image"])

    def _edit_new_from_request(self, request: Dict, target: str):
        r = {"prompt": request["prompt"], "image": request["image"], "target": target}
        _, vision_reps, query_reps, ans_reps = self.get_reps_for_edit(self.vllm, r)
        patch_reps = self._encode_patches(request["image"]) if self.patch_grounding_enabled else None
        new_edit = self.get_new_edit(vision_reps, query_reps, ans_reps, patch_reps)
        if len(new_edit) == 5:
            return new_edit
        eqr, evr, moe_c, moe_r = new_edit
        return eqr, evr, moe_c, moe_r, None

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...}
        In AR mode (ar_mode=True), performs chunk-wise edits and appends one edit per chunk;
        on empty chunks or any failure, falls back to single-pass edit with full target_new.
        """
        self.is_editing = True
        ar_mode = getattr(self, 'ar_mode', False)
        if not ar_mode:
            self._edit_one_piece_single(request)
            self.is_editing = False
            return
        # AR path: chunk-wise edit loop with safe fallback
        saved_len = len(self.requests_pool)
        try:
            tokenizer = self.vllm.get_llm_tokenizer()
            chunk_size = getattr(self, 'chunk_size', 16)
            max_chunks = getattr(self, 'max_chunks', None)
            chunks = split_target_into_chunks(
                tokenizer, request['target_new'], chunk_size, max_chunks
            )
            if not chunks:
                raise ValueError("AR mode: no chunks from target_new")
            for t in range(len(chunks)):
                target_t = reconstruct_text_from_chunks(tokenizer, chunks[: t + 1])
                eqr, evr, moe_c, moe_r, evr_patch = self._edit_new_from_request(request, target_t)
                self._append_edit_to_pool(request, eqr, evr, moe_c, moe_r, chunk_index=t, evr_patch=evr_patch)
            self.is_editing = False
            return
        except Exception:
            # Fallback: truncate any partial appends and do single-pass edit
            self.requests_pool = self.requests_pool[:saved_len]
            self.eqr_pool = self.eqr_pool[:saved_len]
            self.evr_pool = self.evr_pool[:saved_len]
            self.evr_patch_pool = self.evr_patch_pool[:saved_len]
            self.moe_cs_pool = self.moe_cs_pool[:saved_len]
            self.moe_rs_pool = self.moe_rs_pool[:saved_len]
            if getattr(self, 'chunk_index_pool', None) is not None:
                self.chunk_index_pool = self.chunk_index_pool[:saved_len]
            self.vit_patch_pool = self.vit_patch_pool[:saved_len]
            self.dino_key_pool = self.dino_key_pool[:saved_len]
        self._edit_one_piece_single(request)
        self.is_editing = False

    def _edit_one_piece_single(self, request: Dict) -> None:
        """Single-pass edit: one (eqr, evr, moe_c, moe_r) for full target_new. Used by edit_one_piece."""
        eqr, evr, moe_c, moe_r, evr_patch = self._edit_new_from_request(request, request["target_new"])
        self._append_edit_to_pool(request, eqr, evr, moe_c, moe_r, chunk_index=-1, evr_patch=evr_patch)

    def edit_batch(self, requests: List[Dict]):
        raise
    
    ############################################################################
    ####################### Training Functions #################################
    ############################################################################
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        modules = {'edit_extractor': self.edit_extractor, 'inpt_extractor': self.inpt_extractor,
            'moegen_c': self.moegen_c, 'moegen_r': self.moegen_r,
            'instant_reps_norm': self.instant_reps_norm}
        if self.patch_grounding_enabled and self.patch_grounding_module is not None:
            modules['patch_grounding_module'] = self.patch_grounding_module
        if self.patch_expert_gen_enabled and self.moegen_c_patch is not None:
            modules['moegen_c_patch'] = self.moegen_c_patch
        return modules

    def load_ckpt(self, ckpt_path, restrict = True, load_opt = True):
        from editor.vllm_editors.base import resolve_ckpt_path
        ckpt_path = resolve_ckpt_path(ckpt_path)
        ckpt = torch.load(ckpt_path, 'cpu')
        train_modules = self.get_modules_for_training()
        for k, mod in train_modules.items():
            if k in ckpt.get('train_modules', {}):
                mod.load_state_dict(ckpt['train_modules'][k], restrict)
            else:
                print('[LiveEdit] %s not in checkpoint — keeping initialized weights' % k)
        if load_opt and 'opt' in ckpt:
            try:
                self.opt.load_state_dict(ckpt['opt'])
            except Exception as exc:
                print('[LiveEdit] optimizer state not loaded: %s' % exc)
            if self.lr_scheduler is not None and ckpt.get('lr_scheduler') is not None:
                try:
                    self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                except Exception as exc:
                    print('[LiveEdit] lr_scheduler state not loaded: %s' % exc)
        print('Load %s checkpoint from %s.' % (self.name_of_editor_and_model()[0], ckpt_path))
        return ckpt['i'], ckpt['epoch'], ckpt['loss'], ckpt.get('ema_loss')
    
    def reinit_train_parameters(self):
        self.edit_extractor.reset_parameters()
        self.inpt_extractor.reset_parameters()
        self.moegen_c.reset_parameters()
        self.moegen_r.reset_parameters()
        self.instant_reps_norm.reset_parameters()
        if self.patch_grounding_module is not None:
            self.patch_grounding_module.reset_parameters()
        if self.moegen_c_patch is not None:
            self.moegen_c_patch.reset_parameters()

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data_with_img
        
        
    def organize_batch_data(self, a_batch_raw_data:List):
        with self._model_forward_lock:
            return self._organize_batch_data_impl(a_batch_raw_data)

    def _organize_batch_data_impl(self, a_batch_raw_data:List):
        vllm = self.vllm_data_proc # use the VLLM used to preprocess data
        batch_size = len(a_batch_raw_data)
        batch_edit_signal = []
        batch_edit_packs = []  # parallel to batch_edit_signal: (xym, mid, reps) per request
        rel_edit_i = [] 
        batch_rel_data = [] # for each: {'prompts': [str], 'imgs': [PILImage], 'targets': [str]}
        batch_gen_data = {gen_name: [] for gen_name in a_batch_raw_data[0]['generality']}
        batch_loc_data = {loc_name: [] for loc_name in a_batch_raw_data[0]['locality']
                if a_batch_raw_data[0]['locality'][loc_name][0]['image'] != None}
        for d in a_batch_raw_data:
            # Edit signal (single mid-forward per request)
            packs = [self._pack_edit_forward(vllm, r['prompt'], r['image'], r['target_new'])
                     for r in d['requests']]
            batch_edit_packs.append(packs)
            batch_edit_signal.append([p[2] for p in packs])
            # Reliability 
            i = self.rng_data_proc.integers(0, len(d['requests']))
            rel_edit_i.append(i)
            r = d['requests'][i]
            batch_rel_data.append({'prompts': [r['prompt']], 'imgs': [r['image']], 'targets': [r['target_new']]})
            # Generality
            for gen_name in batch_gen_data.keys():
                i = self.rng_data_proc.integers(0, len(d['generality'][gen_name]))
                g = d['generality'][gen_name][i]
                batch_gen_data[gen_name].append({'prompts': [g['prompt']], 'imgs': [g['image']], 'targets': [g['target']]})
            # Locality
            for loc_name in batch_loc_data.keys():
                i = self.rng_data_proc.integers(0, len(d['locality'][loc_name]))
                l = d['locality'][loc_name][i]
                batch_loc_data[loc_name].append({'prompts': [l['prompt']], 'imgs': [l['image']], 'targets': [l['target']]})
        # Fuse moe masks
        rel_edit_i_request_indices = list(rel_edit_i)  # [batch] request index per batch item (for AR chunk indexing)
        edit_ns = torch.tensor([len(bes) for bes in batch_edit_signal], device=self.device)
        cols = torch.sum(edit_ns)
        start_indices = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), edit_ns[:-1]]), dim=0).unsqueeze(1)
        rel_edit_i = start_indices[:, 0] + torch.tensor(rel_edit_i, device=self.device)
        rel_moe_mask = torch.zeros((batch_size, cols), dtype=torch.int, device=self.device)
        rel_moe_mask[torch.arange(batch_size), rel_edit_i] = 1
        idx = torch.arange(cols, device = self.device).expand(batch_size, cols)
        gen_moe_mask = ((idx >= start_indices) & (idx < (start_indices + edit_ns.unsqueeze(1)))).int()
        loc_moe_mask = torch.zeros_like(gen_moe_mask) 
        for i in range(batch_size):
            ns = self.rng_train.integers(0, int(cols) + 1, 3)
            rel_moe_mask[i, :ns[0]], gen_moe_mask[i, :ns[1]], loc_moe_mask[i, :ns[2]] = 1, 1, 1
        rel_moe_mask, gen_moe_mask, loc_moe_mask = rel_moe_mask.bool(), gen_moe_mask.bool(), loc_moe_mask.bool()
        # Reliability: reuse the already-computed edit pack for the chosen request (no extra LLaVA forward)
        rel_xyms, rel_mid_reps, rel_edit_reps = [], [], []
        for j, ri in enumerate(rel_edit_i_request_indices):
            xym, mid, reps = batch_edit_packs[j][ri]
            rel_xyms.append(xym)
            rel_mid_reps.append(mid)
            rel_edit_reps.append(reps)
        packed_rel_data = (rel_xyms, rel_mid_reps, rel_edit_reps)
        # Generality: one mid-forward each (was mid + get_reps ≈ 2)
        packed_gen_data = {}
        for k, v in batch_gen_data.items():
            packs = [self._pack_edit_forward(vllm, g['prompts'][0], g['imgs'][0], g['targets'][0]) for g in v]
            packed_gen_data[k] = (
                [p[0] for p in packs],
                [p[1] for p in packs],
                [p[2] for p in packs],
            )
        # Locality: one mid-forward + one forward_from_mid (was mid + forward_from_mid + get_reps)
        packed_loc_data = {}
        for k, v in batch_loc_data.items():
            packs = [self._pack_edit_forward(vllm, l['prompts'][0], l['imgs'][0], l['targets'][0]) for l in v]
            loc_xyms = [p[0] for p in packs]
            loc_mid_reps = [p[1] for p in packs]
            loc_edit_reps = [p[2] for p in packs]
            pre_logits = [
                vllm.forward_from_mid_layer(
                    input_embeds, vt_range, lmr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i
                ).logits
                for lmr, ((input_embeds, vt_range), label_ids, label_masks) in zip(loc_mid_reps, loc_xyms)
            ]
            packed_loc_data[k] = (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps)
        # Retrieval contrastive learning data for neighbor loss
        def get_rand_gn_ln():
            gn = list(batch_gen_data.keys())[self.rng_data_proc.integers(0, len(batch_gen_data.keys()))]
            ln = list(batch_loc_data.keys())[self.rng_data_proc.integers(0, len(batch_loc_data.keys()))]
            return gn, ln
        batch_retr_neib_data = [[], []]
        batch_retr_neib_images = [[], []]
        for j in range(batch_size):
            # neighbor 1
            n = self.rng_data_proc.integers(0, 3) # 0: rel, 1: gen, 2: loc. Where rel and gen are in the same neighbor.
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            img = [batch_rel_data, batch_gen_data[gn], batch_loc_data[ln]][n][j]['imgs'][0]
            batch_retr_neib_data[0].append(d)
            batch_retr_neib_images[0].append(img)
            # neighbor 2 
            n = self.rng_data_proc.integers(0, 2) if n != 2 else n # reselect in the same neighbor
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            img = [batch_rel_data, batch_gen_data[gn], batch_loc_data[ln]][n][j]['imgs'][0]
            batch_retr_neib_data[1].append(d)
            batch_retr_neib_images[1].append(img)
        # Retrieval contrastive learning data for prototype loss 
        batch_retr_prot_data = [[], []]
        batch_retr_prot_images = [[], []]
        for j in range(batch_size):
            n = self.rng_data_proc.integers(0, 2) # 0: r/g, 1: l
            # data_1
            gn, ln = get_rand_gn_ln()
            src_i = self.rng_data_proc.integers(0, 2)
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][src_i], packed_loc_data[ln][-1]][n][j][1:3]
            img = [[batch_rel_data, batch_gen_data[gn]][src_i], batch_loc_data[ln]][n][j]['imgs'][0]
            batch_retr_prot_data[0].append(d)
            batch_retr_prot_images[0].append(img)
            # data_2
            gn, ln = get_rand_gn_ln()
            src_i = self.rng_data_proc.integers(0, 2)
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][src_i], packed_loc_data[ln][-1]][1-n][j][1:3]
            img = [[batch_rel_data, batch_gen_data[gn]][src_i], batch_loc_data[ln]][1-n][j]['imgs'][0]
            batch_retr_prot_data[1].append(d)
            batch_retr_prot_images[1].append(img)
        batch_edit_patch_reps = None
        batch_adv_specs: List[Dict[str, Any]] = []
        if float(getattr(self, "adversarial_grounding_lambda", 0.0)) > 0:
            for d in a_batch_raw_data:
                req = d["requests"][0]
                spec: Dict[str, Any] = {
                    "prompt": req["prompt"],
                    "rel_image": req["image"],
                    "target": req["target_new"],
                    "swap_image": None,
                    "hard_neg_image": None,
                }
                gen = d.get("generality", {})
                if gen.get("image_swap"):
                    spec["swap_image"] = gen["image_swap"][0]["image"]
                if gen.get("hard_negative"):
                    spec["hard_neg_image"] = gen["hard_negative"][0]["image"]
                batch_adv_specs.append(spec)
        if self.patch_grounding_enabled:
            batch_edit_patch_reps = [
                [self._encode_patches(r['image']) for r in d['requests']]
                for d in a_batch_raw_data
            ]
        # AR mode: attach chunk metadata per request when ar_mode=True (Task 3)
        batch_ar_chunks: Optional[List[List[List[List[int]]]]] = None  # [batch][request][chunk][token_id]
        if getattr(self, 'ar_mode', False):
            tokenizer = vllm.get_llm_tokenizer()
            chunk_size = getattr(self, 'chunk_size', 16)
            max_chunks = getattr(self, 'max_chunks', None)
            batch_ar_chunks = []
            for d in a_batch_raw_data:
                request_chunks = [
                    split_target_into_chunks(tokenizer, r['target_new'], chunk_size, max_chunks)
                    for r in d['requests']
                ]
                batch_ar_chunks.append(request_chunks)
        a_batch_organized_data = (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask,
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data, batch_retr_prot_data,
            batch_ar_chunks, batch_edit_patch_reps, batch_retr_neib_images, batch_retr_prot_images,
            batch_adv_specs)
        if getattr(self, 'ar_mode', False):
            a_batch_organized_data = (*a_batch_organized_data, rel_edit_i_request_indices)
        return move_to_device(a_batch_organized_data, self.device)

    def train_a_batch(self, a_batch_organized_data):
        with self._model_forward_lock:
            return self._train_a_batch_impl(a_batch_organized_data)

    def _train_a_batch_impl(self, a_batch_organized_data):
        eps = 1e-8
        vllm = self.vllm
        (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask,
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data,
            batch_retr_prot_data, batch_ar_chunks, batch_edit_patch_reps,
            batch_retr_neib_images, batch_retr_prot_images) = a_batch_organized_data[:14]
        batch_adv_specs = (
            a_batch_organized_data[14]
            if len(a_batch_organized_data) > 14
            and isinstance(a_batch_organized_data[14], list)
            and (not a_batch_organized_data[14] or isinstance(a_batch_organized_data[14][0], dict))
            else []
        )
        rel_edit_i = (
            a_batch_organized_data[15] if len(a_batch_organized_data) > 15
            else (a_batch_organized_data[14] if len(a_batch_organized_data) > 14 and not batch_adv_specs else None)
        )
        # initialize losses
        loss = 0
        log_dict = {}
        # Prepare edit data eqr, evr, moe_c, moe_r
        new_edit = []
        if batch_edit_patch_reps is not None:
            for bes, patch_list in zip(batch_edit_signal, batch_edit_patch_reps):
                for (pre_vision_reps, vision_reps, query_reps, ans_reps), patch_reps in zip(bes, patch_list):
                    new_edit.append(self.get_new_edit(vision_reps, query_reps, ans_reps, patch_reps))
        else:
            new_edit = [self.get_new_edit(vision_reps, query_reps, ans_reps)
                        for bes in batch_edit_signal
                        for pre_vision_reps, vision_reps, query_reps, ans_reps in bes]
        eqrs = torch.cat([ne[0] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        edit_eqrs = eqrs  # keep for AVBL/DEK (eqrs is overwritten by soft-routing later)
        # evrs = torch.cat([ne[1] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        moe_cs = torch.cat([ne[2] for ne in new_edit], 0) # [batch_request_n, rank_n, d1]
        moe_rs = torch.cat([ne[3] for ne in new_edit], 0) # [batch_request_n, rank_n, d2]
        # Reliability edit loss (AR path: chunk-step loop with simple aggregate when ar_mode=True)
        rel_loss = 0
        rel_xyms, rel_mid_reps, rel_edit_reps = packed_rel_data
        if getattr(self, 'ar_mode', False) and batch_ar_chunks is not None and rel_edit_i is not None:
            for b, (xyms, mr, er, mm) in enumerate(zip(rel_xyms, rel_mid_reps, rel_edit_reps, rel_moe_mask)):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps)
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm])
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                chunks_b = batch_ar_chunks[b][rel_edit_i[b]]
                if not chunks_b:
                    continue
                for t, chunk_t in enumerate(chunks_b):
                    start_pos = sum(len(chunks_b[k]) for k in range(t))
                    end_pos = start_pos + len(chunk_t)
                    chunk_mask = torch.zeros_like(label_masks, device=label_masks.device, dtype=label_masks.dtype)
                    chunk_mask[:, start_pos:end_pos] = label_masks[:, start_pos:end_pos]
                    if chunk_mask.sum() == 0:
                        continue
                    rel_loss += vllm.label_loss(logits, label_ids, chunk_mask, True)
            rel_loss = rel_loss / batch_size if batch_size > 0 else rel_loss
        else:
            for xyms, mr, er, mm in zip(rel_xyms, rel_mid_reps, rel_edit_reps, rel_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                rel_loss += vllm.label_loss(logits, label_ids, label_masks, True)
            rel_loss /= batch_size
        log_dict['Reliability loss'] = float(rel_loss)
        loss += rel_loss * self.cfg.train_cfg.rel_lambda
        # Generality edit losses
        gen_loss = 0
        for gen_name, (gen_xyms, gen_mid_reps, gen_edit_reps) in packed_gen_data.items():
            gen_name_loss = 0
            for xyms, mr, er, mm in zip(gen_xyms, gen_mid_reps, gen_edit_reps, gen_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                gen_name_loss += vllm.label_loss(logits, label_ids, label_masks, True)
            gen_name_loss /= batch_size
            log_dict['Generality loss %s'%gen_name] = float(gen_name_loss)
            gen_loss += gen_name_loss
        log_dict['Generality loss'] = float(gen_loss)
        loss += gen_loss * self.cfg.train_cfg.gen_lambda
        # Locality edit losses
        loc_loss = 0
        for loc_name, (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps) in packed_loc_data.items():
            loc_name_loss = 0
            for xyms, mr, pl, er, mm in zip(loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps, loc_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                loc_name_loss += vllm.logit_KL_loss(logits, pl, label_masks, True)
            loc_name_loss /= batch_size
            log_dict['Locality loss %s'%loc_name] = float(loc_name_loss)
            loc_loss += loc_name_loss
        log_dict['Locality loss'] = float(loc_loss)
        loss += loc_loss * self.cfg.train_cfg.loc_lambda
        # Moe soft routing loss
        iqrs = torch.cat([self.inpt_extractor.extract_query(d[1]) for d in batch_retr_neib_data[0]], 0) # [b, eqe_n, module_dim]
        eqrs = torch.cat([self.edit_extractor.extract_query(d[1]) for d in batch_retr_neib_data[1]], 0) # [b, eqe_n, module_dim]
        rela_sim, abs_sim = self.get_moe_fuse_coe(iqrs, eqrs, True) # [b, b]
        moe_soft_routing_loss_rela = - torch.log(torch.diag(rela_sim) + eps).mean(0)
        abs_pos_sim = torch.diag(abs_sim)
        abs_neg_sim = torch.diag(abs_sim.roll(1, 1))
        moe_soft_routing_loss_abs = - (torch.log(abs_pos_sim + eps) + torch.log(1 - abs_neg_sim + eps)).mean()
        moe_soft_routing_loss = moe_soft_routing_loss_rela + moe_soft_routing_loss_abs
        log_dict['MoE soft routing relative loss'] = float(moe_soft_routing_loss_rela)
        log_dict['MoE soft routing abstract loss'] = float(moe_soft_routing_loss_abs)
        log_dict['MoE soft routing mean abstract positive similarity'] = float(abs_pos_sim.mean(0)) 
        log_dict['MoE soft routing mean abstract negative similarity'] = float(abs_neg_sim.mean(0))
        log_dict['MoE soft routing loss'] = float(moe_soft_routing_loss)
        loss += moe_soft_routing_loss * self.cfg.train_cfg.soft_routing_lambda
        # Hard Routing losses
        def hard_routing(inpt_reps, edit_reps):
            ivrs = torch.cat([self.inpt_extractor.extract_vision(d[1], d[0]) for d in inpt_reps], 0) # [b, eqe_n, module_dim]
            evrs = torch.cat([self.edit_extractor.extract_vision(d[1], d[0]) for d in edit_reps], 0) # [b, eqe_n, module_dim]
            sim = torch.einsum('bed,med->bme', ivrs, evrs).mean(2) * self.sim_scale # [b, b]
            ivrs_prot = torch.cat([self.inpt_extractor.extract_from_visprot(d[1]) for d in inpt_reps]) # [b, eqe_n, module_dim]
            sim_prot = torch.einsum('bed,bed->be', ivrs, ivrs_prot).mean(1, True) * self.sim_scale # [b,1]
            sim = torch.softmax(torch.cat([sim, sim_prot], 1), 1) # [b, b+1]
            return sim # [b, b+1]
        # Hard routing neighbor loss
        sim = hard_routing(*batch_retr_neib_data) # [b, b+1]
        loss_retr_neb = - torch.log(torch.diag(sim) + eps).mean(0)
        log_dict['MoE hard routing neighbor loss'] = float(loss_retr_neb)
        # Hard routing prototype loss 
        sim = hard_routing(*batch_retr_prot_data) # [b, b+1]
        loss_retr_prot = - torch.log(sim[:, -1] + eps).mean(0)
        log_dict['MoE hard routing prototype loss'] = float(loss_retr_prot)
        hard_routing_loss = loss_retr_neb + loss_retr_prot 
        log_dict['MoE hard routing loss'] = float(hard_routing_loss)
        loss += hard_routing_loss * self.cfg.train_cfg.hard_routing_lambda
        if self.patch_grounding_enabled and self.patch_grounding_module is not None:
            def patch_hard_routing(inpt_reps, edit_reps, inpt_imgs, edit_imgs):
                ivrs = torch.cat([
                    self._extract_patch_routing(d[1], self._encode_patches(img))
                    for d, img in zip(inpt_reps, inpt_imgs)
                ], 0)
                evrs = torch.cat([
                    self._extract_patch_routing(d[1], self._encode_patches(img))
                    for d, img in zip(edit_reps, edit_imgs)
                ], 0)
                per_eqe = torch.einsum('bed,med->bme', ivrs, evrs) * self.sim_scale
                mode = getattr(self, 'routing_pool_mode', 'mean')
                if mode == 'max':
                    sim = per_eqe.max(dim=2).values
                elif mode == 'attn':
                    weights = torch.softmax(per_eqe, dim=2)
                    sim = (weights * per_eqe).sum(dim=2)
                else:
                    sim = per_eqe.mean(2)
                ivrs_prot = torch.cat([
                    self._extract_patch_prototype_routing(d[1])
                    for d in inpt_reps
                ], 0)
                sim_prot = torch.einsum('bed,bed->be', ivrs, ivrs_prot).mean(1, True) * self.sim_scale
                return torch.softmax(torch.cat([sim, sim_prot], 1), 1)

            neib_inpt = list(zip(batch_retr_neib_data[0], batch_retr_neib_images[0]))
            neib_edit = list(zip(batch_retr_neib_data[1], batch_retr_neib_images[1]))
            sim_p = patch_hard_routing(
                [x[0] for x in neib_inpt], [x[0] for x in neib_edit],
                [x[1] for x in neib_inpt], [x[1] for x in neib_edit],
            )
            loss_patch_neb = - torch.log(torch.diag(sim_p) + eps).mean(0)
            prot_inpt = list(zip(batch_retr_prot_data[0], batch_retr_prot_images[0]))
            prot_edit = list(zip(batch_retr_prot_data[1], batch_retr_prot_images[1]))
            sim_p = patch_hard_routing(
                [x[0] for x in prot_inpt], [x[0] for x in prot_edit],
                [x[1] for x in prot_inpt], [x[1] for x in prot_edit],
            )
            loss_patch_prot = - torch.log(sim_p[:, -1] + eps).mean(0)
            patch_routing_loss = loss_patch_neb + loss_patch_prot
            log_dict['Patch hard routing neighbor loss'] = float(loss_patch_neb)
            log_dict['Patch hard routing prototype loss'] = float(loss_patch_prot)
            log_dict['Patch hard routing loss'] = float(patch_routing_loss)
            loss += patch_routing_loss * self.cfg.train_cfg.hard_routing_lambda
        adv_lambda = float(getattr(self, "adversarial_grounding_lambda", 0.0))
        if adv_lambda > 0 and batch_adv_specs and self.patch_grounding_enabled:
            evrs = torch.cat([ne[1] for ne in new_edit], 0)
            evr_patch = (
                torch.cat([ne[4] for ne in new_edit], 0)
                if len(new_edit[0]) > 4 and new_edit[0][4] is not None else None
            )
            saved_pool = (self.evr_pool, self.evr_patch_pool, self.eqr_pool)
            self.evr_pool = evrs
            self.eqr_pool = edit_eqrs  # DEK dual-gate AVBL needs batch prompt keys
            if evr_patch is not None:
                self.evr_patch_pool = evr_patch
            batch_edit_images = [spec.get("rel_image") for spec in batch_adv_specs]
            avbl_total = torch.tensor(0.0, device=self.device)
            avbl_n = 0
            for j, spec in enumerate(batch_adv_specs):
                adv_imgs = [img for img in (spec.get("swap_image"), spec.get("hard_neg_image")) if img]
                if not adv_imgs:
                    continue
                avbl_loss, avbl_log = self.compute_avbl_losses(
                    vllm, spec["prompt"], spec["rel_image"], adv_imgs, [j],
                    spec.get("target", "."),
                    batch_edit_images=batch_edit_images,
                    batch_moe_cs=moe_cs,
                    batch_moe_rs=moe_rs,
                    batch_eqrs=edit_eqrs,
                )
                avbl_total = avbl_total + avbl_loss
                avbl_n += 1
                for k, v in avbl_log.items():
                    log_dict['AVBL %s' % k] = log_dict.get('AVBL %s' % k, 0.0) + v
            if avbl_n > 0:
                avbl_total = avbl_total / avbl_n
                log_dict['AVBL loss'] = float(avbl_total.detach())
                loss += avbl_total * adv_lambda
            # Cross-batch mining buffer: store this batch's experts for later refuse mining.
            self._push_avbl_pool_buffer(batch_edit_images, moe_cs, moe_rs, edit_eqrs)
            log_dict['AVBL pool buffer size'] = float(len(getattr(self, "_avbl_pool_buffer", [])))
            self.evr_pool, self.evr_patch_pool, self.eqr_pool = saved_pool
            self._clear_query_patch_cache()
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
        log_dict['Learning rate'] = float(self.lr_scheduler.get_last_lr()[0])
        return float(loss), log_dict
    
    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        from torch.optim.lr_scheduler import LambdaLR
        params = [*self.edit_extractor.parameters(), *self.inpt_extractor.parameters(),
                  *self.instant_reps_norm.parameters(), *self.moegen_c.parameters(),
                  *self.moegen_r.parameters()]
        if self.patch_grounding_module is not None:
            params.extend(self.patch_grounding_module.parameters())
        if self.moegen_c_patch is not None:
            params.extend(self.moegen_c_patch.parameters())
        opt = Adam(params, self.cfg.train_cfg.lr)
        lr_cut_it = torch.tensor(self.cfg.train_cfg.lr_cut_it, device=self.device)
        def lr_lambda(step):
            return self.cfg.train_cfg.lr_cut_rate**(step > lr_cut_it).sum()
        lr_scheduler = LambdaLR(opt, lr_lambda)
        return opt, lr_scheduler

    def set_train(self, is_train:bool):
        self.is_train = is_train
        self.vllm.model.requires_grad_(False)
        self.vllm.model.train(False)
        if hasattr(self, 'vllm_data_proc') and self.vllm_data_proc != None:
            self.vllm_data_proc.model.requires_grad_(False)
            self.vllm_data_proc.model.train(False)
        self.edit_extractor.train(is_train)
        self.edit_extractor.requires_grad_(is_train)
        self.inpt_extractor.train(is_train)
        self.inpt_extractor.requires_grad_(is_train)
        self.instant_reps_norm.train(is_train)
        self.instant_reps_norm.requires_grad_(is_train)
        self.moegen_c.train(is_train)
        self.moegen_c.requires_grad_(is_train)
        self.moegen_r.train(is_train)
        self.moegen_r.requires_grad_(is_train)
        if getattr(self, "patch_grounding_module", None) is not None:
            self.patch_grounding_module.train(is_train)
            self.patch_grounding_module.requires_grad_(is_train)
        if getattr(self, "moegen_c_patch", None) is not None:
            self.moegen_c_patch.train(is_train)
            self.moegen_c_patch.requires_grad_(is_train)

    def other_train_init_begin(self):
        self.rng_data_proc = np.random.default_rng(self.random_seed)
        self.rng_train = np.random.default_rng(self.random_seed + 1)
