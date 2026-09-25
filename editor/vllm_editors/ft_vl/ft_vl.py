from ...vllms_for_edit.base import BaseVLLMForEdit
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from ..base import VLLMBaseEditor
from ...base import BaseConfig
from ... import nethook
from copy import deepcopy
import numpy as np
import torch

@dataclass
class FTvlConfig(BaseConfig):
    edit_model_name:str
    # Method
    rewrite_module_tmp: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    norm_constraint: float
    batch_size: int = 128


class FTvl(VLLMBaseEditor):

    def __init__(self, vllm: BaseVLLMForEdit, config:FTvlConfig, device='cuda:0',
                 verbose = False):
        super().__init__(vllm, device)
        self.cfg = config
        self.verbose = verbose
        self.original_w = {
            n: p.clone()
            for n, p in self.vllm.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        } 
 
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'ft_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True

    def restore_to_original_model(self):
        with torch.no_grad():
            for n, p0 in self.original_w.items():
                p = nethook.get_parameter(self.vllm.model, n)
                p[...] = p0.to(dtype=p.dtype, device=p.device)

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'image': PILImage, 'prompt': str, 'target_new': str, ...},
            {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...] '''
        deltas = self.execute_ft(requests)
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                if self.verbose:
                    print('edit:', w_name, 'delta_norm=', float(upd_matrix.float().norm()))
                w = nethook.get_parameter(self.vllm.model, w_name)
                w[...] += upd_matrix.to(dtype=w.dtype, device=w.device)

    ######################################################################################
    ############################# Fine-Tune Utilization ##################################
    ######################################################################################
    def execute_ft(self, requests: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Executes the FT update algorithm for the specified update at the specified layer
        Invariant: model at beginning of function == model at end of function

        Optimizes float32 copies of rewrite weights (BLIP2 is staged fp16; AdamW on
        half weights collapses decode to <s>).
        """
        requests = deepcopy(requests)
        for request in requests:
            if request["target_new"] and request["target_new"][0] != " ":
                request["target_new"] = " " + request["target_new"]
            if self.verbose:
                print(
                    f"Executing FT algo for: "
                    f"[{request['prompt']}] -> [{request['target_new']}]"
                )

        # Live model weights (often fp16)
        weights = {
            n: p
            for n, p in self.vllm.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        }
        if not weights:
            raise RuntimeError(
                "FT found no rewrite weights for template %r layers=%r"
                % (self.cfg.rewrite_module_tmp, self.cfg.layers)
            )

        # Float32 editable copies + backup of live tensors
        weights_live_copy = {k: v.detach().clone() for k, v in weights.items()}
        weights_fp32 = {
            k: v.detach().float().clone().requires_grad_(True)
            for k, v in weights.items()
        }
        if self.verbose:
            print(f"Weights to be updated: {list(weights.keys())}")

        images = [r["image"] for r in requests]
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        opt = torch.optim.AdamW(
            list(weights_fp32.values()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # Only rewrite params need grads through the frozen backbone path
        req_grad_backup = {
            n: p.requires_grad for n, p in self.vllm.model.named_parameters()
        }
        for name, w in self.vllm.model.named_parameters():
            w.requires_grad = False

        loss_meter = AverageMeter()
        try:
            # Rewrite weights must require grad *before* forward so the graph attaches.
            for k, v in weights.items():
                v.requires_grad_(True)

            for it in range(self.cfg.num_steps):
                if self.verbose:
                    print(20 * "=")
                    print(f"Epoch: {it}")
                    print(20 * "=")
                loss_meter.reset()

                for imgs, txt, tgt in zip(
                    chunks(images, self.cfg.batch_size),
                    chunks(texts, self.cfg.batch_size),
                    chunks(targets, self.cfg.batch_size),
                ):
                    # Sync live (often fp16) weights from fp32 candidates
                    with torch.no_grad():
                        for k, v in weights.items():
                            v.copy_(weights_fp32[k].to(dtype=v.dtype))

                    (input_embeds, vt_range), label_ids, label_masks = (
                        self.vllm.prompts_imgs_target_to_xym(txt, imgs, tgt)
                    )
                    opt.zero_grad(set_to_none=True)
                    for k, v in weights.items():
                        if v.grad is not None:
                            v.grad = None
                    logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
                    loss = label_loss(logits, label_ids, label_masks)
                    if self.verbose:
                        print(f"Batch loss {loss.item()}")
                    loss_meter.update(loss.item(), n=label_ids.shape[0])

                    if not torch.isfinite(loss):
                        if self.verbose:
                            print("Non-finite FT loss; skipping step")
                        continue

                    loss.backward()
                    for k, v in weights.items():
                        if v.grad is None:
                            continue
                        g32 = v.grad.detach().float()
                        if weights_fp32[k].grad is None:
                            weights_fp32[k].grad = g32.clone()
                        else:
                            weights_fp32[k].grad.copy_(g32)
                        v.grad = None

                    if loss.item() >= 1e-2:
                        opt.step()

                    if isinstance(self.cfg.norm_constraint, float):
                        eps = self.cfg.norm_constraint
                        with torch.no_grad():
                            for k, v32 in weights_fp32.items():
                                base = weights_live_copy[k].float()
                                v32.clamp_(min=base - eps, max=base + eps)

                if self.verbose:
                    print(f"Total loss {loss_meter.avg}")
                if loss_meter.avg < 1e-2:
                    break
        finally:
            for name, p in self.vllm.model.named_parameters():
                p.requires_grad = req_grad_backup.get(name, False)

        deltas = {
            k: (weights_fp32[k].detach() - weights_live_copy[k].float()).to(
                dtype=weights_live_copy[k].dtype
            )
            for k in weights
        }

        # Restore live weights to pre-execute_ft state
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_live_copy[k]

        if self.verbose:
            print(f"Deltas successfully computed for {list(weights.keys())}")

        return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self): 
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def label_loss(logits, label_ids, masks, average = True):
    # logits: [batch_size, total_l, d], label_ids/masks: [batch_size, short_l]
    logits = logits[:, -label_ids.shape[1]:]
    log_pre_p = torch.log_softmax(logits, -1)
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss