"""Fast AVBL direction check before full VG-Edit training.

Builds a mini edit pool, measures fused binding + routing on rel/swap/hard-neg,
runs a few AVBL-only gradient steps on the patch module, and re-measures.

Pass (directional, single seed):
  - swap or hard-neg fused expert_mass drops by >= 0.10, OR
  - swap/hard-neg false_cross drops by >= 10 pp
  - reliability own_prob drop <= 0.10

Usage:
  python tools/meeting_validation/step6/verify_avbl_direction.py \\
    -ckpt "records/liveedit/.../epoch-20-..." -dvc cuda:0 -dsn 15 -sen 5 -seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH


def _probe_specs(sample: Dict[str, Any]) -> List[Tuple[str, str, Any, str]]:
    req = sample["requests"][0]
    specs = [("reliability", req["prompt"], req["image"], req["target_new"])]
    gen = sample.get("generality", {})
    for name in ("image_swap", "hard_negative"):
        if name in gen and gen[name]:
            g = gen[name][0]
            specs.append((name, g["prompt"], g["image"], g["target"]))
    return specs


def _own_indices(sample_id: int, pool_sample_ids: List[int]) -> List[int]:
    return [i for i, sid in enumerate(pool_sample_ids) if sid == sample_id]


@torch.no_grad()
def measure_metrics(
    editor,
    samples: List[Dict[str, Any]],
    pool_sample_ids: List[int],
) -> Dict[str, float]:
    editor.set_train(False)
    rows: List[Dict[str, float]] = []
    for sid, sample in enumerate(samples):
        own = _own_indices(sid, pool_sample_ids)
        for probe, prompt, image, target in _probe_specs(sample):
            routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
            q_idx = routing.get("qformer_retrieved_indices", routing.get("retrieved_indices", []))
            false_cross = any(
                pool_sample_ids[i] != sid for i in q_idx if i < len(pool_sample_ids)
            )
            false_own = any(
                pool_sample_ids[i] == sid for i in q_idx if i < len(pool_sample_ids)
            )
            v, q = editor._forward_query_reps(editor.vllm, prompt, image, target)
            dist = editor._fused_binding_distribution(
                v, q, image,
                for_avbl=float(getattr(editor, "adversarial_grounding_lambda", 0.0)) > 0,
            )
            expert_mass = float(dist[:-1].sum()) if dist is not None else float("nan")
            own_prob = float(dist[own].sum()) if dist is not None and own else float("nan")
            rows.append({
                "probe": probe,
                "expert_mass": expert_mass,
                "own_prob": own_prob,
                "false_cross": float(false_cross),
                "false_own": float(false_own and probe != "reliability"),
            })

    def _mean(probe: str, key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r["probe"] == probe and r[key] == r[key]]
        return float(np.mean(vals)) if vals else None

    return {
        "reliability_own_prob": _mean("reliability", "own_prob"),
        "swap_expert_mass": _mean("image_swap", "expert_mass"),
        "hard_neg_expert_mass": _mean("hard_negative", "expert_mass"),
        "swap_false_cross": _mean("image_swap", "false_cross"),
        "hard_neg_false_cross": _mean("hard_negative", "false_cross"),
        "swap_false_own": _mean("image_swap", "false_own"),
    }


def _build_pool(editor, samples: List[Dict[str, Any]], sequential_n: int) -> List[int]:
    editor.restore_to_original_model()
    pool_ids: List[int] = []
    for sid, sample in enumerate(samples):
        editor.edit_one_piece(sample["requests"][0])
        n_new = len(editor.requests_pool) - len(pool_ids)
        pool_ids.extend([sid] * n_new)
        if sid + 1 >= sequential_n:
            break
    return pool_ids


def _avbl_train_steps(
    editor,
    samples: List[Dict[str, Any]],
    pool_sample_ids: List[int],
    sequential_n: int,
    steps: int,
    lr: float,
) -> List[Dict[str, float]]:
    editor.set_train(True)
    params = list(editor.patch_grounding_module.parameters())
    opt = Adam(params, lr=lr)
    history: List[Dict[str, float]] = []
    vllm = editor.vllm
    # Negatives need to dominate early; otherwise the positive step can globally
    # increase expert activation and make hard-neg worse (even if swap is trained).
    neg_weight = 2.0

    for step in range(steps):
        step_log: Dict[str, float] = {"step": float(step)}
        n_terms = 0
        for sid, sample in enumerate(samples):
            if sid >= sequential_n:
                break
            own = _own_indices(sid, pool_sample_ids)
            if not own:
                continue
            req = sample["requests"][0]
            gen = sample.get("generality", {})
            adv: List[Any] = []
            if gen.get("image_swap"):
                adv.append(gen["image_swap"][0]["image"])
            if gen.get("hard_negative"):
                adv.append(gen["hard_negative"][0]["image"])

            opt.zero_grad()
            adv_imgs = [a for a in adv if a is not None]
            loss, llog = editor.compute_avbl_losses(
                vllm, req["prompt"], req["image"], adv_imgs, own,
                target=req["target_new"], neg_weight=neg_weight,
            )
            llog_p = {k: v for k, v in llog.items() if k.startswith("avbl_rel")}
            llog_n_all = {k: v for k, v in llog.items() if not k.startswith("avbl_rel")}
            if float(loss.detach()) > 0:
                loss.backward()
                opt.step()
            editor._clear_query_patch_cache()

            if adv:
                n_terms += 1
                step_log["avbl_loss"] = step_log.get("avbl_loss", 0.0) + float(loss.detach())
                for k, v in {**llog_n_all, **llog_p}.items():
                    step_log[k] = step_log.get(k, 0.0) + v

        new_pool = _build_pool(editor, samples, sequential_n)
        pool_sample_ids.clear()
        pool_sample_ids.extend(new_pool)
        if n_terms:
            step_log["avbl_loss"] = step_log["avbl_loss"] / n_terms
            for k in list(step_log.keys()):
                if k not in ("step", "avbl_loss"):
                    step_log[k] = step_log[k] / n_terms
            history.append(step_log)
    editor.set_train(False)
    return history


def run_verify(
    ckpt: str,
    device: str,
    sample_n: int,
    sequential_n: int,
    seed: int,
    steps: int,
    lr: float,
    data_path: str,
    img_root: str,
    patch_ckpt_load: bool = False,
) -> Dict[str, Any]:
    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    if patch_ckpt_load:
        editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, None, False)
        editor.enable_patch_grounding(encoder="dinov2", expert_gen=True)
        editor.load_ckpt(ckpt, True, False)
    else:
        editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    editor.enable_adversarial_grounding(
        lambda_adv=0.2, neg_weight=2.0, hinge_tau=0.3, use_contrastive=True,
        targeted_false_cross=True, rel_anchor_min=0.5,
    )
    editor.patch_routing_alpha = 0.5
    # patch_routing_only is for ablation only; AVBL binding uses for_avbl fusion in loss.

    eval_data = GroundingVLKEB(data_path, img_root, sample_n)
    samples = copy.deepcopy(eval_data.data_with_img)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    samples = samples[:sample_n]

    pool_ids = _build_pool(editor, samples, sequential_n)
    probe_samples = samples[:sequential_n]

    before = measure_metrics(editor, probe_samples, pool_ids)
    train_hist = []
    if steps > 0:
        train_hist = _avbl_train_steps(editor, probe_samples, pool_ids, sequential_n, steps=steps, lr=lr)
    after = measure_metrics(editor, probe_samples, pool_ids) if steps > 0 else before

    swap_mass_drop = (before.get("swap_expert_mass") or 0) - (after.get("swap_expert_mass") or 0)
    hard_mass_drop = (before.get("hard_neg_expert_mass") or 0) - (after.get("hard_neg_expert_mass") or 0)
    swap_fc_drop = (before.get("swap_false_cross") or 0) - (after.get("swap_false_cross") or 0)
    hard_fc_drop = (before.get("hard_neg_false_cross") or 0) - (after.get("hard_neg_false_cross") or 0)
    rel_own_drop = (before.get("reliability_own_prob") or 0) - (after.get("reliability_own_prob") or 0)

    mass_ok = swap_mass_drop >= 0.10 or hard_mass_drop >= 0.10
    fc_ok = swap_fc_drop >= 0.10 or hard_fc_drop >= 0.10
    rel_ok = rel_own_drop <= 0.10
    passed = steps == 0 or ((mass_ok or fc_ok) and rel_ok)

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": ckpt,
            "seed": seed,
            "sample_n": sample_n,
            "sequential_n": sequential_n,
            "avbl_steps": steps,
            "lr": lr,
        },
        "before": before,
        "after": after,
        "deltas": {
            "swap_expert_mass_drop": swap_mass_drop,
            "hard_neg_expert_mass_drop": hard_mass_drop,
            "swap_false_cross_drop_pp": swap_fc_drop * 100,
            "hard_neg_false_cross_drop_pp": hard_fc_drop * 100,
            "reliability_own_prob_drop": rel_own_drop,
        },
        "train_history": train_hist[-5:],
        "pass": passed,
        "pass_reason": (
            "Measure-only (no AVBL steps)"
            if steps == 0 else
            (
                "AVBL reduced adversarial expert mass / false-cross with bounded rel cost"
                if passed else
                "Insufficient AVBL signal — do not start full train yet"
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="AVBL fast direction check")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, required=True)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument("-dsn", "--data_sample_n", type=int, default=15)
    ap.add_argument("-sen", "--sequential_edit_n", type=int, default=5)
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument(
        "-dpath", "--grounding_eval_json", type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", type=str, default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("-out", "--output_json", type=str,
                    default="doc/results/meeting_validation/step6/avbl_direction_check.json")
    ap.add_argument(
        "--patch_grounding",
        action="store_true",
        help="Load AVBT ckpt with patch module weights (enable patch before load_ckpt).",
    )
    args = ap.parse_args()

    result = run_verify(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        sample_n=args.data_sample_n,
        sequential_n=args.sequential_edit_n,
        seed=args.eval_seed,
        steps=args.steps,
        lr=args.lr,
        data_path=args.grounding_eval_json,
        img_root=args.img_root,
        patch_ckpt_load=getattr(args, "patch_grounding", False),
    )

    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== AVBL direction check ===")
    print("Before:", json.dumps(result["before"], indent=2))
    print("After: ", json.dumps(result["after"], indent=2))
    print("Deltas:", json.dumps(result["deltas"], indent=2))
    print("PASS" if result["pass"] else "FAIL", "—", result["pass_reason"])
    print("Wrote", out_path)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
