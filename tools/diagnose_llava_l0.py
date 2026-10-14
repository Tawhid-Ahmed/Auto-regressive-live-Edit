"""G5 LLaVA routing diagnosis + Step-1 routing health gate.

Modes
-----
1. Default (L0 diagnosis): train-split reliability routing, hidden-state scale,
   query slice. Writes JSON for ``G5_LLAVA_L0_DIAGNOSIS.md``.

2. ``--health_gate`` (Step 1 of ``G5_LLAVA_DEK_ROOT_CAUSE.md``): after S2/DEK,
   probe reliability + image_swap on the train split and fail fast if the
   vision gate is open, swap own-margin is non-negative, or text top-1 dropped
   more than ``--text_top1_min_delta_pp`` vs an L0 baseline JSON.

Usage (lab)::

  # L0 diagnosis (unchanged)
  python -u tools/diagnose_llava_l0.py -ckpt ... -out llava_l0_diagnosis.json

  # Routing health gate after S2/DEK
  python -u tools/diagnose_llava_l0.py --health_gate \\
    -ckpt ... -out routing_health_s2.json \\
    --baseline_json doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH


def _ensure_image(img):
    if img is None:
        return None
    if isinstance(img, str):
        with Image.open(img) as im:
            return im.convert("RGB").copy()
    return img


def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"numel": 0}
    abs_x = x.abs()
    if t.dim() >= 2:
        per_ch = t.detach().float().abs().mean(dim=tuple(range(t.dim() - 1)))
        topk = min(8, int(per_ch.numel()))
        vals, idxs = torch.topk(per_ch, k=topk)
        outliers = [
            {"channel": int(i), "mean_abs": float(v)}
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]
    else:
        outliers = []
    return {
        "shape": list(t.shape),
        "norm_l2": float(torch.linalg.vector_norm(x)),
        "mean_token_l2": float(
            torch.linalg.vector_norm(t.detach().float(), dim=-1).mean()
        ),
        "max_abs": float(abs_x.max()),
        "mean_abs": float(abs_x.mean()),
        "std": float(x.std()),
        "p99_abs": float(torch.quantile(abs_x, 0.99)),
        "outlier_channels": outliers,
    }


def _channel_energy_report(t: torch.Tensor, *, topk: int = 16) -> Dict[str, Any]:
    """Per-channel |·| energy for extractor keys [..., dim]. M10 pre-check."""
    if t is None or t.numel() == 0:
        return {"numel": 0}
    x = t.detach().float()
    per_ch = x.abs().mean(dim=tuple(range(x.dim() - 1)))  # [dim]
    total = float(per_ch.sum().clamp_min(1e-12))
    k = min(int(topk), int(per_ch.numel()))
    vals, idxs = torch.topk(per_ch, k=k)
    top = [
        {
            "channel": int(i),
            "mean_abs": float(v),
            "share_of_l1": float(v) / total,
        }
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]
    shares = [c["share_of_l1"] for c in top]
    return {
        "shape": list(x.shape),
        "dim": int(per_ch.numel()),
        "mean_abs": float(per_ch.mean()),
        "max_abs": float(per_ch.max()),
        "std_abs": float(per_ch.std()),
        "p99_abs": float(torch.quantile(per_ch, 0.99)),
        "top1_share": round(shares[0], 6) if shares else None,
        "top8_share": round(sum(shares[:8]), 6) if len(shares) >= 8 else round(sum(shares), 6),
        "top16_share": round(sum(shares[:16]), 6) if len(shares) >= 16 else round(sum(shares), 6),
        "top_channels": top,
        "dominated": bool(shares and shares[0] >= 0.05),  # heuristic: one ch >=5% of L1
    }


def _accumulate_channel_abs(acc: Optional[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """Running sum of per-channel mean |·| (last dim = channels). Handles variable seq len."""
    x = t.detach().float()
    per_ch = x.abs().mean(dim=tuple(range(x.dim() - 1)))  # [dim]
    return per_ch if acc is None else acc + per_ch


def _report_from_acc(acc: Optional[torch.Tensor], n: int, *, topk: int = 16) -> Dict[str, Any]:
    if acc is None or n <= 0:
        return {"numel": 0}
    return _channel_energy_report((acc / float(n)).unsqueeze(0), topk=topk)


@torch.no_grad()
def _measure_extractor_energy(editor, samples: List[Dict[str, Any]], n_query: int = 16) -> Dict[str, Any]:
    """Channel energy on pool keys (evr/eqr) and query-time ivr / visprot."""
    out: Dict[str, Any] = {
        "evr_pool": _channel_energy_report(getattr(editor, "evr_pool", None)),
        "eqr_pool": _channel_energy_report(getattr(editor, "eqr_pool", None)),
    }
    vllm = editor.vllm
    ivrs: List[torch.Tensor] = []
    iqrs: List[torch.Tensor] = []
    prots: List[torch.Tensor] = []
    # Variable-length hidden states: accumulate per-channel means (no cat)
    hid_vis_acc: Optional[torch.Tensor] = None
    hid_qry_acc: Optional[torch.Tensor] = None
    n = min(int(n_query), len(samples))
    was = editor.is_editing
    editor.is_editing = True
    try:
        for sample in samples[:n]:
            req = sample["requests"][0]
            prompt, image, target = req["prompt"], _ensure_image(req["image"]), req["target_new"]
            (input_embeds, vt_range), _, label_masks = vllm.prompts_imgs_target_to_xym(
                [prompt], [image], [target]
            )
            query_end = input_embeds["inputs_embeds"].shape[1] - label_masks.shape[1] + 1
            mid = vllm.get_mid_module_outpt(input_embeds, vt_range, editor.edit_layer_path)
            reps = mid.float()
            vision_reps = reps[:, vt_range[0] : vt_range[1]]
            query_reps = reps[:, vt_range[1] : query_end]
            ivrs.append(editor.inpt_extractor.extract_vision(query_reps, vision_reps))
            iqrs.append(editor.inpt_extractor.extract_query(query_reps))
            prots.append(editor.inpt_extractor.extract_from_visprot(query_reps))
            hid_vis_acc = _accumulate_channel_abs(hid_vis_acc, vision_reps)
            hid_qry_acc = _accumulate_channel_abs(hid_qry_acc, query_reps)
    finally:
        editor.is_editing = was
    if ivrs:
        out["ivr_query"] = _channel_energy_report(torch.cat(ivrs, 0))
        out["iqr_query"] = _channel_energy_report(torch.cat(iqrs, 0))
        out["ivr_visprot"] = _channel_energy_report(torch.cat(prots, 0))
        out["hidden_vision_layer"] = _report_from_acc(hid_vis_acc, n)
        out["hidden_query_layer"] = _report_from_acc(hid_qry_acc, n)
    out["n_query_samples"] = n
    return out


def _decode_query_slice(vllm, prompt: str, image, target: str) -> Dict[str, Any]:
    (input_embeds, vt_range), _, label_masks = vllm.prompts_imgs_target_to_xym(
        [prompt], [image], [target]
    )
    query_end = input_embeds["inputs_embeds"].shape[1] - label_masks.shape[1] + 1
    vt0, vt1 = int(vt_range[0]), int(vt_range[1])
    q_len = int(query_end - vt1)
    tok = vllm.get_llm_tokenizer()
    decoded = None
    q_ids: Optional[List[int]] = None
    try:
        proc = getattr(vllm, "processor", None)
        ist = vllm.get_img_special_token_str()
        tgt = (
            (" " + target)
            if prompt[-1] not in [" ", "\n"] and target[0] not in [" ", "\n"]
            else target
        )
        inpt_str = (ist + "\n" + prompt if ist and prompt.find(ist) == -1 else prompt) + tgt
        if proc is not None:
            inpt = proc(text=[inpt_str], images=[image], return_tensors="pt", padding=True)
            input_ids = inpt["input_ids"][0].tolist()
            q_ids = input_ids[vt1:query_end]
            decoded = tok.decode(q_ids, skip_special_tokens=False)
            tokens = tok.convert_ids_to_tokens(q_ids)
        else:
            tokens = None
    except Exception as e:
        decoded = f"<decode_failed: {e}>"
        tokens = None
    return {
        "prompt": prompt,
        "vt_range": [vt0, vt1],
        "query_end": int(query_end),
        "vision_token_n": vt1 - vt0,
        "query_token_n": q_len,
        "label_token_n": int(label_masks.shape[1]),
        "decoded_query_slice": decoded,
        "query_tokens": tokens,
        "query_token_ids_head": (q_ids[:24] if q_ids else None),
    }


@torch.no_grad()
def _probe_one(
    editor,
    prompt,
    image,
    target,
    own_idx: int,
    *,
    with_hidden: bool = False,
) -> Dict[str, Any]:
    image = _ensure_image(image)
    vllm = editor.vllm
    routing = editor.probe_routing(vllm, prompt, image, target=target)
    txt = routing.get("txt_sim") or []
    text_top1 = int(np.argmax(txt)) if txt else None
    own_text_rank = None
    if txt and 0 <= own_idx < len(txt):
        own_text_rank = int(sum(1 for s in txt if s > txt[own_idx])) + 1
    vis = routing.get("vis_sim") or []
    own_vis_rank = None
    own_vis_margin = None
    n_pool = int(routing.get("n_pool") or len(vis) or 0)
    if vis and 0 <= own_idx < len(vis):
        own_vis_rank = int(sum(1 for s in vis if s > vis[own_idx])) + 1
        prot = routing.get("vis_sim_prot")
        if prot is not None:
            own_vis_margin = float(vis[own_idx]) - float(prot)
    vision_ok = list(routing.get("vision_ok_indices") or [])
    text_ok = set(routing.get("text_ok_indices") or [])
    retrieved = set(routing.get("retrieved_indices") or [])
    n_vision_ok = len(vision_ok)
    out: Dict[str, Any] = {
        "routing": {
            "n_retrieved": routing.get("n_retrieved"),
            "n_pool": n_pool,
            "n_vision_ok": n_vision_ok,
            "vision_ok_pct_of_pool": (
                round(100.0 * n_vision_ok / n_pool, 2) if n_pool else None
            ),
            "own_retrieved": own_idx in retrieved,
            "own_in_vision_ok": own_idx in set(vision_ok),
            "own_in_text_ok": own_idx in text_ok,
            "text_top1": text_top1,
            "text_top1_is_own": text_top1 == own_idx,
            "own_text_rank": own_text_rank,
            "own_vis_rank": own_vis_rank,
            "own_vis_margin": own_vis_margin,
            "vis_sim_prot": (
                float(routing["vis_sim_prot"])
                if routing.get("vis_sim_prot") is not None
                else None
            ),
            "own_txt_sim": float(txt[own_idx]) if txt and own_idx < len(txt) else None,
            "max_txt_sim": float(max(txt)) if txt else None,
            "max_fuse_coe": float(max(routing["fuse_coe"])) if routing.get("fuse_coe") else None,
            "ivr_eqe_std": routing.get("ivr_eqe_std"),
            "query_token_n": routing.get("query_token_n"),
            "vis_token_n": routing.get("vis_token_n"),
            "routing_rule": routing.get("routing_rule"),
        },
    }
    if with_hidden:
        (input_embeds, vt_range), _, label_masks = vllm.prompts_imgs_target_to_xym(
            [prompt], [image], [target]
        )
        query_end = input_embeds["inputs_embeds"].shape[1] - label_masks.shape[1] + 1
        was = editor.is_editing
        editor.is_editing = True
        try:
            mid = vllm.get_mid_module_outpt(input_embeds, vt_range, editor.edit_layer_path)
        finally:
            editor.is_editing = was
        reps = mid.float()
        vision_reps = reps[:, vt_range[0] : vt_range[1]]
        query_reps = reps[:, vt_range[1] : query_end]
        out["hidden"] = {
            "edit_layer_path": str(editor.edit_layer_path),
            "query_reps": _tensor_stats(query_reps),
            "vision_reps": _tensor_stats(vision_reps),
        }
        out["query_slice"] = _decode_query_slice(vllm, prompt, image, target)
    return out


def _swap_probe(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gen = sample.get("generality") or {}
    swaps = gen.get("image_swap") or []
    return swaps[0] if swaps else None


def _agg_rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    def rate(key: str) -> Optional[float]:
        xs = [r[key] for r in rows if isinstance(r.get(key), bool)]
        return round(100.0 * sum(xs) / len(xs), 2) if xs else None

    def mean(key: str) -> Optional[float]:
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(float(np.mean(xs)), 6) if xs else None

    return {
        "n": len(rows),
        "own_retrieved_pct": rate("own_retrieved"),
        "own_in_vision_ok_pct": rate("own_in_vision_ok"),
        "own_in_text_ok_pct": rate("own_in_text_ok"),
        "text_top1_is_own_pct": rate("text_top1_is_own"),
        "mean_own_text_rank": mean("own_text_rank"),
        "mean_own_vis_rank": mean("own_vis_rank"),
        "mean_own_txt_sim": mean("own_txt_sim"),
        "mean_max_fuse_coe": mean("max_fuse_coe"),
        "mean_ivr_eqe_std": mean("ivr_eqe_std"),
        "mean_query_token_n": mean("query_token_n"),
        "mean_vis_token_n": mean("vis_token_n"),
        "mean_n_retrieved": mean("n_retrieved"),
        "mean_n_vision_ok": mean("n_vision_ok"),
        "mean_vision_ok_pct_of_pool": mean("vision_ok_pct_of_pool"),
        "mean_own_vis_margin": mean("own_vis_margin"),
        "own_vis_margin_neg_pct": (
            round(
                100.0
                * sum(1 for r in rows if r.get("own_vis_margin") is not None and r["own_vis_margin"] < 0)
                / max(1, sum(1 for r in rows if r.get("own_vis_margin") is not None)),
                2,
            )
            if any(r.get("own_vis_margin") is not None for r in rows)
            else None
        ),
    }


def _evaluate_health_gate(
    health: Dict[str, Any],
    *,
    max_vision_ok_pct: float,
    require_swap_margin_neg: bool,
    baseline_text_top1: Optional[float],
    text_top1_min_delta_pp: float,
) -> Tuple[bool, List[str], List[str]]:
    """Return (passed, failures, notes)."""
    fails: List[str] = []
    notes: List[str] = []
    rel = health.get("reliability") or {}
    swap = health.get("image_swap") or {}

    vok = rel.get("mean_vision_ok_pct_of_pool")
    if vok is None:
        fails.append("missing reliability.mean_vision_ok_pct_of_pool")
    elif float(vok) > max_vision_ok_pct:
        fails.append(
            "vision gate open on reliability: mean_vision_ok_pct_of_pool="
            f"{float(vok):.1f} > {max_vision_ok_pct:g} (plan: <=20/100 pool)"
        )
    else:
        notes.append(f"vision gate OK: mean_vision_ok_pct_of_pool={float(vok):.1f}")

    margin = swap.get("mean_own_vis_margin")
    if require_swap_margin_neg:
        if margin is None:
            fails.append("missing image_swap.mean_own_vis_margin (no swap probes?)")
        elif float(margin) >= 0:
            fails.append(
                "own-vs-prototype margin on image_swap is non-negative: "
                f"mean_own_vis_margin={float(margin):.3f} (need < 0)"
            )
        else:
            notes.append(f"swap margin OK: mean_own_vis_margin={float(margin):.3f}")

    text_top1 = rel.get("text_top1_is_own_pct")
    if baseline_text_top1 is not None and text_top1 is not None:
        floor = float(baseline_text_top1) + float(text_top1_min_delta_pp)
        if float(text_top1) < floor:
            fails.append(
                "text top-1 dropped vs L0 baseline: "
                f"{float(text_top1):.1f}% < floor {floor:.1f}% "
                f"(baseline {float(baseline_text_top1):.1f}% + delta {text_top1_min_delta_pp:g} pp)"
            )
        else:
            notes.append(
                f"text top-1 OK: {float(text_top1):.1f}% >= floor {floor:.1f}% "
                f"(baseline {float(baseline_text_top1):.1f}%)"
            )
    elif baseline_text_top1 is None:
        notes.append("no --baseline_json: skipped text-top-1 vs L0 check")

    stab = health.get("text_argmax_stable_under_swap_pct")
    if stab is not None:
        notes.append(f"text-argmax stability under image_swap: {float(stab):.1f}% (info only)")

    return (len(fails) == 0), fails, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", required=True)
    ap.add_argument(
        "-dpath",
        default="data/VLKEB/grounding/grounding_train_short.json",
        help="Train-split grounding JSON (same file used for L0 train)",
    )
    ap.add_argument("-img", default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("-dsn", type=int, default=50)
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-enp", default="llava_l0_diag_train50")
    ap.add_argument(
        "-out",
        default="doc/results/cvpr2027/g5/llava_l0_diagnosis.json",
    )
    ap.add_argument(
        "--n_detail",
        type=int,
        default=5,
        help="How many reliability probes get full hidden/slice dumps (L0 mode)",
    )
    # Step-1 health gate
    ap.add_argument(
        "--health_gate",
        action="store_true",
        help="Run reliability+image_swap health checks; exit non-zero on fail",
    )
    ap.add_argument(
        "--baseline_json",
        default=None,
        help="L0 diagnosis JSON; used for text_top1_is_own_pct floor under --health_gate",
    )
    ap.add_argument(
        "--max_vision_ok_pct",
        type=float,
        default=20.0,
        help="Fail if mean vision-ok %% of pool on reliability exceeds this (default 20)",
    )
    ap.add_argument(
        "--no_require_swap_margin_neg",
        action="store_true",
        help="Do not require mean own_vis_margin < 0 on image_swap",
    )
    ap.add_argument(
        "--text_top1_min_delta_pp",
        type=float,
        default=-5.0,
        help="Allowed drop in text top-1 vs baseline, in percentage points (default -5)",
    )
    ap.add_argument(
        "--dek_dual_gate",
        action="store_true",
        help="Enable DEK dual-gate routing flags on the loaded editor (match DEK eval)",
    )
    ap.add_argument(
        "--extractor_energy",
        action="store_true",
        help="After building the pool, measure per-channel L1 energy on evr/eqr/ivr "
        "(M10 pre-check). Also records layer-21 hidden channel energy.",
    )
    ap.add_argument(
        "--extractor_energy_n",
        type=int,
        default=16,
        help="Query samples for ivr/hidden channel energy (default 16)",
    )
    args = ap.parse_args()

    from dataset.vllm_grounding import GroundingVLKEB
    from utils import load_vllm_editor

    dpath = args.dpath if os.path.isabs(args.dpath) else os.path.join(ROOT_PATH, args.dpath)
    img = args.img if os.path.isabs(args.img) else os.path.join(ROOT_PATH, args.img)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT_PATH, args.ckpt)

    print("[diag] loading editor", flush=True)
    editor = load_vllm_editor("liveedit", "llava-v1.5-7b", args.dvc, None, ckpt, False)
    # Flag-only: do NOT call enable_dek() after load_ckpt — it may rebuild patch
    # modules and wipe trained weights. Mirror DEK eval routing flags only.
    if args.dek_dual_gate:
        editor.dek_dual_gate_enabled = True
        if getattr(editor, "routing_rule", "liveedit") in ("liveedit", "vision_only"):
            editor.routing_rule = "and"
    sim_scale = float(editor.sim_scale)
    module_dim = int(editor.cfg.retrieval_editor.module_dim)
    edit_layer_i = int(editor.cfg.edit_layer_i)

    print("[diag] loading train split dsn=%d seed=%d" % (args.dsn, args.seed), flush=True)
    data = GroundingVLKEB(dpath, img, args.dsn, sample_seed=args.seed)
    samples = deepcopy(data.data_with_img)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(samples)

    print("[diag] editing pool n=%d" % len(samples), flush=True)
    editor.restore_to_original_model()
    for sid, sample in enumerate(samples):
        with torch.no_grad():
            req = sample["requests"][0]
            req = {**req, "image": _ensure_image(req["image"])}
            editor.edit_one_piece(req)
        if (sid + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    rel_rows: List[Dict[str, Any]] = []
    swap_rows: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    n_text_stable = 0
    n_swap_paired = 0

    for sid, sample in enumerate(samples):
        req = sample["requests"][0]
        rel = _probe_one(
            editor,
            req["prompt"],
            req["image"],
            req["target_new"],
            sid,
            with_hidden=(not args.health_gate) and len(detail) < args.n_detail,
        )
        r = rel["routing"]
        rel_rows.append({"sample_id": sid, "probe": "reliability", **r})
        if "hidden" in rel:
            detail.append({"sample_id": sid, **rel})

        sp = _swap_probe(sample)
        if sp is not None:
            swap = _probe_one(
                editor,
                sp.get("prompt") or req["prompt"],
                sp["image"],
                sp.get("target") or req["target_new"],
                sid,
                with_hidden=False,
            )
            sr = swap["routing"]
            swap_rows.append({"sample_id": sid, "probe": "image_swap", **sr})
            if r.get("text_top1") is not None and sr.get("text_top1") is not None:
                n_swap_paired += 1
                if r["text_top1"] == sr["text_top1"]:
                    n_text_stable += 1

        if (sid + 1) % 10 == 0:
            print("[diag] probed %d/%d" % (sid + 1, len(samples)), flush=True)

    reliability_summary = _agg_rates(rel_rows)
    # Keep L0-diagnosis key names for backcompat
    train_split_reliability = {
        "n": reliability_summary["n"],
        "own_retrieved_pct": reliability_summary["own_retrieved_pct"],
        "own_in_vision_ok_pct": reliability_summary["own_in_vision_ok_pct"],
        "own_in_text_ok_pct": reliability_summary["own_in_text_ok_pct"],
        "text_top1_is_own_pct": reliability_summary["text_top1_is_own_pct"],
        "mean_own_text_rank": reliability_summary["mean_own_text_rank"],
        "mean_own_vis_rank": reliability_summary["mean_own_vis_rank"],
        "mean_own_txt_sim": reliability_summary["mean_own_txt_sim"],
        "mean_max_fuse_coe": reliability_summary["mean_max_fuse_coe"],
        "mean_ivr_eqe_std": reliability_summary["mean_ivr_eqe_std"],
        "mean_query_token_n": reliability_summary["mean_query_token_n"],
        "mean_vis_token_n": reliability_summary["mean_vis_token_n"],
        "mean_n_retrieved": reliability_summary["mean_n_retrieved"],
        "mean_n_vision_ok": reliability_summary["mean_n_vision_ok"],
        "mean_vision_ok_pct_of_pool": reliability_summary["mean_vision_ok_pct_of_pool"],
        "mean_own_vis_margin": reliability_summary["mean_own_vis_margin"],
    }

    health = {
        "reliability": reliability_summary,
        "image_swap": _agg_rates(swap_rows),
        "text_argmax_stable_under_swap_pct": (
            round(100.0 * n_text_stable / n_swap_paired, 2) if n_swap_paired else None
        ),
        "n_swap_paired": n_swap_paired,
    }

    baseline_text_top1 = None
    if args.baseline_json:
        bpath = (
            args.baseline_json
            if os.path.isabs(args.baseline_json)
            else os.path.join(ROOT_PATH, args.baseline_json)
        )
        with open(bpath, encoding="utf-8") as f:
            base = json.load(f)
        baseline_text_top1 = (base.get("train_split_reliability") or {}).get(
            "text_top1_is_own_pct"
        )
        if baseline_text_top1 is None:
            print("[diag] WARNING: baseline missing text_top1_is_own_pct", flush=True)

    gate_passed = True
    gate_fails: List[str] = []
    gate_notes: List[str] = []
    if args.health_gate:
        gate_passed, gate_fails, gate_notes = _evaluate_health_gate(
            health,
            max_vision_ok_pct=args.max_vision_ok_pct,
            require_swap_margin_neg=not args.no_require_swap_margin_neg,
            baseline_text_top1=baseline_text_top1,
            text_top1_min_delta_pp=args.text_top1_min_delta_pp,
        )

    extractor_energy = None
    if args.extractor_energy:
        print("[diag] measuring extractor channel energy…", flush=True)
        try:
            extractor_energy = _measure_extractor_energy(
                editor, samples, n_query=args.extractor_energy_n
            )
            for key in ("evr_pool", "ivr_query", "ivr_visprot", "hidden_vision_layer"):
                rep = (extractor_energy or {}).get(key) or {}
                if rep.get("top1_share") is not None:
                    print(
                        f"[energy] {key}: top1_share={rep['top1_share']:.4f} "
                        f"top8_share={rep.get('top8_share')} dominated={rep.get('dominated')} "
                        f"top_ch={[(c['channel'], round(c['share_of_l1'], 4)) for c in (rep.get('top_channels') or [])[:4]]}",
                        flush=True,
                    )
        except Exception as e:
            extractor_energy = {"error": repr(e)}
            print("[diag] WARNING: extractor_energy failed:", e, flush=True)

    mode = "l0_diagnosis"
    if args.health_gate and args.extractor_energy:
        mode = "health_gate+extractor_energy"
    elif args.health_gate:
        mode = "health_gate"
    elif args.extractor_energy:
        mode = "extractor_energy"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "ckpt": ckpt,
        "dpath": dpath,
        "dsn": args.dsn,
        "seed": args.seed,
        "edit_layer_i": edit_layer_i,
        "module_dim": module_dim,
        "sim_scale": sim_scale,
        "sim_scale_formula": "1/sqrt(module_dim)",
        "dek_dual_gate": bool(args.dek_dual_gate),
        "train_split_reliability": train_split_reliability,
        "routing_health": health,
        "health_gate": {
            "enabled": bool(args.health_gate),
            "passed": gate_passed if args.health_gate else None,
            "max_vision_ok_pct": args.max_vision_ok_pct,
            "require_swap_margin_neg": not args.no_require_swap_margin_neg,
            "baseline_json": args.baseline_json,
            "baseline_text_top1_is_own_pct": baseline_text_top1,
            "text_top1_min_delta_pp": args.text_top1_min_delta_pp,
            "failures": gate_fails,
            "notes": gate_notes,
        },
        "extractor_energy": extractor_energy,
        "detail_probes": detail,
        "all_reliability_rows": rel_rows,
        "all_image_swap_rows": swap_rows if args.health_gate else [],
    }
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT_PATH, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(train_split_reliability, indent=2))
    if args.health_gate:
        print("[health] image_swap:", json.dumps(health["image_swap"], indent=2))
        print(
            "[health] text_argmax_stable_under_swap_pct:",
            health["text_argmax_stable_under_swap_pct"],
        )
        for n in gate_notes:
            print("[health] NOTE:", n, flush=True)
        for fail in gate_fails:
            print("[health] FAIL:", fail, flush=True)
        print(
            "[health] RESULT:",
            "PASS" if gate_passed else "FAIL",
            flush=True,
        )
    print("[diag] wrote", out_path, flush=True)
    if args.health_gate and not gate_passed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
