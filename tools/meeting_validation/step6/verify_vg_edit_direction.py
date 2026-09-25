"""Small-scale VG-Edit direction check before full grounding eval.

Builds a mini sequential pool, probes routing on swap/hard-neg/reliability,
and compares baseline vs decode vision gate. Target: lower false-fire on
adversarial probes without large reliability miss rate.

Usage:
  python tools/meeting_validation/step6/verify_vg_edit_direction.py \\
    -ckpt "records/liveedit/.../epoch-20-..." -dvc cuda:0 -dsn 25 -sen 5 -seed 42

Pass criteria (directional, single seed):
  - hard_negative or image_swap false-fire drops by >= 10 pp vs baseline, OR
  - reliability false-cross drops by >= 10 pp
  - reliability miss increase <= 5 pp
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


def _classify(probe: str, sample_id: int, retrieved: List[int], pool_ids: List[int]) -> Dict[str, bool]:
    expected = {i for i, sid in enumerate(pool_ids) if sid == sample_id}
    retr = set(retrieved)
    out = {"false_fire_own": False, "false_cross": False, "miss": False}
    if probe in ("hard_negative", "image_swap"):
        out["false_fire_own"] = any(
            pool_ids[i] == sample_id for i in retrieved if i < len(pool_ids)
        )
    elif probe == "reliability":
        out["miss"] = len(expected & retr) == 0 and len(expected) > 0
        out["false_cross"] = any(
            pool_ids[i] != sample_id for i in retrieved if i < len(pool_ids)
        )
    return out


def _build_pool(editor, samples: List[Dict[str, Any]], sequential_n: int, margin: float, abs_threshold: float) -> List[int]:
    editor.restore_to_original_model()
    editor.enable_decode_vision_gate(margin=margin, abs_threshold=abs_threshold)
    pool_ids: List[int] = []
    for sid, sample in enumerate(samples):
        req = copy.deepcopy(sample["requests"][0])
        editor.edit_one_piece(req)
        n_new = len(editor.requests_pool) - len(pool_ids)
        pool_ids.extend([sid] * n_new)
        if sid + 1 >= sequential_n:
            break
    return pool_ids


def _run_routing_probes(
    editor,
    samples: List[Dict[str, Any]],
    pool_ids: List[int],
    gate_enabled: bool,
    abs_threshold: float,
) -> Dict[str, Any]:
    prev_gate = bool(getattr(editor, "decode_vision_gate_enabled", False))
    prev_abs = getattr(editor, "decode_vision_abs_threshold", None)
    editor.decode_vision_gate_enabled = gate_enabled
    editor.decode_vision_abs_threshold = abs_threshold if gate_enabled else prev_abs
    try:
        return _run_routing_probes_inner(editor, samples, pool_ids)
    finally:
        editor.decode_vision_gate_enabled = prev_gate
        editor.decode_vision_abs_threshold = prev_abs


def _run_routing_probes_inner(editor, samples: List[Dict[str, Any]], pool_ids: List[int]) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    for sid, sample in enumerate(samples):
        for probe, prompt, image, target in _probe_specs(sample):
            routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
            cls = _classify(probe, sid, routing["retrieved_indices"], pool_ids)
            logs.append({"probe": probe, "sample_id": sid, **routing, **cls})

    def rate(probe: str, key: str) -> Optional[float]:
        vals = [e[key] for e in logs if e["probe"] == probe]
        return float(np.mean(vals)) if vals else None

    return {
        "hard_neg_false_fire": rate("hard_negative", "false_fire_own"),
        "swap_false_fire": rate("image_swap", "false_fire_own"),
        "rel_false_cross": rate("reliability", "false_cross"),
        "rel_miss": rate("reliability", "miss"),
        "mean_n_retrieved_hard_neg": float(
            np.mean([e["n_retrieved"] for e in logs if e["probe"] == "hard_negative"])
        )
        if any(e["probe"] == "hard_negative" for e in logs)
        else None,
        "probes": logs,
    }


def _evaluate_config(
    editor,
    samples: List[Dict[str, Any]],
    pool_ids: List[int],
    gate: bool,
    margin: float,
    abs_threshold: float,
) -> Dict[str, Any]:
    summary = _run_routing_probes(editor, samples, pool_ids, gate_enabled=gate, abs_threshold=abs_threshold)
    summary["decode_vision_gate"] = gate
    summary["decode_vision_margin"] = margin
    summary["decode_vision_abs_threshold"] = abs_threshold
    return summary


def _passes_direction_check(base: Dict[str, Any], gated: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    adv_drop = 0.0
    for key in ("hard_neg_false_fire", "swap_false_fire"):
        b, g = base.get(key), gated.get(key)
        if b is not None and g is not None:
            adv_drop = max(adv_drop, (b - g) * 100)
    rel_cross_drop = 0.0
    if base.get("rel_false_cross") is not None and gated.get("rel_false_cross") is not None:
        rel_cross_drop = (base["rel_false_cross"] - gated["rel_false_cross"]) * 100
    miss_increase = 0.0
    if base.get("rel_miss") is not None and gated.get("rel_miss") is not None:
        miss_increase = (gated["rel_miss"] - base["rel_miss"]) * 100

    ok_adv = adv_drop >= 10.0
    ok_cross = rel_cross_drop >= 10.0
    ok_miss = miss_increase <= 5.0
    if ok_adv:
        reasons.append("adversarial false-fire dropped >= 10 pp")
    if ok_cross:
        reasons.append("reliability false-cross dropped >= 10 pp")
    if not ok_miss:
        reasons.append("reliability miss increased > 5 pp (FAIL)")
    passed = (ok_adv or ok_cross) and ok_miss
    return passed, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="VG-Edit small-scale direction verification")
    parser.add_argument("-ckpt", required=True)
    parser.add_argument("-dvc", default="cuda:0")
    parser.add_argument("-dsn", type=int, default=25)
    parser.add_argument("-sen", type=int, default=5, help="Sequential edits before probe")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--abs-threshold", type=float, default=0.45, dest="abs_threshold")
    parser.add_argument(
        "-dpath",
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    parser.add_argument("--img_root", default="data/VLKEB/VLKEB_images/mmkb_images")
    parser.add_argument("-out", default=None)
    args = parser.parse_args()

    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    data_path = args.dpath
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    img_root = args.img_root
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    data = GroundingVLKEB(data_path, img_root, args.dsn)
    rows = data.data_with_img
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    samples = [rows[int(i)] for i in order[: args.dsn]]

    print("Loading editor + checkpoint...")
    editor = load_vllm_editor(
        "liveedit", get_full_model_name("blip2"), args.dvc, None, args.ckpt, False
    )

    print("Building mini pool (sen=%d, dsn=%d)..." % (args.sen, args.dsn))
    pool_ids = _build_pool(editor, samples, args.sen, args.margin, args.abs_threshold)
    print("Pool size:", len(pool_ids))

    probe_samples = samples[: args.sen]
    print("Probing baseline (no gate)...")
    base = _evaluate_config(editor, probe_samples, pool_ids, gate=False, margin=args.margin, abs_threshold=args.abs_threshold)
    print("Probing VG-Edit v2 (decode gate margin=%.3f abs=%.3f)..." % (args.margin, args.abs_threshold))
    gated = _evaluate_config(editor, probe_samples, pool_ids, gate=True, margin=args.margin, abs_threshold=args.abs_threshold)

    passed, reasons = _passes_direction_check(base, gated)
    report = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt,
        "dsn": args.dsn,
        "sequential_n": args.sen,
        "seed": args.seed,
        "margin": args.margin,
        "abs_threshold": args.abs_threshold,
        "pool_size": len(pool_ids),
        "baseline": {k: v for k, v in base.items() if k != "probes"},
        "vg_edit": {k: v for k, v in gated.items() if k != "probes"},
        "direction_check_passed": passed,
        "direction_check_reasons": reasons,
    }

    print("\n=== VG-Edit direction check ===")
    for label, block in [("Baseline", base), ("VG-Edit", gated)]:
        print(
            "%s: hard_neg_fire=%.1f%% swap_fire=%.1f%% rel_cross=%.1f%% rel_miss=%.1f%%"
            % (
                label,
                (block.get("hard_neg_false_fire") or 0) * 100,
                (block.get("swap_false_fire") or 0) * 100,
                (block.get("rel_false_cross") or 0) * 100,
                (block.get("rel_miss") or 0) * 100,
            )
        )
    print("PASS" if passed else "FAIL", "—", "; ".join(reasons) or "no improvement")

    out = args.out or os.path.join(
        ROOT_PATH,
        "doc/results/meeting_validation/step6/vg_edit_direction_check_%s.json"
        % datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote", out)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
