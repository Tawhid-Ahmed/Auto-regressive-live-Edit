"""Run Step 1 grounding probes on one specific sample (detailed report).

Opt-in patch — does not change batch eval. Use to inspect a single fact
with per-probe accuracy, prediction snippet, and optional MoE routing.

Examples (CMD, from repo root):

  REM List first 20 samples
  python tools/run_grounding_single_sample.py --list 20

  REM Isolated: only this edit in the pool (source_index from JSON)
  python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/.../epoch-20-..." -dvc cuda:0

  REM Lifelike pool: 49 distractor edits before target, then test target
  python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/..." -dvc cuda:0 --pool_before 49 -seed 42

  REM Save JSON report
  python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/..." -dvc cuda:0 -out doc/results/grounding_single_idx0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

PROBE_ORDER = (
    "reliability",
    "text_rephrase",
    "image_rephrase",
    "image_swap",
    "hard_negative",
    "modality_consistency",
    "prompt_conflict",
    "text_loc",
    "image_loc",
)


def _resolve(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT_PATH, path)


def _load_json_rows(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


def _list_samples(data_path: str, n: int) -> None:
    rows = _load_json_rows(data_path)
    print("%-6s %-6s %s" % ("row", "idx", "prompt / alt_short"))
    print("-" * 90)
    for i, r in enumerate(rows[:n]):
        alt_short = r.get("alt_short") or (r.get("alt", "")[:40] + "...")
        print("%-6d %-6s %s" % (i, r.get("source_index", "?"), r.get("src", "")[:50]))
        print("       alt: %s" % alt_short)
        gp = r.get("grounding_probes") or {}
        if gp:
            print("       swap partner row=%s  hard-neg row=%s" % (
                gp.get("image_swap_partner_index"), gp.get("hard_neg_partner_index")))
        print()


def _find_row_index(rows: List[Dict], source_index: int) -> int:
    for i, r in enumerate(rows):
        if r.get("source_index") == source_index:
            return i
    if 0 <= source_index < len(rows):
        return source_index
    raise IndexError("source_index %s not found" % source_index)


def _build_result_template(ed: Dict) -> Dict:
    rd = {
        "reliability": [{"target": ed["requests"][0]["target_new"]}],
        "generality": {k: [{}] for k in ed["generality"]},
        "locality": {k: [{}] for k in ed["locality"]},
    }
    return rd


def _probe_specs(ed: Dict, rd: Dict) -> List[tuple]:
    specs = [("reliability", ed["requests"][0], rd["reliability"][0])]
    for name in ed["generality"]:
        specs.append((name, ed["generality"][name][0], rd["generality"][name][0]))
    for name in ed["locality"]:
        specs.append((name, ed["locality"][name][0], rd["locality"][name][0]))
    return specs


def _short(text: str, n: int = 120) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def run_single(
    ckpt: str,
    device: str,
    data_path: str,
    img_root: str,
    source_index: int,
    pool_before: int,
    seed: int,
    ar_mode: bool,
    chunk_size: int,
    with_routing: bool,
    preview_chars: int,
) -> Dict[str, Any]:
    from dataset.vllm_grounding import GroundingVLKEB
    from evaluation.vllm_editor_eval import VLLMEditorEvaluation
    from utils import get_full_model_name, load_vllm_editor

    rows = _load_json_rows(data_path)
    target_row = _find_row_index(rows, source_index)
    target_raw = rows[target_row]

    if pool_before > 0:
        import numpy as np
        rng = np.random.default_rng(seed)
        candidates = [i for i in range(len(rows)) if i != target_row]
        rng.shuffle(candidates)
        context_rows = candidates[:pool_before]
        row_indices = context_rows + [target_row]
        eval_data = GroundingVLKEB.from_row_indices(data_path, img_root, row_indices)
        target_pos = len(row_indices) - 1
    else:
        eval_data = GroundingVLKEB.from_source_index(data_path, img_root, source_index)
        row_indices = [target_row]
        target_pos = 0

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), device, None, ckpt, False)
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = ar_mode
        editor.chunk_size = chunk_size

    editor.restore_to_original_model()
    samples = eval_data.data_with_img
    samples_path = eval_data.data_with_img_path

    # Locality probes need pre-edit predictions (same as batch eval)
    import torch
    tokenizer = editor.vllm.get_llm_tokenizer()
    for sample, sample_path in zip(samples, samples_path):
        for loc_name in sample["locality"]:
            for edl, rdl in zip(sample["locality"][loc_name], sample_path["locality"][loc_name]):
                (input_embeds, vt_range), label_ids, label_masks = editor.vllm.prompts_imgs_target_to_xym(
                    [edl["prompt"]], [edl["image"]], [edl["target"]])
                logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:]
                edl["before_edit_ids"] = before_edit_ids

    for sample in samples:
        editor.edit_one_piece(sample["requests"][0])

    ed = deepcopy(samples[target_pos])
    rd = _build_result_template(ed)

    ev = VLLMEditorEvaluation(editor, eval_data, "single_sample", "eval_results")
    ev.editor = editor
    rd = VLLMEditorEvaluation.__get_results_after_edit__(ev, editor.vllm, ed, rd)

    probe_results = []
    for name, ed_spec, rd_spec in _probe_specs(ed, rd):
        img = ed_spec.get("image")
        if img is not None and not isinstance(img, str):
            img = getattr(img, "filename", None) or str(type(img).__name__)
        entry = {
            "probe": name,
            "prompt": ed_spec["prompt"],
            "image": img,
            "acc": rd_spec.get("acc"),
            "em": rd_spec.get("em"),
            "target_token_count": rd_spec.get("target_token_count"),
            "predict_preview": _short(rd_spec.get("predict_after_edit", ""), preview_chars),
        }
        if with_routing and name not in ("text_loc",):
            target = ed_spec.get("target") or ed_spec.get("target_new") or "."
            image = ed_spec.get("image")
            if image is not None:
                routing = editor.probe_routing(editor.vllm, ed_spec["prompt"], image, target=target)
                entry["routing"] = {
                    "n_retrieved": routing["n_retrieved"],
                    "retrieved_indices": routing["retrieved_indices"],
                    "vis_sim_prot": routing["vis_sim_prot"],
                    "n_pool": routing["n_pool"],
                }
                entry["routing"]["expected_index"] = target_pos
                entry["routing"]["hits_expected"] = target_pos in routing["retrieved_indices"]
                entry["routing"]["false_cross"] = any(
                    idx != target_pos for idx in routing["retrieved_indices"]
                )
        probe_results.append(entry)

    rel_acc = next(p["acc"] for p in probe_results if p["probe"] == "reliability")
    swap_acc = next((p["acc"] for p in probe_results if p["probe"] == "image_swap"), None)

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_index": target_raw.get("source_index", source_index),
            "row_index": target_row,
            "pool_before": pool_before,
            "pool_size": len(samples),
            "target_pool_index": target_pos,
            "context_row_indices": row_indices[:-1] if pool_before else [],
            "seed": seed if pool_before else None,
            "ckpt": ckpt,
            "ar_mode": ar_mode,
        },
        "sample": {
            "prompt": target_raw.get("src"),
            "alt_short": target_raw.get("alt_short"),
            "pred_prior": target_raw.get("pred"),
            "image": target_raw.get("image"),
            "image_rephrase": target_raw.get("image_rephrase"),
            "grounding_probes": target_raw.get("grounding_probes"),
        },
        "summary": {
            "reliability_acc": rel_acc,
            "image_swap_acc": swap_acc,
            "swap_gap_pp": (rel_acc - swap_acc) * 100 if swap_acc is not None else None,
        },
        "probes": probe_results,
    }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    s = report["sample"]
    gp = s.get("grounding_probes") or {}
    print("=" * 72)
    print("SAMPLE source_index=%s  row=%s  pool_size=%s  target@pool=%s" % (
        report["meta"]["source_index"],
        report["meta"]["row_index"],
        report["meta"]["pool_size"],
        report["meta"]["target_pool_index"],
    ))
    print("Prompt: %s" % s.get("prompt"))
    print("Alt:    %s" % s.get("alt_short"))
    print("Image:  %s" % s.get("image"))
    if gp:
        print("Swap image (partner row %s): %s" % (
            gp.get("image_swap_partner_index"), gp.get("image_swap")))
        print("Hard-neg (partner row %s):   %s" % (
            gp.get("hard_neg_partner_index"), gp.get("hard_neg_image")))
        print("Conflict prompt: %s" % _short(gp.get("conflict_prompt", ""), 100))
    print("-" * 72)
    print("%-22s %8s  %s" % ("PROBE", "ACC", "prediction preview"))
    for p in report["probes"]:
        acc = p.get("acc")
        acc_s = "%.2f%%" % (acc * 100) if acc is not None else "n/a"
        print("%-22s %8s  %s" % (p["probe"], acc_s, p.get("predict_preview", "")))
        r = p.get("routing")
        if r:
            print("                       routing: retrieved=%s expected=%s hits=%s false_cross=%s prot=%.4f" % (
                r["retrieved_indices"], r["expected_index"], r["hits_expected"],
                r["false_cross"], r["vis_sim_prot"]))
    sm = report["summary"]
    if sm.get("swap_gap_pp") is not None:
        print("-" * 72)
        print("Swap-gap (reliability - image_swap): %.2f pp" % sm["swap_gap_pp"])
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 1 grounding probes on one sample")
    ap.add_argument("--list", type=int, default=0, metavar="N", help="List first N samples and exit")
    ap.add_argument("-idx", "--source_index", type=int, default=0,
                    help="source_index in grounding JSON (default 0 = Kankakee example)")
    ap.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=None)
    ap.add_argument("-dvc", "--device", type=str, default="cuda:0")
    ap.add_argument(
        "-dpath", "--grounding_eval_json", type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", type=str, default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("--pool_before", type=int, default=0,
                    help="Edit this many random OTHER samples before target (0=isolated)")
    ap.add_argument("-seed", "--eval_seed", type=int, default=42)
    ap.add_argument("--ar_mode", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--routing", action="store_true", help="Include MoE routing per probe")
    ap.add_argument("--preview", type=int, default=160, help="Chars of prediction to print")
    ap.add_argument("-out", "--output_json", type=str, default=None)
    args = ap.parse_args()

    data_path = _resolve(args.grounding_eval_json)
    img_root = _resolve(args.img_root)

    if args.list:
        _list_samples(data_path, args.list)
        return 0

    if not args.editor_ckpt_path:
        ap.error("-ckpt is required unless --list is used")

    report = run_single(
        ckpt=args.editor_ckpt_path,
        device=args.device,
        data_path=data_path,
        img_root=img_root,
        source_index=args.source_index,
        pool_before=args.pool_before,
        seed=args.eval_seed,
        ar_mode=args.ar_mode,
        chunk_size=args.chunk_size,
        with_routing=args.routing,
        preview_chars=args.preview,
    )
    _print_report(report)

    if args.output_json:
        out_path = _resolve(args.output_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
