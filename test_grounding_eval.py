"""Meeting-validation grounding probe eval (Steps 1–2 patch).

Opt-in runner — does not modify ``test_vllm_edit.py``. Uses ``GroundingVLKEB``
and ``data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`` by default.

Examples:
  # Step 1 baseline
  python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 `
    -dsn 500 -seed 42 -ckpt "records/liveedit/.../epoch-20-..." -enp step1_grounding_baseline

  # Step 2 resolution sweep (224 default; try 336 / 448)
  python -W ignore test_grounding_eval.py ... -enp step2_res336 --meeting_step 2 --image_size 336
"""
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
from datetime import datetime
import os
import argparse
import time
import json


def get_attr():
    parser = argparse.ArgumentParser(description="Step 1 grounding probe eval (opt-in patch)")
    parser.add_argument("-en", "--editor_name", type=str, required=True)
    parser.add_argument("-mn", "--edit_model_name", type=str, required=True)
    parser.add_argument("-sen", "--sequential_edit_n", type=int, required=True)
    parser.add_argument("-enp", "--eval_name_postfix", type=str, default="")
    parser.add_argument("-dvc", "--device", type=str, required=True)
    parser.add_argument("-ckpt", "--editor_ckpt_path", type=str, default=None)
    parser.add_argument("-dsn", "--data_sample_n", type=int, default=None)
    parser.add_argument("-seed", "--eval_seed", type=int, default=None)
    parser.add_argument("--ar_mode", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument(
        "-dpath",
        "--grounding_eval_json",
        type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
        help="Grounding probe JSON from tools/build_grounding_eval_json.py",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="data/VLKEB/VLKEB_images/mmkb_images",
    )
    parser.add_argument(
        "--meeting_step",
        type=int,
        default=1,
        choices=[1, 2, 6],
        help="Isolated eval_results subfolder (step1=grounding, step2=resolution, step6=ablation)",
    )
    parser.add_argument(
        "--ablation_layer",
        type=str,
        default=None,
        help="Step 6 layer id (L0..L5, VG); applies ablation flags after ckpt load",
    )
    parser.add_argument(
        "--decode_vision_gate",
        action="store_true",
        help="VG-Edit: enable decode-time DINO competitive-margin gate (overrides L0)",
    )
    parser.add_argument(
        "--decode_vision_margin",
        type=float,
        default=0.0,
        help="VG-Edit margin tau: sim(own) >= sim(best_other) + tau (default 0)",
    )
    parser.add_argument(
        "--decode_vision_abs_threshold",
        type=float,
        default=0.45,
        help="VG-Edit v2: min DINO cosine sim(query_img, expert_img) to inject (default 0.45)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Eval-only BLIP2 processor resize (default: model native 224). Step 2: 224/336/448.",
    )
    parser.add_argument(
        "--patch_routing_only",
        action="store_true",
        help="Use patch-only routing at inference (usually leave off; AVBL trains via for_avbl fusion)",
    )
    parser.add_argument(
        "--patch_grounding",
        action="store_true",
        help="Enable patch grounding module before ckpt load (AVBL / step6 ckpts)",
    )
    parser.add_argument(
        "--fused_expert_key",
        action="store_true",
        help="LEGACY DEK: QFormer+patch embed fuse (collapses rel — prefer --dek_dual_gate)",
    )
    parser.add_argument(
        "--dek_dual_gate",
        action="store_true",
        help="DEK: retrieve only if visual gate AND prompt-key agreement",
    )
    parser.add_argument(
        "--dek_text_margin",
        type=float,
        default=0.0,
        help="DEK dual-gate: allow txt_sim within margin of global max (0=argmax only)",
    )
    parser.add_argument(
        "--pool_dedup",
        action="store_true",
        help="DEK: merge/skip near-duplicate pool entries (DINO sim >= threshold)",
    )
    parser.add_argument(
        "--pool_dedup_threshold",
        type=float,
        default=0.95,
        help="DEK pool dedup cosine threshold (default 0.95)",
    )
    parser.add_argument(
        "--pool_dedup_mode",
        type=str,
        default="merge",
        choices=["merge", "skip"],
        help="DEK pool hygiene: merge overwrite vs skip new edit",
    )
    parser.add_argument(
        "--log_routing",
        action="store_true",
        help="After token eval, log false_cross / false_activation routing rates into mean_results.json",
    )
    args = parser.parse_args()
    if isinstance(args.editor_ckpt_path, str) and args.editor_ckpt_path.strip().lower() in {"none", "null", ""}:
        args.editor_ckpt_path = None
    return args


if __name__ == "__main__":
    cfg = get_attr()
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)

    _parts = ["VLKEB_GROUNDING"]
    if cfg.ar_mode:
        _parts.append("ar")
    if cfg.eval_name_postfix:
        _parts.append(cfg.eval_name_postfix)
    _parts.append(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    cfg.evaluation_name = "-".join(_parts)
    print(cfg)

    ablation_layer = getattr(cfg, "ablation_layer", None)
    needs_patch_modules = (
        (ablation_layer and ablation_layer.upper() not in ("L0", "BASELINE"))
        or getattr(cfg, "patch_grounding", False)
    )
    ckpt = cfg.editor_ckpt_path
    editor = load_vllm_editor(
        cfg.editor_name, cfg.edit_model_name, cfg.device, None,
        None if needs_patch_modules else ckpt,
        False,
    )
    if getattr(cfg, "patch_grounding", False):
        editor.enable_patch_grounding(encoder="dinov2", expert_gen=True)
        if getattr(cfg, "patch_routing_only", False):
            editor.patch_routing_only = True
    if (
        getattr(cfg, "fused_expert_key", False)
        or getattr(cfg, "pool_dedup", False)
        or getattr(cfg, "dek_dual_gate", False)
    ):
        editor.enable_dek(
            fused_expert_key=getattr(cfg, "fused_expert_key", False),
            pool_dedup=getattr(cfg, "pool_dedup", False),
            pool_dedup_threshold=getattr(cfg, "pool_dedup_threshold", 0.95),
            dual_gate=getattr(cfg, "dek_dual_gate", False),
            text_margin=getattr(cfg, "dek_text_margin", 0.0),
            pool_dedup_mode=getattr(cfg, "pool_dedup_mode", "merge"),
        )
    if ablation_layer:
        from tools.meeting_validation.step6.layer_ablation import AblationConfig, apply_ablation_config
        apply_ablation_config(editor, AblationConfig.for_layer(ablation_layer, chunk_size=getattr(cfg, "chunk_size", 16)))
    elif getattr(cfg, "decode_vision_gate", False):
        editor.enable_decode_vision_gate(
            margin=getattr(cfg, "decode_vision_margin", 0.0),
            abs_threshold=getattr(cfg, "decode_vision_abs_threshold", 0.45),
        )
    if needs_patch_modules and ckpt:
        editor.load_ckpt(ckpt, True, False)
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = getattr(cfg, "ar_mode", False)
        editor.chunk_size = getattr(cfg, "chunk_size", 16)
        editor.max_chunks = getattr(cfg, "max_chunks", None)
    image_size = getattr(cfg, "image_size", None)
    if image_size is not None and hasattr(editor.vllm, "set_eval_image_size"):
        editor.vllm.set_eval_image_size(image_size)

    from dataset.vllm_grounding import GroundingEVQA, GroundingVLKEB

    data_path = cfg.grounding_eval_json
    if data_path and not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    img_root_dir = cfg.img_root
    if img_root_dir and not os.path.isabs(img_root_dir):
        img_root_dir = os.path.join(ROOT_PATH, img_root_dir)

    use_evqa = (
        "easy-edit-mm" in data_path.replace("\\", "/").lower()
        or "evqa" in os.path.basename(data_path).lower()
    )
    if use_evqa:
        eval_data = GroundingEVQA(
            data_path, img_root_dir, cfg.data_sample_n,
            sample_seed=getattr(cfg, "eval_seed", None),
        )
    else:
        eval_data = GroundingVLKEB(
            data_path, img_root_dir, cfg.data_sample_n,
            sample_seed=getattr(cfg, "eval_seed", None),
        )

    use_random = getattr(cfg, "eval_seed", None) is not None
    eval_seed = getattr(cfg, "eval_seed", None)
    meeting_step = getattr(cfg, "meeting_step", 1)
    eval_results_root = f"eval_results/meeting_validation/step{meeting_step}"
    routing_result = None
    if getattr(cfg, "log_routing", False):
        from copy import deepcopy
        import numpy as np
        from tools.log_grounding_diagnostics import collect_grounding_routing_diagnostics
        routing_samples = deepcopy(eval_data.data_with_img)
        if use_random and eval_seed is not None:
            np.random.default_rng(eval_seed).shuffle(routing_samples)
        routing_result = collect_grounding_routing_diagnostics(
            editor,
            routing_samples,
            meta={
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ckpt": ckpt,
                "data_path": data_path,
                "seed": eval_seed if use_random else 42,
                "device": cfg.device,
                "ar_mode": getattr(cfg, "ar_mode", False),
            },
        )
        editor.restore_to_original_model()
    ev = VLLMEditorEvaluation(
        editor, eval_data, cfg.evaluation_name, eval_results_root,
        ar_chunk_size=getattr(cfg, "chunk_size", 16),
    )
    t0 = time.time()
    ev.evaluate_sequential_edit(cfg.sequential_edit_n, use_random, eval_seed)
    elapsed = time.time() - t0

    mean_results_fname = ("seed_%s_mean_results.json" % eval_seed) if use_random else "mean_results.json"
    mean_results_path = os.path.join(
        ev.result_dir, "sequential_edit_%s" % cfg.sequential_edit_n, mean_results_fname
    )
    if os.path.exists(mean_results_path):
        with open(mean_results_path, "r") as f:
            data = json.load(f)
        data["runtime_seconds"] = round(elapsed, 2)
        data["runtime_minutes"] = round(elapsed / 60, 2)
        data["grounding_eval_json"] = data_path
        data["meeting_step"] = meeting_step
        if image_size is not None:
            data["image_size"] = image_size
        elif hasattr(editor.vllm, "eval_image_size") and editor.vllm.eval_image_size is not None:
            data["image_size"] = editor.vllm.eval_image_size
        if routing_result is not None:
            data["routing_diagnostics"] = routing_result["summary"]
            routing_out = os.path.join(ev.result_dir, "routing_diagnostics.json")
            with open(routing_out, "w", encoding="utf-8") as rf:
                json.dump(routing_result, rf, indent=2)
        with open(mean_results_path, "w") as f:
            json.dump(data, f, indent=4)
    print("[Timer] Total elapsed: %.2f s (%.2f min)" % (elapsed, elapsed / 60))
