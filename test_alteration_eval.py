"""Step 3 patch: evaluate LiveEdit when test images are altered vs edit-time image.

Probes: low-res, rotate 90°, mirror, center crop, mild blur/brightness.
Edit always uses the original image; generality uses on-the-fly PIL transforms.

Example:
  python -W ignore test_alteration_eval.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 `
    -dsn 500 -seed 42 -ckpt "records/liveedit/.../epoch-20-..." -enp step3_alteration_baseline
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
    parser = argparse.ArgumentParser(description="Step 3 image alteration probe eval (opt-in patch)")
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
        help="Source eval JSON (same rows as Step 1 grounding build)",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="data/VLKEB/VLKEB_images/mmkb_images",
    )
    args = parser.parse_args()
    if isinstance(args.editor_ckpt_path, str) and args.editor_ckpt_path.strip().lower() in {"none", "null", ""}:
        args.editor_ckpt_path = None
    return args


if __name__ == "__main__":
    cfg = get_attr()
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)

    _parts = ["VLKEB_ALTERATION"]
    if cfg.ar_mode:
        _parts.append("ar")
    if cfg.eval_name_postfix:
        _parts.append(cfg.eval_name_postfix)
    _parts.append(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    cfg.evaluation_name = "-".join(_parts)
    print(cfg)

    editor = load_vllm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, None, cfg.editor_ckpt_path, False)
    if hasattr(editor, "ar_mode"):
        editor.ar_mode = getattr(cfg, "ar_mode", False)
        editor.chunk_size = getattr(cfg, "chunk_size", 16)
        editor.max_chunks = getattr(cfg, "max_chunks", None)

    from dataset.vllm_alteration import AlterationVLKEB

    data_path = cfg.grounding_eval_json
    if data_path and not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    img_root_dir = cfg.img_root
    if img_root_dir and not os.path.isabs(img_root_dir):
        img_root_dir = os.path.join(ROOT_PATH, img_root_dir)

    eval_data = AlterationVLKEB(data_path, img_root_dir, cfg.data_sample_n)

    use_random = getattr(cfg, "eval_seed", None) is not None
    eval_seed = getattr(cfg, "eval_seed", None)
    eval_results_root = "eval_results/meeting_validation/step3"
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
        data["meeting_step"] = 3
        data["probe_type"] = "image_alteration"
        with open(mean_results_path, "w") as f:
            json.dump(data, f, indent=4)
    print("[Timer] Total elapsed: %.2f s (%.2f min)" % (elapsed, elapsed / 60))
