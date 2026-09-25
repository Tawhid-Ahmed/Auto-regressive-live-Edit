"""Before vs after edit predictions on grounding probes (teacher-forced decode)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _adjacent_word_repeats(text: str) -> list:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return [(words[i], i) for i in range(len(words) - 1) if words[i] == words[i + 1]]


def _teacher_forced_predict(vllm, prompt, image, target, tokenizer):
    import torch

    (input_embeds, vt_range), label_ids, label_masks = vllm.prompts_imgs_target_to_xym(
        [prompt], [image], [target]
    )
    l2 = label_ids.shape[1]
    logits = vllm.get_llm_outpt(input_embeds, vt_range).logits
    pre_y = torch.softmax(logits, -1).argmax(-1)[:, -l2:]
    if pre_y.shape[1] < l2:
        pre_y = torch.cat(
            [torch.zeros(1, l2 - pre_y.shape[1], dtype=pre_y.dtype, device=pre_y.device), pre_y],
            dim=1,
        )
    pre_y = pre_y.cpu()
    label_ids_c = label_ids.cpu()
    label_masks_c = label_masks.cpu()
    denom = label_masks_c.sum()
    acc = float(((pre_y == label_ids_c) * label_masks_c).sum() / denom) if denom > 0 else 0.0
    pred = tokenizer.decode(pre_y[label_masks_c.to(bool)])
    return acc, pred.strip()


def run(args) -> dict:
    import numpy as np
    import torch

    from dataset.vllm_grounding import GroundingVLKEB
    from utils import get_full_model_name, load_vllm_editor

    data_path = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    img_root = args.img_root if os.path.isabs(args.img_root) else os.path.join(ROOT, args.img_root)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)

    with open(data_path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["data"] if isinstance(payload, dict) else payload
    target_row = next(i for i, r in enumerate(rows) if r.get("source_index") == args.source_index)
    target_raw = rows[target_row]

    if args.pool_before > 0:
        rng = np.random.default_rng(args.seed)
        candidates = [i for i in range(len(rows)) if i != target_row]
        rng.shuffle(candidates)
        row_indices = candidates[: args.pool_before] + [target_row]
        eval_data = GroundingVLKEB.from_row_indices(data_path, img_root, row_indices)
    else:
        eval_data = GroundingVLKEB.from_source_index(data_path, img_root, args.source_index)
        row_indices = [target_row]

    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), args.device, None, ckpt, False)
    editor.restore_to_original_model()
    vllm = editor.vllm
    tokenizer = vllm.get_llm_tokenizer()
    samples = eval_data.data_with_img
    target_pos = len(samples) - 1
    ed = samples[target_pos]

    probes = {
        "reliability": ed["requests"][0],
        "image_swap": ed["generality"]["image_swap"][0],
    }

    def score_probe(name, spec):
        acc, pred = _teacher_forced_predict(
            vllm, spec["prompt"], spec["image"], spec.get("target") or spec.get("target_new"), tokenizer
        )
        return {
            "probe": name,
            "acc": acc,
            "prediction": pred,
            "adjacent_repeats": _adjacent_word_repeats(pred),
            "n_adjacent_repeats": len(_adjacent_word_repeats(pred)),
        }

    before = {name: score_probe(name, spec) for name, spec in probes.items()}

    for sample in samples:
        editor.edit_one_piece(sample["requests"][0])

    after = {name: score_probe(name, spec) for name, spec in probes.items()}

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_index": args.source_index,
            "row_index": target_row,
            "pool_before": args.pool_before,
            "pool_size": len(samples),
            "seed": args.seed if args.pool_before else None,
            "ckpt": ckpt,
        },
        "sample": {
            "prompt": target_raw["src"],
            "alt_short": target_raw.get("alt_short"),
            "dataset_alt_preview": target_raw["alt"][:200] + "...",
            "correct_image": target_raw["image"],
            "swap_image": (target_raw.get("grounding_probes") or {}).get("image_swap"),
        },
        "before_edit": before,
        "after_edit": after,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-idx", "--source_index", type=int, default=428)
    ap.add_argument("-ckpt", required=True)
    ap.add_argument("-dvc", "--device", default="cuda:0")
    ap.add_argument(
        "-dpath", "--data",
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("--img_root", default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument("--pool_before", type=int, default=49)
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("-out", default="doc/results/meeting_validation/step1/tours_sanity_before_after.json")
    args = ap.parse_args()

    report = run(args)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Wrote %s" % out_path)
    for phase in ("before_edit", "after_edit"):
        print("\n=== %s ===" % phase.replace("_", " ").upper())
        for name in ("reliability", "image_swap"):
            p = report[phase][name]
            print("%s acc=%.1f%% adjacent_repeats=%d" % (name, p["acc"] * 100, p["n_adjacent_repeats"]))
            if p["adjacent_repeats"][:8]:
                print("  repeats: %s" % ", ".join("%s@%d" % (w, i) for w, i in p["adjacent_repeats"][:8]))
            print("  pred: %s" % (p["prediction"][:320] + ("..." if len(p["prediction"]) > 320 else "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
