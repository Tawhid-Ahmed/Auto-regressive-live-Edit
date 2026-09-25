"""Step 1: build adversarial / grounding probe dataset from VLKEB-long eval JSON.

Adds per-sample probe metadata consumed by ``dataset.vllm.VLKEB`` as extra
generality keys: image_swap, hard_negative, modality_consistency, prompt_conflict.

Usage (from repo root):
  python tools/build_grounding_eval_json.py \\
    -dpath "../vlkeb_long_ollama/out/eval_longform_ollama.json" \\
    --img_root "data/VLKEB/VLKEB_images/mmkb_images" \\
    -out data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json

  python tools/build_grounding_eval_json.py -dpath data/VLKEB/eval.json -out data/VLKEB/grounding_eval_short.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH


def _entity_id(image_path: str, mode: str = "folder") -> str:
    """Partner exclusion key.

    - ``folder``: VLKEB-style first path component (entity id folder).
    - ``coco``: COCO file id so EVQA rows under ``val2014/`` are not all one entity.
    """
    if mode == "coco":
        base = os.path.basename(image_path.replace("\\", "/"))
        m = re.search(r"(\d{12})", base) or re.search(r"(\d+)", base)
        return m.group(1) if m else base
    return image_path.split("/")[0].split("\\")[0]


def _prompt_skeleton(src: str) -> str:
    """Coarse question-type bucket for pairing similar prompts."""
    s = re.sub(r"\s+", " ", src.lower().strip())
    words = s.split()
    return " ".join(words[:7])


def _pred_display(pred: str) -> str:
    return pred.replace("_", " ").strip()


def _build_conflict_prompt(src: str, pred: str, alt_short: Optional[str]) -> str:
    """Misleading language prior toward ``pred`` while the image supports ``alt``."""
    pred_clean = _pred_display(pred) or "another entity"
    alt_hint = (alt_short or "").strip()
    if alt_hint and alt_hint.lower() not in src.lower():
        return (
            f"(Most viewers would guess {pred_clean}, not {alt_hint}.) {src}"
        )
    return f"(The photograph is often misread as depicting {pred_clean}.) {src}"


def _image_exists(img_root: str, rel_path: str) -> bool:
    return os.path.isfile(os.path.join(img_root, rel_path))


def _rank_partners(
    idx: int,
    rows: List[Dict[str, Any]],
    skeleton_index: Dict[str, List[int]],
    entity_mode: str = "folder",
) -> Tuple[List[int], List[int]]:
    """Return (hard_negative_candidates, swap_candidates) sorted by preference."""
    row = rows[idx]
    my_entity = _entity_id(row["image"], entity_mode)
    my_sk = _prompt_skeleton(row["src"])

    hard: List[Tuple[int, int, int]] = []
    swap: List[Tuple[int, int]] = []
    for j, other in enumerate(rows):
        if j == idx:
            continue
        if _entity_id(other["image"], entity_mode) == my_entity:
            continue
        sk_match = _prompt_skeleton(other["src"]) == my_sk
        same_sk_list = skeleton_index.get(my_sk, [])
        sk_score = 2 if sk_match else (1 if j in same_sk_list else 0)
        if sk_score > 0:
            hard.append((sk_score, abs(j - idx), j))
        swap.append((sk_score, j))

    hard.sort(key=lambda x: (-x[0], x[1]))
    swap.sort(key=lambda x: (-x[0], x[1]))
    hard_ids = [j for _, _, j in hard]
    swap_ids = [j for _, j in swap]
    if not hard_ids:
        hard_ids = [
            j
            for j in range(len(rows))
            if j != idx and _entity_id(rows[j]["image"], entity_mode) != my_entity
        ]
    if not swap_ids:
        swap_ids = hard_ids[:]
    return hard_ids, swap_ids


def build_grounding_rows(
    rows: List[Dict[str, Any]],
    img_root: str,
    seed: int,
    max_samples: Optional[int] = None,
    entity_mode: str = "folder",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_samples is not None:
        rows = rows[:max_samples]

    rng = np.random.default_rng(seed)
    skeleton_index: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        skeleton_index[_prompt_skeleton(row["src"])].append(i)

    out: List[Dict[str, Any]] = []
    stats = {
        "n_input": len(rows),
        "n_output": 0,
        "missing_image_swap": 0,
        "missing_hard_neg": 0,
        "skeleton_match_hard_neg": 0,
        "skeleton_match_swap": 0,
        "entity_mode": entity_mode,
    }

    for i, row in enumerate(rows):
        hard_ids, swap_ids = _rank_partners(i, rows, skeleton_index, entity_mode=entity_mode)
        hard_j = None
        for j in hard_ids:
            if _image_exists(img_root, rows[j]["image"]):
                hard_j = j
                break
        swap_j = None
        for j in swap_ids:
            if j != hard_j and _image_exists(img_root, rows[j]["image"]):
                swap_j = j
                break
        if swap_j is None:
            for j in swap_ids:
                if _image_exists(img_root, rows[j]["image"]):
                    swap_j = j
                    break

        if hard_j is None or swap_j is None:
            if hard_j is None:
                stats["missing_hard_neg"] += 1
            if swap_j is None:
                stats["missing_image_swap"] += 1
            continue

        if _prompt_skeleton(rows[hard_j]["src"]) == _prompt_skeleton(row["src"]):
            stats["skeleton_match_hard_neg"] += 1
        if _prompt_skeleton(rows[swap_j]["src"]) == _prompt_skeleton(row["src"]):
            stats["skeleton_match_swap"] += 1

        alt_short = row.get("alt_short")
        conflict_prompt = _build_conflict_prompt(row["src"], row.get("pred", ""), alt_short)
        consistency_image = row.get("image_rephrase") or row["image"]

        new_row = {k: v for k, v in row.items() if not str(k).startswith("grounding_")}
        new_row["source_index"] = int(i)
        new_row["grounding_probes"] = {
            "image_swap": rows[swap_j]["image"],
            "image_swap_partner_index": int(swap_j),
            "hard_neg_image": rows[hard_j]["image"],
            "hard_neg_partner_index": int(hard_j),
            "conflict_prompt": conflict_prompt,
            "modality_consistency_image": consistency_image,
            "pairing_seed": seed,
        }
        out.append(new_row)

    stats["n_output"] = len(out)
    stats["skeleton_match_hard_neg_rate"] = (
        stats["skeleton_match_hard_neg"] / stats["n_output"] if stats["n_output"] else 0.0
    )
    stats["skeleton_match_swap_rate"] = (
        stats["skeleton_match_swap"] / stats["n_output"] if stats["n_output"] else 0.0
    )
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Step 1 grounding eval JSON")
    ap.add_argument(
        "-dpath", "--vlkeb_eval_json", type=str,
        default="../vlkeb_long_ollama/out/eval_longform_ollama.json",
        help="Source VLKEB eval JSON (long-form or standard)",
    )
    ap.add_argument(
        "--img_root", type=str,
        default="data/VLKEB/VLKEB_images/mmkb_images",
        help="Image root for existence checks",
    )
    ap.add_argument(
        "-out", "--output_json", type=str,
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
    )
    ap.add_argument("-seed", "--pairing_seed", type=int, default=42)
    ap.add_argument("-max", "--max_samples", type=int, default=None,
                    help="Cap rows (smoke tests); default keeps all valid rows")
    ap.add_argument(
        "--entity_mode",
        type=str,
        default="folder",
        choices=["folder", "coco"],
        help="Partner exclusion: folder=VLKEB entity dir; coco=COCO image id (EVQA/EIC)",
    )
    args = ap.parse_args()

    data_path = args.vlkeb_eval_json
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT_PATH, data_path)
    img_root = args.img_root
    if not os.path.isabs(img_root):
        img_root = os.path.join(ROOT_PATH, img_root)

    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]

    out_rows, stats = build_grounding_rows(
        rows,
        img_root,
        seed=args.pairing_seed,
        max_samples=args.max_samples,
        entity_mode=args.entity_mode,
    )

    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT_PATH, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_eval_json": data_path,
        "img_root": img_root,
        "pairing_seed": args.pairing_seed,
        "entity_mode": args.entity_mode,
        "probes": {
            "image_swap": "Wrong-entity image at test time; expect low acc vs reliability",
            "hard_negative": "Visually confusable partner image; false activation if acc high",
            "modality_consistency": "Image rephrase; same answer as reliability",
            "prompt_conflict": "Misleading prompt prefix; image-grounded editor should still answer alt",
        },
        "stats": stats,
    }

    payload = {"meta": meta, "data": out_rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  samples: {stats['n_output']} / {stats['n_input']}")
    print(f"  hard-neg skeleton match: {stats['skeleton_match_hard_neg_rate']*100:.1f}%")
    print(f"  swap skeleton match: {stats['skeleton_match_swap_rate']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
