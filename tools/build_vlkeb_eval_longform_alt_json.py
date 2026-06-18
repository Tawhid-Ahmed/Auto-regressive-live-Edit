"""
Build a VLKEB-shaped eval JSON where `alt` is long-form (for pipeline / bin checks).
Writes: data/VLKEB/eval_longform_alt_10.json (first 10 eval.json rows, long `alt`).

Run: python tools/build_vlkeb_eval_longform_alt_json.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer  # noqa: E402

from utils.GLOBAL import model_path_map  # noqa: E402


def _ntok(tok, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def _build_long_alt(tok, short_answer: str, lo: int, hi: int) -> str:
    base = (str(short_answer).strip() or "answer").replace("\n", " ")
    if lo <= 24:
        glue = " It matches the scene."
        s = base + "."
        while _ntok(tok, s) < lo:
            s += glue
        while _ntok(tok, s) > hi and len(s) > len(base):
            s = s[: max(int(len(s) * 0.85), len(base))]
        return s

    core = (
        f"The visually grounded answer is {base}. "
        "The photograph provides contextual cues such as signage, skyline, and terrain "
        "that jointly support this identification rather than nearby confusable locations."
    )
    pad = (
        " Additional inspection of lighting, weather, and street layout further reinforces "
        "the same conclusion without introducing details that are not justified by the image."
    )
    s = core
    while _ntok(tok, s) < lo:
        s += pad
    while _ntok(tok, s) > hi:
        s = s[: int(len(s) * 0.92)]
    return s


def main() -> int:
    mp = model_path_map["blip2-opt-2.7b"]
    tok = AutoTokenizer.from_pretrained(mp, use_fast=False, local_files_only=True)
    eval_path = os.path.join(ROOT, "data", "VLKEB", "eval.json")
    out_path = os.path.join(ROOT, "data", "VLKEB", "eval_longform_alt_10.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        rows = json.load(f)[:10]
    bands = [
        (8, 15),
        (8, 15),
        (8, 15),
        (22, 45),
        (22, 45),
        (22, 45),
        (40, 63),
        (40, 63),
        (70, 120),
        (70, 120),
    ]
    out = []
    for i, row in enumerate(rows):
        lo, hi = bands[i]
        r = {k: v for k, v in row.items() if k != "alt_long"}
        r["alt"] = _build_long_alt(tok, row.get("alt", "answer"), lo, hi)
        out.append(r)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    lens = [_ntok(tok, r["alt"]) for r in out]
    print("Wrote:", out_path)
    print("alt token lengths:", lens, "min", min(lens), "max", max(lens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
