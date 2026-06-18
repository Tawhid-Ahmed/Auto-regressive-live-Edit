"""
Build "VLKEB-long" JSONs: VLKEB-shaped files whose `alt` targets are long-form.

This repo's VLKEB loader (`dataset/vllm.py::__init_eic_evqa__`) consumes keys:
  src, rephrase, alt, image, image_rephrase, loc, loc_ans, m_loc, m_loc_q, m_loc_a

"VLKEB-long" is not a separate public download; it's a *derived* version of VLKEB
that keeps the same schema but uses longer targets in `alt` (bounded by token bins).

Examples (from repo root):
  python tools/build_vlkeb_longform_json.py --split eval  --n 500 --seed 42
  python tools/build_vlkeb_longform_json.py --split train --n 500 --seed 42

Outputs (defaults, kept separate from official VLKEB):
  data/VLKEB_long/eval_longform.json
  data/VLKEB_long/train_longform.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer  # noqa: E402

from utils.GLOBAL import model_path_map  # noqa: E402


@dataclass(frozen=True)
class Bin:
    lo: int
    hi: int


def _ntok(tok, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def _trim_to_hi(tok, s: str, hi: int, min_chars: int = 8) -> str:
    s2 = s.strip()
    while _ntok(tok, s2) > hi and len(s2) > min_chars:
        s2 = s2[: int(len(s2) * 0.93)].rstrip()
    return s2


def _pad_to_lo(tok, s: str, lo: int, pad: str) -> str:
    s2 = s.strip()
    while _ntok(tok, s2) < lo:
        s2 = (s2 + " " + pad).strip()
    return s2


def _build_long_alt(tok, short_answer: str, b: Bin) -> str:
    base = (str(short_answer).strip() or "answer").replace("\n", " ").strip()
    base = base.rstrip(".")

    templates = [
        (
            f"The visually grounded answer is {base}. "
            "The image supports this through consistent contextual cues (e.g., scene layout, landmarks, and text). "
            "These cues jointly disambiguate it from nearby confusable alternatives."
        ),
        (
            f"{base}. "
            "This answer is supported by the overall visual evidence. "
            "Key cues such as environment, visible structures, and perspective align with this identification."
        ),
        (
            f"In this image, the correct answer is {base}. "
            "Multiple independent visual hints point to the same conclusion, "
            "so the response remains consistent under reasonable rephrasings of the question."
        ),
    ]
    pad = (
        "Additional inspection of lighting, viewpoint, and surrounding details further reinforces the same conclusion "
        "without adding information that is not justified by the image."
    )
    s = random.choice(templates)
    s = _pad_to_lo(tok, s, b.lo, pad)
    s = _trim_to_hi(tok, s, b.hi)
    return s


def _parse_bins(s: str) -> List[Bin]:
    """
    Parse bins string like: "8-16,17-64,65-150"
    """
    out: List[Bin] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s = part.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"Invalid bin: {part!r}")
        out.append(Bin(lo=lo, hi=hi))
    if not out:
        raise ValueError("No bins parsed.")
    return out


def _validate_vlkeb_row(row: Dict[str, Any]) -> None:
    req = ["src", "rephrase", "alt", "image", "image_rephrase", "loc", "loc_ans", "m_loc", "m_loc_q", "m_loc_a"]
    missing = [k for k in req if k not in row]
    if missing:
        raise KeyError(f"VLKEB row missing keys: {missing}")


def build(args: argparse.Namespace) -> Tuple[str, List[int]]:
    mp = model_path_map["blip2-opt-2.7b"]
    tok = AutoTokenizer.from_pretrained(mp, use_fast=False, local_files_only=True)

    in_path = args.in_json or os.path.join(ROOT, "data", "VLKEB", f"{args.split}.json")
    default_out_dir = os.path.join(ROOT, "data", "VLKEB_long")
    out_path = args.out_json or os.path.join(default_out_dir, f"{args.split}_longform.json")

    with open(in_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if args.n is not None:
        rows = rows[: int(args.n)]

    bins = _parse_bins(args.bins)
    random.seed(args.seed)

    out: List[Dict[str, Any]] = []
    lens: List[int] = []
    for i, row in enumerate(rows):
        _validate_vlkeb_row(row)
        b = bins[i % len(bins)] if args.bin_schedule == "round_robin" else random.choice(bins)
        r = dict(row)
        if args.keep_short:
            r["alt_short"] = row.get("alt", "")
        r["alt"] = _build_long_alt(tok, row.get("alt", "answer"), b)
        lens.append(_ntok(tok, r["alt"]))
        out.append(r)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out_path, lens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], default="eval", help="Which VLKEB split to transform.")
    ap.add_argument("--in_json", default=None, help="Override input JSON path (defaults to data/VLKEB/<split>.json).")
    ap.add_argument(
        "--out_json",
        default=None,
        help="Override output JSON path (default: data/VLKEB_long/<split>_longform.json).",
    )
    ap.add_argument("--n", type=int, default=None, help="Limit to first N rows (useful for pilot).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (bin assignment + template choice).")
    ap.add_argument(
        "--bins",
        default="8-16,17-64,65-150",
        help="Target token bins for long `alt` (comma-separated lo-hi).",
    )
    ap.add_argument(
        "--bin_schedule",
        choices=["round_robin", "random"],
        default="round_robin",
        help="How to assign bins to rows.",
    )
    ap.add_argument(
        "--keep_short",
        action="store_true",
        help="If set, preserve original short answer under `alt_short`.",
    )
    args = ap.parse_args()

    out_path, lens = build(args)
    print("Wrote:", out_path)
    if lens:
        print(
            "alt token lengths:",
            f"min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

