"""Workstream E1: offline sequence-level metrics over free-running predictions.

Reads `*freegen_results.json` files (produced by --free_running eval) and scores
`prediction_freegen` vs `target_text` for every reliability sample. ROUGE-L /
ROUGE-1 / token-F1 / EM are computed with no third-party dependency (LCS + unigram
overlap). BERTScore is optional (needs the `bert_score` package + a model).

Usage:
  python tools/score_freegen.py \
    --run Baseline=eval_results/.../seed_42_freegen_results.json \
    --run "AR (cs 16)"=eval_results/.../seed_42_freegen_results.json \
    --out doc/results/freegen_scores.md
  # add --bertscore (slow, downloads a model) for BERTScore-F1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from typing import Dict, List, Tuple

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm_tokens(s: str) -> List[str]:
    s = str(s).lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip().split()


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if n_pred == 0 or n_ref == 0 or overlap == 0:
        return 0.0
    p = overlap / n_pred
    r = overlap / n_ref
    return 2 * p * r / (p + r)


def _rouge_l_f1(pred: str, ref: str) -> float:
    p, r = _norm_tokens(pred), _norm_tokens(ref)
    return _f1(_lcs_len(p, r), len(p), len(r))


def _rouge_1_f1(pred: str, ref: str) -> float:
    from collections import Counter
    p, r = _norm_tokens(pred), _norm_tokens(ref)
    overlap = sum((Counter(p) & Counter(r)).values())
    return _f1(overlap, len(p), len(r))


def _token_f1(pred: str, ref: str) -> float:
    # same as rouge-1 here (unigram overlap F1); kept as an explicit alias
    return _rouge_1_f1(pred, ref)


def _em(pred: str, ref: str) -> float:
    return float(" ".join(_norm_tokens(pred)) == " ".join(_norm_tokens(ref)))


def _collect(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds, refs = [], []
    # results structure: list of splits -> list of records -> {'reliability': [ {...} ]}
    for split in data:
        for rec in split:
            for rr in rec.get("reliability", []):
                if "prediction_freegen" in rr and "target_text" in rr:
                    preds.append(rr["prediction_freegen"])
                    refs.append(rr["target_text"])
    return preds, refs


def _parse_run(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise SystemExit(f"--run must be 'Label=path', got: {item}")
    label, path = item.split("=", 1)
    return label.strip(), path.strip()


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=path to *freegen_results.json")
    ap.add_argument("--out", default="doc/results/freegen_scores.md")
    ap.add_argument("--bertscore", action="store_true", help="Also compute BERTScore-F1 (needs bert_score pkg).")
    ap.add_argument("--bertscore_model", default="roberta-large")
    ap.add_argument("--bertscore_device", default="cuda:0")
    args = ap.parse_args()

    runs = [_parse_run(r) for r in args.run]
    rows = []
    for label, path in runs:
        preds, refs = _collect(path)
        n = len(preds)
        rl = _mean([_rouge_l_f1(p, r) for p, r in zip(preds, refs)])
        r1 = _mean([_rouge_1_f1(p, r) for p, r in zip(preds, refs)])
        tf = _mean([_token_f1(p, r) for p, r in zip(preds, refs)])
        em = _mean([_em(p, r) for p, r in zip(preds, refs)])
        bs = None
        if args.bertscore and n > 0:
            from bert_score import score as bertscore
            _, _, F = bertscore(preds, refs, model_type=args.bertscore_model,
                                lang="en", device=args.bertscore_device, verbose=False)
            bs = float(F.mean().item())
        rows.append((label, n, rl, r1, tf, em, bs))
        bss = f"{bs * 100:.2f}" if bs is not None else "-"
        print(f"{label:18s} n={n:4d}  ROUGE-L={rl*100:5.2f}  ROUGE-1={r1*100:5.2f}  "
              f"tokF1={tf*100:5.2f}  EM={em*100:5.2f}  BERTScore={bss}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Free-running sequence-level scores (VLKEB-long)\n\n")
        f.write("Non-teacher-forced generation; metrics computed on decoded text vs target. "
                "Values are percentages.\n\n")
        f.write("| Run | N | ROUGE-L | ROUGE-1 | Token-F1 | EM | BERTScore-F1 |\n")
        f.write("|-----|---|---------|---------|----------|----|--------------|\n")
        for (label, n, rl, r1, tf, em, bs) in rows:
            bss = f"{bs * 100:.2f}" if bs is not None else "-"
            f.write(f"| {label} | {n} | {rl*100:.2f} | {r1*100:.2f} | {tf*100:.2f} | {em*100:.2f} | {bss} |\n")
        f.write("\n*ROUGE-L = LCS-based F1; ROUGE-1/Token-F1 = unigram-overlap F1; "
                "EM = normalized exact match. BERTScore-F1 only present when --bertscore is set.*\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
