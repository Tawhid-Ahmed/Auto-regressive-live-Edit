"""CHECK 5.3: quick BERTScore on GPU (or CPU). Run: python tools/bertscore_smoke.py"""
from __future__ import annotations

import sys

import torch

from bert_score import score as bert_score_fn


def main() -> int:
    cands = ["The dog sat on the mat in the living room near the door"]
    refs = ["The cat sat on the mat in the living room near the window"]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"BERTScore smoke on {dev} ...", flush=True)
    P, R, F1 = bert_score_fn(cands, refs, lang="en", rescale_with_baseline=True, device=dev)
    print(f"  P={float(P.mean()):.4f} R={float(R.mean()):.4f} F1={float(F1.mean()):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
