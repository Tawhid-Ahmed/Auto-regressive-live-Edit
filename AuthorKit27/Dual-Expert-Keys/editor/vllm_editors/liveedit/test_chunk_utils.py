"""
Unit-style checks for AR chunk split+reconstruct utility.
Run from repo root with conda env liveedit:
  conda activate liveedit
  python -m editor.vllm_editors.liveedit.test_chunk_utils
Or: path/to/liveedit/python.exe -m editor.vllm_editors.liveedit.test_chunk_utils

Requires: transformers (and optionally VLKEB eval.json for sample checks).
"""
import sys
import os

# Allow running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from editor.vllm_editors.liveedit.chunk_utils import (
    split_target_into_chunks,
    reconstruct_text_from_chunks,
)


def _normalize(s: str) -> str:
    """Normalize for comparison when tokenizer round-trip may change spaces."""
    return " ".join(s.split())


def run_edge_case_checks(tokenizer):
    """Edge cases: short target, exact multiple, long target."""
    from typing import List

    passed = 0
    failed = 0

    # --- Short target (fewer tokens than chunk_size) ---
    short_target = "Hello."
    chunks = split_target_into_chunks(tokenizer, short_target, chunk_size=8)
    recon = reconstruct_text_from_chunks(tokenizer, chunks, clean_up_tokenization_spaces=False)
    if _normalize(recon) == _normalize(short_target):
        passed += 1
        print("[PASS] Short target: reconstruct(chunks) == original")
    else:
        failed += 1
        print(f"[FAIL] Short target: got recon={repr(recon)}")

    # --- Exact multiple of chunk_size ---
    exact_target = "One two three four five six."  # tune to get 6 tokens per chunk for chunk_size=6, or use fixed token count
    chunks = split_target_into_chunks(tokenizer, exact_target, chunk_size=6)
    recon = reconstruct_text_from_chunks(tokenizer, chunks, clean_up_tokenization_spaces=False)
    if _normalize(recon) == _normalize(exact_target):
        passed += 1
        print("[PASS] Exact multiple: reconstruct(chunks) == original")
    else:
        failed += 1
        print(f"[FAIL] Exact multiple: got recon={repr(recon)}")

    # --- Long target ---
    long_target = "This is a longer target sentence with many tokens so that we get several chunks when we use a small chunk size."
    chunk_size = 8
    chunks = split_target_into_chunks(tokenizer, long_target, chunk_size=chunk_size)
    recon = reconstruct_text_from_chunks(tokenizer, chunks, clean_up_tokenization_spaces=False)
    if _normalize(recon) == _normalize(long_target):
        passed += 1
        print("[PASS] Long target: reconstruct(chunks) == original")
    else:
        failed += 1
        print(f"[FAIL] Long target: got recon={repr(recon)}")

    # --- max_chunks cap ---
    chunks_capped = split_target_into_chunks(tokenizer, long_target, chunk_size=chunk_size, max_chunks=2)
    if len(chunks_capped) == 2:
        passed += 1
        print("[PASS] max_chunks: returns at most 2 chunks")
    else:
        failed += 1
        print(f"[FAIL] max_chunks: expected 2 chunks, got {len(chunks_capped)}")

    # --- Empty target ---
    empty_chunks = split_target_into_chunks(tokenizer, "  ", chunk_size=4)
    empty_recon = reconstruct_text_from_chunks(tokenizer, [])
    if empty_chunks == [] and empty_recon == "":
        passed += 1
        print("[PASS] Empty target / empty chunks handled")
    else:
        failed += 1
        print(f"[FAIL] Empty: chunks={empty_chunks}, recon={repr(empty_recon)}")

    return passed, failed


def run_vlkeb_sample_checks(tokenizer, sample_n: int = 5):
    """Reconstruct(chunks) == original_target on sampled VLKEB entries (JSON only, no images)."""
    import json
    data_path = os.path.join(ROOT, "data", "VLKEB", "eval.json")
    if not os.path.isfile(data_path):
        print(f"[SKIP] VLKEB data not found at {data_path}")
        return 0, 0
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] VLKEB JSON not loaded: {e}")
        return 0, 0
    # VLKEB format: list of { "requests": [ {"target_new": str, ...}, ... ], ... }
    passed = 0
    failed = 0
    for i, d in enumerate(data[:sample_n]):
        for r in d.get("requests", []):
            target_new = r.get("target_new", "")
            if not target_new.strip():
                continue
            for chunk_size in [8, 16, 32]:
                chunks = split_target_into_chunks(tokenizer, target_new, chunk_size=chunk_size)
                recon = reconstruct_text_from_chunks(tokenizer, chunks, clean_up_tokenization_spaces=False)
                if _normalize(recon) == _normalize(target_new):
                    passed += 1
                else:
                    failed += 1
                    print(f"[FAIL] VLKEB sample i={i} target={repr(target_new)[:60]}... chunk_size={chunk_size} recon={repr(recon)[:60]}...")
    if failed == 0 and passed > 0:
        print(f"[PASS] VLKEB: reconstruct == original for {passed} (target x chunk_size) checks")
    return passed, failed


def main():
    print("Chunk utility unit-style checks (Task 2)")
    print("-" * 50)

    # Prefer BLIP2 tokenizer to match LiveEdit; fallback to GPT-2 for CI without full model.
    tokenizer = None
    try:
        from editor.vllms_for_edit.blip2.blip2 import BLIP2OPTForEdit
        from transformers import Blip2Processor
        model_path = os.environ.get("BLIP2_MODEL_PATH", "Salesforce/blip2-opt-2.7b")
        processor = Blip2Processor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        print(f"Using BLIP2 tokenizer from {model_path}")
    except Exception as e:
        print(f"BLIP2 tokenizer not available: {e}")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            print("Using GPT-2 tokenizer as fallback")
        except Exception as e:
            print(f"Cannot load tokenizer: {e}")
            sys.exit(1)

    p1, f1 = run_edge_case_checks(tokenizer)
    p2, f2 = run_vlkeb_sample_checks(tokenizer, sample_n=5)

    total_pass = p1 + p2
    total_fail = f1 + f2
    print("-" * 50)
    print(f"Total: {total_pass} passed, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
