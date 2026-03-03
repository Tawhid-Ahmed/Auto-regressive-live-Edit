"""
Validate Task 5 AR inference loop: edit_one_piece in both modes and safe fallback.
- ar_mode=False: single edit, pool +1.
- ar_mode=True: chunk-wise edits, pool +N (N = number of chunks).
- ar_mode=True with empty target_new: fallback to single-pass, no crash, pool +1.
Run: python validate_ar_inference.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor


def main():
    device = "cuda:0"
    extra_devices = [0]
    data_n = 2

    print("Loading editor (LiveEdit + BLIP2)...")
    editor = load_vllm_editor("LiveEdit", "blip2", device, extra_devices, None, True)

    print("Loading VLKEB (tiny subset)...")
    from dataset.vllm import VLKEB
    data_path = os.path.join(ROOT_PATH, "data/VLKEB/train.json")
    img_root_dir = os.path.join(ROOT_PATH, "data/VLKEB/VLKEB_images/mmkb_images")
    train_data = VLKEB(data_path, img_root_dir, data_n)

    # One request from first sample
    sample = train_data.data_with_img[0]
    assert sample["requests"], "Need at least one request"
    request = sample["requests"][0].copy()
    assert "prompt" in request and "image" in request and "target_new" in request

    # --- Test 1: ar_mode=False -> single edit, pool +1 ---
    print("\nTest 1: ar_mode=False (single-pass)...")
    editor.restore_to_original_model()
    editor.ar_mode = False
    n_before = len(editor.requests_pool)
    editor.edit_one_piece(request)
    n_after = len(editor.requests_pool)
    assert n_after == n_before + 1, f"Expected pool +1, got {n_before} -> {n_after}"
    assert len(editor.eqr_pool) == len(editor.requests_pool)
    print("  [PASS] Single edit, pool size +1.")

    # --- Test 2: ar_mode=True -> chunk-wise edits, pool +N ---
    print("\nTest 2: ar_mode=True (chunk-wise)...")
    editor.restore_to_original_model()
    editor.ar_mode = True
    editor.chunk_size = 16
    editor.max_chunks = None
    n_before = len(editor.requests_pool)
    editor.edit_one_piece(request)
    n_after = len(editor.requests_pool)
    assert n_after > n_before, f"Expected pool to grow, got {n_before} -> {n_after}"
    assert len(editor.eqr_pool) == len(editor.requests_pool)
    # With chunk_size=16, a typical target may yield multiple chunks
    num_edits = n_after - n_before
    print(f"  [PASS] Chunk-wise edits, pool size +{num_edits}.")

    # --- Test 3: ar_mode=True with empty target -> fallback to single-pass ---
    print("\nTest 3: ar_mode=True with empty target_new (fallback)...")
    editor.restore_to_original_model()
    editor.ar_mode = True
    request_empty = {**request, "target_new": "   "}
    n_before = len(editor.requests_pool)
    editor.edit_one_piece(request_empty)
    n_after = len(editor.requests_pool)
    # Fallback: single edit (empty chunks -> ValueError -> fallback -> _edit_one_piece_single)
    assert n_after == n_before + 1, f"Fallback should add 1 edit, got {n_before} -> {n_after}"
    assert len(editor.eqr_pool) == len(editor.requests_pool)
    print("  [PASS] Fallback to single-pass, pool +1, no crash.")

    print("\nValidation passed: Task 5 AR inference loop (edit_one_piece both modes + fallback).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
