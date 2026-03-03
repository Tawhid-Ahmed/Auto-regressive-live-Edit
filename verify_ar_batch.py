"""
Quick verification that AR chunk metadata is attached in batch organization (Task 3).
Run with: python verify_ar_batch.py
Uses liveedit conda env; requires VLKEB data and BLIP2 model.
"""
import os
import sys
from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor

def main():
    device = "cuda:0"
    extra_devices = [0]
    data_n = 2
    batch_size = 2

    print("Loading editor (LiveEdit + BLIP2)...")
    editor = load_vllm_editor("LiveEdit", "blip2", device, extra_devices, None, True)

    print("Loading VLKEB (tiny subset)...")
    from dataset.vllm import VLKEB
    data_path = os.path.join(ROOT_PATH, "data/VLKEB/train.json")
    img_root_dir = os.path.join(ROOT_PATH, "data/VLKEB/VLKEB_images/mmkb_images")
    train_data = VLKEB(data_path, img_root_dir, data_n)

    training_data = editor.preprocess_train_data(train_data)
    assert len(training_data) >= batch_size, "Need at least batch_size samples"

    # Test 1: ar_mode=False -> batch_ar_chunks is None
    editor.ar_mode = False
    editor.chunk_size = 16
    editor.max_chunks = None
    editor.set_random_seeds(42)
    editor.other_train_init_begin()
    a_batch_raw = [training_data[i] for i in range(batch_size)]
    batch_organized = editor.organize_batch_data(a_batch_raw)
    assert len(batch_organized) == 11, f"Expected 11 elements, got {len(batch_organized)}"
    batch_ar_chunks = batch_organized[10]
    assert batch_ar_chunks is None, f"With ar_mode=False expected batch_ar_chunks None, got {type(batch_ar_chunks)}"
    print("  [PASS] ar_mode=False: batch has 11 elements, batch_ar_chunks is None")

    # Test 2: ar_mode=True -> batch has 12 elements (batch_ar_chunks + rel_edit_i)
    editor.ar_mode = True
    batch_organized_ar = editor.organize_batch_data(a_batch_raw)
    assert len(batch_organized_ar) == 12, f"With ar_mode=True expected 12 elements, got {len(batch_organized_ar)}"
    batch_ar_chunks_ar = batch_organized_ar[10]
    rel_edit_i = batch_organized_ar[11]
    assert batch_ar_chunks_ar is not None, "With ar_mode=True expected batch_ar_chunks non-None"
    assert rel_edit_i is not None and len(rel_edit_i) == batch_size, "rel_edit_i should be list of length batch_size"
    assert len(batch_ar_chunks_ar) == batch_size, f"Expected {batch_size} batch items, got {len(batch_ar_chunks_ar)}"
    for i, request_chunks in enumerate(batch_ar_chunks_ar):
        assert isinstance(request_chunks, list), f"Batch item {i} should be list of request chunks"
        for j, chunks in enumerate(request_chunks):
            assert isinstance(chunks, list), f"Request {j} should be list of chunks"
            for c in chunks:
                assert isinstance(c, list) and all(isinstance(t, int) for t in c), "Chunk should be list of token ids"
    print("  [PASS] ar_mode=True: batch has 12 elements, batch_ar_chunks + rel_edit_i for Task 4")

    print("\nVerification passed: AR metadata is correctly attached in batch organization (Task 3).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
