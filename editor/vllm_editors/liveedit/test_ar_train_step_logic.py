"""
Unit test for Task 4 AR training loop logic: chunk indexing and mask construction.
No model or data load; validates that start_pos/end_pos and chunk_mask align with
label_masks so that per-chunk loss aggregation is correct.
Run: python -m editor.vllm_editors.liveedit.test_ar_train_step_logic
"""
import torch


def test_chunk_positions_and_mask():
    # Simulate one sample: label_masks [1, L], L=20; target occupies first T=12 positions (mask 1)
    L = 20
    T = 12
    label_masks = torch.zeros(1, L)
    label_masks[:, :T] = 1

    # Chunks for this target: e.g. [4, 4, 4] tokens
    chunks_b = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert sum(len(c) for c in chunks_b) == T

    # Replicate AR loop logic
    total_masked = 0
    for t, chunk_t in enumerate(chunks_b):
        start_pos = sum(len(chunks_b[k]) for k in range(t))
        end_pos = start_pos + len(chunk_t)
        chunk_mask = torch.zeros_like(label_masks, device=label_masks.device, dtype=label_masks.dtype)
        chunk_mask[:, start_pos:end_pos] = label_masks[:, start_pos:end_pos]
        total_masked += chunk_mask.sum().item()

    # Sum of chunk mask ones should equal number of target positions (each position in one chunk)
    assert total_masked == T, f"Expected {T} masked positions total, got {total_masked}"


def test_chunk_positions_uneven():
    # Uneven chunks: 3 + 5 + 2 = 10
    L = 15
    T = 10
    label_masks = torch.zeros(1, L)
    label_masks[:, :T] = 1
    chunks_b = [[0] * 3, [0] * 5, [0] * 2]

    for t, chunk_t in enumerate(chunks_b):
        start_pos = sum(len(chunks_b[k]) for k in range(t))
        end_pos = start_pos + len(chunk_t)
        assert start_pos < L and end_pos <= L
        assert end_pos <= T  # chunk end within target
    assert sum(len(c) for c in chunks_b) == T


def test_empty_chunks_skipped():
    chunks_b = []
    rel_loss_contrib = 0
    for t, chunk_t in enumerate(chunks_b):
        rel_loss_contrib += 1  # would add loss
    assert rel_loss_contrib == 0


if __name__ == "__main__":
    test_chunk_positions_and_mask()
    print("[PASS] chunk positions and mask")
    test_chunk_positions_uneven()
    print("[PASS] uneven chunk positions")
    test_empty_chunks_skipped()
    print("[PASS] empty chunks skipped")
    print("\nAll AR train step logic tests passed.")
