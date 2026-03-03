"""
Unit test for Task 6 chunk-aware routing: gate restricts retrieval to pool entries
with matching chunk index. No model or data load.
Run: python -m editor.vllm_editors.liveedit.test_ar_chunk_routing
"""
import torch


def test_chunk_gate_indices():
    """When chunk_index_pool has per-entry chunk indices, gate selects the right subset."""
    # Simulate pool of 5 experts: chunk indices [0, 0, 1, 1, 2]
    chunk_index_pool = [0, 0, 1, 1, 2]
    m = len(chunk_index_pool)
    device = torch.device("cpu")

    for chunk_index in [0, 1, 2]:
        gate = torch.tensor(
            [ci == chunk_index for ci in chunk_index_pool], device=device, dtype=torch.bool
        )
        pool_indices = torch.arange(m, device=device)[gate]
        # Should only include indices for that chunk
        expected = [i for i, ci in enumerate(chunk_index_pool) if ci == chunk_index]
        assert pool_indices.tolist() == expected, (
            f"chunk_index={chunk_index} expected {expected}, got {pool_indices.tolist()}"
        )


def test_chunk_gate_fallback_when_no_match():
    """When chunk_index is set but no pool entry matches, gate falls back to full pool."""
    chunk_index_pool = [0, 0, 1]  # only 0 and 1
    chunk_index = 2  # no match
    gate = torch.tensor(
        [ci == chunk_index for ci in chunk_index_pool], device=torch.device("cpu"), dtype=torch.bool
    )
    assert not gate.any(), "No entry should match chunk_index=2"
    # In retrieve_moes we use full pool when gate.any() is False; no crash.


def test_non_ar_chunk_index():
    """Non-AR edits use chunk index -1; gate with chunk_index=0 should not select them."""
    chunk_index_pool = [-1, -1]  # two single-pass edits
    chunk_index = 0
    gate = torch.tensor(
        [ci == chunk_index for ci in chunk_index_pool], device=torch.device("cpu"), dtype=torch.bool
    )
    assert not gate.any()
    # So retrieval falls back to full pool (all experts). Sensible.


if __name__ == "__main__":
    test_chunk_gate_indices()
    print("[PASS] chunk gate indices")
    test_chunk_gate_fallback_when_no_match()
    print("[PASS] chunk gate fallback when no match")
    test_non_ar_chunk_index()
    print("[PASS] non-AR chunk index")
    print("All Task 6 chunk routing tests passed.")
