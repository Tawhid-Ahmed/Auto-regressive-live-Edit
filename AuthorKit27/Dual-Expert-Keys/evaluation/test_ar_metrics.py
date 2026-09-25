"""
Validation tests for AR-specific evaluation metrics (Task 7: chunk and length-bin).
Run: python -m pytest evaluation/test_ar_metrics.py -v
  or: python evaluation/test_ar_metrics.py
"""
import sys
import os
import json
import tempfile

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from evaluation.vllm_editor_eval import (
    _length_bin_name,
    _chunk_em_f1,
    LENGTH_BIN_EDGES,
    VLLMEditorEvaluation,
)


def test_length_bin_name():
    """Length bins: <=16, 17-32, 33-64, 65+."""
    assert _length_bin_name(0) == "<=16"
    assert _length_bin_name(16) == "<=16"
    assert _length_bin_name(17) == "17-32"
    assert _length_bin_name(32) == "17-32"
    assert _length_bin_name(33) == "33-64"
    assert _length_bin_name(64) == "33-64"
    assert _length_bin_name(65) == "65+"
    assert _length_bin_name(100) == "65+"


def test_chunk_em_f1_perfect_match():
    """All chunks correct -> chunk_em=1, chunk_f1=1."""
    L = 20
    ids = torch.randint(0, 1000, (1, L))
    mask = torch.ones(1, L)
    ce, cf = _chunk_em_f1(ids, ids, mask, chunk_size=8)
    assert ce == 1.0
    assert cf == 1.0


def test_chunk_em_f1_no_match():
    """Prediction wrong everywhere -> chunk_em=0, chunk_f1=0."""
    L = 20
    ref = torch.ones(1, L, dtype=torch.long)
    pred = torch.zeros(1, L, dtype=torch.long)
    mask = torch.ones(1, L)
    ce, cf = _chunk_em_f1(pred, ref, mask, chunk_size=8)
    assert ce == 0.0
    assert cf == 0.0


def test_chunk_em_f1_empty():
    """No tokens -> (0, 0)."""
    ce, cf = _chunk_em_f1(
        torch.zeros(1, 5), torch.zeros(1, 5), torch.zeros(1, 5), chunk_size=8
    )
    assert ce == 0.0
    assert cf == 0.0


def test_get_mean_results_includes_ar_metrics():
    """get_mean_results returns reliability.em/chunk_em/chunk_f1 and length_bin."""
    class MockEditor:
        def name_of_editor_and_model(self):
            return "liveedit", "blip2"

    class MockData:
        data_with_img = []
        data_with_img_path = []

    ev = VLLMEditorEvaluation(MockEditor(), MockData(), ar_chunk_size=16)

    # Mock results with AR-specific keys (as produced by __get_results_after_edit__)
    results = [
        {
            "reliability": [
                {
                    "acc": 0.8,
                    "edit_time": 0.5,
                    "target_token_count": 10,
                    "em": 0.0,
                    "chunk_em": 0.5,
                    "chunk_f1": 0.7,
                },
                {
                    "acc": 1.0,
                    "edit_time": 0.3,
                    "target_token_count": 40,
                    "em": 1.0,
                    "chunk_em": 1.0,
                    "chunk_f1": 1.0,
                },
            ],
            "generality": {},
            "locality": {},
        },
    ]

    mean_res = ev.get_mean_results(results)

    # Existing reliability aggregates
    assert "reliability" in mean_res
    assert mean_res["reliability"]["acc"] == 0.9
    assert mean_res["reliability"]["em"] == 0.5
    assert mean_res["reliability"]["chunk_em"] == 0.75
    assert mean_res["reliability"]["chunk_f1"] == 0.85

    # Length-bin present and correct
    assert "length_bin" in mean_res
    for bin_name in ["<=16", "17-32", "33-64", "65+"]:
        assert bin_name in mean_res["length_bin"]
        b = mean_res["length_bin"][bin_name]
        assert "count" in b
        assert "em" in b
        assert "acc" in b
        assert "chunk_em" in b
        assert "chunk_f1" in b

    # <=16 has 1 sample (10 tokens), 17-32 has 0, 33-64 has 1 (40 tokens), 65+ has 0
    assert mean_res["length_bin"]["<=16"]["count"] == 1
    assert mean_res["length_bin"]["<=16"]["em"] == 0.0
    assert mean_res["length_bin"]["17-32"]["count"] == 0
    assert mean_res["length_bin"]["33-64"]["count"] == 1
    assert mean_res["length_bin"]["33-64"]["em"] == 1.0
    assert mean_res["length_bin"]["65+"]["count"] == 0


def test_save_results_rounds_ar_metrics():
    """save_results produces JSON with length_bin (schema not broken)."""
    class MockEditor:
        def name_of_editor_and_model(self):
            return "liveedit", "blip2"

    class MockData:
        data_with_img = []
        data_with_img_path = []

    ev = VLLMEditorEvaluation(MockEditor(), MockData())
    mean_results = {
        "reliability": {"acc": 0.9, "em": 0.5, "chunk_em": 0.75, "chunk_f1": 0.85},
        "generality": {},
        "locality": {},
        "length_bin": {
            "<=16": {"count": 5, "em": 0.8, "acc": 0.82, "chunk_em": 0.7, "chunk_f1": 0.75},
            "17-32": {"count": 3, "em": 0.33, "acc": 0.5, "chunk_em": 0.4, "chunk_f1": 0.45},
            "33-64": {"count": 0, "em": None, "acc": None, "chunk_em": None, "chunk_f1": None},
            "65+": {"count": 0, "em": None, "acc": None, "chunk_em": None, "chunk_f1": None},
        },
        "sample_count": 8,
    }

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mean_results.json")
        ev.save_results(path, mean_results)
        with open(path) as f:
            loaded = json.load(f)
    assert "length_bin" in loaded
    assert loaded["length_bin"]["<=16"]["em"] == 0.8
    assert loaded["reliability"]["chunk_f1"] == 0.85


if __name__ == "__main__":
    test_length_bin_name()
    print("test_length_bin_name OK")
    test_chunk_em_f1_perfect_match()
    print("test_chunk_em_f1_perfect_match OK")
    test_chunk_em_f1_no_match()
    print("test_chunk_em_f1_no_match OK")
    test_chunk_em_f1_empty()
    print("test_chunk_em_f1_empty OK")
    test_get_mean_results_includes_ar_metrics()
    print("test_get_mean_results_includes_ar_metrics OK")
    test_save_results_rounds_ar_metrics()
    print("test_save_results_rounds_ar_metrics OK")
    print("All validation tests passed.")
