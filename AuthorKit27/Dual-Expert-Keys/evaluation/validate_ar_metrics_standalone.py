"""
Standalone validation for AR metrics logic (no transformers/editor deps).
Run from repo root: python evaluation/validate_ar_metrics_standalone.py
"""
import os
import sys
import json
import tempfile

import numpy as np
import torch

# Copy of length-bin and chunk logic so we can test without importing vllm_editor_eval
LENGTH_BIN_EDGES = (16, 32, 64)


def length_bin_name(num_tokens: int) -> str:
    if num_tokens <= LENGTH_BIN_EDGES[0]:
        return "<=16"
    if num_tokens <= LENGTH_BIN_EDGES[1]:
        return "17-32"
    if num_tokens <= LENGTH_BIN_EDGES[2]:
        return "33-64"
    return "65+"


def chunk_em_f1(pre_y, label_ids, label_masks, chunk_size):
    L = label_ids.shape[1]
    n = label_masks.sum().item()
    if n == 0 or chunk_size <= 0:
        return 0.0, 0.0
    ems, f1s = [], []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        m = label_masks[:, start:end]
        if m.sum() == 0:
            continue
        pred = pre_y[:, start:end]
        ref = label_ids[:, start:end]
        m_bool = m.bool() if m.dtype != torch.bool else m
        chunk_em = ((pred == ref) | ~m_bool).all().item()
        correct = ((pred == ref) * m).sum().item()
        chunk_acc = correct / m.sum().item()
        ems.append(float(chunk_em))
        f1s.append(float(chunk_acc))
    if not ems:
        return 0.0, 0.0
    return float(np.mean(ems)), float(np.mean(f1s))


def compute_length_bin_aggregates(results):
    """Replicate get_mean_results length_bin section."""
    bin_names = ["<=16", "17-32", "33-64", "65+"]
    bin_agg = {b: {"em": [], "acc": [], "chunk_em": [], "chunk_f1": []} for b in bin_names}
    for r in results:
        for rr in r.get("reliability", []):
            n = rr.get("target_token_count")
            if n is None:
                continue
            b = length_bin_name(int(n))
            if "em" in rr:
                bin_agg[b]["em"].append(rr["em"])
            if "acc" in rr:
                bin_agg[b]["acc"].append(rr["acc"])
            if "chunk_em" in rr:
                bin_agg[b]["chunk_em"].append(rr["chunk_em"])
            if "chunk_f1" in rr:
                bin_agg[b]["chunk_f1"].append(rr["chunk_f1"])
    out = {}
    for b in bin_names:
        arr = bin_agg[b]
        count = len(arr["em"]) if arr["em"] else 0
        out[b] = {
            "count": count,
            "em": float(np.mean(arr["em"])) if arr["em"] else None,
            "acc": float(np.mean(arr["acc"])) if arr["acc"] else None,
            "chunk_em": float(np.mean(arr["chunk_em"])) if arr["chunk_em"] else None,
            "chunk_f1": float(np.mean(arr["chunk_f1"])) if arr["chunk_f1"] else None,
        }
    return out


def main():
    errors = []

    # 1. Length bin names
    if length_bin_name(16) != "<=16" or length_bin_name(17) != "17-32" or length_bin_name(65) != "65+":
        errors.append("length_bin_name")
    print("  length_bin_name: OK")

    # 2. Chunk EM/F1
    L = 20
    ids = torch.randint(0, 1000, (1, L))
    mask = torch.ones(1, L, dtype=torch.bool)
    ce, cf = chunk_em_f1(ids, ids, mask, chunk_size=8)
    if ce != 1.0 or cf != 1.0:
        errors.append("chunk_em_f1_perfect")
    print("  chunk_em_f1 (perfect match): OK")

    ref = torch.ones(1, L, dtype=torch.long)
    pred = torch.zeros(1, L, dtype=torch.long)
    mask2 = torch.ones(1, L, dtype=torch.bool)
    ce, cf = chunk_em_f1(pred, ref, mask2, chunk_size=8)
    if ce != 0.0 or cf != 0.0:
        errors.append("chunk_em_f1_no_match")
    print("  chunk_em_f1 (no match): OK")

    # 3. Length-bin aggregation
    results = [
        {
            "reliability": [
                {"acc": 0.8, "target_token_count": 10, "em": 0.0, "chunk_em": 0.5, "chunk_f1": 0.7},
                {"acc": 1.0, "target_token_count": 40, "em": 1.0, "chunk_em": 1.0, "chunk_f1": 1.0},
            ],
        },
    ]
    length_bin = compute_length_bin_aggregates(results)
    if length_bin["<=16"]["count"] != 1 or length_bin["<=16"]["em"] != 0.0:
        errors.append("length_bin_<=16")
    if length_bin["33-64"]["count"] != 1 or length_bin["33-64"]["em"] != 1.0:
        errors.append("length_bin_33-64")
    if length_bin["17-32"]["count"] != 0:
        errors.append("length_bin_17-32_empty")
    print("  length_bin aggregation: OK")

    # 4. JSON round-trip (schema)
    payload = {
        "reliability": {"acc": 0.9, "em": 0.5, "chunk_em": 0.75, "chunk_f1": 0.85},
        "length_bin": {
            "<=16": {"count": 5, "em": 0.8, "acc": 0.82, "chunk_em": 0.7, "chunk_f1": 0.75},
            "17-32": {"count": 3, "em": 0.33, "acc": 0.5, "chunk_em": 0.4, "chunk_f1": 0.45},
            "33-64": {"count": 0, "em": None, "acc": None, "chunk_em": None, "chunk_f1": None},
            "65+": {"count": 0, "em": None, "acc": None, "chunk_em": None, "chunk_f1": None},
        },
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        with open(path) as f:
            loaded = json.load(f)
    if "length_bin" not in loaded or loaded["length_bin"]["<=16"]["em"] != 0.8:
        errors.append("json_schema")
    print("  JSON schema (length_bin): OK")

    if errors:
        print("FAILED:", errors)
        sys.exit(1)
    print("All AR metrics validation checks passed.")


if __name__ == "__main__":
    main()
