"""M5: build a pair-closed grounding pool of target size n.

Every kept row's image_swap and hard_negative partners are also in the pool.
Partner indices are remapped to the new local 0..len-1 indexing.

Usage:
  python tools/build_pair_closed_pool.py \\
    -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json \\
    -n 100 -seed 42 \\
    -out data/VLKEB/meeting_validation/step1/grounding/grounding_pairclosed_n100_s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"], payload.get("meta") or {}
    return payload, {}


def _partner_indices(row: Dict[str, Any]) -> List[int]:
    gp = row.get("grounding_probes") or {}
    out = []
    for k in ("image_swap_partner_index", "hard_neg_partner_index"):
        if gp.get(k) is not None:
            out.append(int(gp[k]))
    return out


def _closure(rows: List[Dict[str, Any]], seed_idx: int) -> Set[int]:
    stack = [int(seed_idx)]
    seen: Set[int] = set()
    n = len(rows)
    while stack:
        j = stack.pop()
        if j in seen or j < 0 or j >= n:
            continue
        seen.add(j)
        for p in _partner_indices(rows[j]):
            if p not in seen:
                stack.append(p)
    return seen


def pair_closed_indices(rows: List[Dict[str, Any]], n: int, seed: int) -> List[int]:
    """Grow a partner-closed index set until size >= n (may slightly exceed n)."""
    rng = np.random.default_rng(int(seed))
    order = [int(i) for i in rng.permutation(len(rows))]
    selected: Set[int] = set()
    for i in order:
        if len(selected) >= n:
            break
        selected |= _closure(rows, i)
    # If still short (pathological), pad with remaining rows + their closures
    if len(selected) < n:
        for i in order:
            if len(selected) >= n:
                break
            if i not in selected:
                selected |= _closure(rows, i)
    # Deterministic order for the file: sort by original index, then we shuffle at eval
    return sorted(selected)


def remap_rows(rows: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
    old_to_new = {old: new for new, old in enumerate(indices)}
    out = []
    n_remap_ok = 0
    n_remap_miss = 0
    for new_i, old_i in enumerate(indices):
        row = {k: v for k, v in rows[old_i].items()}
        gp = dict(row.get("grounding_probes") or {})
        for key in ("image_swap_partner_index", "hard_neg_partner_index"):
            if gp.get(key) is None:
                continue
            old_p = int(gp[key])
            if old_p in old_to_new:
                gp[key] = int(old_to_new[old_p])
                n_remap_ok += 1
            else:
                # Should not happen if set is closed; keep old and flag
                n_remap_miss += 1
        row["grounding_probes"] = gp
        row["source_index"] = int(old_i)
        row["pair_closed_local_index"] = int(new_i)
        out.append(row)
    return out, {"partner_remap_ok": n_remap_ok, "partner_remap_miss": n_remap_miss}


def verify_closed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    miss_swap = miss_hn = 0
    for i, row in enumerate(rows):
        gp = row.get("grounding_probes") or {}
        ps = gp.get("image_swap_partner_index")
        ph = gp.get("hard_neg_partner_index")
        if ps is None or not (0 <= int(ps) < n):
            miss_swap += 1
        if ph is None or not (0 <= int(ph) < n):
            miss_hn += 1
    return {
        "n": n,
        "partner_in_pool_swap_rate": 1.0 - miss_swap / n if n else None,
        "partner_in_pool_hard_neg_rate": 1.0 - miss_hn / n if n else None,
        "missing_swap_partner": miss_swap,
        "missing_hard_neg_partner": miss_hn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 pair-closed grounding pool builder")
    ap.add_argument(
        "-dpath",
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json",
    )
    ap.add_argument("-n", type=int, required=True, help="Target pool size (closed set may be slightly larger)")
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("-out", required=True)
    args = ap.parse_args()

    dpath = args.dpath if os.path.isabs(args.dpath) else os.path.join(ROOT, args.dpath)
    rows, meta = _load_rows(dpath)
    idxs = pair_closed_indices(rows, args.n, args.seed)
    new_rows, remap_stats = remap_rows(rows, idxs)
    verify = verify_closed(new_rows)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "meta": {
            **meta,
            "pair_closed": True,
            "pair_closed_target_n": int(args.n),
            "pair_closed_actual_n": len(new_rows),
            "pair_closed_seed": int(args.seed),
            "source_grounding_json": dpath.replace("\\", "/"),
            "created": datetime.now().isoformat(timespec="seconds"),
            "remap_stats": remap_stats,
            "verify": verify,
            "probes": {
                **(meta.get("probes") or {}),
                "pair_closed": "M5: every row's image_swap and hard_neg partners are in this pool",
            },
        },
        "data": new_rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    print("Wrote", out_path)
    print(
        "target_n=%d actual_n=%d swap_in_pool=%.1f%% hn_in_pool=%.1f%% remap_miss=%d"
        % (
            args.n,
            len(new_rows),
            100 * (verify["partner_in_pool_swap_rate"] or 0),
            100 * (verify["partner_in_pool_hard_neg_rate"] or 0),
            remap_stats["partner_remap_miss"],
        )
    )
    if verify["missing_swap_partner"] or verify["missing_hard_neg_partner"] or remap_stats["partner_remap_miss"]:
        print("WARNING: pool not fully closed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
