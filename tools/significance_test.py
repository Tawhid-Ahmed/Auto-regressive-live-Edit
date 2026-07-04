"""Workstream D: multi-seed aggregation + paired significance test (baseline vs AR).

Reads `seed_<seed>_mean_results.json` files for baseline and AR across several
seeds, reports mean +/- std per metric, and runs a paired test on the per-seed
AR-minus-baseline deltas (pairing by seed, since the same seed = same eval
shuffle for both runs).

Usage:
  python tools/significance_test.py \
    --baseline 42=path/seed_42_mean_results.json --baseline 123=path/seed_123_mean_results.json ... \
    --ar 42=path/seed_42_mean_results.json --ar 123=path/seed_123_mean_results.json ... \
    --out doc/results/significance.md

No third-party stats dependency required (paired t-test computed directly with an
embedded two-tailed t critical-value table).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

# two-tailed t critical values by degrees of freedom
T_CRIT_05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}
T_CRIT_01 = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707,
             7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169, 11: 3.106, 12: 3.055,
             13: 3.012, 14: 2.977, 15: 2.947, 20: 2.845, 30: 2.750}

METRICS = [
    ("reliability.acc", "Reliability acc"),
    ("reliability.chunk_em", "Chunk EM"),
    ("reliability.chunk_f1", "Chunk F1"),
    ("generality.text_rephrase.acc", "Text generality"),
    ("generality.image_rephrase.acc", "Image generality"),
    ("locality.text_loc.acc", "Locality (text)"),
    ("locality.image_loc.acc", "Locality (image)"),
]


def _get(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("total_mean", data)


def _parse(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--baseline/--ar must be 'seed=path', got: {it}")
        seed, path = it.split("=", 1)
        out[seed.strip()] = path.strip()
    return out


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _t_crit(table: Dict[int, float], df: int) -> float:
    if df in table:
        return table[df]
    keys = sorted(table)
    for k in keys:
        if k >= df:
            return table[k]
    return table[keys[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="append", required=True, help="seed=path")
    ap.add_argument("--ar", action="append", required=True, help="seed=path")
    ap.add_argument("--out", default="doc/results/significance.md")
    args = ap.parse_args()

    base = _parse(args.baseline)
    ar = _parse(args.ar)
    seeds = sorted(set(base) & set(ar), key=lambda s: int(s) if s.isdigit() else s)
    if not seeds:
        raise SystemExit("No common seeds between --baseline and --ar.")

    base_tm = {s: _load(base[s]) for s in seeds}
    ar_tm = {s: _load(ar[s]) for s in seeds}

    rows = []
    for key, label in METRICS:
        b_vals = [(_get(base_tm[s], key) or 0.0) * 100 for s in seeds]
        a_vals = [(_get(ar_tm[s], key) or 0.0) * 100 for s in seeds]
        deltas = [a - b for a, b in zip(a_vals, b_vals)]
        bm, bs = _mean_std(b_vals)
        am, as_ = _mean_std(a_vals)
        dm, ds = _mean_std(deltas)
        n = len(deltas)
        df = n - 1
        if ds > 0 and n > 1:
            t = dm / (ds / math.sqrt(n))
            tc05 = _t_crit(T_CRIT_05, df)
            tc01 = _t_crit(T_CRIT_01, df)
            sig = "** (p<0.01)" if abs(t) >= tc01 else ("* (p<0.05)" if abs(t) >= tc05 else "ns")
            half = tc05 * ds / math.sqrt(n)
            ci = f"[{dm - half:+.2f}, {dm + half:+.2f}]"
            tstr = f"{t:+.2f}"
        else:
            sig = "n/a"
            ci = "n/a"
            tstr = "n/a"
        rows.append((label, bm, bs, am, as_, dm, ds, tstr, ci, sig))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Multi-seed significance: baseline vs AR (cs=16)\n\n")
        f.write(f"Seeds (n={len(seeds)}): {', '.join(seeds)}\n\n")
        f.write("Paired by seed (same seed = same eval shuffle). Values are percentages.\n\n")
        f.write("| Metric | Baseline (mean+/-std) | AR (mean+/-std) | Delta (mean+/-std) | t | 95% CI | Sig |\n")
        f.write("|--------|----------------------|-----------------|--------------------|---|--------|-----|\n")
        for (label, bm, bs, am, as_, dm, ds, tstr, ci, sig) in rows:
            f.write(f"| {label} | {bm:.2f}+/-{bs:.2f} | {am:.2f}+/-{as_:.2f} | "
                    f"{dm:+.2f}+/-{ds:.2f} | {tstr} | {ci} | {sig} |\n")
        f.write("\n*Paired two-tailed t-test on per-seed deltas. "
                "* = p<0.05, ** = p<0.01, ns = not significant.*\n")
    print(f"Wrote {args.out}")
    for (label, bm, bs, am, as_, dm, ds, tstr, ci, sig) in rows:
        print(f"{label:18s} base={bm:6.2f}  ar={am:6.2f}  d={dm:+.2f}+/-{ds:.2f}  t={tstr}  {sig}")


if __name__ == "__main__":
    main()
