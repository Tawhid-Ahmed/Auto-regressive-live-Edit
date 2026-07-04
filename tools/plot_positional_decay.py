"""Workstream A: plot positional efficacy decay (accuracy vs position in target).

Reads one or more `*_mean_results.json` files produced by the eval (each must
contain `total_mean.position_curve`, added by evaluation/vllm_editor_eval.py) and
plots normalized-position decile accuracy for each run on one chart.

Usage:
  python tools/plot_positional_decay.py \
    --run "Baseline=eval_results/.../VLKEB-VLKEB_long_baseline_full-.../sequential_edit_50/seed_42_mean_results.json" \
    --run "AR cs=16=eval_results/.../VLKEB-ar-VLKEB_long_ar_full-.../sequential_edit_50/seed_42_mean_results.json" \
    --out doc/results/results_positional_decay.png

Each --run is "Label=path". Output is a PNG plus a small markdown table next to it.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_curve(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tm = data.get("total_mean", data)
    pc = tm.get("position_curve")
    if pc is None:
        raise SystemExit(
            f"No 'position_curve' in {path}. Re-run eval after instrumenting "
            f"evaluation/vllm_editor_eval.py (Workstream A1/A2)."
        )
    return pc


def _parse_run(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise SystemExit(f"--run must be 'Label=path', got: {s}")
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=path to *_mean_results.json")
    ap.add_argument("--out", default="doc/results/results_positional_decay.png")
    args = ap.parse_args()

    runs: List[Tuple[str, str]] = [_parse_run(r) for r in args.run]

    plt.figure(figsize=(7, 4.5))
    table_rows = []
    for label, path in runs:
        pc = _load_curve(path)
        accs = pc["acc"]
        n_bins = pc.get("n_bins", len(accs))
        xs = [(i + 0.5) / n_bins for i in range(n_bins)]
        ys = [(a * 100.0) if a is not None else None for a in accs]
        xs_p = [x for x, y in zip(xs, ys) if y is not None]
        ys_p = [y for y in ys if y is not None]
        plt.plot(xs_p, ys_p, marker="o", label=label)
        first = next((y for y in ys if y is not None), None)
        last = next((y for y in reversed(ys) if y is not None), None)
        drop = (first - last) if (first is not None and last is not None) else None
        table_rows.append((label, first, last, drop))

    plt.xlabel("Normalized position within target (0 = start, 1 = end)")
    plt.ylabel("Token accuracy (%)")
    plt.title("Positional efficacy decay on VLKEB-long")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    md = args.out.rsplit(".", 1)[0] + "_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Positional efficacy decay summary\n\n")
        f.write("| Run | First-decile acc | Last-decile acc | Decay (pp) |\n")
        f.write("|-----|------------------|-----------------|------------|\n")
        for label, first, last, drop in table_rows:
            fs = f"{first:.1f}%" if first is not None else "-"
            ls = f"{last:.1f}%" if last is not None else "-"
            ds = f"{drop:.1f}" if drop is not None else "-"
            f.write(f"| {label} | {fs} | {ls} | {ds} |\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
