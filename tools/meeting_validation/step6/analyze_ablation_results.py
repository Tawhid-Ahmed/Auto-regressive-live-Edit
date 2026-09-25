"""Analyze Step 6 layer ablation mean_results.json files.

Usage:
  python tools/meeting_validation/step6/analyze_ablation_results.py \\
    -manifest doc/results/meeting_validation/step6/step6_run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.analyze_grounding_results import analyze_mean
from tools.meeting_validation.step6.layer_ablation import LAYER_TABLE, LAYER_NAMES


def _row_from_analysis(layer: str, analysis: Dict[str, Any], table_row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "layer": layer,
        "component": table_row["component"],
        "motivation": table_row["motivation"],
        "primary_metric": table_row["primary_metric"],
        "reliability_acc": analysis.get("reliability_acc"),
        "hard_negative_acc": analysis.get("probe_acc", {}).get("hard_negative"),
        "swap_gap_pp": analysis.get("swap_gap_pp"),
        "modality_consistency_acc": analysis.get("modality_consistency_acc"),
        "locality_acc": analysis.get("locality_acc"),
        "path": analysis.get("path"),
    }


def build_summary(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    table_by_layer = {r["layer"]: r for r in LAYER_TABLE}
    rows: List[Dict[str, Any]] = []
    for layer in LAYER_NAMES:
        run = manifest.get("runs", {}).get(layer)
        if not run or not run.get("mean_results"):
            continue
        path = run["mean_results"]
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        analysis = analyze_mean(path, layer)
        mean_raw = json.load(open(path, encoding="utf-8"))
        total = mean_raw.get("total_mean", mean_raw)
        loc = total.get("locality", {})
        locality_vals = [v.get("acc") for v in loc.values() if isinstance(v, dict) and v.get("acc") is not None]
        analysis["locality_acc"] = min(locality_vals) if locality_vals else None
        rows.append(_row_from_analysis(layer, analysis, table_by_layer[layer]))

    out = {
        "manifest": manifest_path,
        "layers": rows,
        "motivation_table": LAYER_TABLE,
    }
    return out


def write_markdown_table(summary: Dict[str, Any], out_path: str) -> None:
    lines = [
        "# Step 6 — Layer ablation (motivation → impact)",
        "",
        "| Layer | Component | Motivation | Rel acc | Hard-neg acc | Swap gap (pp) | Locality |",
        "|-------|-----------|------------|--------:|-------------:|--------------:|---------:|",
    ]
    for r in summary.get("layers", []):
        rel = r.get("reliability_acc")
        hn = r.get("hard_negative_acc")
        sg = r.get("swap_gap_pp")
        loc = r.get("locality_acc")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                r["layer"],
                r["component"],
                r["motivation"][:48] + ("…" if len(r["motivation"]) > 48 else ""),
                "%.1f%%" % (rel * 100) if rel is not None else "—",
                "%.1f%%" % (hn * 100) if hn is not None else "—",
                "%.1f" % sg if sg is not None else "—",
                "%.1f%%" % (loc * 100) if loc is not None else "—",
            )
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-manifest",
        type=str,
        default="doc/results/meeting_validation/step6/step6_run_manifest.json",
    )
    ap.add_argument(
        "-out",
        type=str,
        default="doc/results/meeting_validation/step6/step6_ablation_summary.json",
    )
    args = ap.parse_args()
    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(ROOT, manifest_path)
    if not os.path.isfile(manifest_path):
        print("ERROR: manifest not found:", manifest_path)
        return 1
    summary = build_summary(manifest_path)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md_path = os.path.join(os.path.dirname(out_path), "step6_ablation_table.md")
    write_markdown_table(summary, md_path)
    print("Wrote", out_path)
    print("Wrote", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
