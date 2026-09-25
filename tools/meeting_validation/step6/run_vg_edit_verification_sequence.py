"""Run VG-Edit verification pipeline sequentially (memory-safe, one GPU job at a time).

Steps:
  1. Unit smoke test (no GPU)
  2. Routing direction check (dsn=25, sen=5)
  3. If direction PASS: grounding eval L0 baseline (dsn=100, sen=10)
  4. If direction PASS: grounding eval VG-Edit gate (same protocol)
  5. Write combined analysis JSON + markdown

Usage:
  conda activate liveedit
  python tools/meeting_validation/step6/run_vg_edit_verification_sequence.py -dvc cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

DEFAULT_CKPT = (
    "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/"
    "checkpoints/epoch-20-i-27800-ema_loss-0.7936"
)


def _run(cmd: list, step: str) -> int:
    print("\n" + "=" * 72)
    print("STEP:", step)
    print("CMD:", " ".join(cmd))
    print("=" * 72)
    rc = subprocess.call(cmd, cwd=ROOT_PATH)
    print("[exit %d] %s" % (rc, step))
    _gpu_cleanup()
    return rc


def _gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _analyze_mean(path: str, label: str) -> Dict[str, Any]:
    from tools.analyze_grounding_results import analyze_mean

    return analyze_mean(path, label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-dvc", default="cuda:0")
    parser.add_argument("-ckpt", default=DEFAULT_CKPT)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--direction-dsn", type=int, default=25)
    parser.add_argument("--direction-sen", type=int, default=5)
    parser.add_argument("--eval-dsn", type=int, default=100)
    parser.add_argument("--eval-sen", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--skip-eval", action="store_true", help="Only smoke + direction check")
    args = parser.parse_args()

    py = sys.executable
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_PATH, "doc/results/meeting_validation/step6")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "vg_edit_verification_%s.json" % ts)
    md_path = os.path.join(out_dir, "VG_EDIT_VERIFICATION_ANALYSIS.md")

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt,
        "seed": args.seed,
        "direction": {"dsn": args.direction_dsn, "sen": args.direction_sen, "margin": args.margin},
        "eval": {"dsn": args.eval_dsn, "sen": args.eval_sen},
        "steps": [],
    }

    # 1 — smoke
    rc = _run(
        [py, "tools/meeting_validation/step6/smoke_test_decode_vision_gate.py"],
        "smoke_test_decode_vision_gate",
    )
    report["steps"].append({"name": "smoke", "exit_code": rc})
    if rc != 0:
        _write_report(report, report_path, md_path, failed="smoke test")
        sys.exit(rc)

    # 2 — direction verify
    direction_json = os.path.join(out_dir, "vg_edit_direction_check_%s.json" % ts)
    rc = _run(
        [
            py,
            "tools/meeting_validation/step6/verify_vg_edit_direction.py",
            "-ckpt", args.ckpt,
            "-dvc", args.dvc,
            "-dsn", str(args.direction_dsn),
            "-sen", str(args.direction_sen),
            "-seed", str(args.seed),
            "--margin", str(args.margin),
            "-out", direction_json,
        ],
        "verify_vg_edit_direction",
    )
    direction_passed = False
    direction_data: Optional[Dict[str, Any]] = None
    if os.path.exists(direction_json):
        with open(direction_json, encoding="utf-8") as f:
            direction_data = json.load(f)
        direction_passed = bool(direction_data.get("direction_check_passed"))
    report["direction_check"] = direction_data
    report["steps"].append({"name": "direction", "exit_code": rc, "passed": direction_passed})
    if rc != 0 and not direction_passed:
        _write_report(report, report_path, md_path, failed="direction check")
        sys.exit(rc)

    if args.skip_eval or not direction_passed:
        note = "skipped eval (flag or direction fail)" if not direction_passed else "skipped eval (--skip-eval)"
        report["note"] = note
        _write_report(report, report_path, md_path, failed=None if direction_passed else "direction check")
        sys.exit(0 if direction_passed else 1)

    # 3 — L0 grounding eval (separate process → fresh GPU memory)
    enp_l0 = "vg_verify_l0_dsn%d_sen%d" % (args.eval_dsn, args.eval_sen)
    rc_l0 = _run(
        [
            py, "-W", "ignore", "test_grounding_eval.py",
            "-en", "liveedit", "-mn", "blip2",
            "-sen", str(args.eval_sen),
            "-dvc", args.dvc,
            "-dsn", str(args.eval_dsn),
            "-seed", str(args.seed),
            "-ckpt", args.ckpt,
            "-enp", enp_l0,
            "--meeting_step", "6",
        ],
        "grounding_eval_L0_baseline",
    )
    report["steps"].append({"name": "eval_l0", "exit_code": rc_l0, "enp": enp_l0})

    # 4 — VG grounding eval
    enp_vg = "vg_verify_gate_dsn%d_sen%d" % (args.eval_dsn, args.eval_sen)
    rc_vg = _run(
        [
            py, "-W", "ignore", "test_grounding_eval.py",
            "-en", "liveedit", "-mn", "blip2",
            "-sen", str(args.eval_sen),
            "-dvc", args.dvc,
            "-dsn", str(args.eval_dsn),
            "-seed", str(args.seed),
            "-ckpt", args.ckpt,
            "-enp", enp_vg,
            "--meeting_step", "6",
            "--decode_vision_gate",
            "--decode_vision_margin", str(args.margin),
        ],
        "grounding_eval_VG_gate",
    )
    report["steps"].append({"name": "eval_vg", "exit_code": rc_vg, "enp": enp_vg})

    # 5 — locate mean results + analyze
    def _find_mean(enp: str) -> Optional[str]:
        import glob

        pat = os.path.join(
            ROOT_PATH,
            "eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b",
            "VLKEB_GROUNDING-%s-*" % enp,
            "sequential_edit_%s" % args.eval_sen,
            "seed_%s_mean_results.json" % args.seed,
        )
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        return hits[0] if hits else None

    l0_path = _find_mean(enp_l0)
    vg_path = _find_mean(enp_vg)
    report["eval_paths"] = {"l0": l0_path, "vg": vg_path}
    if l0_path and vg_path:
        l0_a = _analyze_mean(l0_path, "L0_baseline")
        vg_a = _analyze_mean(vg_path, "VG_Edit_gate")
        report["eval_analysis"] = {"l0": l0_a, "vg": vg_a}
        report["deltas_pp"] = _deltas(l0_a, vg_a)

    _write_report(report, report_path, md_path, failed=None)
    print("\nWrote", report_path)
    print("Wrote", md_path)
    sys.exit(0 if rc_l0 == 0 and rc_vg == 0 else 1)


def _deltas(l0: Dict[str, Any], vg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("reliability_acc", "swap_gap_pp"):
        a, b = l0.get(key), vg.get(key)
        if a is not None and b is not None:
            out[key] = (b - a) * (100 if key == "reliability_acc" else 1)
    for probe in ("image_swap", "hard_negative"):
        a = (l0.get("probe_acc") or {}).get(probe)
        b = (vg.get("probe_acc") or {}).get(probe)
        if a is not None and b is not None:
            out[probe + "_acc_pp"] = (b - a) * 100
    return out


def _write_report(
    report: Dict[str, Any],
    json_path: str,
    md_path: str,
    failed: Optional[str],
) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# VG-Edit Verification Analysis",
        "",
        "**Date:** %s" % report.get("timestamp", ""),
        "**Checkpoint:** `%s`" % report.get("ckpt", ""),
        "",
    ]
    if failed:
        lines.append("**Status:** FAILED at `%s`" % failed)
    elif report.get("direction_check", {}).get("direction_check_passed"):
        lines.append("**Status:** Direction check PASSED")
    else:
        lines.append("**Status:** Direction check did not pass thresholds")

    dc = report.get("direction_check") or {}
    if dc:
        b, g = dc.get("baseline", {}), dc.get("vg_edit", {})
        lines += [
            "",
            "## Step 2 — Routing direction check (small scale)",
            "",
            "| Metric | Baseline | VG-Edit |",
            "|--------|----------|---------|",
        ]
        for k, label in [
            ("hard_neg_false_fire", "Hard-neg false-fire"),
            ("swap_false_fire", "Swap false-fire"),
            ("rel_false_cross", "Rel false-cross"),
            ("rel_miss", "Rel miss"),
        ]:
            bv = b.get(k)
            gv = g.get(k)
            if bv is not None and gv is not None:
                lines.append("| %s | %.1f%% | %.1f%% |" % (label, bv * 100, gv * 100))

    ea = report.get("eval_analysis")
    if ea:
        l0, vg = ea.get("l0", {}), ea.get("vg", {})
        lines += [
            "",
            "## Step 3–4 — Grounding eval (token accuracy)",
            "",
            "| Metric | L0 | VG-Edit | Δ (VG−L0) |",
            "|--------|-----|---------|-----------|",
        ]
        d = report.get("deltas_pp") or {}
        rows = [
            ("Reliability", "reliability_acc", "reliability_acc", True),
            ("Image swap", "probe_acc.image_swap", None, True),
            ("Hard-negative", "probe_acc.hard_negative", None, True),
            ("Swap-gap (pp)", "swap_gap_pp", "swap_gap_pp", False),
        ]
        for label, l0k, dk, is_pct in rows:
            if l0k == "probe_acc.image_swap":
                lv = (l0.get("probe_acc") or {}).get("image_swap")
                vv = (vg.get("probe_acc") or {}).get("image_swap")
                delta = d.get("image_swap_acc_pp")
            elif l0k == "probe_acc.hard_negative":
                lv = (l0.get("probe_acc") or {}).get("hard_negative")
                vv = (vg.get("probe_acc") or {}).get("hard_negative")
                delta = d.get("hard_negative_acc_pp")
            else:
                lv, vv = l0.get(l0k), vg.get(l0k)
                delta = d.get(dk) if dk else None
            if lv is None or vv is None:
                continue
            if is_pct and l0k != "swap_gap_pp":
                lines.append(
                    "| %s | %.2f%% | %.2f%% | %+.2f pp |"
                    % (label, lv * 100, vv * 100, (delta or 0))
                )
            else:
                lines.append(
                    "| %s | %.2f | %.2f | %+.2f |"
                    % (label, lv, vv, (delta or 0))
                )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Direction check** uses routing probes only (fast); full eval uses teacher-forced token accuracy.",
        "- **Proceed to dsn=500 / multi-seed** only if swap-gap drops meaningfully with reliability cost < 5 pp.",
        "",
        "Raw JSON: `%s`" % os.path.relpath(json_path, ROOT_PATH),
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
