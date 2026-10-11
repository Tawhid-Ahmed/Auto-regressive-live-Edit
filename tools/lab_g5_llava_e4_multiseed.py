"""G5 E4: LLaVA VLKEB multi-seed leak taxonomy (seeds 42,0,1,2,3).

L0 / S2 / DEK × 5 seeds, dsn=100, sen=10, --routing_only.
Writes mean±std under doc/results/cvpr2027/g5/.

Usage (lab):
  nohup python -u tools/lab_g5_llava_e4_multiseed.py \\
    > doc/results/cvpr2027/g5/logs/e4_multiseed_outer.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)

L0_CKPT = (
    ROOT / "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/"
    "checkpoints/epoch-20-i-10200-ema_loss-2.2061"
)
S2_CKPT = (
    ROOT / "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_S2_dn500_e1_s42_20261011_104554-2026.10.11-10.45.58/"
    "checkpoints/epoch-1-i-500-ema_loss-3.8904"
)
DEK_CKPT = (
    ROOT / "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06/"
    "checkpoints/epoch-1-i-500-ema_loss-3.8851"
)
VLKEB_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
VLKEB_IMG = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
OUT_DIR = ROOT / "doc/results/cvpr2027/g5"
DEFAULT_SEEDS = (42, 0, 1, 2, 3)

METHODS = {
    "L0": {"ckpt": L0_CKPT, "extra": []},
    "S2": {"ckpt": S2_CKPT, "extra": ["--patch_grounding"]},
    "DEK": {"ckpt": DEK_CKPT, "extra": ["--patch_grounding", "--dek_dual_gate"]},
}


def _find_latest_run(enp: str) -> Path | None:
    root = ROOT / "eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b"
    if not root.is_dir():
        return None
    hits = sorted(
        root.glob(f"VLKEB_GROUNDING-{enp}-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def run_job(method: str, seed: int, dsn: int, sen: int, device: str, log_dir: Path) -> dict:
    cfg = METHODS[method]
    enp = f"g5_llava_ms_{method}_s{seed}_d{dsn}"
    log_path = log_dir / f"{enp}.log"
    cmd = [
        str(PY), "-u", "-W", "ignore", "test_grounding_eval.py",
        "-en", "liveedit", "-mn", "llava",
        "-sen", str(sen), "-dvc", device,
        "-dsn", str(dsn), "-seed", str(seed),
        "-ckpt", str(cfg["ckpt"]),
        "-enp", enp,
        "-dpath", str(VLKEB_JSON),
        "--img_root", str(VLKEB_IMG),
        "--meeting_step", "6",
        "--log_routing",
        "--routing_only",
        *cfg["extra"],
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n===== {method} seed={seed} {stamp} =====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NVIDIA_TF32_OVERRIDE": "1",
    }
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    elapsed = time.time() - t0
    run_dir = _find_latest_run(enp)
    return {
        "method": method,
        "seed": seed,
        "enp": enp,
        "exit_code": proc.returncode,
        "runtime_sec": round(elapsed, 1),
        "run_dir": str(run_dir).replace("\\", "/") if run_dir else None,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }


def score_and_aggregate(results, out_dir: Path, seeds, dsn: int):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.score_leakage_taxonomy import score_run

    by_method = {m: [] for m in METHODS}
    for r in results:
        if r.get("exit_code") != 0 or not r.get("run_dir"):
            continue
        try:
            s = score_run(
                r["run_dir"], f"{r['method']}_s{r['seed']}", None,
                (0.1,), False, r["seed"], None, "auto", False, strict=False,
            )
            sw = s["image_swap"]["binary"]
            hn = s["hard_negative"]["binary"]
            rl = s["reliability"]["binary"]
            row = {
                "seed": r["seed"],
                "swap_total_leak": sw["total_leak"],
                "swap_stale": sw["stale_leak"],
                "swap_foreign": sw["foreign_leak"],
                "correct_route": sw["correct_route"],
                "abstain": sw["abstain"],
                "hn_total_leak": hn["total_leak"],
                "rel_hit": rl["hit"],
                "only_own": rl["only_own"],
            }
            r["taxonomy"] = row
            by_method[r["method"]].append(row)
        except Exception as e:
            r["taxonomy_error"] = repr(e)

    summary = {}
    for m, rows in by_method.items():
        if not rows:
            continue
        summary[m] = {}
        for key in (
            "swap_total_leak", "hn_total_leak", "rel_hit", "only_own",
            "correct_route", "abstain", "swap_stale", "swap_foreign",
        ):
            vals = np.array([x[key] for x in rows], dtype=float)
            summary[m][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "per_seed": {str(rows[i]["seed"]): float(vals[i]) for i in range(len(rows))},
            }

    path = out_dir / f"multiseed_llava_vlkeb_d{dsn}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": "llava-v1.5-7b",
                "dataset": "vlkeb",
                "dsn": dsn,
                "seeds": list(seeds),
                "summary": summary,
                "job_results": results,
            },
            f, indent=2,
        )

    md = out_dir / f"multiseed_llava_vlkeb_d{dsn}.md"
    lines = [
        f"# LLaVA VLKEB multi-seed (dsn={dsn}, seeds={list(seeds)})",
        "",
        "| method | swap TL mean±std ↓ | hn TL mean±std ↓ | Rel hit mean±std ↑ | only-own mean±std ↑ | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(
            "| %s | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f | %d |"
            % (
                m,
                100 * s["swap_total_leak"]["mean"],
                100 * s["swap_total_leak"]["std"],
                100 * s["hn_total_leak"]["mean"],
                100 * s["hn_total_leak"]["std"],
                100 * s["rel_hit"]["mean"],
                100 * s["rel_hit"]["std"],
                100 * s["only_own"]["mean"],
                100 * s["only_own"]["std"],
                s["swap_total_leak"]["n"],
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", path)
    print("Wrote", md)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="G5 E4 LLaVA multi-seed")
    ap.add_argument("--seeds", default="42,0,1,2,3")
    ap.add_argument("-dsn", type=int, default=100)
    ap.add_argument("-sen", type=int, default=10)
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for p in (L0_CKPT, S2_CKPT, DEK_CKPT, VLKEB_JSON, VLKEB_IMG):
        if not Path(p).exists():
            print("FAIL: missing", p)
            return 1

    log_dir = OUT_DIR / "logs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = len(seeds) * len(METHODS)
    eta_min = max(10, int(n_jobs * 2.5))  # routing_only ~2–3 min/job
    started = datetime.now(timezone.utc)
    eta_utc = (started + timedelta(minutes=eta_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        "E4 START utc=%s jobs=%d eta_min~%d eta_utc~%s"
        % (started.strftime("%Y-%m-%dT%H:%M:%SZ"), n_jobs, eta_min, eta_utc),
        flush=True,
    )

    manifest = {
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eta_min_estimate": eta_min,
        "eta_utc": eta_utc,
        "seeds": seeds,
        "dsn": args.dsn,
        "results": [],
    }

    for seed in seeds:
        for method in METHODS:
            enp = f"g5_llava_ms_{method}_s{seed}_d{args.dsn}"
            if args.skip_existing:
                existing = _find_latest_run(enp)
                if existing and (existing / "routing_diagnostics.json").is_file():
                    print("SKIP", enp, flush=True)
                    manifest["results"].append({
                        "method": method, "seed": seed, "enp": enp,
                        "exit_code": 0, "skipped": True,
                        "run_dir": str(existing).replace("\\", "/"),
                    })
                    continue
            rec = run_job(method, seed, args.dsn, args.sen, args.dvc, log_dir)
            manifest["results"].append(rec)
            with (OUT_DIR / "multiseed_manifest_llava.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            if rec["exit_code"] != 0:
                print("FAILED", method, "seed", seed, "exit", rec["exit_code"], flush=True)

    print("\nScoring + aggregate…", flush=True)
    score_and_aggregate(manifest["results"], OUT_DIR, seeds, args.dsn)
    finished = datetime.now(timezone.utc)
    manifest["finished"] = finished.strftime("%Y-%m-%dT%H:%M:%SZ")
    with (OUT_DIR / "multiseed_manifest_llava.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_fail = sum(
        1 for r in manifest["results"]
        if r.get("exit_code", 1) != 0 and not r.get("skipped")
    )
    print("E4 DONE. failures=%d / %d  finished_utc=%s" % (
        n_fail, len(manifest["results"]), manifest["finished"],
    ))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
