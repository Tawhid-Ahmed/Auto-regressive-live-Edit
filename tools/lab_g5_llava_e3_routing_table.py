"""G5 E3: LLaVA VLKEB routing-rule control table (G2 matrix, lab ckpts).

L0 × {liveedit, vision_only, text_only, and, or, sum, carml, always_reject}
DEK × {and, liveedit}
dsn=100, sen=10, seed=42, --routing_only (taxonomy-focused; matches G2 scoring).

Usage (lab):
  nohup python -u tools/lab_g5_llava_e3_routing_table.py \\
    > doc/results/cvpr2027/g5/logs/e3_routing_outer.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)

L0_CKPT = (
    ROOT / "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/"
    "checkpoints/epoch-20-i-10200-ema_loss-2.2061"
)
DEK_CKPT = (
    ROOT / "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06/"
    "checkpoints/epoch-1-i-500-ema_loss-3.8851"
)
VLKEB_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
VLKEB_IMG = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
OUT_DIR = ROOT / "doc/results/cvpr2027/g5"

L0_RULES = (
    "liveedit",
    "vision_only",
    "text_only",
    "and",
    "or",
    "sum",
    "carml",
    "always_reject",
)
DEK_RULES = ("and", "liveedit")


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


def build_jobs(rules_filter):
    jobs = []
    for rule in L0_RULES:
        if rules_filter and rule not in rules_filter:
            continue
        jobs.append({
            "name": f"llava_L0_{rule}",
            "enp": f"g5_llava_L0_{rule}_s42",
            "ckpt": L0_CKPT,
            "rule": rule,
            "dek": False,
        })
    for rule in DEK_RULES:
        if rules_filter and rule not in rules_filter and f"dek_{rule}" not in rules_filter:
            if rules_filter and rule not in rules_filter:
                continue
        jobs.append({
            "name": f"llava_DEK_{rule}",
            "enp": f"g5_llava_DEK_{rule}_s42",
            "ckpt": DEK_CKPT,
            "rule": rule,
            "dek": True,
        })
    return jobs


def run_job(job, device, sen, dsn, seed, log_dir: Path) -> dict:
    enp = job["enp"]
    log_path = log_dir / f"{enp}.log"
    cmd = [
        str(PY), "-u", "-W", "ignore", "test_grounding_eval.py",
        "-en", "liveedit", "-mn", "llava",
        "-sen", str(sen), "-dvc", device,
        "-dsn", str(dsn), "-seed", str(seed),
        "-ckpt", str(job["ckpt"]),
        "-enp", enp,
        "-dpath", str(VLKEB_JSON),
        "--img_root", str(VLKEB_IMG),
        "--meeting_step", "6",
        "--log_routing",
        "--routing_only",
        "--routing_rule", job["rule"],
    ]
    if job["dek"]:
        cmd += ["--patch_grounding", "--dek_dual_gate"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n===== {job['name']} {stamp} =====", flush=True)
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
        "name": job["name"],
        "enp": enp,
        "rule": job["rule"],
        "dek": job["dek"],
        "exit_code": proc.returncode,
        "runtime_sec": round(elapsed, 1),
        "run_dir": str(run_dir).replace("\\", "/") if run_dir else None,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }


def score_jobs(results, out_dir: Path, seed: int):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.score_leakage_taxonomy import score_run

    scored = []
    for r in results:
        if r.get("exit_code") != 0 or not r.get("run_dir"):
            continue
        try:
            s = score_run(
                r["run_dir"], r["name"], None, (0.1,), False, seed, None, "auto", False, strict=False
            )
            sw = s["image_swap"]["binary"]
            hn = s["hard_negative"]["binary"]
            rl = s["reliability"]["binary"]
            r["taxonomy"] = {
                "image_swap_total_leak": sw["total_leak"],
                "image_swap_stale": sw["stale_leak"],
                "image_swap_foreign": sw["foreign_leak"],
                "image_swap_correct_route": sw["correct_route"],
                "image_swap_abstain": sw["abstain"],
                "hard_negative_total_leak": hn["total_leak"],
                "rel_hit": rl["hit"],
                "rel_only_own": rl["only_own"],
            }
            scored.append({**s, "e3_meta": {"name": r["name"], "rule": r["rule"], "dek": r["dek"]}})
        except Exception as e:
            r["taxonomy_error"] = repr(e)

    path = out_dir / f"routing_table_llava_vlkeb_s{seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"backbone": "llava-v1.5-7b", "dataset": "vlkeb", "seed": seed,
             "runs": scored, "job_results": results},
            f, indent=2,
        )

    md = out_dir / f"routing_table_llava_vlkeb_s{seed}.md"
    lines = [
        "| run | rule | dek | swap stale | swap foreign | swap total | correct_route | abstain | hn total | rel hit | only-own |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.get("taxonomy") or {}
        if not t:
            continue
        lines.append(
            "| %s | %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (
                r["name"],
                r["rule"],
                "Y" if r["dek"] else "N",
                100 * t.get("image_swap_stale", float("nan")),
                100 * t.get("image_swap_foreign", float("nan")),
                100 * t.get("image_swap_total_leak", float("nan")),
                100 * t.get("image_swap_correct_route", float("nan")),
                100 * t.get("image_swap_abstain", float("nan")),
                100 * t.get("hard_negative_total_leak", float("nan")),
                100 * t.get("rel_hit", float("nan")),
                100 * t.get("rel_only_own", float("nan")),
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", path)
    print("Wrote", md)


def main() -> int:
    ap = argparse.ArgumentParser(description="G5 E3 LLaVA routing-rule table")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("-dsn", type=int, default=100)
    ap.add_argument("-sen", type=int, default=10)
    ap.add_argument("--rules", default=None, help="Comma-separated subset of rules")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    for p in (L0_CKPT, DEK_CKPT, VLKEB_JSON, VLKEB_IMG):
        if not Path(p).exists():
            print("FAIL: missing", p)
            return 1

    rules_filter = None
    if args.rules:
        rules_filter = {x.strip() for x in args.rules.split(",") if x.strip()}

    log_dir = OUT_DIR / "logs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(rules_filter)
    started = datetime.now(timezone.utc)
    # ~4 min/job routing_only LLaVA (load+routing); 10 jobs → ~40 min
    eta_min = max(5, int(len(jobs) * 4))
    from datetime import timedelta

    eta_utc = (started + timedelta(minutes=eta_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        "E3 START utc=%s jobs=%d eta_min~%d eta_utc~%s"
        % (started.strftime("%Y-%m-%dT%H:%M:%SZ"), len(jobs), eta_min, eta_utc),
        flush=True,
    )
    print("Planned:", [j["name"] for j in jobs], flush=True)

    manifest = {
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eta_min_estimate": eta_min,
        "eta_utc": eta_utc,
        "jobs_planned": [j["name"] for j in jobs],
        "results": [],
    }

    for job in jobs:
        if args.skip_existing:
            existing = _find_latest_run(job["enp"])
            if existing and (existing / "routing_diagnostics.json").is_file():
                print("SKIP", job["enp"], existing, flush=True)
                manifest["results"].append({
                    "name": job["name"], "enp": job["enp"], "rule": job["rule"],
                    "dek": job["dek"], "exit_code": 0, "skipped": True,
                    "run_dir": str(existing).replace("\\", "/"),
                })
                continue
        rec = run_job(job, args.dvc, args.sen, args.dsn, args.seed, log_dir)
        manifest["results"].append(rec)
        with (OUT_DIR / "routing_table_manifest_llava.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if rec["exit_code"] != 0:
            print("FAILED", job["name"], "exit", rec["exit_code"], flush=True)

    print("\nScoring taxonomy…", flush=True)
    score_jobs(manifest["results"], OUT_DIR, args.seed)
    finished = datetime.now(timezone.utc)
    manifest["finished"] = finished.strftime("%Y-%m-%dT%H:%M:%SZ")
    with (OUT_DIR / "routing_table_manifest_llava.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_fail = sum(
        1 for r in manifest["results"]
        if r.get("exit_code", 1) != 0 and not r.get("skipped")
    )
    print("E3 DONE. failures=%d / %d  finished_utc=%s" % (
        n_fail, len(manifest["results"]), manifest["finished"],
    ))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
