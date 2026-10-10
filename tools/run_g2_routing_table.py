"""G2: seed-42 routing-rule table on frozen checkpoints (VLKEB + EVQA).

Runs ``test_grounding_eval.py`` with ``--log_routing`` for each (dataset, ckpt, rule)
row, then scores leakage taxonomy into ``doc/results/cvpr2027/g2/``.

Default matrix (Plan G2 decision table):
  L0 ckpt  × {liveedit, vision_only, text_only, and, or, sum, carml, always_reject}
  DEK ckpt × {and, liveedit}   # AVBT+AND and AVBT with liveedit routing

Usage:
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g2_routing_table.py
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g2_routing_table.py --only vlkeb --rules and,carml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
if not PY.is_file():
    PY = Path(sys.executable)

L0_CKPT = (
    ROOT / "records/liveedit/blip2-opt-2.7b/"
    "VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201"
)
DEK_VLKEB_CKPT = (
    ROOT / "records/liveedit/blip2-opt-2.7b/"
    "VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100"
)
DEK_EVQA_CKPT = (
    ROOT / "records/liveedit/blip2-opt-2.7b/"
    "EVQA_avbt_DEK_gnd-2026.07.25-11.47.44/checkpoints/epoch-1-i-125-ema_loss-4.4475"
)
EVQA_L0_CKPT = (
    ROOT / "records/liveedit/blip2-opt-2.7b/"
    "EVQA_L0_full-2026.07.24-12.05.12/checkpoints/epoch-20-i-29400-ema_loss-0.1996"
)

VLKEB_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
VLKEB_IMG = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
EVQA_JSON = ROOT / "data/easy-edit-mm/grounding/grounding_eval_short.json"
EVQA_IMG = ROOT / "data/easy-edit-mm/images"

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
DEK_RULES = ("and", "liveedit")  # DEK full vs AVBT with liveedit routing


def _find_latest_run(enp: str) -> Path | None:
    root = ROOT / "eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b"
    hits = sorted(root.glob(f"VLKEB_GROUNDING-{enp}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def build_jobs(datasets, rules_filter):
    jobs = []
    for ds in datasets:
        if ds == "vlkeb":
            l0_ckpt, dek_ckpt = L0_CKPT, DEK_VLKEB_CKPT
            dpath, img, dsn = VLKEB_JSON, VLKEB_IMG, 100
            tag = "vlkeb"
        else:
            l0_ckpt, dek_ckpt = EVQA_L0_CKPT, DEK_EVQA_CKPT
            dpath, img, dsn = EVQA_JSON, EVQA_IMG, 300
            tag = "evqa"

        for rule in L0_RULES:
            if rules_filter and rule not in rules_filter:
                continue
            jobs.append({
                "name": f"{tag}_L0_{rule}",
                "enp": f"g2_{tag}_L0_{rule}_s42",
                "ckpt": l0_ckpt,
                "dpath": dpath,
                "img": img,
                "dsn": dsn,
                "rule": rule,
                "dek": False,
            })
        for rule in DEK_RULES:
            if rules_filter and rule not in rules_filter and f"dek_{rule}" not in rules_filter:
                # allow filtering "and" to still run DEK and via name
                if rule not in (rules_filter or []):
                    continue
            jobs.append({
                "name": f"{tag}_DEK_{rule}",
                "enp": f"g2_{tag}_DEK_{rule}_s42",
                "ckpt": dek_ckpt,
                "dpath": dpath,
                "img": img,
                "dsn": dsn,
                "rule": rule,
                "dek": True,
            })
    return jobs


def run_job(job, device, sen, seed, log_dir: Path) -> dict:
    enp = job["enp"]
    log_path = log_dir / f"{enp}.log"
    cmd = [
        str(PY), "-u", "-W", "ignore", "test_grounding_eval.py",
        "-en", "liveedit", "-mn", "blip2",
        "-sen", str(sen), "-dvc", device,
        "-dsn", str(job["dsn"]), "-seed", str(seed),
        "-ckpt", str(job["ckpt"]),
        "-enp", enp,
        "-dpath", str(job["dpath"]),
        "--img_root", str(job["img"]),
        "--meeting_step", "6",
        "--log_routing",
        "--routing_rule", job["rule"],
    ]
    if job["dek"]:
        cmd += ["--patch_grounding", "--dek_dual_gate"]
        # dual_gate is overridden by routing_rule when set; keep for patch module load path
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n===== {job['name']} {stamp} =====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"}
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
        "log": str(log_path).replace("\\", "/"),
    }


def score_jobs(results, out_dir: Path):
    """Score taxonomy for finished runs grouped by dataset."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.score_leakage_taxonomy import score_run

    by_ds = {"vlkeb": [], "evqa": []}
    for r in results:
        if r.get("exit_code") != 0 or not r.get("run_dir"):
            continue
        ds = "evqa" if "_evqa_" in r["enp"] or r["name"].startswith("evqa") else "vlkeb"
        label = r["name"]
        try:
            scored = score_run(
                r["run_dir"], label, None, (0.1,), False, 42, None, "auto", False, strict=False
            )
            by_ds[ds].append(scored)
            r["taxonomy"] = {
                "image_swap_total_leak": scored["image_swap"]["binary"]["total_leak"],
                "image_swap_stale": scored["image_swap"]["binary"]["stale_leak"],
                "image_swap_foreign": scored["image_swap"]["binary"]["foreign_leak"],
                "image_swap_correct_route": scored["image_swap"]["binary"]["correct_route"],
                "image_swap_abstain": scored["image_swap"]["binary"]["abstain"],
                "hard_negative_total_leak": scored["hard_negative"]["binary"]["total_leak"],
                "rel_hit": scored["reliability"]["binary"]["hit"],
                "rel_only_own": scored["reliability"]["binary"]["only_own"],
            }
        except Exception as e:
            r["taxonomy_error"] = repr(e)

    for ds, runs in by_ds.items():
        if not runs:
            continue
        path = out_dir / f"routing_table_{ds}_s42.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "seed": 42, "runs": runs}, f, indent=2)
        # compact md
        md = out_dir / f"routing_table_{ds}_s42.md"
        lines = [
            "| run | swap stale | swap foreign | swap total | correct_route | abstain | hn total | rel hit | only_own |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in runs:
            sw, hn, rl = s["image_swap"]["binary"], s["hard_negative"]["binary"], s["reliability"]["binary"]
            lines.append(
                "| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
                % (
                    s["label"],
                    100 * sw["stale_leak"], 100 * sw["foreign_leak"], 100 * sw["total_leak"],
                    100 * sw["correct_route"], 100 * sw["abstain"],
                    100 * hn["total_leak"], 100 * rl["hit"], 100 * rl["only_own"],
                )
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Wrote", path)
        print("Wrote", md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["vlkeb", "evqa", "both"], default="both")
    ap.add_argument("--rules", default=None, help="Comma-separated rule filter")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-sen", type=int, default=10)
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true", help="Skip if enp result dir already exists")
    args = ap.parse_args()

    datasets = ["vlkeb", "evqa"] if args.only == "both" else [args.only]
    rules_filter = [x.strip() for x in args.rules.split(",")] if args.rules else None

    # Fix DEK job filter when rules_filter is set: include DEK rows for listed rules
    jobs = build_jobs(datasets, None)
    if rules_filter:
        jobs = [j for j in jobs if j["rule"] in rules_filter]

    out_dir = ROOT / "doc/results/cvpr2027/g2"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "jobs_planned": [j["name"] for j in jobs],
        "results": [],
    }
    print("Planned %d jobs:" % len(jobs), [j["name"] for j in jobs], flush=True)

    for job in jobs:
        if args.skip_existing:
            existing = _find_latest_run(job["enp"])
            if existing and (existing / "routing_diagnostics.json").is_file():
                print("SKIP existing", job["enp"], existing)
                manifest["results"].append({
                    "name": job["name"], "enp": job["enp"], "rule": job["rule"],
                    "dek": job["dek"], "exit_code": 0, "skipped": True,
                    "run_dir": str(existing).replace("\\", "/"),
                })
                continue
        rec = run_job(job, args.dvc, args.sen, args.seed, log_dir)
        manifest["results"].append(rec)
        with (out_dir / "routing_table_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if rec["exit_code"] != 0:
            print("FAILED", job["name"], "exit", rec["exit_code"], flush=True)
            # continue to next job rather than aborting the whole matrix

    print("\nScoring taxonomy…", flush=True)
    score_jobs(manifest["results"], out_dir)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    with (out_dir / "routing_table_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_fail = sum(1 for r in manifest["results"] if r.get("exit_code", 1) != 0 and not r.get("skipped"))
    print("DONE. failures=%d / %d" % (n_fail, len(manifest["results"])))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
