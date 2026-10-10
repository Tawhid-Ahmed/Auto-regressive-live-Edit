"""G3: pool-size curve on pair-closed VLKEB pools (M5).

Compares L0_liveedit vs DEK_and at sen ∈ {100, 300, 1000} (target sizes;
actual closed set may be slightly larger). Seed 42, one edit order.

Builds missing pair-closed JSONs, then runs test_grounding_eval.py with
--log_routing, scores leakage taxonomy into doc/results/cvpr2027/g3/.

Default job order: 1000 first (slowest), then 300, then 100.

Usage:
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g3_pool_size_curve.py
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g3_pool_size_curve.py --sizes 1000
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g3_pool_size_curve.py --smoke
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
DEK_CKPT = (
    ROOT / "records/liveedit/blip2-opt-2.7b/"
    "VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100"
)
SRC_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
IMG = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
POOL_DIR = ROOT / "data/VLKEB/meeting_validation/step1/grounding/pair_closed"

SIZES = (1000, 300, 100)  # run order: largest first


def pool_path(n: int, seed: int) -> Path:
    return POOL_DIR / f"grounding_pairclosed_n{n}_s{seed}.json"


def ensure_pool(n: int, seed: int) -> dict:
    out = pool_path(n, seed)
    if out.is_file():
        with out.open(encoding="utf-8") as f:
            meta = (json.load(f).get("meta") or {})
        return {
            "path": out,
            "target_n": n,
            "actual_n": int(meta.get("pair_closed_actual_n") or n),
            "built": False,
            "verify": meta.get("verify"),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), "-u", str(ROOT / "tools/build_pair_closed_pool.py"),
        "-dpath", str(SRC_JSON),
        "-n", str(n),
        "-seed", str(seed),
        "-out", str(out),
    ]
    print("Building pair-closed pool:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError("pair-closed build failed for n=%d" % n)
    with out.open(encoding="utf-8") as f:
        meta = (json.load(f).get("meta") or {})
    return {
        "path": out,
        "target_n": n,
        "actual_n": int(meta.get("pair_closed_actual_n") or n),
        "built": True,
        "verify": meta.get("verify"),
    }


def _find_latest_run(enp: str) -> Path | None:
    root = ROOT / "eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b"
    hits = sorted(root.glob(f"VLKEB_GROUNDING-{enp}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def build_jobs(sizes, seed: int, include_natural: bool):
    jobs = []
    for n in sizes:
        pool = ensure_pool(n, seed)
        actual = pool["actual_n"]
        dpath = pool["path"]
        for method, ckpt, rule, dek in (
            ("L0", L0_CKPT, "liveedit", False),
            ("DEK", DEK_CKPT, "and", True),
        ):
            jobs.append({
                "name": f"vlkeb_{method}_n{n}",
                "enp": f"g3_vlkeb_{method}_pc_n{n}_s{seed}",
                "ckpt": ckpt,
                "dpath": dpath,
                "dsn": actual,
                "sen": actual,
                "rule": rule,
                "dek": dek,
                "target_n": n,
                "actual_n": actual,
                "pair_closed": True,
            })
    if include_natural:
        # Supplementary: natural partner-absent fraction at n≈100 (head/subsample of full JSON).
        for method, ckpt, rule, dek in (
            ("L0", L0_CKPT, "liveedit", False),
            ("DEK", DEK_CKPT, "and", True),
        ):
            jobs.append({
                "name": f"vlkeb_{method}_natural_n100",
                "enp": f"g3_vlkeb_{method}_nat_n100_s{seed}",
                "ckpt": ckpt,
                "dpath": SRC_JSON,
                "dsn": 100,
                "sen": 100,
                "rule": rule,
                "dek": dek,
                "target_n": 100,
                "actual_n": 100,
                "pair_closed": False,
            })
    return jobs


def run_job(job, device, seed, log_dir: Path) -> dict:
    enp = job["enp"]
    log_path = log_dir / f"{enp}.log"
    cmd = [
        str(PY), "-u", "-W", "ignore", "test_grounding_eval.py",
        "-en", "liveedit", "-mn", "blip2",
        "-sen", str(job["sen"]), "-dvc", device,
        "-dsn", str(job["dsn"]), "-seed", str(seed),
        "-ckpt", str(job["ckpt"]),
        "-enp", enp,
        "-dpath", str(job["dpath"]),
        "--img_root", str(IMG),
        "--meeting_step", "6",
        "--log_routing",
        "--routing_rule", job["rule"],
    ]
    if job["dek"]:
        cmd += ["--patch_grounding", "--dek_dual_gate"]
    # DEK at sen=1000 OOMs during the second sequential-eval pool build on 16GB;
    # routing taxonomy is what G3 scores, so use routing_only for large DEK jobs.
    if job.get("routing_only") or (job["dek"] and int(job["actual_n"]) >= 800):
        cmd += ["--routing_only"]
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n===== {job['name']} {stamp} =====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        # Prefer PYTORCH_ALLOC_CONF (PyTorch ≥2.x); keep legacy alias if already set.
        "PYTORCH_ALLOC_CONF": os.environ.get(
            "PYTORCH_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ),
    }
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.write("ENV PYTORCH_ALLOC_CONF=%s\n" % env["PYTORCH_ALLOC_CONF"])
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
        "target_n": job["target_n"],
        "actual_n": job["actual_n"],
        "pair_closed": job["pair_closed"],
        "exit_code": proc.returncode,
        "runtime_sec": round(elapsed, 1),
        "run_dir": str(run_dir).replace("\\", "/") if run_dir else None,
        "log": str(log_path).replace("\\", "/"),
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
            sw, hn, rl = s["image_swap"]["binary"], s["hard_negative"]["binary"], s["reliability"]["binary"]
            r["taxonomy"] = {
                "image_swap_total_leak": sw["total_leak"],
                "image_swap_stale": sw["stale_leak"],
                "image_swap_foreign": sw["foreign_leak"],
                "image_swap_correct_route": sw["correct_route"],
                "image_swap_abstain": sw["abstain"],
                "hard_negative_total_leak": hn["total_leak"],
                "rel_hit": rl["hit"],
                "rel_only_own": rl["only_own"],
                "swap_partner_in_pool_rate": s.get("swap_partner_in_pool_rate"),
                "hard_neg_partner_in_pool_rate": s.get("hard_neg_partner_in_pool_rate"),
            }
            scored.append({**s, "g3_meta": {
                "target_n": r["target_n"], "actual_n": r["actual_n"],
                "pair_closed": r["pair_closed"], "name": r["name"],
            }})
        except Exception as e:
            r["taxonomy_error"] = repr(e)

    path = out_dir / f"pool_size_curve_vlkeb_s{seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"dataset": "vlkeb", "seed": seed, "runs": scored, "job_results": results}, f, indent=2)

    md = out_dir / f"pool_size_curve_vlkeb_s{seed}.md"
    lines = [
        "| run | target_n | actual_n | closed | swap stale | swap foreign | swap total | correct_route | abstain | hn total | rel hit | swap partner in pool | hn partner in pool |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.get("taxonomy") or {}
        if not t:
            continue
        lines.append(
            "| %s | %s | %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (
                r["name"],
                r.get("target_n"),
                r.get("actual_n"),
                "Y" if r.get("pair_closed") else "N",
                100 * t.get("image_swap_stale", float("nan")),
                100 * t.get("image_swap_foreign", float("nan")),
                100 * t.get("image_swap_total_leak", float("nan")),
                100 * t.get("image_swap_correct_route", float("nan")),
                100 * t.get("image_swap_abstain", float("nan")),
                100 * t.get("hard_negative_total_leak", float("nan")),
                100 * t.get("rel_hit", float("nan")),
                100 * (t.get("swap_partner_in_pool_rate") or 0),
                100 * (t.get("hard_neg_partner_in_pool_rate") or 0),
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", path)
    print("Wrote", md)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description="G3 pool-size curve (M5 pair-closed)")
    ap.add_argument("--sizes", default=None, help="Comma-separated targets, default 1000,300,100")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Build n=20 pool and run L0+DEK only")
    ap.add_argument("--natural", action="store_true", help="Also run natural-fraction n=100 side jobs")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--build_only", action="store_true", help="Only build pair-closed pools")
    args = ap.parse_args()

    if args.smoke:
        sizes = [20]
    elif args.sizes:
        sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    else:
        sizes = list(SIZES)

    out_dir = ROOT / "doc/results/cvpr2027/g3"
    if args.smoke:
        out_dir = out_dir / "smoke"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build pools
    pools = {}
    for n in sizes:
        pools[n] = ensure_pool(n, args.seed)
        v = pools[n].get("verify") or {}
        print(
            "pool n=%d actual=%d swap_in=%.1f%% hn_in=%.1f%%"
            % (
                n,
                pools[n]["actual_n"],
                100 * (v.get("partner_in_pool_swap_rate") or 0),
                100 * (v.get("partner_in_pool_hard_neg_rate") or 0),
            ),
            flush=True,
        )
    if args.build_only:
        return 0

    jobs = build_jobs(sizes, args.seed, include_natural=args.natural and not args.smoke)
    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "smoke": bool(args.smoke),
        "seed": args.seed,
        "pools": {str(k): {
            "path": str(v["path"]).replace("\\", "/"),
            "actual_n": v["actual_n"],
            "verify": v.get("verify"),
        } for k, v in pools.items()},
        "jobs_planned": [j["name"] for j in jobs],
        "results": [],
    }
    print("Planned %d jobs:" % len(jobs), [j["name"] for j in jobs], flush=True)

    for job in jobs:
        if args.skip_existing:
            existing = _find_latest_run(job["enp"])
            if existing and (existing / "routing_diagnostics.json").is_file():
                print("SKIP existing", job["enp"], existing, flush=True)
                manifest["results"].append({
                    "name": job["name"], "enp": job["enp"], "rule": job["rule"],
                    "dek": job["dek"], "target_n": job["target_n"],
                    "actual_n": job["actual_n"], "pair_closed": job["pair_closed"],
                    "exit_code": 0, "skipped": True,
                    "run_dir": str(existing).replace("\\", "/"),
                })
                continue
        rec = run_job(job, args.dvc, args.seed, log_dir)
        manifest["results"].append(rec)
        with (out_dir / "pool_size_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if rec["exit_code"] != 0:
            print("FAILED", job["name"], "exit", rec["exit_code"], flush=True)

    print("\nScoring taxonomy…", flush=True)
    score_jobs(manifest["results"], out_dir, args.seed)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    with (out_dir / "pool_size_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_fail = sum(1 for r in manifest["results"] if r.get("exit_code", 1) != 0 and not r.get("skipped"))
    print("DONE. failures=%d / %d" % (n_fail, len(manifest["results"])))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
