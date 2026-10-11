"""G5 E2: LLaVA pair-closed pool-size curve on lab (G3 protocol, LLaVA ckpts).

L0 (liveedit) vs DEK (--routing_rule and + --dek_dual_gate) at target_n in
{100, 300, 1000}. Seed 42. Writes taxonomy under doc/results/cvpr2027/g5/.

Usage (lab, conda liveedit):
  python -u tools/lab_g5_llava_e2_pair_closed.py
  python -u tools/lab_g5_llava_e2_pair_closed.py --sizes 100
  nohup python -u tools/lab_g5_llava_e2_pair_closed.py \\
    > doc/results/cvpr2027/g5/logs/e2_pair_closed_outer.log 2>&1 &
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
SRC_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
IMG = ROOT / "data/VLKEB/VLKEB_images/mmkb_images"
POOL_DIR = ROOT / "data/VLKEB/meeting_validation/step1/grounding/pair_closed"
OUT_DIR = ROOT / "doc/results/cvpr2027/g5"
# Smallest first so early failures surface before multi-hour n=1000 jobs.
SIZES = (100, 300, 1000)


def pool_path(n: int, seed: int) -> Path:
    return POOL_DIR / f"grounding_pairclosed_n{n}_s{seed}.json"


def load_pool(n: int, seed: int) -> dict:
    out = pool_path(n, seed)
    if not out.is_file():
        raise FileNotFoundError("Missing pair-closed pool: %s" % out)
    with out.open(encoding="utf-8") as f:
        meta = (json.load(f).get("meta") or {})
    return {
        "path": out,
        "target_n": n,
        "actual_n": int(meta.get("pair_closed_actual_n") or n),
        "verify": meta.get("verify"),
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


def build_jobs(sizes, seed: int):
    jobs = []
    for n in sizes:
        pool = load_pool(n, seed)
        actual = pool["actual_n"]
        for method, ckpt, rule, dek in (
            ("L0", L0_CKPT, "liveedit", False),
            ("DEK", DEK_CKPT, "and", True),
        ):
            jobs.append({
                "name": f"llava_{method}_n{n}",
                "enp": f"g5_llava_{method}_pc_n{n}_s{seed}",
                "ckpt": ckpt,
                "dpath": pool["path"],
                "dsn": actual,
                "sen": actual,
                "rule": rule,
                "dek": dek,
                "target_n": n,
                "actual_n": actual,
                "pair_closed": True,
                # LLaVA-7B: full sequential eval at large sen is heavy; G3 used
                # routing_only for DEK>=800 on BLIP-2. Use for both methods at n>=300.
                "routing_only": actual >= 300,
            })
    return jobs


def run_job(job, device, seed, log_dir: Path) -> dict:
    enp = job["enp"]
    log_path = log_dir / f"{enp}.log"
    cmd = [
        str(PY), "-u", "-W", "ignore", "test_grounding_eval.py",
        "-en", "liveedit", "-mn", "llava",
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
    if job.get("routing_only"):
        cmd += ["--routing_only"]
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n===== {job['name']} {stamp} =====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTORCH_ALLOC_CONF": os.environ.get(
            "PYTORCH_ALLOC_CONF", "expandable_segments:True"
        ),
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
        "target_n": job["target_n"],
        "actual_n": job["actual_n"],
        "pair_closed": job["pair_closed"],
        "routing_only": bool(job.get("routing_only")),
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
                "swap_partner_in_pool_rate": s.get("swap_partner_in_pool_rate"),
                "hard_neg_partner_in_pool_rate": s.get("hard_neg_partner_in_pool_rate"),
            }
            scored.append({
                **s,
                "e2_meta": {
                    "target_n": r["target_n"],
                    "actual_n": r["actual_n"],
                    "pair_closed": r["pair_closed"],
                    "name": r["name"],
                    "routing_only": r.get("routing_only"),
                },
            })
        except Exception as e:
            r["taxonomy_error"] = repr(e)

    path = out_dir / f"pool_size_curve_llava_s{seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"backbone": "llava-v1.5-7b", "dataset": "vlkeb", "seed": seed,
             "runs": scored, "job_results": results},
            f, indent=2,
        )

    md = out_dir / f"pool_size_curve_llava_s{seed}.md"
    lines = [
        "| run | target_n | actual_n | closed | routing_only | swap stale | swap foreign | swap total | correct_route | abstain | hn total | rel hit | swap partner in pool | hn partner in pool |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r.get("taxonomy") or {}
        if not t:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (
                r["name"],
                r.get("target_n"),
                r.get("actual_n"),
                "Y" if r.get("pair_closed") else "N",
                "Y" if r.get("routing_only") else "N",
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
    ap = argparse.ArgumentParser(description="G5 E2 LLaVA pair-closed pool curve")
    ap.add_argument("--sizes", default=None, help="Comma-separated targets, default 100,300,1000")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    sizes = (
        [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
        if args.sizes
        else list(SIZES)
    )

    for p in (L0_CKPT, DEK_CKPT, IMG, SRC_JSON):
        if not Path(p).exists():
            print("FAIL: missing", p)
            return 1

    log_dir = OUT_DIR / "logs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pools = {}
    for n in sizes:
        pools[n] = load_pool(n, args.seed)
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

    jobs = build_jobs(sizes, args.seed)
    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "backbone": "llava-v1.5-7b",
        "pools": {
            str(k): {
                "path": str(v["path"]).replace("\\", "/"),
                "actual_n": v["actual_n"],
                "verify": v.get("verify"),
            }
            for k, v in pools.items()
        },
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
                    "actual_n": job["actual_n"], "pair_closed": True,
                    "routing_only": job.get("routing_only"),
                    "exit_code": 0, "skipped": True,
                    "run_dir": str(existing).replace("\\", "/"),
                })
                continue
        rec = run_job(job, args.dvc, args.seed, log_dir)
        manifest["results"].append(rec)
        with (OUT_DIR / "pool_size_manifest_llava.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if rec["exit_code"] != 0:
            print("FAILED", job["name"], "exit", rec["exit_code"], flush=True)

    print("\nScoring taxonomy…", flush=True)
    score_jobs(manifest["results"], OUT_DIR, args.seed)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    with (OUT_DIR / "pool_size_manifest_llava.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n_fail = sum(
        1 for r in manifest["results"]
        if r.get("exit_code", 1) != 0 and not r.get("skipped")
    )
    print("E2 DONE. failures=%d / %d" % (n_fail, len(manifest["results"])))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
