"""Must-do reliability: multi-seed + larger dsn EVQA grounding eval.

Default: seeds 42,0,1,2,3 and dsn=300 for L0 / S2_gnd / DEK_gnd.
Reuses fixed checkpoints (eval-only; no retrain).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(r"E:\MSC\MSc thesis project\LiveEdit\LiveEdit")
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
LOG_DIR = ROOT / "doc" / "results" / "meeting_validation" / "step6" / "stress" / "logs"
OUT_DIR = ROOT / "doc" / "results" / "meeting_validation" / "step6" / "stress"
RECORDS = ROOT / "records" / "liveedit" / "blip2-opt-2.7b"
EVAL_JSON = ROOT / "data" / "easy-edit-mm" / "grounding" / "grounding_eval_short.json"
IMG_ROOT = "data/easy-edit-mm/images"

SEEDS = [42, 0, 1, 2, 3]
DSN = 300
SEN = 10


def resolve_ckpts():
    l0 = (
        RECORDS
        / "EVQA_L0_full-2026.07.24-12.05.12"
        / "checkpoints"
        / "epoch-20-i-29400-ema_loss-0.1996"
    )
    s2_runs = sorted(RECORDS.glob("EVQA_avbt_S2_gnd-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    dek_runs = sorted(RECORDS.glob("EVQA_avbt_DEK_gnd-*"), key=lambda p: p.stat().st_mtime, reverse=True)

    def latest_epoch(run):
        ckpts = [
            c
            for c in sorted((run / "checkpoints").glob("epoch-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ".CORRUPT" not in c.name
        ]
        return ckpts[0] if ckpts else None

    return {
        "L0": {"ckpt": l0, "extra": [], "enp": "evqa_gnd_ms_L0"},
        "S2": {"ckpt": latest_epoch(s2_runs[0]), "extra": ["--patch_grounding"], "enp": "evqa_gnd_ms_S2"},
        "DEK": {
            "ckpt": latest_epoch(dek_runs[0]),
            "extra": ["--patch_grounding", "--dek_dual_gate"],
            "enp": "evqa_gnd_ms_DEK",
        },
    }


def extract_metrics(mean_path: Path) -> dict:
    d = json.loads(mean_path.read_text(encoding="utf-8"))
    tm = d["total_mean"]
    rd = d.get("routing_diagnostics") or {}
    return {
        "rel_acc": tm["reliability"]["acc"],
        "swap_acc": tm["generality"].get("image_swap", {}).get("acc"),
        "hard_acc": tm["generality"].get("hard_negative", {}).get("acc"),
        "fc_swap": rd.get("image_swap_false_cross_rate"),
        "fc_hard": rd.get("hard_negative_false_cross_rate"),
        "mean_n_retrieved": rd.get("mean_n_retrieved"),
        "result_json": str(mean_path).replace("\\", "/"),
    }


from typing import Optional
...
def find_mean_results(enp: str, seed: int) -> Optional[Path]:
    hits = sorted(
        [
            p
            for p in (ROOT / "eval_results" / "meeting_validation" / "step6").rglob(
                f"seed_{seed}_mean_results.json"
            )
            if f"VLKEB_GROUNDING-{enp}-" in str(p)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def main():
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else SEEDS)]
    dsn = int(sys.argv[2]) if len(sys.argv) > 2 else DSN
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    runs = resolve_ckpts()
    for name, cfg in runs.items():
        if cfg["ckpt"] is None or not Path(cfg["ckpt"]).is_file():
            print(f"Missing ckpt for {name}", file=sys.stderr)
            return 1

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    results = {
        "protocol": {
            "dsn": dsn,
            "sen": SEN,
            "seeds": seeds,
            "eval_json": str(EVAL_JSON).replace("\\", "/"),
            "note": "Must-do: multi-seed + dsn=300 with seeded subsample from full eval JSON",
        },
        "checkpoints": {k: str(v["ckpt"]).replace("\\", "/") for k, v in runs.items()},
        "by_seed": {},
        "runs": [],
    }

    for seed in seeds:
        results["by_seed"][str(seed)] = {}
        for name, cfg in runs.items():
            # unique enp per seed so result folders don't collide on name alone
            enp = f"{cfg['enp']}_sub_s{seed}_d{dsn}"
            log_file = LOG_DIR / f"eval_gnd_ms_{name}_s{seed}_d{dsn}.log"
            cmd = [
                str(PY),
                "-u",
                "-W",
                "ignore",
                "test_grounding_eval.py",
                "-en",
                "liveedit",
                "-mn",
                "blip2",
                "-sen",
                str(SEN),
                "-dvc",
                "cuda:0",
                "-dsn",
                str(dsn),
                "-seed",
                str(seed),
                "-ckpt",
                str(cfg["ckpt"]),
                "-enp",
                enp,
                "-dpath",
                str(EVAL_JSON),
                "--img_root",
                IMG_ROOT,
                "--meeting_step",
                "6",
                "--log_routing",
                *cfg["extra"],
            ]
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            print(f"=== {name} seed={seed} dsn={dsn} {stamp} ===", flush=True)
            with log_file.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"CMD: {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except UnicodeEncodeError:
                        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
                rc = proc.wait()
                log.write(f"\nEXIT_CODE={rc}\n")
            if rc != 0:
                print(f"FAILED {name} seed={seed} rc={rc}", file=sys.stderr)
                return rc
            mean_path = find_mean_results(enp, seed)
            if mean_path is None:
                print(f"Could not find mean_results for {enp}", file=sys.stderr)
                return 1
            metrics = extract_metrics(mean_path)
            metrics.update({"seed": seed, "method": name, "dsn": dsn, "enp": enp})
            results["by_seed"][str(seed)][name] = metrics
            results["runs"].append(metrics)
            # incremental save
            (OUT_DIR / "aaai_stress_evqa_grounding_multiseed.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

    # aggregate
    import numpy as np

    agg = {}
    for name in runs:
        rels = [results["by_seed"][str(s)][name]["rel_acc"] for s in seeds]
        fcs = [results["by_seed"][str(s)][name]["fc_swap"] for s in seeds]
        swaps = [results["by_seed"][str(s)][name]["swap_acc"] for s in seeds]
        agg[name] = {
            "rel_mean": float(np.mean(rels)),
            "rel_std": float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0,
            "fc_swap_mean": float(np.mean(fcs)),
            "fc_swap_std": float(np.std(fcs, ddof=1)) if len(fcs) > 1 else 0.0,
            "swap_mean": float(np.mean(swaps)),
            "swap_std": float(np.std(swaps, ddof=1)) if len(swaps) > 1 else 0.0,
            "n_seeds": len(seeds),
            "per_seed_rel": rels,
            "per_seed_fc_swap": fcs,
        }
    results["aggregate"] = agg
    l0, dek = agg["L0"], agg["DEK"]
    results["gate"] = {
        "rel_held": dek["rel_mean"] >= l0["rel_mean"] - 0.05,
        "fc_drop": dek["fc_swap_mean"] < l0["fc_swap_mean"] - 0.2,
    }
    results["gate"]["pass"] = bool(results["gate"]["rel_held"] and results["gate"]["fc_drop"])
    out_path = OUT_DIR / "aaai_stress_evqa_grounding_multiseed.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print(json.dumps(agg, indent=2))
    print("GATE", results["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
