"""EVQA grounding eval (Rel / Swap / Hard / FC) for L0, S2_gnd, DEK_gnd."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\MSC\MSc thesis project\LiveEdit\LiveEdit")
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
LOG_DIR = ROOT / "doc" / "results" / "meeting_validation" / "step6" / "stress" / "logs"
RECORDS = ROOT / "records" / "liveedit" / "blip2-opt-2.7b"
EVAL_JSON = ROOT / "data" / "easy-edit-mm" / "grounding" / "grounding_eval_short.json"
IMG_ROOT = "data/easy-edit-mm/images"

RUNS = {
    "L0": {
        "ckpt": RECORDS
        / "EVQA_L0_full-2026.07.24-12.05.12"
        / "checkpoints"
        / "epoch-20-i-29400-ema_loss-0.1996",
        "enp": "evqa_gnd_L0",
        "extra": [],
    },
    "S2": {
        "ckpt": RECORDS
        / "EVQA_avbt_S2_gnd-2026.07.25-11.32.38"
        / "checkpoints"
        / "epoch-1-i-125-ema_loss-3.8897",
        "enp": "evqa_gnd_S2",
        "extra": ["--patch_grounding"],
    },
    "DEK": {
        "ckpt": RECORDS
        / "EVQA_avbt_DEK_gnd-2026.07.25-11.49.05"
        / "checkpoints"
        / "epoch-1-i-125-ema_loss-3.9200",
        "enp": "evqa_gnd_DEK",
        "extra": ["--patch_grounding", "--dek_dual_gate"],
    },
}


def main():
    # Allow DEK ckpt path discovery if timestamp differs
    dek_runs = sorted(RECORDS.glob("EVQA_avbt_DEK_gnd-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if dek_runs:
        ckpts = sorted((dek_runs[0] / "checkpoints").glob("epoch-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        ckpts = [c for c in ckpts if ".CORRUPT" not in c.name]
        if ckpts:
            RUNS["DEK"]["ckpt"] = ckpts[0]
    s2_runs = sorted(RECORDS.glob("EVQA_avbt_S2_gnd-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if s2_runs:
        ckpts = sorted((s2_runs[0] / "checkpoints").glob("epoch-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        ckpts = [c for c in ckpts if ".CORRUPT" not in c.name]
        if ckpts:
            RUNS["S2"]["ckpt"] = ckpts[0]

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["L0", "S2", "DEK"] if which == "all" else [which.upper()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for name in order:
        cfg = RUNS[name]
        if not Path(cfg["ckpt"]).is_file():
            print(f"Missing ckpt {name}: {cfg['ckpt']}", file=sys.stderr)
            return 1
        log_file = LOG_DIR / f"eval_gnd_{name}.log"
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
            "10",
            "-dvc",
            "cuda:0",
            "-dsn",
            "100",
            "-seed",
            "42",
            "-ckpt",
            str(cfg["ckpt"]),
            "-enp",
            cfg["enp"],
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
        print(f"=== GND EVAL {name} {stamp} ckpt={cfg['ckpt']} ===")
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"GND EVAL {name} {stamp}\nCMD: {' '.join(cmd)}\n")
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
                    sys.stdout.buffer.flush()
            rc = proc.wait()
            log.write(f"\nEXIT_CODE={rc}\n")
            print(f"EXIT_CODE={name}={rc}")
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
