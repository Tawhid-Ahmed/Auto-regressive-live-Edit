"""Run EVQA Rel/Gen/Loc eval for L0 / S2 / DEK (logged)."""
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

RUNS = {
    "L0": {
        "ckpt": RECORDS
        / "EVQA_L0_full-2026.07.24-12.05.12"
        / "checkpoints"
        / "epoch-20-i-29400-ema_loss-0.1996",
        "enp": "stress_blip2_L0_evqa",
        "extra": [],
    },
    "S2": {
        "ckpt": RECORDS
        / "EVQA_avbt_S2_stress-2026.07.25-11.04.33"
        / "checkpoints"
        / "epoch-1-i-125-ema_loss-1.5223",
        "enp": "stress_blip2_S2_evqa",
        "extra": ["--patch_grounding"],
    },
    "DEK": {
        "ckpt": RECORDS
        / "EVQA_avbt_DEK_stress-2026.07.25-11.12.24"
        / "checkpoints"
        / "epoch-1-i-125-ema_loss-1.5539",
        "enp": "stress_blip2_DEK_evqa",
        "extra": ["--patch_grounding", "--dek_dual_gate"],
    },
}


def main():
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
        if not cfg["ckpt"].is_file():
            print(f"Missing ckpt for {name}: {cfg['ckpt']}", file=sys.stderr)
            return 1
        log_file = LOG_DIR / f"eval_{name}.log"
        cmd = [
            str(PY),
            "-u",
            "-W",
            "ignore",
            "test_vllm_edit.py",
            "-en",
            "liveedit",
            "-mn",
            "blip2",
            "-dn",
            "EVQA",
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
            *cfg["extra"],
        ]
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"=== EVAL {name} {stamp} ===")
        with log_file.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"EVAL {name} {stamp}\nCMD: {' '.join(cmd)}\n")
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
