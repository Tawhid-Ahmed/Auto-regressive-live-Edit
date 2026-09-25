"""Run train_vllm_editor.py with UTF-8 tee to a log file (Windows-safe)."""
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
# AAAI protocol: full EVQA train (6345), same scale spirit as VLKEB L0 (5000).
# AVBT/DEK fine-tunes later still use dn=500 to match the main-paper DEK stage.
TRAIN_NAME_PREFIX = "EVQA_L0_full"
LOG_FILE = LOG_DIR / "EVQA_L0_full.train.log"
RUN_DIR_FILE = LOG_DIR / "EVQA_L0_full_run_dir.txt"
PID_FILE = LOG_DIR / "EVQA_L0_full.pid"


def latest_run():
    runs = sorted(
        RECORDS.glob(f"{TRAIN_NAME_PREFIX}-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def latest_ckpt(run: Path):
    ckpts = sorted(
        (run / "checkpoints").glob("epoch-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Skip files marked corrupt / incomplete (power loss mid-save).
    for c in ckpts:
        if c.name.endswith(".CORRUPT") or c.suffix == ".CORRUPT":
            continue
        if ".CORRUPT" in c.name:
            continue
        # quick zip/header check via torch
        try:
            import torch

            torch.load(c, map_location="cpu")
            return c
        except Exception as e:
            print(f"Skipping unreadable ckpt {c.name}: {e}", flush=True)
            try:
                c.rename(c.with_name(c.name + ".CORRUPT"))
            except Exception:
                pass
    return None


def main():
    resume = "--resume" in sys.argv[1:]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    args = [
        str(PY),
        "-u",
        "-W",
        "ignore",
        "train_vllm_editor.py",
        "-en",
        "liveedit",
        "-mn",
        "blip2",
        "-dna",
        "EVQA",
        "-bs",
        "4",
        "-dvc",
        "cuda:0",
        "-edvc",
        "0",
        "-eps",
        "20",
        "-tnp",
        TRAIN_NAME_PREFIX,
        # Omit -dn → full EVQA train set (6345).
        "-rs",
        "42",
        # ~1586 steps/epoch at bs=4; save often (loadshedding).
        "-sci",
        "200",
    ]

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if resume:
        run = latest_run()
        if run is None:
            print(f"No {TRAIN_NAME_PREFIX}-* run to resume.", file=sys.stderr)
            return 1
        ckpt = latest_ckpt(run)
        if ckpt is None:
            print(f"No checkpoint under {run / 'checkpoints'}", file=sys.stderr)
            return 1
        args += ["-lkpt", str(ckpt), "--resume", "--resume_run_name", run.name]
        header = f"RESUMING FULL-EVQA run={run.name} ckpt={ckpt.name} {stamp}\n"
        RUN_DIR_FILE.write_text(run.name, encoding="utf-8")
    else:
        header = f"FRESH START {TRAIN_NAME_PREFIX} (full EVQA, AAAI protocol) {stamp}\n"

    if LOG_FILE.exists():
        bak = LOG_DIR / f"EVQA_L0_full.train.{time.strftime('%Y%m%d_%H%M%S')}.log.bak"
        LOG_FILE.rename(bak)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Windows consoles are often cp1252; tqdm uses Unicode blocks.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with LOG_FILE.open("w", encoding="utf-8", errors="replace") as log:
        log.write(header)
        log.write("CMD: " + " ".join(args) + "\n")
        log.flush()
        print(header.strip())
        print("Log:", LOG_FILE)

        proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        print("PID=", proc.pid)

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
            # Keep run-dir pointer updated when records appear
            run = latest_run()
            if run is not None:
                RUN_DIR_FILE.write_text(run.name, encoding="utf-8")

        rc = proc.wait()
        log.write(f"\nEXIT_CODE={rc} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print("EXIT_CODE=", rc)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
