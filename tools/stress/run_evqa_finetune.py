"""EVQA S2 / DEK fine-tune (dn=500) — logged, resumable, AAAI protocol.

Usage:
  python tools/stress/run_evqa_finetune.py --mode s2
  python tools/stress/run_evqa_finetune.py --mode dek
  python tools/stress/run_evqa_finetune.py --mode s2 --resume
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\MSC\MSc thesis project\LiveEdit\LiveEdit")
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
LOG_DIR = ROOT / "doc" / "results" / "meeting_validation" / "step6" / "stress" / "logs"
RECORDS = ROOT / "records" / "liveedit" / "blip2-opt-2.7b"
L0_CKPT = (
    RECORDS
    / "EVQA_L0_full-2026.07.24-12.05.12"
    / "checkpoints"
    / "epoch-20-i-29400-ema_loss-0.1996"
)
GROUNDING_TRAIN = (
    ROOT / "data" / "easy-edit-mm" / "grounding" / "grounding_train_short.json"
)

MODE_CFG = {
    "s2": {
        "prefix": "EVQA_avbt_S2_gnd",
        "extra": [
            "-dpath",
            str(GROUNDING_TRAIN),
            "--patch_grounding",
            "--adversarial_grounding",
            "--adv_grounding_lambda",
            "0.3",
            "--adv_grounding_neg_weight",
            "2.5",
            "--avbl_hinge_tau",
            "0.3",
            "--avbl_contrastive",
        ],
        "label": "AVBT-only S2 (EVQA grounding)",
    },
    "dek": {
        "prefix": "EVQA_avbt_DEK_gnd",
        "extra": [
            "-dpath",
            str(GROUNDING_TRAIN),
            "--patch_grounding",
            "--adversarial_grounding",
            "--adv_grounding_lambda",
            "0.3",
            "--adv_grounding_neg_weight",
            "2.5",
            "--avbl_hinge_tau",
            "0.3",
            "--avbl_contrastive",
            "--dek_dual_gate",
        ],
        "label": "AVBT+DEK (EVQA grounding)",
    },
}


def latest_run(prefix: str):
    runs = sorted(
        RECORDS.glob(f"{prefix}-*"),
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
    for c in ckpts:
        if ".CORRUPT" in c.name:
            continue
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("s2", "dek"), required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = MODE_CFG[args.mode]
    prefix = cfg["prefix"]
    log_file = LOG_DIR / f"{prefix}.train.log"
    run_dir_file = LOG_DIR / f"{prefix}_run_dir.txt"
    pid_file = LOG_DIR / f"{prefix}.pid"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    if not L0_CKPT.is_file():
        print(f"Missing L0 ckpt: {L0_CKPT}", file=sys.stderr)
        return 1
    if not GROUNDING_TRAIN.is_file():
        print(f"Missing grounding train JSON: {GROUNDING_TRAIN}", file=sys.stderr)
        return 1

    cmd = [
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
        "1",
        "-tnp",
        prefix,
        "-dn",
        "500",
        "-rs",
        "42",
        "-sci",
        "125",  # 500/bs4 ≈ 125 steps/epoch — save final iter (matches VLKEB short)
        *cfg["extra"],
    ]

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if args.resume:
        run = latest_run(prefix)
        if run is None:
            print(f"No {prefix}-* run to resume.", file=sys.stderr)
            return 1
        ckpt = latest_ckpt(run)
        if ckpt is None:
            print(f"No readable checkpoint under {run / 'checkpoints'}", file=sys.stderr)
            return 1
        cmd += ["-lkpt", str(ckpt), "--resume", "--resume_run_name", run.name]
        header = f"RESUMING {cfg['label']} run={run.name} ckpt={ckpt.name} {stamp}\n"
        run_dir_file.write_text(run.name, encoding="utf-8")
    else:
        cmd += ["-lkpt", str(L0_CKPT)]
        header = (
            f"FRESH START {prefix} ({cfg['label']}, dn=500 from L0 full) {stamp}\n"
            f"L0_CKPT={L0_CKPT}\n"
        )

    if log_file.exists():
        bak = LOG_DIR / f"{prefix}.train.{time.strftime('%Y%m%d_%H%M%S')}.log.bak"
        log_file.rename(bak)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write(header)
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        print(header.strip())
        print("Log:", log_file)

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
        pid_file.write_text(str(proc.pid), encoding="utf-8")
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
            run = latest_run(prefix)
            if run is not None:
                run_dir_file.write_text(run.name, encoding="utf-8")

        rc = proc.wait()
        log.write(f"\nEXIT_CODE={rc} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print("EXIT_CODE=", rc)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
