"""Run VLKEB LLaVA L0 with dn=500 (deadline-aware backbone stress)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\MSC\MSc thesis project\LiveEdit\LiveEdit")
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
LOG_DIR = ROOT / "doc" / "results" / "meeting_validation" / "step6" / "stress" / "logs"
RECORDS = ROOT / "records" / "liveedit" / "llava-v1.5-7b"
# AAAI deadline protocol: same n as S2/DEK fine-tune; disclose as reduced-data L0.
TRAIN_NAME_PREFIX = "VLKEB_llava_L0_dn500"
LOG_FILE = LOG_DIR / "VLKEB_llava_L0_dn500.train.log"
RUN_DIR_FILE = LOG_DIR / "VLKEB_llava_L0_dn500_run_dir.txt"
PID_FILE = LOG_DIR / "VLKEB_llava_L0_dn500.pid"
DATA_N = 500
EPOCHS = 3  # deadline-aware LLaVA stress (disclose in paper; not matched to BLIP2 eps=20)
SAVE_EVERY = 100


def latest_run():
    if not RECORDS.is_dir():
        return None
    runs = sorted(
        RECORDS.glob(f"{TRAIN_NAME_PREFIX}-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def latest_ckpt(run: Path):
    ckpt_dir = run / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("epoch-*"), key=lambda p: p.stat().st_mtime, reverse=True)
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
    resume = "--resume" in sys.argv[1:]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    train_argv = [
        "train_vllm_editor.py",
        "-en", "liveedit",
        "-mn", "llava",
        "-dna", "VLKEB",
        "-bs", "1",
        "-dvc", "cuda:0",
        "-edvc", "0",
        "-eps", str(EPOCHS),
        "-tnp", TRAIN_NAME_PREFIX,
        "-dn", str(DATA_N),
        "-rs", "42",
        "-sci", str(SAVE_EVERY),
        "-dpath", "data/VLKEB/train.json",
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
        train_argv += ["-lkpt", str(ckpt), "--resume", "--resume_run_name", run.name]
        header = f"RESUMING LLaVA L0 dn={DATA_N} run={run.name} ckpt={ckpt.name} {stamp}\n"
        RUN_DIR_FILE.write_text(run.name, encoding="utf-8")
    else:
        header = (
            f"FRESH START {TRAIN_NAME_PREFIX} (dn={DATA_N}, eps={EPOCHS}, "
            f"deadline-aware backbone stress) {stamp}\n"
        )

    if LOG_FILE.exists():
        bak = LOG_DIR / f"VLKEB_llava_L0_dn500.train.{time.strftime('%Y%m%d_%H%M%S')}.log.bak"
        LOG_FILE.rename(bak)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Enable TF32 then run train_vllm_editor with the intended argv.
    wrapper = (
        "import sys, torch, runpy\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "torch.backends.cudnn.allow_tf32 = True\n"
        f"sys.argv = {train_argv!r}\n"
        "runpy.run_path('train_vllm_editor.py', run_name='__main__')\n"
    )
    cmd = [str(PY), "-u", "-c", wrapper]

    with LOG_FILE.open("w", encoding="utf-8", errors="replace") as log:
        log.write(header)
        log.write("CMD: " + " ".join(train_argv) + "\n")
        log.write("TF32: enabled\n")
        log.flush()
        print(header.strip())
        print("Log:", LOG_FILE)

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
            run = latest_run()
            if run is not None:
                RUN_DIR_FILE.write_text(run.name, encoding="utf-8")

        rc = proc.wait()
        log.write(f"\nEXIT_CODE={rc} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        print("EXIT_CODE=", rc)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
