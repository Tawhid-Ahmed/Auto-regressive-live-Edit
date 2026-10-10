"""G5 lab pipeline: fresh LLaVA L0 -> S2 -> DEK on lab GPU (no old checkpoints).

Designed for RTX 4090 (~24 GB). Keeps only the latest 3 checkpoints per run.
Writes experiment tracking under doc/results/cvpr2027/g5/ (mirrored on lab).

Usage (on lab, repo root, conda activate liveedit):
  python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 S2 DEK
  python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0          # start with L0 only
  python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 DEK --from_l0_run <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK_DIR = ROOT / "doc" / "results" / "cvpr2027" / "g5"
TRACKER_MD = TRACK_DIR / "EXPERIMENT_TRACKER.md"
TRACKER_JSONL = TRACK_DIR / "experiments.jsonl"
LAB_BACKUP = Path(
    os.environ.get(
        "LAB_EXPERIMENT_BACKUP",
        "/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5",
    )
)

# 4090-friendly defaults
DEFAULT_BS = 1
DEFAULT_DN = 500
DEFAULT_EPS = 1
DEFAULT_SCI = 100  # ~5 saves / 500-iter epoch; retain 3
DEFAULT_KEEP = 3
DEFAULT_SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_dpath() -> Path:
    candidates = [
        ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json",
        ROOT / "data/VLKEB/grounding/grounding_train_short.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("No grounding_train_short.json found")


def latest_ckpt(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    files = [p for p in ckpt_dir.iterdir() if p.is_file() and p.name.startswith("epoch-")]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def find_run_dir(prefix: str) -> Path | None:
    bases = [
        ROOT / "records" / "lab_g5" / "liveedit" / "llava-v1.5-7b",
        ROOT / "records" / "liveedit" / "llava-v1.5-7b",
    ]
    cands = []
    for base in bases:
        if not base.is_dir():
            continue
        cands.extend(
            [p for p in base.iterdir() if p.is_dir() and prefix in p.name]
        )
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def append_tracker(event: dict) -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    LAB_BACKUP.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    for path in (TRACKER_JSONL, LAB_BACKUP / "experiments.jsonl"):
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    # Markdown mirror (append section)
    md_lines = [
        f"\n### {event.get('stage', '?')} — `{event.get('run_name', '')}`\n",
        f"- **time (UTC):** {event.get('time')}\n",
        f"- **status:** {event.get('status')}\n",
        f"- **host:** {event.get('host')}\n",
        f"- **cmd:** `{event.get('cmd', '')}`\n",
    ]
    if event.get("run_dir"):
        md_lines.append(f"- **run_dir:** `{event['run_dir']}`\n")
    if event.get("ckpt"):
        md_lines.append(f"- **ckpt:** `{event['ckpt']}`\n")
    if event.get("log"):
        md_lines.append(f"- **log:** `{event['log']}`\n")
    if event.get("exit_code") is not None:
        md_lines.append(f"- **exit:** {event['exit_code']}\n")
    if event.get("note"):
        md_lines.append(f"- **note:** {event['note']}\n")
    body = "".join(md_lines)
    for path in (TRACKER_MD, LAB_BACKUP / "EXPERIMENT_TRACKER.md"):
        if not path.exists():
            path.write_text(
                "# G5 LLaVA lab experiment tracker\n\n"
                "Fresh trains on lab GPU (no reused home-PC checkpoints). "
                "Keep latest 3 checkpoints per run.\n",
                encoding="utf-8",
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(body)


def run_train(stage: str, lkpt: Path | None, args: argparse.Namespace) -> tuple[int, Path | None, Path]:
    import socket

    host = socket.gethostname()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lab_g5_llava_{stage}_dn{args.dn}_s{args.rs}_{stamp}"
    dpath = resolve_dpath()
    log_dir = TRACK_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.train.log"

    cmd = [
        sys.executable, "-u", str(ROOT / "train_vllm_editor.py"),
        "-en", "liveedit", "-mn", "llava", "-dna", "VLKEB",
        "-bs", str(args.bs), "-dvc", args.dvc, "-edvc", "0",
        "-eps", str(args.eps), "-tnp", run_name,
        "-dn", str(args.dn), "-rs", str(args.rs),
        "-sci", str(args.sci), "-lpi", "1",
        "-dpath", str(dpath),
        "--keep_latest_ckpts", str(args.keep_latest_ckpts),
        "--records_dir", str(ROOT / "records" / "lab_g5"),
    ]
    if lkpt is not None:
        cmd.extend(["-lkpt", str(lkpt)])
    if stage in ("S2", "DEK"):
        cmd.extend([
            "--patch_grounding", "--adversarial_grounding",
            "--adv_grounding_lambda", "0.3",
            "--adv_grounding_neg_weight", "2.5",
            "--avbl_hinge_tau", "0.3", "--avbl_contrastive",
        ])
    if stage == "DEK":
        cmd.append("--dek_dual_gate")

    meta = {
        "time": utc_now(),
        "stage": stage,
        "run_name": run_name,
        "status": "started",
        "host": host,
        "cmd": " ".join(cmd),
        "log": str(log_path.relative_to(ROOT)),
        "note": "fresh train; no home-PC checkpoints; keep_latest_ckpts=%d" % args.keep_latest_ckpts,
        "config": {
            "bs": args.bs, "dn": args.dn, "eps": args.eps, "sci": args.sci,
            "seed": args.rs, "dvc": args.dvc, "lkpt": str(lkpt) if lkpt else None,
        },
    }
    append_tracker(meta)
    print("===== START", stage, run_name, "=====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    print("LOG:", log_path, flush=True)

    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NVIDIA_TF32_OVERRIDE": "1",
    }
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.write("started: " + utc_now() + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)

    run_dir = find_run_dir(run_name)
    # records path may be records/lab_g5/liveedit/llava-v1.5-7b/<run_name>
    if run_dir is None:
        alt = ROOT / "records" / "lab_g5" / "liveedit" / "llava-v1.5-7b"
        if alt.is_dir():
            cands = sorted(
                [p for p in alt.iterdir() if p.is_dir() and run_name in p.name],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            run_dir = cands[0] if cands else None

    ckpt = latest_ckpt(run_dir) if run_dir else None
    done = {
        "time": utc_now(),
        "stage": stage,
        "run_name": run_name,
        "status": "finished" if proc.returncode == 0 else "failed",
        "host": host,
        "cmd": " ".join(cmd),
        "log": str(log_path.relative_to(ROOT)),
        "exit_code": proc.returncode,
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else None,
        "ckpt": str(ckpt.relative_to(ROOT)) if ckpt else None,
    }
    append_tracker(done)
    # also copy tracker snippet to lab backup run folder
    if run_dir:
        try:
            (run_dir / "pipeline_event.json").write_text(
                json.dumps(done, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    print("===== END", stage, "exit", proc.returncode, "ckpt", ckpt, "=====", flush=True)
    return proc.returncode, ckpt, run_dir or Path(".")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fresh G5 LLaVA lab pipeline")
    ap.add_argument("--stages", nargs="+", choices=["L0", "S2", "DEK"], default=["L0", "S2", "DEK"])
    ap.add_argument("--from_l0_ckpt", default=None, help="Existing L0 ckpt path (lab-trained only)")
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-bs", type=int, default=DEFAULT_BS)
    ap.add_argument("-dn", type=int, default=DEFAULT_DN)
    ap.add_argument("-eps", type=int, default=DEFAULT_EPS)
    ap.add_argument("-sci", type=int, default=DEFAULT_SCI)
    ap.add_argument("-rs", type=int, default=DEFAULT_SEED)
    ap.add_argument("--keep_latest_ckpts", type=int, default=DEFAULT_KEEP)
    args = ap.parse_args()

    os.chdir(ROOT)
    l0_ckpt = Path(args.from_l0_ckpt) if args.from_l0_ckpt else None
    if l0_ckpt and not l0_ckpt.exists():
        print("FAIL: --from_l0_ckpt missing:", l0_ckpt)
        return 1

    for stage in args.stages:
        if stage == "L0":
            code, ckpt, _ = run_train("L0", None, args)
            if code != 0:
                return code
            l0_ckpt = ckpt
        elif stage == "S2":
            if l0_ckpt is None:
                print("FAIL: S2 needs L0 ckpt (train L0 first or pass --from_l0_ckpt)")
                return 1
            code, _, _ = run_train("S2", l0_ckpt, args)
            if code != 0:
                return code
        elif stage == "DEK":
            if l0_ckpt is None:
                print("FAIL: DEK needs L0 ckpt (train L0 first or pass --from_l0_ckpt)")
                return 1
            code, _, _ = run_train("DEK", l0_ckpt, args)
            if code != 0:
                return code
    print("PIPELINE OK", args.stages, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
