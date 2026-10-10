"""G5: train LLaVA S2 (AVBT) or DEK from the existing L0 dn=500 ckpt.

Usage (repo root):
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g5_llava_train.py --stage S2
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/run_g5_llava_train.py --stage DEK
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"E:\anaconda\envs\liveedit\python.exe")
if not PY.is_file():
    PY = Path(sys.executable)

L0_CKPT = (
    ROOT / "records/liveedit/llava-v1.5-7b/"
    "VLKEB_llava_L0_dn500-2026.07.25-19.25.39/checkpoints/epoch-3-i-1900-ema_loss-8.1575"
)
# Prefer meeting_validation grounding train if present; else AAAI path.
CANDIDATE_DPATHS = [
    ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json",
    ROOT / "data/VLKEB/grounding/grounding_train_short.json",
]


def resolve_dpath() -> Path:
    for p in CANDIDATE_DPATHS:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No grounding_train_short.json found. Tried:\n  " + "\n  ".join(str(p) for p in CANDIDATE_DPATHS)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="G5 LLaVA S2/DEK train launcher")
    ap.add_argument("--stage", choices=["S2", "DEK"], required=True)
    ap.add_argument("-dvc", default="cuda:0")
    ap.add_argument("-lkpt", default=str(L0_CKPT))
    ap.add_argument("-dn", type=int, default=500)
    ap.add_argument("-eps", type=int, default=1)
    ap.add_argument("-rs", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.lkpt).exists():
        print("FAIL: L0 ckpt missing:", args.lkpt)
        return 1
    dpath = resolve_dpath()
    tnp = "g5_vlkeb_llava_S2_dn500" if args.stage == "S2" else "g5_vlkeb_llava_DEK_dn500"
    log_dir = ROOT / "doc/results/cvpr2027/g5/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tnp}.train.log"

    cmd = [
        str(PY), "-u", "train_vllm_editor.py",
        "-en", "liveedit", "-mn", "llava", "-dna", "VLKEB",
        "-bs", "1", "-dvc", args.dvc, "-edvc", "0",
        "-lkpt", args.lkpt,
        "-eps", str(args.eps), "-tnp", tnp,
        "-dn", str(args.dn), "-rs", str(args.rs), "-sci", "125",
        "-dpath", str(dpath),
        "--patch_grounding", "--adversarial_grounding",
        "--adv_grounding_lambda", "0.3",
        "--adv_grounding_neg_weight", "2.5",
        "--avbl_hinge_tau", "0.3", "--avbl_contrastive",
    ]
    if args.stage == "DEK":
        cmd.append("--dek_dual_gate")

    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"===== G5 LLaVA {args.stage} {stamp} =====", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    print("LOG:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.write(f"started: {stamp}\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )
    print("exit", proc.returncode, flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
