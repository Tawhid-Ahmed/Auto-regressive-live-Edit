@echo off
REM Task 9: Small ablation set for AR-LiveEdit (chunk_size, routing_gate, max_chunks).
REM Run from repo root (LiveEdit folder). Runs 12 AR evals then writes ablation table + recommendation.
REM
REM Usage:
REM   run_ar_ablation.bat
REM   run_ar_ablation.bat --skip_run   (build table from existing results only)
REM
REM Set CKPT to your AR (or baseline) checkpoint; leave empty to run with untrained editor.

set CKPT=
set DEVICE=cuda:0
set DSN=100
set SEN=25
set SEED=42

call conda activate liveedit 2>nul
if errorlevel 1 (
  echo Conda env liveedit not activated (optional).
)

echo Running AR-LiveEdit ablation (chunk_size=8,16,32 x routing_gate=on,off x max_chunks=unlim,4)...
python -W ignore run_ar_ablation.py -dvc %DEVICE% -ckpt "%CKPT%" -dsn %DSN% -sen %SEN% -seed %SEED%
if errorlevel 1 (
  echo Ablation script failed.
  exit /b 1
)
echo Done. Check eval_results\ablation\ar_ablation_seed%SEED%.json
exit /b 0
