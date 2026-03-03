@echo off
REM Task 8: First LiveEdit vs AR-LiveEdit controlled comparison on VLKEB.
REM Run from repo root (LiveEdit folder). Uses same seed for both evals, then compares metrics.
REM
REM Usage:
REM   run_liveedit_ar_comparison.bat
REM   run_liveedit_ar_comparison.bat --skip_eval
REM
REM Set CKPT to your baseline checkpoint; set AR_CKPT to AR-trained checkpoint (or leave empty to use same ckpt).

set CKPT=records\liveedit\blip2-opt-2.7b\<your_train_name>\checkpoints\epoch-1-i-100-loss-0.1234.pt
set AR_CKPT=
set DEVICE=cuda:0
set DSN=200
set SEN=50
set SEED=42

call conda activate liveedit 2>nul
if errorlevel 1 (
  echo Conda env liveedit not activated \(optional\).
)

if "%AR_CKPT%"=="" set AR_CKPT=%CKPT%

echo Running LiveEdit vs AR-LiveEdit comparison on VLKEB (seed=%SEED%, n=%DSN%)...
python -W ignore run_liveedit_ar_comparison.py -dvc %DEVICE% -ckpt "%CKPT%" --ar_ckpt "%AR_CKPT%" -dsn %DSN% -sen %SEN% -seed %SEED%
if errorlevel 1 (
  echo Comparison script failed.
  exit /b 1
)
echo Done. Check eval_results\comparison\liveedit_vs_ar_seed%SEED%.json
exit /b 0
