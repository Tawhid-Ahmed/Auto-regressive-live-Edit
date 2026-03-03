@echo off
REM Run AR Task 4 validation on liveedit profile: one step + one short epoch.
REM Use from repo root (LiveEdit folder) with: run_ar_validation_liveedit.bat

call conda activate liveedit
if errorlevel 1 (
  echo Failed to activate conda env liveedit.
  exit /b 1
)

echo === 1. One AR training step (validate_ar_train_step.py) ===
python -W ignore validate_ar_train_step.py
if errorlevel 1 (
  echo Step 1 failed.
  exit /b 1
)

echo.
echo === 2. One short epoch (train_vllm_editor.py --ar_mode -eps 1) ===
python -W ignore train_vllm_editor.py -en LiveEdit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 --ar_mode -eps 1 -lpi 1
if errorlevel 1 (
  echo Step 2 failed.
  exit /b 1
)

echo.
echo All AR validations passed.
exit /b 0
