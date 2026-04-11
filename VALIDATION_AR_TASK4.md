# Task 4 AR Training Loop – Validation Summary

**Run both full tests on liveedit profile:**  
`run_ar_validation_liveedit.bat` (from repo root) or the commands in sections 3 and 4 below with `conda activate liveedit` then `python -W ignore ...`.

**Note:** When run on this machine, both "One AR training step" and "One short epoch" failed with **OSError: The paging file is too small for this operation to complete (os error 1455)** during BLIP2 model load. Increase Windows virtual memory or run on a machine with more RAM to complete these tests.

---

## Automated checks (run in repo root)

### 1. Batch organization (Task 3 + 12-element tuple)
```bash
conda run -n liveedit python verify_ar_batch.py
```
- **Result:** PASSED  
- Confirms: `ar_mode=False` → 11 elements, `batch_ar_chunks` is None; `ar_mode=True` → 12 elements with `batch_ar_chunks` and `rel_edit_i`.

### 2. AR chunk indexing and mask logic (no model)
```bash
python -m editor.vllm_editors.liveedit.test_ar_train_step_logic
```
- **Result:** PASSED  
- Confirms: `start_pos` / `end_pos` and per-chunk mask construction are correct; sum of chunk masks equals target length.

## Full training-step validation (requires GPU + liveedit env)

Run these **on the liveedit profile** (conda env `liveedit`). Use the same Python for both:

```bash
conda activate liveedit
# Or use the env Python directly, e.g.:
# E:\anaconda\envs\liveedit\python.exe
```

### 3. One AR training step – finite loss, no NaNs
```bash
python -W ignore validate_ar_train_step.py
```
- Loads LiveEdit + BLIP2, VLKEB tiny subset, runs one batch with `ar_mode=True`, asserts loss is finite and not NaN.
- **If you get** `OSError: The paging file is too small (os error 1455)` **:** increase Windows virtual memory (System → Advanced → Performance → Virtual memory) or free RAM and try again.

### 4. One short epoch (plan validation)
```bash
python -W ignore train_vllm_editor.py -en LiveEdit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 --ar_mode -eps 1 -lpi 1
```
- One epoch on 6 VLKEB samples; confirm it completes and logged loss is finite.
- Same paging-file/memory note as above if model load fails.

## Plan criteria (Task 4)

- [x] Chunk-step loop in `train_a_batch` with simple aggregate (sum per chunk, then mean over batch).
- [x] Code path behind `ar_mode=True`; non-AR path unchanged.
- [x] Batch organization provides `rel_edit_i` for correct request indexing.
- [ ] One short training epoch completes on small VLKEB subset (run 3 or 4 above locally).
- [ ] Loss is finite, no NaNs (asserted in `validate_ar_train_step.py` when run successfully).
