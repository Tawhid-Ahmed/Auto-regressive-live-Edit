# Step 6 — L0 / L5 run status (2026-07-08)

## Attempted (sequential, not parallel)

1. **L0** — `run_ablation_eval.py --layers L0` with `VLKEB_long_baseline` ckpt  
2. **L5** — not started (blocked after L0 load failure)

## Failure

BLIP2 model load crashes consistently at **checkpoint shard 2/2**:

```
Loading checkpoint shards: 50%|#####| 1/2
exit code: -1073741819 (Windows ACCESS_VIOLATION)
```

Reproduced on:
- `test_grounding_eval.py` L0 (dsn 500 and dsn 150)
- Minimal `load_vllm_editor(...)` smoke test (baseline + patch ckpts)

GPU VRAM was mostly free (~472 MiB used); likely **system RAM / Windows page file** exhaustion.

## L0 workaround

**Step 1 baseline** used the **same** checkpoint and protocol:
- Ckpt: `VLKEB_long_baseline-2026.06.02-13.43.00/.../epoch-20-i-27800-ema_loss-0.7936`
- seed 42, dsn 500, sen 50, `grounding_eval_longform.json`
- Results: `eval_results/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step1_grounding_baseline-2026.07.05-12.20.17/...`

Registered as **L0 proxy** in `step6_run_manifest.json`.

## Retry L5 (after reboot / increase page file)

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
conda activate liveedit

python -u -W ignore tools/meeting_validation/step6/run_ablation_eval.py --layers L5 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_patch-2026.07.08-11.37.09/checkpoints/epoch-20-i-26250-ema_loss-1.3055" ^
  -dvc cuda:0 -dsn 500 -seed 42

python tools/meeting_validation/step6/analyze_ablation_results.py
python tools/meeting_validation/step6/plot_ablation_table.py
```

## L5 attempt result (2026-07-08)

L5 started but crashed after **~half of the eval loop** with:

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.15 MiB for an array with shape (224, 224, 3) and data type float64
```

It occurred inside `FrozenPatchEncoder.encode_patches()` while the Transformers image processor was rescaling (DINOv2 patch encoding). This is consistent with **system RAM / Windows pagefile exhaustion**, not just GPU VRAM.

Recommended mitigation for the next retry:
- rerun L5 with smaller `-dsn` (e.g. `150`)
- also consider reducing `-sen` (sequential_edit_n) to keep memory pressure down
- reboot / increase Windows pagefile again

Optional L0 re-run (same harness as Step 6):

```cmd
python -u -W ignore tools/meeting_validation/step6/run_ablation_eval.py --layers L0 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -dvc cuda:0 -dsn 500 -seed 42
```
