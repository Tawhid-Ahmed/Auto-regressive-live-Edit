# G5 — Second backbone setup (CVPR 2027)

**Started:** 2026-10-06  
**Decision due:** 2026-10-15  
**Gate log:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § G5

## Goal

Show DEK is not BLIP-2-only. Ladder: **LLaVA-1.5-7B** → stronger GPU (preferred) → **Qwen2-VL-2B** fallback.

## Local PC status (Oct 7 evening)

Long runs **stopped on this machine**. Continue on a stronger GPU box.

| Asset | Status |
|---|---|
| LLaVA weights `../models/llava-1.5-7b-hf` | present — copy to remote |
| L0 ckpt `epoch-3-i-1900-ema_loss-8.1575` | present — copy (DEK starts from this) |
| S2 ckpt `epoch-1-i-500-ema_loss-6.4867` | present — copy (eval only; S2 train done enough) |
| DEK ckpt | **none yet** — train fresh on remote from L0 |
| Local DEK train | **killed** (only reached ~2/500; no save) |

## What to copy to the remote GPU PC

```
models/llava-1.5-7b-hf/                          # ~14 GB
records/liveedit/llava-v1.5-7b/VLKEB_llava_L0_dn500-2026.07.25-19.25.39/checkpoints/epoch-3-i-1900-ema_loss-8.1575/
records/liveedit/llava-v1.5-7b/g5_vlkeb_llava_S2_dn500-2026.10.06-12.13.09/checkpoints/epoch-1-i-500-ema_loss-6.4867/
data/VLKEB/grounding/grounding_train_short.json
data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
data/VLKEB/VLKEB_images/                         # or whatever img_root you use
repo code (LiveEdit) + configs/liveedit/llava-v1.5-7b.yaml
```

Also useful: `tools/run_g5_llava_train.py`, `tools/smoke_g5_llava_load.py`.

Update `utils/GLOBAL.py` `ROOT_PATH` / model paths on the remote if the drive layout differs.

## Remote commands

```bash
# 1) Smoke (optional)
python -u tools/smoke_g5_llava_load.py -dvc cuda:0

# 2) DEK train from L0 (fresh; ~hours on a stronger GPU)
python -u tools/run_g5_llava_train.py --stage DEK -dvc cuda:0

# Or direct:
python -u train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 \
  -lkpt records/liveedit/llava-v1.5-7b/VLKEB_llava_L0_dn500-2026.07.25-19.25.39/checkpoints/epoch-3-i-1900-ema_loss-8.1575 \
  -eps 1 -tnp g5_vlkeb_llava_DEK_dn500 -dn 500 -rs 42 -sci 125 \
  -dpath data/VLKEB/grounding/grounding_train_short.json \
  --patch_grounding --adversarial_grounding \
  --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 \
  --avbl_hinge_tau 0.3 --avbl_contrastive --dek_dual_gate

# 3) After DEK finishes — grounding eval (L0 / S2 / DEK)
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
IMG=data/VLKEB/VLKEB_images/mmkb_images

python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt <L0_CKPT> -enp g5_llava_L0_short -dpath $PROBE --img_root $IMG --meeting_step 6 --log_routing

python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt <S2_i500_CKPT> -enp g5_llava_S2_short -dpath $PROBE --img_root $IMG --meeting_step 6 \
  --patch_grounding --log_routing

python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt <DEK_CKPT> -enp g5_llava_DEK_short -dpath $PROBE --img_root $IMG --meeting_step 6 \
  --patch_grounding --dek_dual_gate --log_routing --routing_rule and
```

## Checklist

- [x] Local smoke PASS
- [x] S2 usable at i=500 (local)
- [x] Local long trains stopped (Oct 7) — move to stronger GPU
- [ ] Remote: DEK train from L0
- [ ] Remote: grounding eval L0 / S2 / DEK + taxonomy
- [ ] Fill G5 Result in gate log by Oct 15

## Deferred (after G5)

M6 hn method try; `emits_y_i` recovery; EVQA pair-closed curve; full 5-seed; G4; MSCKE; CARML tau.
