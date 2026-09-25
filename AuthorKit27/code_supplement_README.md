# Code and Data Supplement

This archive accompanies the anonymous AAAI-27 submission *Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing*.

It contains the implementation of **Dual-Expert Keys (DEK)**, **adversarial visual binding training (AVBT)**, and the **adversarial grounding probe** evaluation protocol used in the paper (VLKEB and EVQA).

## Contents


| Path                                 | Description                                                 |
| ------------------------------------ | ----------------------------------------------------------- |
| `editor/`                            | Lifelong MoE editor with DEK, AVBT, and patch grounding     |
| `dataset/`                           | VLKEB / EVQA loaders and grounding-probe overlay            |
| `evaluation/`                        | Evaluation harness (reliability, generality, locality, FPR) |
| `configs/`                           | YAML configs for LiveEdit and competitor editors            |
| `utils/`                             | Path helpers and editor loading                             |
| `train_vllm_editor.py`               | Training entry point                                        |
| `test_vllm_edit.py`                  | Standard Rel/Gen/Loc evaluation                             |
| `test_grounding_eval.py`             | Adversarial grounding + FPR evaluation                      |
| `tools/build_grounding_eval_json.py` | Probe construction from VLKEB eval JSON                     |
| `data/*/grounding/*.json`            | Probe-augmented edit metadata (**images not included**)     |




## Prerequisites

The following are **not** included in this archive and must be obtained separately:

1. **BLIP-2 OPT-2.7B weights** — place under `models/blip2-opt-2.7b/`, or update `utils/GLOBAL.py` `model_path_map`.
2. **VLKEB images** — required for training and evaluation; see `data/README.md` for the expected layout.
3. **EVQA / COCO images** — required only for the EVQA stress protocol; see `data/README.md`.
4. **Lifelong baseline checkpoint** — required before training or evaluating the proposed method.



### Baseline checkpoint

The paper compares against a **lifelong LiveEdit baseline** trained on the original VLKEB training edits (no DEK, no AVBT). The DEK+AVBT runs in the paper are fine-tuned from this baseline.

To reproduce from scratch:

1. Train the lifelong baseline on VLKEB:
  ```
   python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 ^
     -eps 1 -tnp VLKEB_baseline -dn 500 -rs 42 -sci 125 -lpi 10
  ```
2. Note the saved checkpoint under `records/liveedit/blip2-opt-2.7b/<run>/checkpoints/`.
3. Pass that path to `-lkpt` when training DEK+AVBT (below), and use the resulting checkpoint with `-ckpt` at evaluation time.

Checkpoints are **not** redistributed here (size limits). Reviewers must train the baseline locally or use a checkpoint they already have from an equivalent VLKEB lifelong run.

## Setup

1. Python 3.10 recommended; install dependencies:
  ```
   pip install -r requirements.txt
  ```
2. Place model weights and images as described above.
3. `utils/GLOBAL.py` sets `ROOT_PATH` to this package directory by default.

**Hardware:** one CUDA GPU with enough memory for BLIP-2 OPT-2.7B (DINOv2 is loaded when patch grounding is enabled).

## Hyperparameters (paper)

- AVBT: `λ=0.3`, `β=2.5` (`--adv_grounding_neg_weight`), hinge `τ=0.3`, contrastive on
- DEK: `--dek_dual_gate`, text margin `m=0`
- Evaluation seeds: `{42, 0, 1, 2, 3}`
- Main VLKEB tables: `100` edits, `10` generations per example
- EVQA stress: `300` edits per seed



## Reproduce: train DEK + AVBT

Fine-tune from the lifelong baseline checkpoint (`-lkpt`):

```
python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 ^
  -eps 1 -tnp VLKEB_avbt_dek -dn 500 -rs 42 -sci 125 -lpi 10 ^
  -lkpt PATH_TO_BASELINE_CHECKPOINT ^
  -dpath data/VLKEB/grounding/grounding_train_short.json ^
  --patch_grounding --adversarial_grounding ^
  --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 ^
  --avbl_hinge_tau 0.3 --avbl_contrastive --dek_dual_gate
```



## Reproduce: evaluate FPR and reliability on grounding probes

```
python test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 ^
  -dsn 100 -seed 42 -ckpt PATH_TO_TRAINED_CHECKPOINT -enp dek_vlkeb ^
  -dpath data/VLKEB/grounding/grounding_eval_short.json ^
  --patch_grounding --dek_dual_gate --log_routing
```

Repeat with `-seed 42|0|1|2|3` for the multi-seed table. Use the lifelong baseline checkpoint with the same flags (except `--dek_dual_gate`) for the baseline row.

## Reproduce: EVQA stress

```
python test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 ^
  -dsn 300 -seed 42 -ckpt PATH_TO_TRAINED_CHECKPOINT -enp dek_evqa ^
  -dpath data/easy-edit-mm/grounding/grounding_eval_short.json ^
  --img_root data/easy-edit-mm/images ^
  --patch_grounding --dek_dual_gate --log_routing
```

Repeat for seeds `{42,0,1,2,3}`.

## Rebuild grounding probes

If needed, reconstruct probe JSON from the original VLKEB eval split:

```
python tools/build_grounding_eval_json.py ^
  -dpath PATH_TO_ORIGINAL_VLKEB_EVAL_JSON ^
  --img_root data/VLKEB/VLKEB_images/mmkb_images ^
  -out data/VLKEB/grounding/grounding_eval_short.json
```
