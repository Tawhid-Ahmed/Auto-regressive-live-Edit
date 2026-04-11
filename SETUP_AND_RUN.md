# LiveEdit: Full Setup and Run Instructions

This document provides step-by-step instructions to set up the environment and run the LiveEdit project from scratch (including datasets, model weights, training, and evaluation).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Project Layout](#2-clone-and-project-layout)
3. [Environment Setup](#3-environment-setup)
4. [Configure Paths](#4-configure-paths)
5. [Download Datasets](#5-download-datasets)
6. [Download Model Weights](#6-download-model-weights)
7. [Verify Setup](#7-verify-setup)
8. [Train an Editor](#8-train-an-editor)
9. [Evaluate an Editor](#9-evaluate-an-editor)
10. [Optional: AR-LiveEdit Validation and Comparison](#10-optional-ar-liveedit-validation-and-comparison)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- **Python**: 3.8+ (3.9 or 3.10 recommended).
- **CUDA**: GPU with CUDA support; install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and matching cuDNN if you plan to use GPU.
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) (recommended for environment isolation).
- **Git**: To clone the repository.
- **Disk**: Enough space for datasets and model weights (tens of GB depending on choices).

---

## 2. Clone and Project Layout

Clone the repository (or use your existing clone):

```bash
git clone <repository-url>
cd LiveEdit
```

Ensure you are in the **project root** (the folder that contains `train_vllm_editor.py`, `test_vllm_edit.py`, `utils/`, `data/`, etc.). All commands below assume you run them from this root.

---

## 3. Environment Setup

### 3.1 Create Conda Environment

Create and activate a dedicated environment (the project uses the name `liveedit` in scripts):

```bash
conda create -n liveedit python=3.10 -y
conda activate liveedit
```

### 3.2 Install PyTorch (GPU)

Install PyTorch with CUDA support matching your driver. Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

### 3.3 Install Dependencies

Install the main Python dependencies:

```bash
pip install transformers
pip install Pillow
pip install tqdm
pip install numpy
pip install pyyaml
```

Optional (needed only if you use MiniGPT-4 or LLaVA and their configs use OmegaConf):

```bash
pip install omegaconf
```

### 3.4 Verify Environment

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## 4. Configure Paths

The project expects two things to be set correctly.

### 4.1 Project Root and Model Paths

Edit **`utils/GLOBAL.py`** and set:

- **`ROOT_PATH`**: Path to the project root (the `LiveEdit` folder). Can be absolute or relative to the current working directory.

- **Model paths**: The script uses paths **relative to `ROOT_PATH`** (e.g. `models/blip2-opt-2.7b`). You only need to set `ROOT_PATH`; model directories are resolved as `ROOT_PATH` + relative path. To add or change a backbone, edit `_model_rel_paths` in `utils/GLOBAL.py`.

Example (Windows):

```python
ROOT_PATH = r'E:\MSC\MSc thesis project\LiveEdit\LiveEdit'
# Model paths are relative to ROOT_PATH (e.g. models/blip2-opt-2.7b)
```

Example (Linux/Mac):

```python
ROOT_PATH = '/home/user/LiveEdit'
# Model paths are relative to ROOT_PATH (e.g. models/blip2-opt-2.7b)
```

You only need to add entries in `_model_rel_paths` for the backbones you actually use (e.g. only `blip2-opt-2.7b` for the examples below).

---

## 5. Download Datasets

Place datasets under the `data/` directory as follows.

### 5.1 MMEdit Dataset (for EVQA / EIC)

- Source: URL from the MMEdit paper *Can We Edit Multimodal Large Language Models?* (EMNLP 2023).
- Extract and place contents so that:
  - **`data/easy-edit-mm/`** contains:
    - `vqa/vqa_train.json`, `vqa/vqa_eval.json`
    - `caption/caption_train_edit.json`, `caption/caption_eval_edit.json`
    - Image folders as referenced in the JSON (e.g. `val2014/` for COCO under `data/easy-edit-mm/`).

### 5.2 VLKEB Dataset (for VLKEB training and evaluation)

- Source: **VLKEB** (NeurIPS 2024 Datasets and Benchmarks) — use the official URL from the [VLKEB repository](https://github.com/VLKEB/VLKEB).
- Place the dataset so that:
  - **`data/VLKEB/`** contains:
    - `train.json`
    - `eval.json`
  - **`data/VLKEB/VLKEB_images/`** (or `data/VLKEB/VLKEB_images/mmkb_images/`) contains the images referenced in the JSON.

If your VLKEB images live under `data/VLKEB/VLKEB_images/mmkb_images`, the code in this repo already uses that path for training/eval.

---

## 6. Download Model Weights

Download the backbone model you want to use and place it under your project root using the **relative** paths defined in `_model_rel_paths` in `utils/GLOBAL.py` (e.g. `models/blip2-opt-2.7b` → `ROOT_PATH/models/blip2-opt-2.7b`).

### 6.1 BLIP2-OPT-2.7B (recommended for quick start)

- Use Hugging Face: e.g. [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) or the variant used by the paper.
- Download/clone to `models/blip2-opt-2.7b` (relative to project root). No path change needed in config if you set `ROOT_PATH` correctly.

### 6.2 Other backbones (optional)

- **LLaVA v1.5 7B**: Place under `models/llava-v1.5-7b-hf` (relative to `ROOT_PATH`).
- **MiniGPT-4 Vicuna 7B**: Place under `models/minigpt-4-vicuna-7b` (relative to `ROOT_PATH`).

---

## 7. Verify Setup

- **Paths**: Check that `ROOT_PATH` exists and that the model directory (e.g. `ROOT_PATH/models/blip2-opt-2.7b`) contains the expected files (e.g. `config.json`, `pytorch_model.bin` or safetensors for BLIP2).
- **Data**: Check that at least one dataset is present (e.g. `data/VLKEB/train.json` and `data/VLKEB/eval.json` and the image directory).
- **Env**: From project root with `conda activate liveedit`:

```bash
python -c "
from utils.GLOBAL import ROOT_PATH, model_path_map
import os
print('ROOT_PATH:', ROOT_PATH, 'exists:', os.path.isdir(ROOT_PATH))
for k, v in model_path_map.items():
    print('  ', k, ':', v, 'exists:', os.path.isdir(v))
"
```

---

## 8. Train an Editor

Training is done with **`train_vllm_editor.py`** from the project root.

### 8.1 Basic training (e.g. EVQA, LiveEdit, BLIP2)

```bash
conda activate liveedit
python train_vllm_editor.py -en liveedit -mn blip2 -dna EVQA -bs 8 -dvc "cuda:0" -edvc 0 -lkpt None -tnp EVQA -eps 50 -sci 500
```

- **`-en`**: Editor name (`liveedit`).
- **`-mn`**: Model short name (`blip2`).
- **`-dna`**: Dataset name (`EVQA`, `EIC`, or `VLKEB`).
- **`-bs`**: Batch size.
- **`-dvc`**: Device (e.g. `cuda:0`).
- **`-edvc`**: Extra device index for data loading (e.g. `0` = same GPU).
- **`-lkpt`**: Load checkpoint path or `None`.
- **`-tnp`**: Train name prefix (used in saving).
- **`-eps`**: Epochs.
- **`-sci`**: Save checkpoint every N iterations.

### 8.2 Training on VLKEB (small subset)

```bash
python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 -edvc 0 -lkpt None -tnp VLKEB -eps 1 -lpi 1
```

- **`-dn 6`**: Use only 6 samples (for a quick run).

### 8.3 AR-LiveEdit training (chunk-wise autoregressive)

Add **`--ar_mode`** and optionally **`--chunk_size`** / **`--max_chunks`**:

```bash
python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 --ar_mode -eps 1 -lpi 1
```

Checkpoints are saved under **`records/liveedit/blip2-opt-2.7b/<train_name_prefix>/checkpoints/`**.

---

## 9. Evaluate an Editor

Evaluation is done with **`test_vllm_edit.py`**.

### 9.1 Without a trained checkpoint (no checkpoint)

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 1000 -dvc cuda:0 -dn EVQA -ckpt None
```

### 9.2 With a trained checkpoint

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 1000 -dvc cuda:0 -dn EVQA -ckpt "records/liveedit/blip2-opt-2.7b/EVQA/checkpoints/ckpt"
```

Replace the `-ckpt` path with your actual checkpoint path or prefix.

### 9.3 VLKEB evaluation (with optional seed for reproducibility)

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB/checkpoints/ckpt" -seed 42
```

- **`-dsn`**: Number of evaluation samples.
- **`-seed`**: Fixed random seed for reproducible splits (useful when comparing baseline vs AR).

### 9.4 AR-LiveEdit evaluation

Add **`--ar_mode`** (and optionally **`--chunk_size`** / **`--max_chunks`** if needed):

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records/..." --ar_mode -seed 42
```

Results are written under **`eval_results/liveedit/blip2-opt-2.7b/<dataset_name>/`**.

---

## 10. Optional: AR-LiveEdit Validation and Comparison

### 10.1 Batch structure verification (no GPU required for first part)

```bash
conda activate liveedit
python verify_ar_batch.py
```

### 10.2 Unit tests (no model load)

```bash
python -m editor.vllm_editors.liveedit.test_ar_train_step_logic
```

### 10.3 One AR training step (GPU + env)

```bash
python -W ignore validate_ar_train_step.py
```

### 10.4 One short AR epoch

```bash
python -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 --ar_mode -eps 1 -lpi 1
```

### 10.5 Full AR validation script (Windows)

From project root:

```batch
run_ar_validation_liveedit.bat
```

This runs the one-step validation and one short epoch (requires conda env `liveedit` and enough RAM/GPU).

### 10.6 LiveEdit vs AR-LiveEdit comparison

1. Set checkpoint paths in **`run_liveedit_ar_comparison.bat`** (variables `CKPT` and optionally `AR_CKPT`), or run the Python script directly.
2. Run:

```batch
run_liveedit_ar_comparison.bat
```

Or:

```bash
python -W ignore run_liveedit_ar_comparison.py -dvc cuda:0 -ckpt "records/.../checkpoints/ckpt" --ar_ckpt "..." -dsn 200 -sen 50 -seed 42
```

Comparison output: **`eval_results/comparison/liveedit_vs_ar_seed42.json`** (or the seed you set).

### 10.7 Task 9: AR ablation (chunk_size, routing gate, max_chunks)

Run the small ablation set (chunk_size: 8/16/32, routing_gate: on/off, max_chunks: unlimited/4), then get one compact table and a recommendation:

```batch
run_ar_ablation.bat
```

Or:

```bash
python -W ignore run_ar_ablation.py -dvc cuda:0 -ckpt "records/.../checkpoints/ckpt" -dsn 100 -sen 25 -seed 42
```

To only build the table from existing eval results (no new eval runs):

```bash
python -W ignore run_ar_ablation.py --skip_run -seed 42
```

Output: **`eval_results/ablation/ar_ablation_seed42.json`** (ablation table + best-config recommendation).

---

## 11. Troubleshooting

### 11.1 `OSError: The paging file is too small for this operation to complete (os error 1455)`

- **Cause**: Not enough virtual memory when loading the model (common on Windows with large models).
- **Fix**: Increase Windows virtual memory (System → Advanced system settings → Performance → Advanced → Virtual memory), or use a machine with more RAM, or reduce batch size and model load (e.g. load on CPU first as in the BLIP2 code).

### 11.2 CUDA out of memory

- Reduce **`-bs`** (batch size).
- Use a single GPU and **`-edvc 0`** so the code does not load a second copy of the model for data processing.
- Close other GPU applications.

### 11.3 Dataset not found

- Confirm **`ROOT_PATH`** in `utils/GLOBAL.py` is the absolute path to the project root.
- Confirm directory layout matches what the code expects (see [Section 5](#5-download-datasets)).
- For VLKEB, ensure image paths in JSON match the actual location (e.g. `VLKEB_images/mmkb_images`).

### 11.4 Model not found

- Ensure the path in **`model_path_map`** points to the correct model folder (with `config.json`, tokenizer files, and weights).
- For BLIP2, the code expects a Hugging Face–style layout (e.g. from `Blip2ForConditionalGeneration.from_pretrained(model_path)`).

### 11.5 Conda env not found

- Create the env first: `conda create -n liveedit python=3.10 -y`, then `conda activate liveedit`, then install dependencies as in [Section 3](#3-environment-setup).
- On Windows, if `run_ar_validation_liveedit.bat` fails to activate, run the same commands manually in a terminal after `conda activate liveedit`.

---

## Quick Reference: Common Commands

| Task | Command (from project root, `conda activate liveedit`) |
|------|--------------------------------------------------------|
| Train (EVQA, 50 epochs) | `python train_vllm_editor.py -en liveedit -mn blip2 -dna EVQA -bs 8 -dvc cuda:0 -edvc 0 -lkpt None -tnp EVQA -eps 50 -sci 500` |
| Train (VLKEB, 1 epoch, AR) | `python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 2 -dvc cuda:0 -dn 6 --ar_mode -eps 1 -lpi 1` |
| Eval (no ckpt) | `python test_vllm_edit.py -en liveedit -mn blip2 -sen 1000 -dvc cuda:0 -dn EVQA -ckpt None` |
| Eval (with ckpt, VLKEB, seed) | `python test_vllm_edit.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records/.../checkpoints/ckpt" -seed 42` |
| AR validation (one step) | `python -W ignore validate_ar_train_step.py` |
| AR validation (batch) | `run_ar_validation_liveedit.bat` |
| LiveEdit vs AR comparison | `run_liveedit_ar_comparison.bat` or `python -W ignore run_liveedit_ar_comparison.py ...` |
| AR ablation (Task 9) | `run_ar_ablation.bat` or `python -W ignore run_ar_ablation.py -ckpt "..." -dsn 100 -sen 25 -seed 42` |

---

*For the paper and citation, see the main [README.md](README.md).*
