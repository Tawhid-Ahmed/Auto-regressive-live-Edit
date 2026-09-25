# Download Guide — Stress Transfer (LLaVA + EVQA/EIC)

**Date:** 2026-07-23  
**For plan:** [`AAAI_STRESS_TRANSFER_PLAN.md`](AAAI_STRESS_TRANSFER_PLAN.md)  
**Repo root:** `E:\MSC\MSc thesis project\LiveEdit\LiveEdit`

---

## What you already have (no re-download)

| Asset | Status | Location |
|-------|--------|----------|
| **BLIP2-OPT-2.7B** | Done | `E:\MSC\MSc thesis project\LiveEdit\models\blip2-opt-2.7b` |
| **VLKEB** JSON + images | Done | `LiveEdit\data\VLKEB\` |
| **Grounding probes** | Done | `data\VLKEB\meeting_validation\step1\grounding\` |
| **MMEdit JSON (EVQA/EIC)** | Mostly done | `data\easy-edit-mm\vqa\`, `caption\` |
| **MMEdit COCO images** | Present but **wrong folder name** | `data\easy-edit-mm\val2014\` (needs fix below) |

---

## Folder map (target layout)

```text
E:\MSC\MSc thesis project\LiveEdit\
├── models\
│   └── blip2-opt-2.7b\          ← already here (GLOBAL.py points here)
└── LiveEdit\                    ← repo (ROOT_PATH)
    ├── models\
    │   ├── llava-v1.5-7b-hf\    ← DOWNLOAD #1 (required for Phase 2)
    │   └── minigpt-4-vicuna-7b\ ← optional Phase 3
    └── data\
        ├── VLKEB\               ← already OK
        └── easy-edit-mm\
            ├── vqa\             ← already OK
            ├── caption\         ← already OK
            ├── locality\        ← already OK
            ├── multimodal_locality\
            └── images\          ← FIX #2 (create; put image folders under this)
                ├── val2014\
                └── val2014_image_rephrase\
```

Paths come from `utils/GLOBAL.py`:

| Key | Resolved path |
|-----|----------------|
| `blip2-opt-2.7b` | `...\LiveEdit\models\blip2-opt-2.7b` |
| `llava-v1.5-7b` | `...\LiveEdit\LiveEdit\models\llava-v1.5-7b-hf` |
| `minigpt-4-vicuna-7b` | `...\LiveEdit\LiveEdit\models\minigpt-4-vicuna-7b` |

---

## Download #1 — LLaVA-1.5-7B (required for backbone stress)

### From where

Hugging Face: **[llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)**

~13–14 GB (weights + processor files).

### Where to put it

```text
E:\MSC\MSc thesis project\LiveEdit\LiveEdit\models\llava-v1.5-7b-hf\
```

After download you should see files like `config.json`, `model-00001-of-00003.safetensors`, `tokenizer.json`, `preprocessor_config.json`, etc. **inside** that folder (not nested one extra level).

### How (recommended: Hugging Face CLI)

In PowerShell / Anaconda Prompt:

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
mkdir models 2>nul

E:\anaconda\envs\liveedit\python.exe -m pip install -U "huggingface_hub[cli]"

E:\anaconda\envs\liveedit\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='llava-hf/llava-1.5-7b-hf', local_dir=r'E:\MSC\MSc thesis project\LiveEdit\LiveEdit\models\llava-v1.5-7b-hf', local_dir_use_symlinks=False)"
```

Or with `hf` CLI (if installed):

```cmd
hf download llava-hf/llava-1.5-7b-hf --local-dir "E:\MSC\MSc thesis project\LiveEdit\LiveEdit\models\llava-v1.5-7b-hf"
```

### Verify

```cmd
E:\anaconda\envs\liveedit\python.exe -c "from utils.GLOBAL import model_path_map; import os; p=model_path_map['llava-v1.5-7b']; print(p, os.path.isdir(p), os.path.isfile(os.path.join(p,'config.json')))"
```

Expect: `... True True`

---

## Fix #2 — MMEdit / EasyEdit-MM image path (you already have the files)

LiveEdit loads EVQA with:

- JSON: `data/easy-edit-mm/vqa/vqa_train.json`
- Images root: `data/easy-edit-mm/images`
- JSON entries look like: `val2014/COCO_val2014_....jpg`

So the full path must be:

```text
data/easy-edit-mm/images/val2014/COCO_....jpg
```

You currently have:

```text
data/easy-edit-mm/val2014\          ← one level too high
data/easy-edit-mm/val2014_image_rephrase\
```

### Fix (junction — no copy, no extra disk)

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit\data\easy-edit-mm"
mkdir images 2>nul
mklink /J images\val2014 val2014
mklink /J images\val2014_image_rephrase val2014_image_rephrase
```

Run Command Prompt **as Administrator** if `mklink` fails (or use `cmd /c mklink ...`).

### Alternative (move folders)

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit\data\easy-edit-mm"
mkdir images
move val2014 images\
move val2014_image_rephrase images\
```

### If images were never downloaded

| What | URL |
|------|-----|
| E-VQA / E-IC JSON | [Google Drive folder](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) (from [EasyEdit MMEdit](https://github.com/zjunlp/EasyEdit)) |
| Shared images archive | [Google Drive file](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view) |
| Baidu mirror | [pan.baidu.com](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) (pwd `ik5c`) |

Unpack so you end with the layout in the folder map above. Docs: [EasyEdit README — MMEdit](https://github.com/zjunlp/EasyEdit).

### Verify EVQA paths

```cmd
E:\anaconda\envs\liveedit\python.exe -c "import json,os; d=json.load(open(r'data/easy-edit-mm/vqa/vqa_train.json',encoding='utf-8')); p=os.path.join('data/easy-edit-mm/images', d[0]['image']); print(p, os.path.isfile(p))"
```

Expect: `... True`

---

## Download #3 — MiniGPT-4 (optional only)

Skip unless Phase 2 (LLaVA) already works.

| Item | Detail |
|------|--------|
| **Where** | LiveEdit expects `models/minigpt-4-vicuna-7b` (Vicuna-7B based MiniGPT-4 weights used by LiveEdit) |
| **Put at** | `E:\MSC\MSc thesis project\LiveEdit\LiveEdit\models\minigpt-4-vicuna-7b\` |
| **Source** | Follow LiveEdit / MiniGPT-4 release notes for the exact Vicuna checkpoint they used; do not mix random MiniGPT builds |

Priority: **LLaVA first**. MiniGPT is Phase 3 optional.

---

## Download #4 — DINOv2 (usually auto)

AVBT / patch grounding uses DINOv2 via Hugging Face Transformers. First run will auto-download into your HF cache (no need to place under `models/`).

If offline later, it will already be cached under something like:

```text
C:\Users\<you>\.cache\huggingface\hub\
```

---

## VLKEB (already OK — only if regenerating)

| What | URL |
|------|-----|
| VLKEB repo / data | [github.com/VLKEB/VLKEB](https://github.com/VLKEB/VLKEB) · paper [arXiv:2403.07350](https://arxiv.org/abs/2403.07350) |
| Put at | `LiveEdit\data\VLKEB\` with `train.json`, `eval.json`, images under `VLKEB_images\` or `mmkb_images\` as your loader expects |

You do **not** need to re-download this for the stress plan.

---

## Checklist (do in order)

1. [ ] Create `LiveEdit\models\` if missing  
2. [ ] Download LLaVA → `models\llava-v1.5-7b-hf\`  
3. [ ] Verify LLaVA `config.json` exists  
4. [ ] Fix EasyEdit images → `data\easy-edit-mm\images\val2014\` (junction or move)  
5. [ ] Verify one EVQA image path resolves to a real file  
6. [ ] (Optional) MiniGPT later  
7. [ ] Return to [`AAAI_STRESS_TRANSFER_PLAN.md`](AAAI_STRESS_TRANSFER_PLAN.md) Phase 0 → Phase 1 (EVQA train)

---

## Disk estimate

| Item | Size |
|------|------|
| LLaVA-1.5-7B HF | ~13–14 GB |
| EasyEdit images (you have) | already on disk |
| MiniGPT (optional) | ~13 GB |
| DINOv2 cache | ~0.3–1 GB |

Free space tip: deleting the safe `records/` smoke folders from earlier (~40 GB) makes room for LLaVA easily.

---

## Quick “am I ready?” script

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
E:\anaconda\envs\liveedit\python.exe -c "
from utils.GLOBAL import model_path_map
import os, json
print('BLIP2', os.path.isfile(os.path.join(model_path_map['blip2-opt-2.7b'],'config.json')))
print('LLaVA', os.path.isfile(os.path.join(model_path_map['llava-v1.5-7b'],'config.json')))
print('EVQA json', os.path.isfile('data/easy-edit-mm/vqa/vqa_train.json'))
d=json.load(open('data/easy-edit-mm/vqa/vqa_train.json',encoding='utf-8'))
img=os.path.join('data/easy-edit-mm/images', d[0]['image'])
print('EVQA img', img, os.path.isfile(img))
print('VLKEB train', os.path.isfile('data/VLKEB/train.json'))
print('Grounding short', os.path.isfile('data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json'))
"
```

When all prints `True` except MiniGPT, you can start **Phase 1 EVQA training** and **Phase 2 LLaVA** from the stress plan.
