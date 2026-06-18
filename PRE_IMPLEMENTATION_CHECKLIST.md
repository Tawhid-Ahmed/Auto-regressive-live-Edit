# Pre-Implementation Checklist — Direction A

Complete **every item** below before writing any new method code.
Mark each item with [x] when done and note the result.

**Automation:** `python test_long_target.py` (Check 1); `python tools/preimplementation_checks.py` (Checks 2–6 log); `python tools/bertscore_smoke.py` (BERTScore GPU); `python tools/build_vlkeb_eval_longform_alt_json.py` (long-`alt` eval JSON).

---

## CHECK 1: GPU & Hardware

### 1.1 Confirm GPU specs

```bash
nvidia-smi
```

- GPU model: **NVIDIA GeForce RTX 5060 Ti** (see `test_long_target.py` / `nvidia-smi -q`)
- VRAM total: **16311 MiB**
- VRAM currently free: *varies with desktop load; snapshot ~1.2 GiB used at idle check*
- Driver version: **595.97**
- CUDA version: **13.2** (driver-reported)

### 1.2 Forward+backward pass with long target

Run a single training step with a ~150-token target to test memory limits.

```python
# Save as: test_long_target.py
# Run: conda activate liveedit && python test_long_target.py

import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_vllm_editor
from utils.GLOBAL import ROOT_PATH

device = 'cuda:0'
editor = load_vllm_editor('liveedit', 'blip2', device, None, None, False)

# Simulate a long target (150 tokens worth of text)
long_target = "The image depicts " + " ".join(["word"] * 140)
short_target = "A cat"

# Test short (should work — baseline)
print("Testing SHORT target...")
try:
    (x, vt), y, m = editor.vllm.prompts_imgs_target_to_xym(
        ["What is in this image? The answer is:"], [None], [short_target])
    print(f"  Input shape: {x['inputs_embeds'].shape}, Label shape: {y.shape}")
    print("  SHORT target: OK")
except Exception as e:
    print(f"  SHORT target FAILED: {e}")

# Test long
print("Testing LONG target (~150 tokens)...")
try:
    (x, vt), y, m = editor.vllm.prompts_imgs_target_to_xym(
        ["What is in this image? The answer is:"], [None], [long_target])
    print(f"  Input shape: {x['inputs_embeds'].shape}, Label shape: {y.shape}")
    print("  LONG target: OK")
except Exception as e:
    print(f"  LONG target FAILED: {e}")

# Check memory after loading
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved() / 1e9
print(f"\nGPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
```

- Short target passes: **yes** (`test_long_target.py`)
- Long target passes: **yes** (~144 OPT tokens, text-only `xym`; vision train step uses dummy image)
- GPU memory after model load: **~7.94 GB** allocated (post-`xym` smoke)
- Maximum batch size at 150 tokens before OOM: **B=16** (batched `prompts_imgs_target_to_xym` only); **B=12** (batched `xym` + LM forward/backward with grad on `inputs_embeds` — surrogate for packed batch VRAM)

### 1.3 Training step with long target

If 1.2 passes, test a full training step (forward + backward + optimizer step).

```python
# Add to the script above, after the long target test:
print("Testing training step with LONG target...")
try:
    editor.reinit_train_parameters()
    opt = editor.get_a_new_optimizer()
    if isinstance(opt, tuple):
        optimizer, scheduler = opt
    else:
        optimizer = opt

    # Make a fake edit sample
    (x, vt), y, m = editor.vllm.prompts_imgs_target_to_xym(
        ["Describe this image in detail:"], [None], [long_target])

    optimizer.zero_grad()
    # Attempt a forward pass through the editor's train path
    # (This is method-specific; adjust if train_a_batch has different signature)
    print("  Forward pass...")
    logits = editor.vllm.get_llm_outpt(x, vt).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, -y.shape[1]:, :].reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction='none'
    )
    loss = (loss * m.reshape(-1)).sum() / m.sum()
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print("  Backward pass: OK")
    optimizer.step()
    print("  Optimizer step: OK")
    print("  TRAINING STEP WITH LONG TARGET: PASSED")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  TRAINING STEP FAILED: {e}")

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  Peak GPU memory: {peak:.2f} GB")
```

- Training step passes: **yes** (LiveEdit reliability path, long target + dummy image)
- Peak GPU memory during training: **~8.44 GB** (single-sample step)
- If OOM: at what batch size? **n/a** on surrogate sweep at B≤12; lower B if full-editor batch OOMs
- Gradient accumulation needed? **not strictly required** for the LM surrogate at B=12; **use if** full `train_a_batch` OOMs at your real batch size

---

## CHECK 2: Tokenizer & Sequence Lengths

### 2.1 Confirm max sequence length

```python
# Run in liveedit env:
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    r"E:\MSC\MSc thesis project\LiveEdit\models\blip2-opt-2.7b",
    use_fast=False)

# Check model's max position embeddings
from transformers import Blip2ForConditionalGeneration
model = Blip2ForConditionalGeneration.from_pretrained(
    r"E:\MSC\MSc thesis project\LiveEdit\models\blip2-opt-2.7b",
    local_files_only=True)
max_pos = model.language_model.config.max_position_embeddings
print(f"Max position embeddings: {max_pos}")

# Check how many tokens BLIP2 QFormer produces
num_query_tokens = model.config.num_query_tokens
print(f"QFormer query tokens (image tokens): {num_query_tokens}")
```

- Max position embeddings: **2048** (OPT-2.7B in BLIP2; from `Blip2Config` / `tools/preimplementation_checks.py`)
- QFormer query tokens: **32**

### 2.2 Token budget calculation

```
Available for prompt + target = max_positions - query_tokens - special_tokens
                              = **2048** - **32** - 2
                              = **2014** tokens (heuristic)

If prompt is ~50 tokens:
  Max target length ≈ **2014** - 50 = **~1964** tokens
```

- Max feasible target length: **~1964** tokens (rough, before other specials / formatting)
- Your planned max target length (<=150 recommended): **150** tokens
- Fits within budget: **yes**

### 2.3 Token length distribution of planned targets

After generating pilot long-form answers (Check 4), run:

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    r"E:\MSC\MSc thesis project\LiveEdit\models\blip2-opt-2.7b",
    use_fast=False)

# Replace with your pilot data path
data = json.load(open("data/VLKEB/eval_longform_pilot.json"))

lengths = []
for d in data:
    n = len(tok.encode(d["alt_long"], add_special_tokens=False))
    lengths.append(n)

import collections
bins = {"<=16": 0, "17-64": 0, "65-150": 0, "150+": 0}
for n in lengths:
    if n <= 16: bins["<=16"] += 1
    elif n <= 64: bins["17-64"] += 1
    elif n <= 150: bins["65-150"] += 1
    else: bins["150+"] += 1

print(f"Total: {len(lengths)}")
print(f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.1f}")
print(f"Bins: {bins}")
```

- Distribution covers all 3 target bins: **yes** (`eval_longform_pilot.json` `alt_long`: bins ≤16, 17–64, 65–150 all non-zero in automated run)
- No target exceeds token budget: **yes** (max `alt_long` length 89 tok in that pilot file)

---

## CHECK 3: Existing Code Compatibility

### 3.1 Dataset loading (dataset/vllm.py)

- Read `VLKEB.__init__` and `__init_eic_evqa__` — does it truncate `alt`?
  - Truncation found: **no** (`target_new = d['alt']` as-is, ~line 87)
  - If yes, where (line number): **—**
- Does the data loader handle multi-sentence strings in `alt`?
  - Tested with: `"alt": "This is a long answer with multiple sentences. It describes the image in detail."`
  - Result: **pass** (temp JSON + `VLKEB(..., preload_images=False)` in `tools/preimplementation_checks.py`)

### 3.2 Embedding construction (editor/vllms_for_edit/base.py)

- Read `prompts_imgs_target_to_xym` — any max_length or truncation?
  - File: `editor/vllms_for_edit/base.py`, line: **~79–113**
  - Truncation found: **no** (tokenizer default; no `max_length` here)
- Read `get_llm_input_embeds` — any sequence length limits?
  - File: `editor/vllms_for_edit/blip2/blip2.py`, line: **~185–233**
  - Limits found: **no** explicit clamp (model still bound by OPT max positions)

### 3.3 BLIP2 processor

- Does `Blip2Processor` or `OPTTokenizer` auto-truncate?
  - Check: `tok("long string " * 100, truncation=False, return_tensors="pt")`
  - **Result:** `truncation=False` → long output; `truncation=True,max_length=128` → 128 tok. **No silent trunc** in `prompts_imgs_target_to_xym` path without `truncation=True`.

### 3.4 Training loop (editor/vllm_editors/base.py)

- Read `train_a_batch` — any assumption about target length?
  - File: line: **~98–105** (abstract); implementation **`liveedit.py` ~479+**
  - Issues found: **none** (no fixed max target length)

### 3.5 Evaluation (evaluation/vllm_editor_eval.py)

- EM computation works for long sequences (no off-by-one errors)?
  - Tested with 100-token target: **yes** (synthetic tensors + `_chunk_em_f1` in `tools/preimplementation_checks.py`)
- `chunk_em` and `chunk_f1` compute correctly for multi-chunk targets?
  - Tested: **yes** (same smoke)
- Length bins populate correctly for targets in 17-64 and 65+ ranges?
  - Tested: **yes** (eval on `data/VLKEB/eval_longform_alt_10.json` with **`test_vllm_edit.py -dpath`**; `alt` token lengths 9–87 include **17–32**, **33–64**, and **65+** → `get_mean_results` **length_bin** gets non-zero **count** for those bins; inspect `eval_results/.../sequential_edit_10/seed_42_mean_results.json` → `total_mean` → `length_bin`)

### 3.6 AR-mode chunk utilities

- Read `chunk_utils.py` — does `split_target_into_chunks` handle 100+ token targets?
  - File: `editor/vllm_editors/liveedit/chunk_utils.py`
  - Tested with 100-token string: **yes** (~241 tokens → **16** chunks @ chunk_size 16)
  - Number of chunks produced: **16**

---

## CHECK 4: Pilot Data Generation

### 4.1 Generate 10-20 pilot long-form answers

Take 10-20 VLKEB eval entries and create long-form `alt` answers.

**Method A (GPT-4V/GPT-4o):**

```
Prompt to GPT-4V:
"Given this image and the question '{src}', provide a detailed answer
in 3-5 sentences (approximately 50-120 words). The answer should be
factually grounded in what is visible in the image. Do not hallucinate
details that are not visible."
```

**Method B (Manual):**
Write 10 answers yourself, 50-100 words each, grounded in the VLKEB images.

- Pilot samples generated: **15** (`alt_long`) + **10** long `alt` train (`pilot_train_longform_10.json`)
- Token length range (`alt_long` in pilot): **9** to **89** tokens (OPT, `add_special_tokens=False`)
- Quality check: answers are factually grounded in images? **Synthetic template** (Method B–style stubs for pipeline testing; replace with GPT‑4V / manual for thesis-quality grounding)
- Saved to: **`data/VLKEB/eval_longform_pilot.json`**, **`data/VLKEB/pilot_train_longform_10.json`**, **`data/VLKEB/eval_longform_alt_10.json`** (rebuild: `python tools/build_vlkeb_eval_longform_alt_json.py`)

### 4.2 End-to-end pipeline test with pilot data

Feed pilot data through the existing training + eval pipeline:

```bash
# 1. Pilot train JSON: data/VLKEB/pilot_train_longform_10.json (committed / regenerated via tools)
# 2. Train 1 epoch (save ckpt each step for short runs; data buffer 0 is clamped to 1 in ParallelDataset)
python -u -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB \
  -bs 1 -dvc cuda:0 -edvc 0 -lkpt none -tnp PILOT_longform \
  -eps 1 -sci 1 -lpi 1 -dbs 0 \
  -dpath data/VLKEB/pilot_train_longform_10.json

# 3a. Eval on standard short eval.json (regression smoke)
python -u -W ignore test_vllm_edit.py -en liveedit -mn blip2 -sen 10 \
  -dvc cuda:0 -dn VLKEB -dsn 10 -seed 42 -enp pilot_longform \
  -ckpt "records/liveedit/blip2-opt-2.7b/PILOT_longform-<timestamp>/checkpoints/epoch-1-i-10-..."

# 3b. Long `alt` smoke + length bins (uses -dpath; optional -ckpt none for pipeline-only)
python tools/build_vlkeb_eval_longform_alt_json.py
python -u -W ignore test_vllm_edit.py -en liveedit -mn blip2 -sen 10 \
  -dvc cuda:0 -dn VLKEB -dsn 10 -seed 42 -enp longform_alt_smoke \
  -ckpt none -dpath data/VLKEB/eval_longform_alt_10.json
```

- Training completes without crash: **yes**
- Eval completes without crash: **yes** (short `eval.json` + pilot ckpt; long `alt` + `-dpath` smoke)
- Length bins in results JSON have non-zero counts for 17-64 or 65+: **yes** (long-form eval file + `-dpath`; bins **17–32**, **33–64**, **65+** per `target_token_count`)
- If failed, error message: **—** (prior `Waiting data` hang fixed: `dataset/__init__.py` clamps `buffer_size` ≥ 1)

---

## CHECK 5: Evaluation Metrics

### 5.1 Decide on metrics

- Primary metric for short targets: EM + Token Accuracy (same as LiveEdit)
- Primary metric for long targets: **ROUGE-L** (+ token accuracy where aligned)
- Secondary metrics: **BERTScore**, **Chunk EM / Chunk F1** (AR path)

### 5.2 ROUGE-L availability

```bash
pip list | findstr rouge
# If not installed:
pip install rouge-score
```

- `rouge-score` package installed: **yes**
- Version: **0.1.2** (`pip show rouge-score`; Apr 2026 env)

### 5.3 BERTScore availability

```bash
pip list | findstr bert-score
# If not installed:
pip install bert-score
```

- `bert-score` package installed: **yes**
- Can run on your GPU alongside the editor? (BERTScore loads a small model)
  - Tested: **yes** (`python tools/bertscore_smoke.py` → **cuda:0**, F1 ≈ **0.91** on close paraphrase pair)
  - If OOM: compute BERTScore separately after eval (save predictions to file)

### 5.4 Metric sanity check

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Perfect match
ref = "The cat sat on the mat in the living room near the window"
hyp = "The cat sat on the mat in the living room near the window"
print("Perfect:", scorer.score(ref, hyp))  # Should be 1.0

# Close match (2 words changed)
hyp2 = "The dog sat on the mat in the living room near the door"
print("Close:", scorer.score(ref, hyp2))  # Should be ~0.8-0.9

# Poor match
hyp3 = "A bird flew over the ocean during sunset"
print("Poor:", scorer.score(ref, hyp3))    # Should be <0.3

# Exact match (EM comparison)
em_perfect = int(ref == hyp)
em_close   = int(ref == hyp2)
print(f"EM perfect: {em_perfect}, EM close: {em_close}")
# Shows why EM is too harsh for long-form
```

- ROUGE-L produces sensible scores: **yes** (perfect 1.0, close ~0.85, poor ~0.10 in `tools/preimplementation_checks.py`)
- EM = 0 for "close" match confirms EM is too harsh for long-form: **yes** (EM 1 vs 0 on close string pair)

---

## CHECK 6: Scope Decisions (write down and freeze)

### 6.1 Target length

- Maximum target token length: **150** tokens (hard cap for Direction A v1)
- Length bins: Short (**≤16**), Medium (**17–64**), Long (**65+**) — eval code also reports **17–32**, **33–64**, **65+** sub-bins
- Minimum samples per bin in eval set: **≥50** per populated bin (target after full dataset build)

### 6.2 Dataset size

- Training samples (long-form): **300–500** (initial target; scale with thesis timeline)
- Eval samples (long-form): **300–500**
- Eval samples (short-form, for regression): **200** (VLKEB-style short `alt`)

### 6.3 Baselines

- Baseline 1: LiveEdit (standard, no AR) — available
- Baseline 2: FT-L (fine-tune last layer) — available
- Baseline 3: **AR-LiveEdit** (chunk-wise; same backbone)
- Baseline 4 (stretch): **MEND / SERAC VL** (only if time)

### 6.4 Ablations

- Ablation 1: **No AR chunking** (single-pass vs chunk path)
- Ablation 2: **Fixed chunk size vs vision-aware chunking** (if implemented)
- Ablation 3: **No cross-modal consistency loss** (if implemented)
- Chunk size sweep: values to test: **{8, 16, 32}**

### 6.5 Lifelong edit counts

- Edit counts to test: 1, 10, 100, **1000** (match LiveEdit where GPU allows)

### 6.6 Random seeds

- Seeds: **42**, **123**, **456** (minimum 3)

### 6.7 VLM backbone

- BLIP2-OPT-2.7B only (single backbone is fine for thesis)
- Second backbone (stretch goal, only if time permits): **none planned** (e.g. LLaVA-7B only if surplus time)

---

## CHECK 7: Theory Decision

### 7.1 Multimodal MI extension

- [x] Option A (thesis-level): 1-paragraph argument that frozen image encoder
means V is a constant prefix, so AnyEdit's proof applies unchanged
- [ ] Option B (paper-level): 1-page formal extension with V as explicit
conditioning variable in the MI chain rule
- [x] Chosen: **A** (Option B only if venue/reviewer demands)

### 7.2 Key sentence to include

Draft the 1-paragraph version now (even if you choose Option B later):

> "Since the image encoder in BLIP2 is frozen during editing, the visual
> representation V enters the language model as a fixed token prefix.
> Consequently, V acts as a constant conditioning variable in the mutual
> information decomposition. AnyEdit's chain rule (Equation 8) generalizes
> directly by absorbing V into the extended input X' = (X, V), yielding
> I(X', Y_1, ..., Y_{k-1}; Y_k | h'_k) for each chunk k. The two
> autoregressive properties (later hidden states do not affect earlier
> outputs; conditioning on Y_k subsumes h'_k) hold unchanged because
> the VLM's language decoder remains autoregressive."

- [x] Drafted and reviewed: **yes** (§7.2 paragraph; align with thesis text)

---

## FINAL GO / NO-GO

All critical checks (1.2, 1.3, 2.2, 3.1-3.5, 4.2, 5.4, 6.*, 7.*) must pass.


| Category                      | Status | Blocker?                |
| ----------------------------- | ------ | ----------------------- |
| GPU handles long targets      | [x]    | Yes                     |
| Pipeline handles long targets | [x]    | Yes                     |
| Token budget sufficient       | [x]    | Yes                     |
| No code truncation issues     | [x]    | Yes (must fix if found) |
| Pilot data generated          | [x]    | Yes                     |
| Metrics decided and working   | [x]    | Yes                     |
| Scope frozen                  | [x]    | Yes                     |
| Theory approach chosen        | [x]    | No (can decide later)   |


**If all "Yes" blockers pass → proceed to Phase 2 (Dataset Construction)**
**If any blocker fails → fix it before proceeding**