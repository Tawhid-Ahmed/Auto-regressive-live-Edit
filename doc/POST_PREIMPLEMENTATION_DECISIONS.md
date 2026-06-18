# Post–Pre-Implementation Checklist: Decisions & Updates

This document records **what was verified**, **what changed in the repo**, and **what was decided** after completing the Direction A pre-implementation checklist (see `PRE_IMPLEMENTATION_CHECKLIST.md`). It is the single place to align thesis scope, engineering, and next steps.

**Date context:** checklist automation and pilot runs completed in **April 2026**. Update this file when scope or defaults change.

---

## 1. Executive summary

- **Hardware / BLIP2:** Long targets (~150 OPT tokens) and a **LiveEdit-style** training step run on **image-only BLIP2-OPT-2.7B** without OOM in the pilot configuration; batch-size headroom was measured with a documented surrogate (see checklist).
- **Tokenizer budget:** OPT max positions **2048**, Q-Former tokens **32** → rough **~2014** tokens for prompt + target before other overhead; **planned cap 150** tokens for long-form v1 **fits**.
- **Code paths:** No silent truncation of `alt` in VLKEB loading; `prompts_imgs_target_to_xym` does not impose `max_length`; training/eval handle long strings if the tokenizer stays under the LM context limit.
- **Pilot & pipeline:** Pilot JSON files, **train** + **eval** smoke (including long-`alt` eval via `-dpath`), **ROUGE-L** and **BERTScore** installs + sanity runs are in place.
- **Thesis scope (frozen for v1):** **Image-only BLIP2**; **no video backbone** in this phase; theory stance **Option A** (frozen encoder as fixed prefix → multimodal MI paragraph) unless venue demands Option B.
- **Dataset strategy (recommended):** **VLKEB-long** (same schema as VLKEB, extended `alt`) as the **primary** long-form benchmark; optional **small** generalization eval on a public long-form image VQA set (e.g. VizWiz-LF or VQAonline) only if time allows.

---

## 2. Checklist completion snapshot

| Area | Outcome |
|------|---------|
| **Check 1 — GPU** | `test_long_target.py`: `nvidia-smi` snippet, short/long `xym`, batch sweep (xym-only vs LM surrogate), one reliability train step, peak memory logged. |
| **Check 2 — Tokenizer** | `tools/preimplementation_checks.py`: `Blip2Config` + tokenizer; budget arithmetic; pilot `alt_long` bin stats. |
| **Check 3 — Code** | VLKEB truncation **no**; multi-sentence `alt` load **pass**; embedding path **no** extra trunc; EM/chunk smoke **pass**; `chunk_utils` **100+** tokens **pass**. |
| **Check 4 — Pilot data** | `data/VLKEB/eval_longform_pilot.json`, `pilot_train_longform_10.json`, `eval_longform_alt_10.json` (rebuild script below). Train/eval smokes documented in checklist. |
| **Check 5 — Metrics** | `rouge-score`, `bert-score` installed; ROUGE sanity in runner; `tools/bertscore_smoke.py` for GPU BERTScore check. |
| **Checks 6–7 — Scope & theory** | Max target **150**; bins and sizes as in checklist; seeds **42, 123, 456**; **Option A** chosen; §7.2 paragraph kept as draft anchor. |

Details and numbers remain in **`PRE_IMPLEMENTATION_CHECKLIST.md`** and the append-only log **`CHECKLIST_RUN_RESULTS.txt`**.

---

## 3. Repository & tooling updates

These changes exist **because** of the checklist; they are part of the new baseline.

| Item | Location / command |
|------|-------------------|
| Long GPU smoke + batch probe | `test_long_target.py` |
| Checks 2–6 runner (append log) | `tools/preimplementation_checks.py` |
| **Data buffer deadlock fix** | `dataset/__init__.py` — `buffer_size` clamped to **`max(1, buffer_size)`** so `-dbs 0` cannot hang the prefetch thread. |
| VLKEB **train** JSON override | `train_vllm_editor.py` — **`-dpath` / `--vlkeb_data_json`** (relative paths resolved under `ROOT_PATH`). |
| VLKEB **eval** JSON override | `test_vllm_edit.py` — **`-dpath` / `--vlkeb_eval_json`**. |
| Long-`alt` eval JSON builder | `tools/build_vlkeb_eval_longform_alt_json.py` → `data/VLKEB/eval_longform_alt_10.json` |
| BERTScore smoke | `tools/bertscore_smoke.py` |

**Example commands (from repo root):**

```bash
conda activate liveedit
python test_long_target.py
python tools/preimplementation_checks.py
python tools/build_vlkeb_eval_longform_alt_json.py
python tools/bertscore_smoke.py
```

**Pilot training (1 epoch, saves every step):**

```bash
python -u -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -lkpt none -tnp PILOT_longform -eps 1 -sci 1 -lpi 1 -dbs 0 -dpath data/VLKEB/pilot_train_longform_10.json
```

**Long-`alt` eval smoke:**

```bash
python tools/build_vlkeb_eval_longform_alt_json.py
python -u -W ignore test_vllm_edit.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dn VLKEB -dsn 10 -seed 42 -enp longform_alt_smoke -ckpt none -dpath data/VLKEB/eval_longform_alt_10.json
```

(Replace `-ckpt none` with a real checkpoint path when evaluating a trained editor.)

---

## 4. Decisions after the checklist

### 4.1 Backbone and modality

- **Decision:** Stay on **image-only BLIP2-OPT-2.7B** for Direction A v1.  
- **Implication:** Video long-form datasets (e.g. LongViTU) are **out of scope** unless the thesis pivots. Public long-form **image** QA datasets remain optional **add-ons**, not requirements.

### 4.2 Long-form data strategy (primary recommendation)

- **Decision:** Treat **VLKEB-long** (same JSON schema as VLKEB, with **long `alt`**) as the **main** long-form benchmark so the task stays aligned with **knowledge-style VLM editing** and the existing LiveEdit codebase.  
- **Quality:** Replace purely **synthetic** pilot strings with **semi-automated** long answers (e.g. expand from short gold `alt` with strict constraints + **spot-check**), or small **manual** / **GPT‑4V** passes where quality matters—document the protocol in the thesis methods section.  
- **Optional:** A **secondary** table on **one** public long-form image dataset (e.g. **VizWiz-LF** or **VQAonline**) for **generalization**, not as a third full training pipeline.

### 4.3 Metrics

- **Short targets (VLKEB regression):** keep **EM + token accuracy** (LiveEdit-style).  
- **Long targets:** **ROUGE-L** primary; **BERTScore** and **chunk EM / chunk F1** (AR path) as secondary where applicable.

### 4.4 Theory (Check 7)

- **Decision:** **Option A** — one-paragraph argument: frozen image encoder → visual prefix as **constant conditioning**; AnyEdit-style decomposition carries over on **X′ = (X, V)**.  
- **Option B** (formal one-page extension) remains **deferred** unless reviewers or venue require it.

---

## 5. Known limitations & honesty notes

- **Pilot `alt_long` / synthetic rows** were for **pipeline** verification, not for claiming human grounding in the final paper.  
- **BERTScore** first run can download large weights; `bertscore_smoke.py` assumes a working GPU (or CPU fallback).  
- **`eval_longform_alt_10.json`** is a **small** smoke file; full-scale train/eval sizes follow **`DIRECTION_A_PLAN.md`** / checklist §6.

---

## 6. Next actions (engineering order)

1. **Protocol write-up:** 1–2 pages: VLKEB-long construction rules, bin targets, quality control sample size, citation of VLKEB + any augmenting tool (e.g. API model version).  
2. **Scale data:** Build train/eval JSON at planned **N** (checklist §6.2) under `data/VLKEB/` or a named subfolder; keep `train_vllm_editor.py` / `test_vllm_edit.py` **`-dpath`** workflow.  
3. **Full runs:** LiveEdit (no AR) vs AR-LiveEdit on VLKEB-long + VLKEB short regression; log to `records/` and `eval_results/` with timestamps and seeds.  
4. **Thesis text:** Problem (short-target limits), gap, method, tables — align with **`DIRECTION_A_PLAN.md`**.

---

## 7. References in-repo

| Document | Role |
|----------|------|
| `PRE_IMPLEMENTATION_CHECKLIST.md` | Item-by-item completion and filled values |
| `DIRECTION_A_PLAN.md` | Phases, experiments, venues |
| `CHECKLIST_RUN_RESULTS.txt` | Append-only machine log from automated runs |

---

*End of document.*
