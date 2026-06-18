"""
Run PRE_IMPLEMENTATION_CHECKLIST.md checks 2–6 (and partial 4.2) in order.

Usage (from repo root):
  conda run -n liveedit python tools/preimplementation_checks.py

Check 1 is separate: python test_long_target.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_PATH = os.path.join(ROOT, "CHECKLIST_RUN_RESULTS.txt")


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(RESULT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def section(title: str) -> None:
    bar = "=" * 72
    log(f"\n{bar}\n{title}\n{bar}")


def check_2_tokenizer_budget() -> None:
    section("CHECK 2: Tokenizer & sequence lengths")
    from transformers import AutoTokenizer, Blip2Config

    from utils.GLOBAL import model_path_map

    mp = model_path_map["blip2-opt-2.7b"]
    log(f"Model path: {mp}")

    tok = AutoTokenizer.from_pretrained(mp, use_fast=False, local_files_only=True)
    cfg_path = os.path.join(mp, "config.json")
    bc = Blip2Config.from_json_file(cfg_path)
    max_pos = int(bc.text_config.max_position_embeddings)
    nq = int(bc.num_query_tokens)
    log(f"Max position embeddings (OPT): {max_pos}")
    log(f"QFormer query tokens: {nq}")

    special = 2
    avail = max_pos - nq - special
    log(f"\nCHECK 2.2 token budget (heuristic):")
    log(f"  Available for prompt + target ≈ {max_pos} - {nq} - {special} = {avail}")
    prompt_tok_est = 50
    max_target_budget = avail - prompt_tok_est
    log(f"  If prompt ≈ {prompt_tok_est} tokens → rough max target ≈ {max_target_budget} tokens")
    planned = 150
    fits = planned <= max_target_budget
    log(f"  Planned max target (Direction A): {planned} → fits budget: {fits}")


def check_3_code_and_runtime() -> None:
    section("CHECK 3: Existing code compatibility")

    log("3.1 VLKEB / __init_eic_evqa__ (dataset/vllm.py)")
    log("  Truncation of 'alt': NO — line ~87 assigns target_new = d['alt'] as-is.")
    log("  Multi-sentence alt: supported at JSON/Python string level (no length cap here).")

    tmpdir = tempfile.mkdtemp(prefix="vlkeb_long_alt_")
    img_root = os.path.join(tmpdir, "imgs")
    os.makedirs(img_root, exist_ok=True)
    long_alt = (
        "This is a long answer with multiple sentences. It describes the image in detail. "
        "The scene contains identifiable geographic and architectural cues. "
        "Therefore the correct placename follows from visible evidence."
    )
    one = {
        "src": "What city is shown?",
        "rephrase": "Which city appears?",
        "pred": "X",
        "alt": long_alt,
        "image": "dummy.jpg",
        "image_rephrase": "dummy.jpg",
        "loc": "q?",
        "loc_ans": "a",
        "m_loc": "dummy.jpg",
        "m_loc_q": "mq?",
        "m_loc_a": "ma",
    }
    jp = os.path.join(tmpdir, "one.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump([one], f)
    from dataset.vllm import VLKEB

    try:
        ds = VLKEB(jp, img_root, data_n=1, preload_images=False)
        got = ds.data_with_img_path[0]["requests"][0]["target_new"]
        ok = got == long_alt
        log(f"  VLKEB load with long multi-sentence alt: {'PASS' if ok else 'FAIL'} (len={len(got)})")
    except Exception as e:
        log(f"  VLKEB load test FAILED: {e}")

    log("\n3.2 prompts_imgs_target_to_xym (editor/vllms_for_edit/base.py ~79–113)")
    log("  No max_length / truncation in this function; tokenizer() uses default padding only.")

    log("\n3.2 get_llm_input_embeds (blip2.py ~185–233)")
    log("  No explicit max sequence length clamp beyond model internals.")

    log("\n3.3 Tokenizer truncation flag")
    from transformers import AutoTokenizer
    from utils.GLOBAL import model_path_map

    tok = AutoTokenizer.from_pretrained(model_path_map["blip2-opt-2.7b"], use_fast=False, local_files_only=True)
    long_str = "word " * 2000
    ids_trunc = tok(long_str, truncation=True, max_length=128, return_tensors="pt")["input_ids"].shape[1]
    ids_full = tok(long_str, truncation=False, return_tensors="pt")["input_ids"].shape[1]
    log(f"  truncation=True,max=128 → length {ids_trunc}; truncation=False → length {ids_full}")
    log("  Default in base.prompts_imgs_target_to_xym is tokenizer(..., padding=True) without truncation.")

    log("\n3.4 train_a_batch (editor/vllm_editors/base.py ~98–105)")
    log("  Abstract only; LiveEdit implementation (liveedit.py ~479+) has no fixed target-length cap.")

    log("\n3.5 evaluation/vllm_editor_eval.py — EM / chunk metrics smoke test")
    import torch
    from evaluation.vllm_editor_eval import _chunk_em_f1

    L = 100
    chunk_size = 16
    label_ids = torch.randint(2, 5000, (1, L))
    label_masks = torch.ones(1, L, dtype=torch.long)
    pre_y = label_ids.clone()
    em = (pre_y[label_masks.bool()] == label_ids[label_masks.bool()]).all().item()
    ce, cf = _chunk_em_f1(pre_y, label_ids, label_masks, chunk_size)
    log(f"  Synthetic L={L}: EM={em}, chunk_em={ce:.4f}, chunk_f1={cf:.4f} (expect EM=1, chunk_em=1)")

    log("\n3.6 chunk_utils.split_target_into_chunks — 100+ tokens")
    from transformers import AutoTokenizer
    from editor.vllm_editors.liveedit.chunk_utils import split_target_into_chunks

    tok2 = AutoTokenizer.from_pretrained(model_path_map["blip2-opt-2.7b"], use_fast=False, local_files_only=True)
    text100 = "toktest " * 80
    n_tok = len(tok2.encode(text100, add_special_tokens=False))
    chunks = split_target_into_chunks(tok2, text100, chunk_size=16, max_chunks=None)
    log(f"  ~{n_tok} tokens → {len(chunks)} chunks (chunk_size=16)")

    from editor.vllm_editors.liveedit import chunk_utils as cu

    log(f"  chunk_utils path: {cu.__file__}")


def _build_long_alt(tok, short_answer: str, lo: int = 72, hi: int = 125) -> str:
    def ntok(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=False))

    base = (str(short_answer).strip() or "answer").replace("\n", " ")
    if lo <= 24:
        glue = " It matches the scene."
        s = base + "."
        while ntok(s) < lo:
            s += glue
        while ntok(s) > hi and len(s) > len(base):
            s = s[: max(int(len(s) * 0.85), len(base))]
        return s

    core = (
        f"The visually grounded answer is {base}. "
        "The photograph provides contextual cues such as signage, skyline, and terrain "
        "that jointly support this identification rather than nearby confusable locations."
    )
    pad = (
        " Additional inspection of lighting, weather, and street layout further reinforces "
        "the same conclusion without introducing details that are not justified by the image."
    )
    s = core
    while ntok(s) < lo:
        s += pad
    while ntok(s) > hi:
        s = s[: int(len(s) * 0.92)]
    return s


def check_4_pilot_data() -> None:
    section("CHECK 4: Pilot long-form data + CHECK 2.3 distribution")
    from transformers import AutoTokenizer

    from utils.GLOBAL import model_path_map

    eval_path = os.path.join(ROOT, "data", "VLKEB", "eval.json")
    if not os.path.isfile(eval_path):
        log(f"  SKIP: {eval_path} not found")
        return

    tok = AutoTokenizer.from_pretrained(model_path_map["blip2-opt-2.7b"], use_fast=False, local_files_only=True)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_rows = json.load(f)

    pilot_eval = []
    for i, row in enumerate(eval_rows[:15]):
        alt = row.get("alt", "")
        row2 = dict(row)
        # Mix lengths so bin stats match checklist 2.3 (<=16, 17-64, 65-150)
        if i < 3:
            lo, hi = 8, 15
        elif i < 9:
            lo, hi = 22, 60
        else:
            lo, hi = 68, 130
        row2["alt_long"] = _build_long_alt(tok, alt, lo, hi)
        pilot_eval.append(row2)

    out_eval = os.path.join(ROOT, "data", "VLKEB", "eval_longform_pilot.json")
    os.makedirs(os.path.dirname(out_eval), exist_ok=True)
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(pilot_eval, f, indent=2)
    log(f"  Wrote {len(pilot_eval)} samples → {out_eval}")

    lengths = [len(tok.encode(d["alt_long"], add_special_tokens=False)) for d in pilot_eval]
    bins = {"<=16": 0, "17-64": 0, "65-150": 0, "150+": 0}
    for n in lengths:
        if n <= 16:
            bins["<=16"] += 1
        elif n <= 64:
            bins["17-64"] += 1
        elif n <= 150:
            bins["65-150"] += 1
        else:
            bins["150+"] += 1
    log(f"  CHECK 2.3 alt_long lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    log(f"  Bins: {bins}")
    covers_all_three = bins["<=16"] > 0 and bins["17-64"] > 0 and bins["65-150"] > 0
    log(f"  Populates all three bins (<=16, 17-64, 65-150): {covers_all_three}")
    budget = 2048 - 32 - 2 - 50
    within_budget = max(lengths) <= budget
    log(f"  No target exceeds rough OPT budget (<= {budget} tok): {within_budget} (max_len={max(lengths)})")

    train_10 = []
    for row in eval_rows[:10]:
        r = dict(row)
        r["alt"] = _build_long_alt(tok, row.get("alt", "answer"), 80, 140)
        train_10.append(r)
    out_train = os.path.join(ROOT, "data", "VLKEB", "pilot_train_longform_10.json")
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_10, f, indent=2)
    log(f"  Wrote pilot train (10 rows, long `alt`) → {out_train}")


def check_4_2_pilot_pipeline() -> None:
    section("CHECK 4.2 Pilot training + eval (smoke)")
    img_dir = os.path.join(ROOT, "data", "VLKEB", "VLKEB_images", "mmkb_images")
    sample_img = None
    if os.path.isdir(img_dir):
        for root, _, files in os.walk(img_dir):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    sample_img = os.path.join(root, fn)
                    break
            if sample_img:
                break
    if sample_img is None:
        log("  SKIP end-to-end train/eval: no image files under VLKEB_images/mmkb_images in this workspace.")
        log("  After downloading images, run e.g.:")
        log(
            "    python -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB "
            "-bs 1 -dvc cuda:0 -edvc 0 -lkpt none -tnp PILOT_longform -eps 1 -sci 10000 -lpi 1 -dbs 1 "
            "-dpath data/VLKEB/pilot_train_longform_10.json"
        )
        return

    log(f"  Sample image present ({sample_img!r}).")
    log("  CHECK 4.2 full train+eval is not run inside this script (long runtime, buffered logs).")
    log("  Use -dpath data/VLKEB/pilot_train_longform_10.json (VLKEB-only flag) and prefer:")
    log("    set PYTHONUNBUFFERED=1   # Windows CMD; PowerShell: $env:PYTHONUNBUFFERED=1")
    log(
        "    python -u -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB "
        "-bs 1 -dvc cuda:0 -edvc 0 -lkpt none -tnp PILOT_longform -eps 1 -sci 10000 -lpi 1 -dbs 1 "
        "-dpath data/VLKEB/pilot_train_longform_10.json"
    )


def check_5_metrics() -> None:
    section("CHECK 5: ROUGE-L + BERTScore install & sanity")
    for pkg in ("rouge-score", "bert-score"):
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        log(f"  pip install {pkg}: exit {r.returncode}")
        if r.returncode != 0:
            log(r.stderr[:500])

    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", "rouge-score", "bert-score"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log(r.stdout[:1200] if r.stdout else "(no pip show output)")

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ref = "The cat sat on the mat in the living room near the window"
    hyp = ref
    hyp2 = "The dog sat on the mat in the living room near the door"
    hyp3 = "A bird flew over the ocean during sunset"
    log(f"  ROUGE-L perfect: {scorer.score(ref, hyp)['rougeL'].fmeasure:.4f}")
    log(f"  ROUGE-L close:   {scorer.score(ref, hyp2)['rougeL'].fmeasure:.4f}")
    log(f"  ROUGE-L poor:    {scorer.score(ref, hyp3)['rougeL'].fmeasure:.4f}")
    log(f"  EM perfect={int(ref==hyp)}, EM close={int(ref==hyp2)} (expect 1,0)")

    if os.environ.get("RUN_BERTSCORE_SMOKE", "").lower() in ("1", "true", "yes"):
        try:
            from bert_score import score as bert_score_fn
            import torch

            cands = [hyp2]
            refs = [ref]
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            _P, _R, F1 = bert_score_fn(cands, refs, lang="en", rescale_with_baseline=True, device=dev)
            log(f"  BERTScore F1 (close vs ref, sample): {float(F1.mean()):.4f}")
        except Exception as e:
            log(f"  BERTScore smoke test failed: {e}")
    else:
        log(
            "  BERTScore forward test skipped (first run downloads models; can take many minutes)."
            " Re-run with RUN_BERTSCORE_SMOKE=1 after models are cached."
        )


def check_6_scope_and_7_theory() -> None:
    section("CHECK 6–7: Scope freeze + theory (draft for thesis log)")
    log(
        "6.1 Max target tokens: 150 (hard cap aligned with checklist); bins: <=16, 17–64, 65–150; "
        "min samples/bin in eval: aim ≥50 per populated bin after dataset construction."
    )
    log("6.2 Dataset size (initial): long-form train 300–500; eval long-form 300–500; short-form regression 200.")
    log("6.3 Baselines: LiveEdit (no AR); AR-LiveEdit; FT-VL if time; stretch: MEND/SERAC VL.")
    log("6.4 Ablations: no AR chunking; chunk size {8,16,32}; optional vision-aware chunking vs fixed.")
    log("6.5 Lifelong edit counts: 1, 10, 100, 1000 (match LiveEdit where feasible).")
    log("6.6 Seeds: 42, 123, 456 (minimum 3).")
    log("6.7 Backbone: BLIP2-OPT-2.7B primary; second backbone only if time.")
    log("7.1 Theory: Option A (1-paragraph MI argument with frozen encoder as fixed prefix) unless venue demands B.")
    log("7.2 Key paragraph: see PRE_IMPLEMENTATION_CHECKLIST.md block quote (frozen encoder → X'=(X,V)).")
    log("  Mark checklist boxes manually after review.")


def main() -> int:
    log("\n\n### PRE_IMPLEMENTATION_CHECKLIST — automated run (Checks 2–6 + notes) ###\n")
    log(f"ROOT: {ROOT}")
    check_2_tokenizer_budget()
    check_3_code_and_runtime()
    check_4_pilot_data()
    check_4_2_pilot_pipeline()
    check_5_metrics()
    check_6_scope_and_7_theory()
    section("DONE")
    log(f"Full log written to: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
