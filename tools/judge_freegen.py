"""Workstream E2: offline LLM-as-judge faithfulness over free-running predictions.

Reads `*freegen_results.json` and asks a local Ollama text model to score how
faithfully each `prediction_freegen` conveys the edited fact / content of
`target_text`. Runs fully offline from saved predictions (no GPU / no editor),
so it can be re-run cheaply and validated against a human spot-check.

Two sub-scores per sample (parsed from strict JSON the judge returns):
  - fact_correct: does the prediction state the SAME final/edited fact (entity,
    name, value) as the target?  (0 or 1)
  - faithfulness: overall semantic agreement with the target paragraph (0-5).

Usage:
  python tools/judge_freegen.py \
    --run Baseline=eval_results/.../seed_42_freegen_results.json \
    --run "AR (cs 16)"=eval_results/.../seed_42_freegen_results.json \
    --out doc/results/freegen_judge.md \
    --model gemma4:31b-cloud --max_samples 200

Add --human_csv to also dump a spot-check sheet for human validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

SYSTEM = (
    "You are a strict evaluator for a knowledge-editing benchmark. You compare a model's "
    "generated answer against a reference answer about an image-grounded fact. The most "
    "important thing is whether the generated answer states the SAME key/edited fact "
    "(the specific entity, name, place, or value) as the reference. Be rigorous and "
    "concise. Always reply with ONLY a JSON object, no prose."
)

PROMPT_TMPL = (
    "Reference answer (ground truth):\n\"\"\"\n{ref}\n\"\"\"\n\n"
    "Generated answer to evaluate:\n\"\"\"\n{pred}\n\"\"\"\n\n"
    "Score the generated answer. Return JSON with exactly these keys:\n"
    "{{\n"
    '  "fact_correct": 0 or 1,   // 1 only if it states the same key/edited fact as the reference\n'
    '  "faithfulness": 0..5,     // overall semantic agreement with the reference (5 = equivalent)\n'
    '  "reason": "one short sentence"\n'
    "}}"
)


def _ollama_chat(base_url: str, model: str, system: str, user: str,
                 timeout: float = 120.0, max_retries: int = 3) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    last = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            return ((data.get("message") or {}).get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    raise last or RuntimeError("ollama chat failed")


def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def _collect(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for split in data:
        for rec in split:
            for rr in rec.get("reliability", []):
                if "prediction_freegen" in rr and "target_text" in rr:
                    pairs.append((rr["prediction_freegen"], rr["target_text"]))
    return pairs


def _parse_run(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise SystemExit(f"--run must be 'Label=path', got: {item}")
    label, path = item.split("=", 1)
    return label.strip(), path.strip()


def _load_cache(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Label=path to *freegen_results.json")
    ap.add_argument("--out", default="doc/results/freegen_judge.md")
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--base_url", default="http://127.0.0.1:11434")
    ap.add_argument("--max_samples", type=int, default=200, help="Cap samples scored per run.")
    ap.add_argument("--cache", default="doc/results/freegen_judge_cache.json")
    ap.add_argument("--human_csv", default=None, help="Optional path to dump a spot-check CSV.")
    args = ap.parse_args()

    runs = [_parse_run(r) for r in args.run]
    cache = _load_cache(args.cache)
    rows = []
    human_rows = []
    for label, path in runs:
        pairs = _collect(path)[: args.max_samples]
        facts, faiths, n_ok = [], [], 0
        for i, (pred, ref) in enumerate(pairs):
            key = f"{args.model}::{hash((label, i, pred, ref))}"
            if key in cache:
                obj = cache[key]
            else:
                raw = _ollama_chat(args.base_url, args.model, SYSTEM,
                                   PROMPT_TMPL.format(ref=ref, pred=pred))
                obj = _parse_json(raw) or {}
                cache[key] = obj
                if i % 10 == 0 and args.cache:
                    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
                    with open(args.cache, "w", encoding="utf-8") as f:
                        json.dump(cache, f)
            fc = obj.get("fact_correct")
            fa = obj.get("faithfulness")
            if fc is not None:
                facts.append(1.0 if float(fc) >= 0.5 else 0.0)
            if fa is not None:
                faiths.append(float(fa) / 5.0)
                n_ok += 1
            if args.human_csv:
                human_rows.append([label, i, ref, pred, fc, fa, obj.get("reason", "")])
        fact_acc = (sum(facts) / len(facts)) if facts else None
        faith = (sum(faiths) / len(faiths)) if faiths else None
        rows.append((label, len(pairs), n_ok, fact_acc, faith))
        print(f"{label:18s} n={len(pairs):4d} judged={n_ok:4d}  "
              f"fact_correct={'-' if fact_acc is None else f'{fact_acc*100:.2f}'}  "
              f"faithfulness={'-' if faith is None else f'{faith*100:.2f}'}")

    if args.cache:
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Free-running LLM-as-judge faithfulness (VLKEB-long)\n\n")
        f.write(f"Judge: `{args.model}` (Ollama). Values are percentages.\n\n")
        f.write("| Run | N | Judged | Fact-correct | Faithfulness (0-5 -> %) |\n")
        f.write("|-----|---|--------|--------------|-------------------------|\n")
        for (label, n, n_ok, fact_acc, faith) in rows:
            fa = "-" if fact_acc is None else f"{fact_acc*100:.2f}"
            ff = "-" if faith is None else f"{faith*100:.2f}"
            f.write(f"| {label} | {n} | {n_ok} | {fa} | {ff} |\n")
        f.write("\n*Fact-correct = % where the judge says the prediction states the same edited "
                "fact as the target. Faithfulness = mean of a 0-5 semantic-agreement score, "
                "rescaled to %. Validate against the human spot-check before reporting.*\n")
    print(f"Wrote {args.out}")

    if args.human_csv and human_rows:
        os.makedirs(os.path.dirname(args.human_csv) or ".", exist_ok=True)
        with open(args.human_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "idx", "reference", "prediction", "judge_fact_correct",
                        "judge_faithfulness", "judge_reason", "human_fact_correct", "human_faithfulness"])
            w.writerows([r + ["", ""] for r in human_rows])
        print(f"Wrote {args.human_csv}")


if __name__ == "__main__":
    main()
