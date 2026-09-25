"""Sample N short grounding probes into a human-audit CSV sheet."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_JSON = ROOT / "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
OUT_DIR = ROOT / "doc/results/meeting_validation/step6/audit"
OUT_CSV = OUT_DIR / "audit_samples_short_n150.csv"
OUT_META = OUT_DIR / "audit_samples_meta.json"

N = 150
SEED = 42
# Match AAAI eval draw size; sheet still samples from full JSON with fixed seed.
DSN_NOTE = 100


def main():
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))["data"]
    rng = random.Random(SEED)
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:N])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_idx",
        "entity_folder",
        "prompt_rephrase",
        "target_alt",
        "edit_image",
        "swap_image",
        "hard_neg_image",
        "modality_image",
        "swap_partner_index",
        "hard_neg_partner_index",
        "swap_score",
        "hard_neg_score",
        "modality_score",
        "notes",
        "rater_id",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in chosen:
            ex = data[i]
            gp = ex.get("grounding_probes") or {}
            img = ex.get("image") or ""
            entity = img.split("/")[0] if img else ""
            w.writerow(
                {
                    "sample_idx": i,
                    "entity_folder": entity,
                    "prompt_rephrase": ex.get("rephrase") or "",
                    "target_alt": ex.get("alt") or "",
                    "edit_image": img,
                    "swap_image": gp.get("image_swap") or "",
                    "hard_neg_image": gp.get("hard_neg_image") or "",
                    "modality_image": gp.get("modality_consistency_image") or "",
                    "swap_partner_index": gp.get("image_swap_partner_index", ""),
                    "hard_neg_partner_index": gp.get("hard_neg_partner_index", ""),
                    "swap_score": "",
                    "hard_neg_score": "",
                    "modality_score": "",
                    "notes": "",
                    "rater_id": "",
                }
            )

    meta = {
        "n": N,
        "seed": SEED,
        "source": str(EVAL_JSON.relative_to(ROOT)).replace("\\", "/"),
        "n_pool": len(data),
        "chosen_indices": chosen,
        "img_root": "data/VLKEB/VLKEB_images/mmkb_images",
        "rubric": "doc/results/meeting_validation/step6/AAAI_HUMAN_AUDIT_PLAN.md",
        "dsn_note": DSN_NOTE,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CSV} ({N} rows)")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
