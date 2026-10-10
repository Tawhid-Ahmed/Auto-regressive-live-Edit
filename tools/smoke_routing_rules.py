"""G2 smoke: exercise every --routing_rule on a tiny pool (CPU/GPU, ~minutes).

Loads one checkpoint, builds a small edit pool, then for each routing rule runs
``probe_routing`` on reliability / image_swap / hard_negative and checks that
logs contain M3 fuse coefficients + routing_rule.

Usage (repo root):
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/smoke_routing_rules.py
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/smoke_routing_rules.py --dsn 10 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.GLOBAL import ROOT_PATH

RULES = (
    "liveedit",
    "vision_only",
    "text_only",
    "and",
    "or",
    "sum",
    "carml",
    "always_reject",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 routing_rule smoke test")
    ap.add_argument("-dvc", "--device", default="cuda:0")
    ap.add_argument("-dsn", type=int, default=10)
    ap.add_argument("-seed", type=int, default=42)
    ap.add_argument(
        "-ckpt",
        default="records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201",
    )
    ap.add_argument(
        "-dpath",
        default="data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json",
    )
    ap.add_argument("--img_root", default="data/VLKEB/VLKEB_images/mmkb_images")
    ap.add_argument(
        "-out",
        default="doc/results/cvpr2027/g2/smoke_routing_rules.json",
    )
    args = ap.parse_args()

    from utils import get_full_model_name, load_vllm_editor
    from dataset.vllm_grounding import GroundingVLKEB

    data_path = args.dpath if os.path.isabs(args.dpath) else os.path.join(ROOT_PATH, args.dpath)
    img_root = args.img_root if os.path.isabs(args.img_root) else os.path.join(ROOT_PATH, args.img_root)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)

    print("Loading editor…")
    # Baseline L0 ckpt: do NOT enable patch_grounding — random patch modules distort vis_sim
    # and make liveedit/vision_only look broken. CARML then uses Q-Former + text channels only.
    editor = load_vllm_editor("liveedit", get_full_model_name("blip2"), args.device, None, ckpt, False)

    eval_data = GroundingVLKEB(data_path, img_root, args.dsn, sample_seed=args.seed)
    samples = deepcopy(eval_data.data_with_img)
    np.random.default_rng(args.seed).shuffle(samples)

    print("Building pool of %d edits…" % len(samples))
    editor.restore_to_original_model()
    for s in samples:
        editor.edit_one_piece(s["requests"][0])

    from tools.log_grounding_diagnostics import _probe_specs

    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": ckpt.replace("\\", "/"),
            "dsn": args.dsn,
            "seed": args.seed,
            "device": args.device,
            "n_pool": int(editor.evr_pool.shape[0]),
        },
        "rules": {},
        "ok": True,
        "errors": [],
    }

    # Probe first sample only per rule for speed; plus always_reject sanity on all.
    sample0 = samples[0]
    for rule in RULES:
        print("=== rule=%s ===" % rule)
        try:
            editor.set_routing_rule(rule)
            rows = []
            for probe_name, prompt, image, target in _probe_specs(sample0):
                if probe_name not in ("reliability", "image_swap", "hard_negative"):
                    continue
                routing = editor.probe_routing(editor.vllm, prompt, image, target=target)
                rows.append({
                    "probe": probe_name,
                    "routing_rule": routing.get("routing_rule"),
                    "n_retrieved": routing.get("n_retrieved"),
                    "retrieved_indices": routing.get("retrieved_indices"),
                    "has_fuse_coe": "fuse_coe" in routing,
                    "has_pool_fuse_coe": bool(routing.get("pool_fuse_coe")),
                    "fuse_coe": routing.get("fuse_coe"),
                    "n_vision_ok": len(routing.get("vision_ok_indices") or []),
                    "n_text_ok": len(routing.get("text_ok_indices") or []),
                })
                # M3 required fields
                assert "fuse_coe" in routing
                assert "pool_fuse_coe" in routing
                assert routing.get("routing_rule") == rule
                if rule == "always_reject":
                    assert routing.get("n_retrieved") == 0, routing.get("retrieved_indices")
            results["rules"][rule] = {"status": "ok", "probes": rows}
            print("  ok:", [(r["probe"], r["n_retrieved"]) for r in rows])
        except Exception as e:
            results["ok"] = False
            results["errors"].append({"rule": rule, "error": repr(e)})
            results["rules"][rule] = {"status": "fail", "error": repr(e)}
            print("  FAIL:", e)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out_path)
    print("SMOKE", "PASS" if results["ok"] else "FAIL")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
