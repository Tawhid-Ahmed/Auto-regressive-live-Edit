import json
import statistics
from pathlib import Path

root = Path("eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b")
seeds = [42, 123, 456, 789, 1024]


def find_mean(prefix: str, seed: int) -> Path:
    if seed == 42:
        if prefix == "DEK":
            pattern = "VLKEB_GROUNDING-aaai_DEK_short-*/sequential_edit_10/seed_42_mean_results.json"
        else:
            pattern = "VLKEB_GROUNDING-aaai_L0_short-*/sequential_edit_10/seed_42_mean_results.json"
    else:
        pattern = (
            f"VLKEB_GROUNDING-aaai_{prefix}_s{seed}-*/sequential_edit_10/"
            f"seed_{seed}_mean_results.json"
        )
    cands = list(root.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"{prefix} seed {seed}: {pattern}")
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def extract(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    rel = j["total_mean"]["reliability"]["acc"]
    swap = j["total_mean"]["generality"]["image_swap"]["acc"]
    hard = j["total_mean"]["generality"]["hard_negative"]["acc"]
    fc = j.get("routing_diagnostics", {}).get("image_swap_false_cross_rate")
    return {
        "rel": rel,
        "swap": swap,
        "hard_neg": hard,
        "fc": fc,
        "swap_gap": rel - swap,
        "eval_path": str(path.parent.parent).replace("\\", "/"),
    }


def summarize(rows):
    out = {}
    for k in ["rel", "swap", "hard_neg", "fc", "swap_gap"]:
        vals = [r[k] for r in rows]
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def main():
    dek_rows, l0_rows = [], []
    per_seed = {}
    for s in seeds:
        dek = extract(find_mean("DEK", s))
        l0 = extract(find_mean("L0", s))
        dek_rows.append(dek)
        l0_rows.append(l0)
        per_seed[str(s)] = {"DEK": dek, "L0": l0}
        print(
            f"seed {s}: DEK swap={dek['swap']:.4f} fc={dek['fc']} | "
            f"L0 swap={l0['swap']:.4f} fc={l0['fc']}"
        )

    dek_sum = summarize(dek_rows)
    l0_sum = summarize(l0_rows)
    fc_diff = [d["fc"] - l["fc"] for d, l in zip(dek_rows, l0_rows)]
    swap_diff = [d["swap"] - l["swap"] for d, l in zip(dek_rows, l0_rows)]
    gap_diff = [d["swap_gap"] - l["swap_gap"] for d, l in zip(dek_rows, l0_rows)]

    wilcoxon = {}
    try:
        from scipy.stats import wilcoxon as wilcoxon_test

        for name, diffs in [("fc", fc_diff), ("swap", swap_diff), ("swap_gap", gap_diff)]:
            try:
                stat, p = wilcoxon_test(diffs)
                wilcoxon[name] = {"stat": float(stat), "p": float(p), "diffs": diffs}
            except Exception as e:
                wilcoxon[name] = {"error": str(e), "diffs": diffs}
    except ImportError:
        wilcoxon = {"error": "scipy not available", "diffs": {
            "fc": fc_diff, "swap": swap_diff, "swap_gap": gap_diff
        }}

    g2 = {
        "dek_mean_fc_le_55": dek_sum["fc"]["mean"] <= 0.55,
        "dek_mean_swap_ge_l0_plus_5pp": dek_sum["swap"]["mean"]
        >= (l0_sum["swap"]["mean"] + 0.05),
        "dek_mean_fc": dek_sum["fc"]["mean"],
        "dek_mean_swap": dek_sum["swap"]["mean"],
        "l0_mean_swap": l0_sum["swap"]["mean"],
        "swap_lift_pp": (dek_sum["swap"]["mean"] - l0_sum["swap"]["mean"]) * 100,
    }

    out = {
        "timestamp": "2026-07-13",
        "protocol": {
            "dsn": 100,
            "sen": 10,
            "seeds": seeds,
            "grounding_eval_json": (
                "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json"
            ),
            "meeting_step": 6,
            "dek_ckpt": (
                "records/liveedit/blip2-opt-2.7b/"
                "VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/"
                "epoch-1-i-125-ema_loss-4.0100"
            ),
            "l0_ckpt": (
                "records/liveedit/blip2-opt-2.7b/"
                "VLKEB_baseline-2026.04.09-08.20.07/checkpoints/"
                "epoch-20-i-25950-ema_loss-0.1201"
            ),
        },
        "per_seed": per_seed,
        "DEK_summary": dek_sum,
        "L0_summary": l0_sum,
        "paired_diffs": {"fc": fc_diff, "swap": swap_diff, "swap_gap": gap_diff},
        "wilcoxon": wilcoxon,
        "g2_gate": g2,
    }
    out_path = Path("doc/results/meeting_validation/step6/aaai_dek_multiseed.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", out_path)
    print("G2", json.dumps(g2, indent=2))
    print(
        "DEK mean±std swap",
        round(dek_sum["swap"]["mean"], 4),
        round(dek_sum["swap"]["std"], 4),
    )
    print(
        "DEK mean±std fc",
        round(dek_sum["fc"]["mean"], 4),
        round(dek_sum["fc"]["std"], 4),
    )
    print(
        "L0 mean±std swap",
        round(l0_sum["swap"]["mean"], 4),
        round(l0_sum["swap"]["std"], 4),
    )
    print(
        "L0 mean±std fc",
        round(l0_sum["fc"]["mean"], 4),
        round(l0_sum["fc"]["std"], 4),
    )
    print("wilcoxon", wilcoxon)


if __name__ == "__main__":
    main()
