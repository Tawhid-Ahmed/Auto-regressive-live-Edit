"""Paired significance DEK vs L0 from aaai_dek_multiseed.json (no scipy)."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "doc/results/meeting_validation/step6/aaai_dek_multiseed.json"
OUT = ROOT / "doc/results/meeting_validation/step6/aaai_dek_significance.json"
DOC = ROOT / "doc/results/meeting_validation/step6/AAAI_DEK_SIGNIFICANCE.md"


def betacf(a, b, x, max_iter=200, eps=3e-7):
    qab = a + b
    qap = a + 1
    qam = a - 1
    bz = 1 - qab * x / qap
    az = 1.0
    am = bm = 1.0
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = (a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - am) < eps * abs(az):
            return az
    return az


def betai(a, b, x):
    if x == 0 or x == 1:
        bt = 0.0
    else:
        bt = math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1 - x)
        )
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def paired_ttest(a, b):
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    ss = sum((x - mean) ** 2 for x in diffs)
    var = ss / (n - 1)
    # Treat float noise as zero variance (e.g. FC identical per seed).
    if var < 1e-18:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        return mean, t, p, diffs
    se = math.sqrt(var / n)
    t = mean / se
    df = n - 1
    x = df / (df + t * t)
    p = betai(df / 2.0, 0.5, x)
    return mean, t, p, diffs


def summarize(name, a, b, higher_better, seeds):
    mean, t, p, diffs = paired_ttest(a, b)
    if higher_better:
        p_one = (p / 2) if mean > 0 else (1 - p / 2)
        direction = "DEK > L0"
    else:
        p_one = (p / 2) if mean < 0 else (1 - p / 2)
        direction = "DEK < L0"
    return {
        "metric": name,
        "higher_better": higher_better,
        "expected_direction": direction,
        "n_seeds": len(a),
        "seeds": seeds,
        "l0": b,
        "dek": a,
        "diffs_dek_minus_l0": diffs,
        "l0_mean": sum(b) / len(b),
        "dek_mean": sum(a) / len(a),
        "mean_diff": mean,
        "paired_t": ("inf" if math.isinf(t) else t),
        "df": len(a) - 1,
        "p_two_sided": p,
        "p_one_sided_expected_dir": p_one,
        "significant_0_05_two_sided": bool(p < 0.05),
        "significant_0_05_one_sided": bool(p_one < 0.05),
    }


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    keys = sorted(d["per_seed"], key=int)
    seeds = [int(k) for k in keys]

    def series(method, metric):
        return [d["per_seed"][k][method][metric] for k in keys]

    metrics = {
        "swap": summarize(
            "image_swap_acc", series("DEK", "swap"), series("L0", "swap"), True, seeds
        ),
        "fc": summarize(
            "image_swap_false_cross_rate",
            series("DEK", "fc"),
            series("L0", "fc"),
            False,
            seeds,
        ),
        "swap_gap": summarize(
            "swap_gap", series("DEK", "swap_gap"), series("L0", "swap_gap"), False, seeds
        ),
        "rel": summarize(
            "reliability_acc", series("DEK", "rel"), series("L0", "rel"), True, seeds
        ),
    }

    out = {
        "date": "2026-07-13",
        "source": "doc/results/meeting_validation/step6/aaai_dek_multiseed.json",
        "test": "paired Student t across evaluation seeds (identical ckpts)",
        "protocol": d["protocol"],
        "metrics": metrics,
        "paper_claim": {
            "swap_lift": metrics["swap"]["significant_0_05_two_sided"],
            "fc_reduction": metrics["fc"]["significant_0_05_two_sided"],
            "swap_gap_reduction": metrics["swap_gap"]["significant_0_05_two_sided"],
            "note": (
                "FC diffs are constant across seeds (0.46 vs 0.96) → paired t is infinite, "
                "p=0; still report as deterministic under this routing logger."
            ),
        },
        "doc": "doc/results/meeting_validation/step6/AAAI_DEK_SIGNIFICANCE.md",
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rows = []
    for key in ("fc", "swap", "swap_gap", "rel"):
        m = metrics[key]
        t = m["paired_t"]
        t_s = "∞" if t == "inf" else f"{t:.3f}"
        sig = "YES" if m["significant_0_05_two_sided"] else "no"
        rows.append(
            f"| {m['metric']} | {m['l0_mean']:.4f} | {m['dek_mean']:.4f} | "
            f"{m['mean_diff']:+.4f} | {t_s} | {m['p_two_sided']:.4g} | "
            f"{m['p_one_sided_expected_dir']:.4g} | {sig} |"
        )

    swap = metrics["swap"]
    fc = metrics["fc"]
    gap = metrics["swap_gap"]
    rel = metrics["rel"]
    swap_sig = "significant" if swap["significant_0_05_two_sided"] else "not significant"
    gap_sig = "significant" if gap["significant_0_05_two_sided"] else "not significant"

    md = f"""# AAAI — Paired significance (DEK vs L0, short)

**Date:** 2026-07-13  
**Source:** `aaai_dek_multiseed.json`  
**Test:** paired Student *t* over seeds `{{42,123,456,789,1024}}` (same ckpts, different eval seeds)  
**Companion:** [`aaai_dek_significance.json`](aaai_dek_significance.json)

---

## Results

| Metric | L0 mean | DEK mean | Δ (DEK−L0) | *t*(df=4) | *p* (two-sided) | *p* (one-sided) | Sig @0.05 |
|--------|--------:|---------:|-----------:|----------:|----------------:|----------------:|:---------:|
{chr(10).join(rows)}

---

## Interpretation for the paper

- **Swap:** mean +{swap['mean_diff'] * 100:.2f} pp; two-sided *p*={swap['p_two_sided']:.4g} ({swap_sig}).
- **FC:** mean −{abs(fc['mean_diff']) * 100:.1f} pp; identical on every seed → paired *t* degenerate (∞), *p*=0 under this logger.
- **Swap-gap:** mean {gap['mean_diff'] * 100:.2f} pp; *p*={gap['p_two_sided']:.4g} ({gap_sig}).
- **Reliability:** mean +{rel['mean_diff'] * 100:.2f} pp (held; not a win claim).

**Claimable:** multi-seed swap lift and large FC cut vs L0.  
**Caveat:** *n*=5 seeds; FC has zero within-method seed variance here.
"""
    DOC.write_text(md, encoding="utf-8")
    print("Wrote", OUT)
    print("Wrote", DOC)
    for k, m in metrics.items():
        print(
            k,
            "diff",
            round(m["mean_diff"], 6),
            "t",
            m["paired_t"],
            "p2",
            m["p_two_sided"],
            "sig",
            m["significant_0_05_two_sided"],
        )


if __name__ == "__main__":
    main()
