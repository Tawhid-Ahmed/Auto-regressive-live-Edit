"""Gate G1: residual leak source analysis from G0 taxonomy records (CPU).

Stratifies DEK residual leaks (stale_leak + foreign_leak) by prompt duplication
and partner-in-pool status; compares leak rates on duplicated vs unique prompts
against the plan's decision rule (>=70% of residual leaks on duplicated prompts
=> text gate is the weak link).

Also reports L0 vs DEK for context. Does not require GPU.
DINOv2 terciles / which-gate-admitted need fields not present in existing logs;
those are noted as deferred (optional short GPU eval).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEAK = ("stale_leak", "foreign_leak")


def _load(path: str) -> Dict[str, Any]:
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_by_label(payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    for r in payload["runs"]:
        if r["label"] == label:
            return r
    raise KeyError(label)


def _probe_rows(run: Dict[str, Any], probe: str) -> List[Dict[str, Any]]:
    return [x for x in run.get("records", []) if x.get("probe") == probe]


def _dup(row: Dict[str, Any]) -> bool:
    return int(row.get("n_pool_sharing_prompt") or 0) > 1


def stratify(rows: List[Dict[str, Any]], variant: str = "binary") -> Dict[str, Any]:
    n = len(rows)
    outcomes = Counter(r["outcome"][variant] for r in rows)
    leaks = [r for r in rows if r["outcome"][variant] in LEAK]
    n_leak = len(leaks)
    dup_all = sum(1 for r in rows if _dup(r))
    uniq_all = n - dup_all
    leak_on_dup = sum(1 for r in leaks if _dup(r))
    leak_on_uniq = n_leak - leak_on_dup

    def rate(num, den):
        return (num / den) if den else None

    # leak rate within each stratum
    dup_rows = [r for r in rows if _dup(r)]
    uniq_rows = [r for r in rows if not _dup(r)]
    dup_leak = sum(1 for r in dup_rows if r["outcome"][variant] in LEAK)
    uniq_leak = sum(1 for r in uniq_rows if r["outcome"][variant] in LEAK)

    by_outcome_dup = Counter(r["outcome"][variant] for r in dup_rows)
    by_outcome_uniq = Counter(r["outcome"][variant] for r in uniq_rows)

    partner_present = [r for r in rows if r.get("partner_in_pool")]
    partner_absent = [r for r in rows if r.get("partner_in_pool") is False]
    # hard_negative uses same key partner_in_pool in scorer

    def partner_block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0}
        oc = Counter(r["outcome"][variant] for r in subset)
        nl = sum(oc[k] for k in LEAK)
        return {
            "n": len(subset),
            "total_leak_rate": rate(nl, len(subset)),
            "outcomes": {k: rate(oc[k], len(subset)) for k in sorted(oc)},
            "residual_share_of_all_leaks": rate(nl, n_leak) if n_leak else None,
        }

    # foreign vs stale among residual, by dup
    foreign = [r for r in leaks if r["outcome"][variant] == "foreign_leak"]
    stale = [r for r in leaks if r["outcome"][variant] == "stale_leak"]

    return {
        "n": n,
        "outcome_rates": {k: rate(outcomes[k], n) for k in outcomes},
        "n_leak": n_leak,
        "total_leak_rate": rate(n_leak, n),
        "prompt_duplication": {
            "n_dup_prompt_probes": dup_all,
            "n_unique_prompt_probes": uniq_all,
            "dup_prompt_rate": rate(dup_all, n),
            "residual_leaks_on_dup": leak_on_dup,
            "residual_leaks_on_unique": leak_on_uniq,
            "share_of_residual_leaks_on_dup": rate(leak_on_dup, n_leak),
            "leak_rate_given_dup_prompt": rate(dup_leak, len(dup_rows)),
            "leak_rate_given_unique_prompt": rate(uniq_leak, len(uniq_rows)),
            "outcomes_given_dup": {k: rate(by_outcome_dup[k], len(dup_rows)) for k in by_outcome_dup} if dup_rows else {},
            "outcomes_given_unique": {k: rate(by_outcome_uniq[k], len(uniq_rows)) for k in by_outcome_uniq} if uniq_rows else {},
        },
        "leak_type_split": {
            "n_foreign": len(foreign),
            "n_stale": len(stale),
            "foreign_share_of_residual": rate(len(foreign), n_leak),
            "stale_share_of_residual": rate(len(stale), n_leak),
            "foreign_on_dup_share": rate(sum(1 for r in foreign if _dup(r)), len(foreign)),
            "stale_on_dup_share": rate(sum(1 for r in stale if _dup(r)), len(stale)),
        },
        "partner_in_pool": {
            "present": partner_block(partner_present),
            "absent": partner_block(partner_absent),
        },
        "dup_count_histogram_residual": dict(Counter(int(r.get("n_pool_sharing_prompt") or 0) for r in leaks)),
        "dup_count_histogram_all": dict(Counter(int(r.get("n_pool_sharing_prompt") or 0) for r in rows)),
    }


def decide_text_vs_vision(dek_swap: Dict[str, Any], dek_hn: Dict[str, Any]) -> Dict[str, Any]:
    """Plan G1 decision rule on residual leaks."""
    # Primary: VLKEB hard-negative + image-swap residual (where DEK still leaks)
    parts = []
    for name, block in (("image_swap", dek_swap), ("hard_negative", dek_hn)):
        share = block["prompt_duplication"]["share_of_residual_leaks_on_dup"]
        n_leak = block["n_leak"]
        parts.append({"probe": name, "n_leak": n_leak, "share_on_dup": share})

    # Combined residual across both probes (equal weight by leak count)
    n_dup = sum(
        (p["share_on_dup"] or 0) * p["n_leak"]
        for p in parts
        if p["n_leak"]
    )
    n_tot = sum(p["n_leak"] for p in parts)
    combined = (n_dup / n_tot) if n_tot else None

    if combined is not None and combined >= 0.70:
        verdict = "text_gate_weak"
        reason = (
            ">=70%% of residual DEK leaks fall on duplicated prompts (combined %.1f%%). "
            "Text gate with margin 0 likely admits every expert sharing p_i."
            % (100 * combined)
        )
    elif combined is not None and combined < 0.50:
        # check if spread: leak rates similar on dup vs unique
        swap_dup_r = dek_swap["prompt_duplication"]["leak_rate_given_dup_prompt"]
        swap_uni_r = dek_swap["prompt_duplication"]["leak_rate_given_unique_prompt"]
        hn_dup_r = dek_hn["prompt_duplication"]["leak_rate_given_dup_prompt"]
        hn_uni_r = dek_hn["prompt_duplication"]["leak_rate_given_unique_prompt"]
        # if unique also leaks a lot, vision/other
        verdict = "vision_or_unexplained"
        reason = (
            "Only %.1f%% of residual DEK leaks are on duplicated prompts (<50%%). "
            "Leak rate dup vs unique: image_swap %s vs %s; hard_negative %s vs %s."
            % (
                100 * combined,
                _pct(swap_dup_r),
                _pct(swap_uni_r),
                _pct(hn_dup_r),
                _pct(hn_uni_r),
            )
        )
    else:
        verdict = "mixed"
        reason = (
            "Residual DEK leaks on duplicated prompts: %.1f%% (between 50 and 70). "
            "Partial support for text-gate story; do not spend the whole residual section on it."
            % (100 * combined if combined is not None else float("nan"))
        )

    return {
        "verdict": verdict,
        "combined_share_of_residual_leaks_on_dup": combined,
        "per_probe": parts,
        "reason": reason,
        "plan_threshold_text_gate": 0.70,
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def analyze_pair(vlkeb_path: str, evqa_path: str) -> Dict[str, Any]:
    vlk = _load(vlkeb_path)
    evq = _load(evqa_path)
    out: Dict[str, Any] = {
        "vlkeb": {},
        "evqa": {},
        "hypothesis_check": {},
        "decision": {},
    }
    for ds, payload, l0, dek in (
        ("vlkeb", vlk, "VLKEB_L0", "VLKEB_DEK"),
        ("evqa", evq, "EVQA_L0", "EVQA_DEK"),
    ):
        out[ds] = {}
        for label in (l0, dek):
            run = _run_by_label(payload, label)
            out[ds][label] = {
                "image_swap": stratify(_probe_rows(run, "image_swap")),
                "hard_negative": stratify(_probe_rows(run, "hard_negative")),
                "swap_prompt_duplicated_rate_meta": run.get("swap_prompt_duplicated_rate"),
            }

    # Hypothesis: VLKEB high dup => more residual; EVQA low dup => less residual after DEK
    v_dek = out["vlkeb"]["VLKEB_DEK"]
    e_dek = out["evqa"]["EVQA_DEK"]
    out["hypothesis_check"] = {
        "claim": (
            "VLKEB templated prompts duplicate in the pool so DEK text gate (margin 0) "
            "collapses toward vision-only; EVQA prompts are less duplicated, hence lower residual."
        ),
        "vlkeb_swap_dup_rate": v_dek["image_swap"]["prompt_duplication"]["dup_prompt_rate"],
        "evqa_swap_dup_rate": e_dek["image_swap"]["prompt_duplication"]["dup_prompt_rate"],
        "vlkeb_dek_swap_total_leak": v_dek["image_swap"]["total_leak_rate"],
        "evqa_dek_swap_total_leak": e_dek["image_swap"]["total_leak_rate"],
        "vlkeb_dek_hn_total_leak": v_dek["hard_negative"]["total_leak_rate"],
        "evqa_dek_hn_total_leak": e_dek["hard_negative"]["total_leak_rate"],
        "vlkeb_share_residual_on_dup_swap": v_dek["image_swap"]["prompt_duplication"][
            "share_of_residual_leaks_on_dup"
        ],
        "vlkeb_share_residual_on_dup_hn": v_dek["hard_negative"]["prompt_duplication"][
            "share_of_residual_leaks_on_dup"
        ],
        "supported": (
            (v_dek["image_swap"]["prompt_duplication"]["dup_prompt_rate"] or 0)
            > (e_dek["image_swap"]["prompt_duplication"]["dup_prompt_rate"] or 0)
            and (v_dek["image_swap"]["total_leak_rate"] or 0)
            > (e_dek["image_swap"]["total_leak_rate"] or 0)
        ),
    }
    out["decision"] = decide_text_vs_vision(v_dek["image_swap"], v_dek["hard_negative"])
    # EVQA residual is tiny on swap; still report hn
    out["decision_evqa_hard_negative"] = {
        "n_leak": e_dek["hard_negative"]["n_leak"],
        "share_on_dup": e_dek["hard_negative"]["prompt_duplication"][
            "share_of_residual_leaks_on_dup"
        ],
        "dup_prompt_rate": e_dek["hard_negative"]["prompt_duplication"]["dup_prompt_rate"],
    }
    out["deferred"] = {
        "dinov2_terciles": (
            "Not in existing routing_diagnostics; needs short GPU pass or rebuild of "
            "hard_neg partners with DINOv2 ranks (plan G1 suggestion)."
        ),
        "which_gate_admitted": (
            "DEK logs only final retrieved_indices; per-channel vision/text pass bits "
            "not stored. Infer for DEK: retrieved => both gates passed. Log explicitly in G2."
        ),
    }
    return out


def print_report(out: Dict[str, Any]) -> None:
    print("=" * 72)
    print("G1 residual leak source (CPU)")
    print("=" * 72)
    for ds, l0, dek in (
        ("vlkeb", "VLKEB_L0", "VLKEB_DEK"),
        ("evqa", "EVQA_L0", "EVQA_DEK"),
    ):
        print("\n--- %s ---" % ds.upper())
        for label in (l0, dek):
            print("[%s]" % label)
            for probe in ("image_swap", "hard_negative"):
                b = out[ds][label][probe]
                pd = b["prompt_duplication"]
                print(
                    "  %s: n=%d total_leak=%s | dup_prompt_rate=%s | "
                    "share_of_residual_on_dup=%s | leak|dup=%s leak|uniq=%s"
                    % (
                        probe,
                        b["n"],
                        _pct(b["total_leak_rate"]),
                        _pct(pd["dup_prompt_rate"]),
                        _pct(pd["share_of_residual_leaks_on_dup"]),
                        _pct(pd["leak_rate_given_dup_prompt"]),
                        _pct(pd["leak_rate_given_unique_prompt"]),
                    )
                )
                lt = b["leak_type_split"]
                print(
                    "    residual: foreign=%d (%.0f%%) stale=%d (%.0f%%); "
                    "foreign_on_dup=%s stale_on_dup=%s"
                    % (
                        lt["n_foreign"],
                        100 * (lt["foreign_share_of_residual"] or 0),
                        lt["n_stale"],
                        100 * (lt["stale_share_of_residual"] or 0),
                        _pct(lt["foreign_on_dup_share"]),
                        _pct(lt["stale_on_dup_share"]),
                    )
                )
                pp = b["partner_in_pool"]
                print(
                    "    partner_in_pool present: n=%s leak=%s | absent: n=%s leak=%s"
                    % (
                        pp["present"].get("n"),
                        _pct(pp["present"].get("total_leak_rate")),
                        pp["absent"].get("n"),
                        _pct(pp["absent"].get("total_leak_rate")),
                    )
                )

    h = out["hypothesis_check"]
    print("\n--- Hypothesis (VLKEB dup => residual; EVQA sparse => clean) ---")
    print("  supported:", h["supported"])
    print(
        "  VLKEB dup_rate=%s DEK swap_leak=%s hn_leak=%s"
        % (_pct(h["vlkeb_swap_dup_rate"]), _pct(h["vlkeb_dek_swap_total_leak"]), _pct(h["vlkeb_dek_hn_total_leak"]))
    )
    print(
        "  EVQA  dup_rate=%s DEK swap_leak=%s hn_leak=%s"
        % (_pct(h["evqa_swap_dup_rate"]), _pct(h["evqa_dek_swap_total_leak"]), _pct(h["evqa_dek_hn_total_leak"]))
    )

    d = out["decision"]
    print("\n--- Decision (VLKEB DEK residual) ---")
    print("  verdict:", d["verdict"])
    print("  combined share of residual leaks on dup prompts:", _pct(d["combined_share_of_residual_leaks_on_dup"]))
    print("  ", d["reason"])
    print("\nDeferred:", out["deferred"]["dinov2_terciles"])
    print("         ", out["deferred"]["which_gate_admitted"])


def main() -> int:
    ap = argparse.ArgumentParser(description="G1 residual leak stratification (CPU)")
    ap.add_argument(
        "--vlkeb",
        default="doc/results/cvpr2027/g0/leakage_taxonomy_vlkeb_s42.json",
    )
    ap.add_argument(
        "--evqa",
        default="doc/results/cvpr2027/g0/leakage_taxonomy_evqa_s42.json",
    )
    ap.add_argument(
        "-out",
        default="doc/results/cvpr2027/g1/residual_leak_source_s42.json",
    )
    args = ap.parse_args()
    out = analyze_pair(args.vlkeb, args.evqa)
    print_report(out)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
