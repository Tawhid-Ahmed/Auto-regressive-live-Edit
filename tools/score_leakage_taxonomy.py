"""Gate G0: honest re-scoring of routing logs into a leakage taxonomy (CPU only).

Re-scores existing ``routing_diagnostics.json`` files (written by
``test_grounding_eval.py --log_routing``) against the grounding probe JSON so
that *stale* activation (the edited expert e_i firing on a mismatched image)
is counted as leakage, and the partner expert e_j firing on its own image is
counted as a correct route rather than an error.

Image-swap probe (p_i, x_j), four mutually exclusive outcomes (priority order):
  stale_leak     e_i fired (alone or together with anything else)
  foreign_leak   no e_i, but some e_k with k not in {i, j} fired
  correct_route  only e_j fired
  abstain        nothing fired
  total_leak = stale_leak + foreign_leak        (headline, lower is better)

Hard-negative probe (p_i, x_hn): the correct outcome is abstain; any firing is
a leak, split into stale (e_i) vs foreign (anything else).

Weight-aware variant: an expert counts as active only if its soft weight
exceeds ``tau`` (default grid 0.05 / 0.1 / 0.2).  The weight is LiveEdit's
residual fuse coefficient (``fuse_coe`` / ``pool_fuse_coe`` from
``get_moe_fuse_coe``) when the log has it (G2 onward, M3); older G0 logs fall
back to a softmax over the retrieved experts' ``vis_sim`` (proxy).  The output
records the branch actually used per run in ``weight_sources``.

Reliability probe (p_i, x_i) is also re-scored (hit / only-own / cross) so a
leak-vs-reliability view is available from the same logs.

Usage (repo root):
  python tools/score_leakage_taxonomy.py \\
    --run eval_results/.../VLKEB_GROUNDING-aaai_L0_short-2026.07.10-19.25.37 \\
    --run eval_results/.../VLKEB_GROUNDING-aaai_DEK_short-2026.07.10-19.30.27 \\
    --labels L0 DEK \\
    -out doc/results/cvpr2027/g0/leakage_taxonomy_vlkeb.json

``--run`` accepts either a result directory (containing routing_diagnostics.json)
or the JSON file itself. The probe JSON defaults to ``meta.data_path`` of the
log, remapped to this checkout if the recorded absolute path does not exist.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SWAP_OUTCOMES = ("stale_leak", "foreign_leak", "correct_route", "abstain")
HN_OUTCOMES = ("stale_leak", "foreign_leak", "abstain")
DEFAULT_TAUS = (0.05, 0.1, 0.2)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve_run(path: str) -> Tuple[str, str]:
    """Return (result_dir, routing_diagnostics.json path)."""
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if os.path.isdir(path):
        return path, os.path.join(path, "routing_diagnostics.json")
    return os.path.dirname(path), path


def _resolve_data_path(recorded: str, override: Optional[str]) -> str:
    if override:
        return override if os.path.isabs(override) else os.path.join(ROOT, override)
    if recorded and os.path.isfile(recorded):
        return recorded
    # Recorded path is absolute from another checkout: keep the part after "data/".
    norm = (recorded or "").replace("\\", "/")
    if "/data/" in norm:
        rel = norm[norm.index("/data/") + 1:]
        cand = os.path.join(ROOT, rel)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "Probe JSON not found (recorded=%r). Pass --probe_json explicitly." % recorded)


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"] if isinstance(payload, dict) and "data" in payload else payload


def _img_key(path: Optional[str]) -> str:
    """Normalise an image path to a comparable 'dir/file' tail."""
    if not path:
        return ""
    p = path.replace("\\", "/").rstrip("/")
    parts = p.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _flatten_results(result_dir: str, seed: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """Load seed_<seed>_results.json (or results.json) from sequential_edit_*/ if present."""
    pattern = "seed_%s_results.json" % seed if seed is not None else "results.json"
    hits = glob.glob(os.path.join(result_dir, "sequential_edit_*", pattern))
    if not hits and seed is not None:
        hits = glob.glob(os.path.join(result_dir, "sequential_edit_*", "results.json"))
    if not hits:
        return None
    with open(hits[0], encoding="utf-8") as f:
        batches = json.load(f)
    if batches and isinstance(batches[0], list):
        return [s for batch in batches for s in batch]
    return batches


def _detect_use_random(result_dir: str, seed: Optional[int]) -> bool:
    """The eval saves seed_<seed>_mean_results.json only when -seed was given."""
    if seed is None:
        return False
    return bool(glob.glob(os.path.join(result_dir, "sequential_edit_*", "seed_%s_mean_results.json" % seed)))


# --------------------------------------------------------------------------- #
# Sample-id -> probe-JSON row reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_sid_to_row(n_rows: int, n_samples: int, seed: Optional[int],
                           subsample: bool, shuffle: bool) -> List[int]:
    """Replicate GroundingVLKEB row selection + the eval/routing shuffle.

    Row selection (GroundingVLKEB._load_rows): ``subsample=True`` reproduces the
    post-Jul-2026 seeded subsample ``default_rng(seed).permutation(len(rows))[:n]``;
    ``subsample=False`` reproduces the older ``rows[:n]``.  The routing probe
    then sees the samples shuffled with ``default_rng(seed).shuffle`` whenever
    the eval was launched with ``-seed`` (``shuffle=True``).
    """
    if subsample and seed is not None and n_samples < n_rows:
        chosen = [int(i) for i in np.random.default_rng(int(seed)).permutation(n_rows)[:n_samples]]
    else:
        chosen = list(range(min(n_rows, n_samples)))
    order = np.arange(len(chosen))
    if shuffle and seed is not None:
        np.random.default_rng(int(seed)).shuffle(order)
    return [chosen[int(k)] for k in order]


def validate_mapping(sid_to_row: List[int], rows: List[Dict[str, Any]],
                     flat_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Cross-check reconstructed mapping against the eval results' image paths."""
    if flat_results is None:
        return {"checked": False, "reason": "no seed_*_results.json next to the log"}
    n = min(len(flat_results), len(sid_to_row))
    mism = []
    for sid in range(n):
        row = rows[sid_to_row[sid]]
        res = flat_results[sid]
        rel = res["reliability"][0]
        ok = _img_key(rel.get("image")) == _img_key(row["image"])
        gp = row.get("grounding_probes") or {}
        swap = (res.get("generality") or {}).get("image_swap")
        if ok and swap and gp.get("image_swap"):
            ok = _img_key(swap[0].get("image")) == _img_key(gp["image_swap"])
        if not ok:
            mism.append(sid)
    return {"checked": True, "n_checked": n, "n_mismatch": len(mism), "mismatch_sids": mism[:20]}


def choose_mapping(n_rows: int, n_samples: int, seed: Optional[int], use_random: bool,
                   sampling: str, rows: List[Dict[str, Any]],
                   flat_results: Optional[List[Dict[str, Any]]]) -> Tuple[List[int], Dict[str, Any]]:
    """Pick the row-selection variant; with ``sampling='auto'`` try both and keep
    the one the eval results validate (older logs: head; newer ``_sub_`` logs: subsample)."""
    if sampling == "auto":
        candidates = [("head", False), ("subsample", True)]
    else:
        candidates = [(sampling, sampling == "subsample")]
    best: Optional[Tuple[List[int], Dict[str, Any]]] = None
    for name, sub in candidates:
        m = reconstruct_sid_to_row(n_rows, n_samples, seed, subsample=sub, shuffle=use_random)
        chk = validate_mapping(m, rows, flat_results)
        chk["sampling"] = name
        if not chk.get("checked"):
            return m, chk  # nothing to validate against; first candidate wins
        if chk["n_mismatch"] == 0:
            return m, chk
        if best is None or chk["n_mismatch"] < best[1]["n_mismatch"]:
            best = (m, chk)
    assert best is not None
    return best


# --------------------------------------------------------------------------- #
# Probe relabelling and expert sets
# --------------------------------------------------------------------------- #
def relabel_probes(probes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Undo the probe-name clobbering in collect_grounding_routing_diagnostics.

    There ``entry = {"probe": probe_name, **routing, **classification}`` and the
    classification dict carries ``probe="image_rephrase"`` for image_swap, so
    image_swap rows are stored as ``image_rephrase`` but (unlike real
    image_rephrase rows) carry a ``false_activation`` key.
    """
    out = []
    n_fixed = 0
    for p in probes:
        q = dict(p)
        if q.get("probe") == "image_rephrase" and "false_activation" in q:
            q["probe"] = "image_swap"
            n_fixed += 1
        out.append(q)
    return out, n_fixed


def _softmax(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def weight_source(entry: Dict[str, Any]) -> str:
    """Which soft weight ``active_experts`` uses for this entry (same preference order)."""
    retrieved = entry.get("retrieved_indices", [])
    if not retrieved:
        return "none_retrieved"
    fuse = entry.get("fuse_coe")
    if isinstance(fuse, list) and len(fuse) == len(retrieved):
        return "fuse_coe"
    pool_fuse = entry.get("pool_fuse_coe") or []
    if pool_fuse and all(int(i) < len(pool_fuse) for i in retrieved):
        return "pool_fuse_coe"
    return "vis_sim_proxy"


def active_experts(entry: Dict[str, Any], tau: Optional[float], include_prot: bool) -> Tuple[Set[int], Dict[int, float]]:
    """Retrieved experts, optionally filtered by soft weight.

    Preference (M3 / G2): ``fuse_coe`` aligned with retrieved_indices, else
    ``pool_fuse_coe[i]`` (real ``get_moe_fuse_coe``), else vision-logit softmax
    proxy used in G0.
    """
    retrieved = [int(i) for i in entry.get("retrieved_indices", [])]
    if not retrieved:
        return set(), {}
    fuse = entry.get("fuse_coe")
    if isinstance(fuse, list) and len(fuse) == len(retrieved):
        weights = {i: float(w) for i, w in zip(retrieved, fuse)}
    else:
        pool_fuse = entry.get("pool_fuse_coe") or []
        if pool_fuse and all(i < len(pool_fuse) for i in retrieved):
            weights = {i: float(pool_fuse[i]) for i in retrieved}
        else:
            vis = entry.get("vis_sim") or []
            logits = [float(vis[i]) for i in retrieved if i < len(vis)]
            if len(logits) != len(retrieved):
                return set(retrieved), {i: float("nan") for i in retrieved}
            if include_prot and entry.get("vis_sim_prot") is not None:
                w = _softmax(logits + [float(entry["vis_sim_prot"])])[: len(retrieved)]
            else:
                w = _softmax(logits)
            weights = {i: float(wi) for i, wi in zip(retrieved, w)}
    if tau is None:
        return set(retrieved), weights
    return {i for i, wi in weights.items() if wi > tau}, weights


def classify_swap(active: Set[int], own: Set[int], partner: Set[int]) -> str:
    if not active:
        return "abstain"
    if active & own:
        return "stale_leak"
    if active - own - partner:
        return "foreign_leak"
    return "correct_route"


def classify_hard_neg(active: Set[int], own: Set[int]) -> str:
    if not active:
        return "abstain"
    if active & own:
        return "stale_leak"
    return "foreign_leak"


# --------------------------------------------------------------------------- #
# Scoring one run
# --------------------------------------------------------------------------- #
def _rates(counter: Counter, n: int, keys: Iterable[str]) -> Dict[str, float]:
    return {k: (counter.get(k, 0) / n if n else float("nan")) for k in keys}


def score_run(
    run: str,
    label: str,
    probe_json: Optional[str],
    taus: Sequence[float],
    include_prot: bool,
    eval_seed: Optional[int],
    force_shuffle: Optional[bool],
    sampling: str,
    keep_records: bool,
    strict: bool,
) -> Dict[str, Any]:
    result_dir, diag_path = _resolve_run(run)
    with open(diag_path, encoding="utf-8") as f:
        diag = json.load(f)
    meta = diag.get("meta", {})
    summary_in = diag.get("summary", {})
    pool_sample_ids: List[int] = [int(x) for x in diag["pool_sample_ids"]]
    probes, n_relabelled = relabel_probes(diag["probes"])

    data_path = _resolve_data_path(meta.get("data_path", ""), probe_json)
    rows = load_rows(data_path)
    n_samples = int(summary_in.get("n_samples") or (max(pool_sample_ids) + 1 if pool_sample_ids else 0))
    seed = eval_seed if eval_seed is not None else meta.get("seed")
    use_random = force_shuffle if force_shuffle is not None else _detect_use_random(result_dir, seed)

    flat_results = _flatten_results(result_dir, seed if use_random else None)
    sid_to_row, mapping_check = choose_mapping(
        len(rows), n_samples, seed, use_random, sampling, rows, flat_results)
    if mapping_check.get("checked") and mapping_check["n_mismatch"]:
        msg = "[%s] sample->row mapping mismatches on %d/%d samples (seed=%s, shuffle=%s, sampling=%s)" % (
            label, mapping_check["n_mismatch"], mapping_check["n_checked"], seed, use_random,
            mapping_check["sampling"])
        if strict:
            raise RuntimeError(msg + ". Check --eval_seed / --shuffle / --no_shuffle / --sampling.")
        print("WARNING: " + msg, file=sys.stderr)

    # pool index sets
    sid_to_pool: Dict[int, Set[int]] = defaultdict(set)
    for pidx, sid in enumerate(pool_sample_ids):
        sid_to_pool[sid].add(pidx)
    row_to_sids: Dict[int, Set[int]] = defaultdict(set)
    for sid, r in enumerate(sid_to_row):
        row_to_sids[r].add(sid)
    prompt_count = Counter(rows[r]["src"].strip() for r in sid_to_row)

    def pool_of_rows(row_ids: Iterable[int]) -> Set[int]:
        s: Set[int] = set()
        for r in row_ids:
            for sid in row_to_sids.get(r, ()):
                s |= sid_to_pool.get(sid, set())
        return s

    tau_keys: List[Tuple[str, Optional[float]]] = [("binary", None)] + [("tau_%g" % t, t) for t in taus]
    swap_counts = {k: Counter() for k, _ in tau_keys}
    hn_counts = {k: Counter() for k, _ in tau_keys}
    swap_flags = {k: Counter() for k, _ in tau_keys}
    hn_flags = {k: Counter() for k, _ in tau_keys}
    rel_counts = {k: Counter() for k, _ in tau_keys}
    n_swap = n_hn = n_rel = 0
    n_partner_in_pool = 0
    n_hn_partner_in_pool = 0
    n_dup_prompt = 0
    records: List[Dict[str, Any]] = []
    weight_sources: Counter = Counter()

    for e in probes:
        probe = e.get("probe")
        if probe not in ("image_swap", "hard_negative", "reliability"):
            continue
        weight_sources[weight_source(e)] += 1
        sid = int(e["sample_id"])
        own = sid_to_pool.get(sid, set())
        row_idx = sid_to_row[sid] if sid < len(sid_to_row) else None
        row = rows[row_idx] if row_idx is not None else {}
        gp = row.get("grounding_probes") or {}
        rec: Dict[str, Any] = {
            "probe": probe, "sample_id": sid, "row_index": row_idx,
            "retrieved_indices": e.get("retrieved_indices", []),
            "vis_sim_prot": e.get("vis_sim_prot"),
            "n_pool_sharing_prompt": prompt_count.get(row.get("src", "").strip(), 0),
        }
        if probe == "image_swap":
            n_swap += 1
            pj = gp.get("image_swap_partner_index")
            partner = pool_of_rows([pj]) if pj is not None else set()
            n_partner_in_pool += bool(partner)
            n_dup_prompt += rec["n_pool_sharing_prompt"] > 1
            rec.update({"partner_row": pj, "partner_in_pool": bool(partner), "outcome": {}})
            for key, tau in tau_keys:
                act, w = active_experts(e, tau, include_prot)
                oc = classify_swap(act, own, partner)
                swap_counts[key][oc] += 1
                swap_flags[key]["own_fired"] += bool(act & own)
                swap_flags[key]["partner_fired"] += bool(act & partner)
                swap_flags[key]["foreign_fired"] += bool(act - own - partner)
                swap_flags[key]["any_fired"] += bool(act)
                rec["outcome"][key] = oc
                if key == "binary":
                    rec["weights"] = {str(i): round(wi, 4) for i, wi in w.items()}
        elif probe == "hard_negative":
            n_hn += 1
            ph = gp.get("hard_neg_partner_index")
            hn_partner = pool_of_rows([ph]) if ph is not None else set()
            n_hn_partner_in_pool += bool(hn_partner)
            rec.update({"partner_row": ph, "partner_in_pool": bool(hn_partner), "outcome": {}})
            for key, tau in tau_keys:
                act, w = active_experts(e, tau, include_prot)
                oc = classify_hard_neg(act, own)
                hn_counts[key][oc] += 1
                hn_flags[key]["own_fired"] += bool(act & own)
                hn_flags[key]["hn_partner_fired"] += bool(act & hn_partner)
                hn_flags[key]["any_fired"] += bool(act)
                rec["outcome"][key] = oc
                if key == "binary":
                    rec["weights"] = {str(i): round(wi, 4) for i, wi in w.items()}
        else:  # reliability
            n_rel += 1
            rec["outcome"] = {}
            for key, tau in tau_keys:
                act, _ = active_experts(e, tau, include_prot)
                hit = bool(act & own)
                rel_counts[key]["hit"] += hit
                rel_counts[key]["only_own"] += bool(act) and act <= own
                rel_counts[key]["cross"] += bool(act - own)
                rel_counts[key]["miss"] += not hit
                rec["outcome"][key] = "hit" if hit else "miss"
        if keep_records:
            records.append(rec)

    def swap_block(key: str) -> Dict[str, Any]:
        r = _rates(swap_counts[key], n_swap, SWAP_OUTCOMES)
        r["total_leak"] = r["stale_leak"] + r["foreign_leak"]
        r.update({k: v / n_swap for k, v in swap_flags[key].items()} if n_swap else {})
        r["counts"] = dict(swap_counts[key])
        return r

    def hn_block(key: str) -> Dict[str, Any]:
        r = _rates(hn_counts[key], n_hn, HN_OUTCOMES)
        r["total_leak"] = r["stale_leak"] + r["foreign_leak"]
        r.update({k: v / n_hn for k, v in hn_flags[key].items()} if n_hn else {})
        r["counts"] = dict(hn_counts[key])
        return r

    def rel_block(key: str) -> Dict[str, Any]:
        return {k: (rel_counts[key].get(k, 0) / n_rel if n_rel else float("nan"))
                for k in ("hit", "only_own", "cross", "miss")}

    out: Dict[str, Any] = {
        "label": label,
        "run_dir": result_dir.replace("\\", "/"),
        "routing_diagnostics": diag_path.replace("\\", "/"),
        "probe_json": data_path.replace("\\", "/"),
        "ckpt": meta.get("ckpt"),
        "eval_seed": seed,
        "shuffled": use_random,
        "row_sampling": mapping_check.get("sampling"),
        "n_samples": n_samples,
        "n_pool_entries": len(pool_sample_ids),
        "n_image_swap_probes": n_swap,
        "n_hard_negative_probes": n_hn,
        "n_reliability_probes": n_rel,
        "n_probes_relabelled_swap": n_relabelled,
        "swap_partner_in_pool_rate": n_partner_in_pool / n_swap if n_swap else None,
        "hard_neg_partner_in_pool_rate": n_hn_partner_in_pool / n_hn if n_hn else None,
        "swap_prompt_duplicated_rate": n_dup_prompt / n_swap if n_swap else None,
        "mapping_check": mapping_check,
        "weight_softmax_includes_prototype": include_prot,
        "weight_sources": dict(weight_sources),
        "weight_definition": (
            "per-entry preference: fuse_coe (get_moe_fuse_coe over retrieved) > pool_fuse_coe > "
            "softmax over vis_sim of retrieved experts%s (proxy); expert active iff weight > tau; "
            "see weight_sources for the branch actually used" % (" + vis_sim_prot" if include_prot else "")),
        "legacy_summary": {k: v for k, v in summary_in.items() if k != "mean_n_retrieved"},
        "image_swap": {key: swap_block(key) for key, _ in tau_keys},
        "hard_negative": {key: hn_block(key) for key, _ in tau_keys},
        "reliability": {key: rel_block(key) for key, _ in tau_keys},
    }
    if keep_records:
        out["records"] = records
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float]) -> str:
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else "%5.1f" % (100 * x)


def print_report(results: List[Dict[str, Any]], taus: Sequence[float]) -> None:
    keys = ["binary"] + ["tau_%g" % t for t in taus]
    for r in results:
        print("=" * 78)
        print("%s  (n=%d edits, pool=%d, seed=%s, shuffle=%s, rows=%s)" % (
            r["label"], r["n_samples"], r["n_pool_entries"], r["eval_seed"], r["shuffled"], r["row_sampling"]))
        print("  log: %s" % r["routing_diagnostics"])
        mc = r["mapping_check"]
        if mc.get("checked"):
            print("  mapping check vs eval results: %d/%d mismatches" % (mc["n_mismatch"], mc["n_checked"]))
        else:
            print("  mapping check: skipped (%s)" % mc.get("reason"))
        print("  swap partner in pool: %s | hard-neg partner in pool: %s | swap prompt duplicated: %s" % (
            _pct(r["swap_partner_in_pool_rate"]).strip() + "%",
            _pct(r["hard_neg_partner_in_pool_rate"]).strip() + "%",
            _pct(r["swap_prompt_duplicated_rate"]).strip() + "%"))
        ls = r["legacy_summary"]
        print("  legacy (AAAI) false_activation: swap=%s%%  hard_neg=%s%%" % (
            _pct(ls.get("image_swap_false_activation_rate")).strip(),
            _pct(ls.get("hard_negative_false_activation_rate")).strip()))
        print("  %-9s | %-44s | %-28s | %s" % ("variant", "IMAGE-SWAP  stale  foreign  TOTAL  correct abstain",
                                               "HARD-NEG stale foreign TOTAL", "REL hit only-own cross"))
        for k in keys:
            s, h, rl = r["image_swap"][k], r["hard_negative"][k], r["reliability"][k]
            print("  %-9s | %12s %8s %6s %8s %7s | %14s %7s %5s | %7s %8s %5s" % (
                k,
                _pct(s["stale_leak"]), _pct(s["foreign_leak"]), _pct(s["total_leak"]),
                _pct(s["correct_route"]), _pct(s["abstain"]),
                _pct(h["stale_leak"]), _pct(h["foreign_leak"]), _pct(h["total_leak"]),
                _pct(rl["hit"]), _pct(rl["only_own"]), _pct(rl["cross"])))
    if len(results) >= 2:
        base = results[0]
        print("=" * 78)
        print("Deltas vs baseline '%s' (percentage points; negative = less leak)" % base["label"])
        for r in results[1:]:
            for k in keys:
                ds = 100 * (r["image_swap"][k]["total_leak"] - base["image_swap"][k]["total_leak"])
                dh = 100 * (r["hard_negative"][k]["total_leak"] - base["hard_negative"][k]["total_leak"])
                dr = 100 * (r["reliability"][k]["hit"] - base["reliability"][k]["hit"])
                print("  %-12s %-9s swap total_leak %+6.1f pp | hard_neg total_leak %+6.1f pp | rel hit %+6.1f pp | %s stale swap %s%%" % (
                    r["label"], k, ds, dh, dr, r["label"], _pct(r["image_swap"][k]["stale_leak"]).strip()))


def to_markdown(results: List[Dict[str, Any]], taus: Sequence[float]) -> str:
    keys = ["binary"] + ["tau_%g" % t for t in taus]
    lines = [
        "| run | variant | swap stale | swap foreign | swap total | swap correct | swap abstain | hn stale | hn foreign | hn total | rel hit | rel only-own |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for k in keys:
            s, h, rl = r["image_swap"][k], r["hard_negative"][k], r["reliability"][k]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["label"], k,
                _pct(s["stale_leak"]).strip(), _pct(s["foreign_leak"]).strip(), _pct(s["total_leak"]).strip(),
                _pct(s["correct_route"]).strip(), _pct(s["abstain"]).strip(),
                _pct(h["stale_leak"]).strip(), _pct(h["foreign_leak"]).strip(), _pct(h["total_leak"]).strip(),
                _pct(rl["hit"]).strip(), _pct(rl["only_own"]).strip()))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="G0 leakage taxonomy re-scoring of routing_diagnostics.json")
    ap.add_argument("--run", action="append", required=True,
                    help="Result dir or routing_diagnostics.json (repeatable; first run is the baseline for deltas)")
    ap.add_argument("--labels", nargs="*", default=None, help="One label per --run")
    ap.add_argument("--probe_json", default=None,
                    help="Grounding probe JSON (default: meta.data_path of each log, remapped to this checkout)")
    ap.add_argument("--taus", nargs="*", type=float, default=list(DEFAULT_TAUS),
                    help="Softmax-weight thresholds for the weight-aware variant")
    ap.add_argument("--weight_with_prot", action="store_true",
                    help="Include vis_sim_prot in the softmax denominator (default: gated experts only)")
    ap.add_argument("--eval_seed", type=int, default=None, help="Override meta.seed for sample-order reconstruction")
    shuffle = ap.add_mutually_exclusive_group()
    shuffle.add_argument("--shuffle", action="store_true", help="Force seeded subsample+shuffle reconstruction")
    shuffle.add_argument("--no_shuffle", action="store_true", help="Force unshuffled order (eval ran without -seed)")
    ap.add_argument("--sampling", choices=["auto", "head", "subsample"], default="auto",
                    help="Row selection: 'head' = rows[:n] (logs before the seeded-subsample loader), "
                         "'subsample' = default_rng(seed).permutation[:n]; 'auto' picks the one the "
                         "eval results validate")
    ap.add_argument("--no_strict", action="store_true",
                    help="Warn instead of failing when the sample->row mapping check finds mismatches")
    ap.add_argument("--dump_records", action="store_true", help="Include per-probe records in the output JSON")
    ap.add_argument("-out", "--output_json", default=None, help="Write summary JSON here")
    ap.add_argument("--md", default=None, help="Also write a Markdown table here")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(_resolve_run(r)[0])) for r in args.run]
    if len(labels) != len(args.run):
        ap.error("--labels must have one entry per --run")
    force_shuffle = True if args.shuffle else (False if args.no_shuffle else None)

    results = [
        score_run(run, label, args.probe_json, args.taus, args.weight_with_prot,
                  args.eval_seed, force_shuffle, args.sampling, args.dump_records,
                  strict=not args.no_strict)
        for run, label in zip(args.run, labels)
    ]
    print_report(results, args.taus)

    if args.output_json:
        out_path = args.output_json if os.path.isabs(args.output_json) else os.path.join(ROOT, args.output_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"taus": list(args.taus), "weight_softmax_includes_prototype": args.weight_with_prot,
                       "runs": results}, f, indent=2)
        print("Wrote %s" % out_path)
    if args.md:
        md_path = args.md if os.path.isabs(args.md) else os.path.join(ROOT, args.md)
        os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(to_markdown(results, args.taus))
        print("Wrote %s" % md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
