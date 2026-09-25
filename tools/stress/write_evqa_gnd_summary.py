import glob
import json

out = {
    "protocol": {
        "dsn": 100,
        "sen": 10,
        "seed": 42,
        "backbone": "blip2-opt-2.7b",
        "train_json": "data/easy-edit-mm/grounding/grounding_train_short.json",
        "dn": 500,
        "eval_json": "data/easy-edit-mm/grounding/grounding_eval_short.json",
    },
    "evqa_grounding": {},
}
for tag in ["L0", "S2", "DEK"]:
    p = sorted(
        glob.glob(
            f"eval_results/meeting_validation/step6/**/VLKEB_GROUNDING-evqa_gnd_{tag}-*/**/seed_42_mean_results.json",
            recursive=True,
        )
    )[-1]
    d = json.load(open(p, encoding="utf-8"))
    tm = d["total_mean"]
    rd = d["routing_diagnostics"]
    out["evqa_grounding"][tag] = {
        "rel_acc": tm["reliability"]["acc"],
        "swap_acc": tm["generality"].get("image_swap", {}).get("acc"),
        "hard_acc": tm["generality"].get("hard_negative", {}).get("acc"),
        "fc_swap": rd["image_swap_false_cross_rate"],
        "fc_hard": rd["hard_negative_false_cross_rate"],
        "fa_swap": rd["image_swap_false_activation_rate"],
        "result_json": p.replace("\\", "/"),
    }
l0, dek = out["evqa_grounding"]["L0"], out["evqa_grounding"]["DEK"]
rel_ok = dek["rel_acc"] >= l0["rel_acc"] - 0.05
fc_ok = dek["fc_swap"] < l0["fc_swap"] - 0.2
out["gate_G_D1_grounding"] = "pass" if (rel_ok and fc_ok) else "fail"
out["gate_note"] = (
    f"Rel {l0['rel_acc']:.3f}->{dek['rel_acc']:.3f}; "
    f"FC_swap {l0['fc_swap']:.2f}->{dek['fc_swap']:.2f}"
)
path = "doc/results/meeting_validation/step6/stress/aaai_stress_evqa_grounding_summary.json"
json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))
print("wrote", path)
