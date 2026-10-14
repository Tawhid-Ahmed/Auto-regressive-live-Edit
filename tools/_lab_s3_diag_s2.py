import json
from pathlib import Path

j = json.load(open("doc/results/cvpr2027/g5/routing_health_s2_s3_bs2.json", encoding="utf-8"))
print("ckpt", j.get("ckpt"))
print("health_gate", json.dumps(j.get("health_gate"), indent=2))
print("reliability", json.dumps(j.get("train_split_reliability"), indent=2))
print("routing_health", json.dumps(j.get("routing_health"), indent=2)[:2000])

ckpt_dir = Path(
    "records/lab_g5/liveedit/llava-v1.5-7b/"
    "lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_100331-2026.10.12-10.03.35/checkpoints"
)
print("ckpts:", sorted(p.name for p in ckpt_dir.glob("epoch-*")))
