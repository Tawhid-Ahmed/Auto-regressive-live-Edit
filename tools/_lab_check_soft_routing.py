from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path("records/lab_g5_smoke/liveedit/llava-v1.5-7b")
cands = sorted(run.glob("s2_step2_smoke_bs2-*/logs"), key=lambda p: p.stat().st_mtime, reverse=True)
print("logdirs", cands[:3])
if not cands:
    raise SystemExit("no logdir")
ea = EventAccumulator(str(cands[0]))
ea.Reload()
tags = ea.Tags().get("scalars", [])
soft = [t for t in tags if "soft" in t.lower() or "MoE" in t]
print("soft_tags", soft)
ok = False
for t in soft:
    vals = [x.value for x in ea.Scalars(t)]
    print(t, "n=", len(vals), "max=", max(vals) if vals else None, "mean=", (sum(vals)/len(vals) if vals else None))
    if "relative" in t.lower() and vals and max(vals) > 0:
        ok = True
print("SOFT_ROUTING_NONZERO", ok)
