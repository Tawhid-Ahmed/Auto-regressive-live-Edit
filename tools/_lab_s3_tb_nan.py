from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logs = sorted(
    Path("records/lab_g5/liveedit/llava-v1.5-7b").glob(
        "lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_100331*/logs"
    ),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print("logs", logs[:1])
ea = EventAccumulator(str(logs[0]))
ea.Reload()
for tag in ["EMA Loss", "Loss", "AVBL loss", "MoE soft routing relative loss", "MoE soft routing loss"]:
    if tag not in ea.Tags().get("scalars", []):
        print(tag, "MISSING")
        continue
    vals = ea.Scalars(tag)
    print(tag, "n=", len(vals))
    for i, s in enumerate(vals[:8]):
        print(" ", s.step, s.value)
    # first nan/inf
    for s in vals:
        if s.value != s.value or abs(s.value) > 1e6:
            print(" first_bad", tag, "step", s.step, "value", s.value)
            break
    else:
        print(" ", "last", vals[-1].step, vals[-1].value if vals else None)
