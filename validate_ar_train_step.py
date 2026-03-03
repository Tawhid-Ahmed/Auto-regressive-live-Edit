"""
Validate Task 4 AR training loop: one batch with ar_mode=True, loss finite and no NaNs.
Run: conda run -n liveedit python validate_ar_train_step.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import traceback

from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor

LOG_PATH = os.path.join(os.path.dirname(__file__) or ".", "ar_validate_result.txt")


def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    device = "cuda:0"
    extra_devices = [0]
    data_n = 4
    batch_size = 2

    log("Loading editor (LiveEdit + BLIP2)...")
    editor = load_vllm_editor("LiveEdit", "blip2", device, extra_devices, None, True)

    log("Loading VLKEB (tiny subset)...")
    from dataset.vllm import VLKEB
    data_path = os.path.join(ROOT_PATH, "data/VLKEB/train.json")
    img_root_dir = os.path.join(ROOT_PATH, "data/VLKEB/VLKEB_images/mmkb_images")
    train_data = VLKEB(data_path, img_root_dir, data_n)

    # AR mode ON for Task 4 validation
    editor.ar_mode = True
    editor.chunk_size = 16
    editor.max_chunks = None

    log("Initializing training (1 batch)...")
    editor.train_init(
        train_data, batch_size,
        train_name_prefix="ar_validate",
        load_ckpt_path=None,
        save_ckpt_per_i=10000,
        log_per_i=1,
        ema_alpha=0.1,
        random_seed=42,
        data_buffer_size=1,
    )

    # Get one organized batch (will have 12 elements when ar_mode=True)
    editor.set_train(True)
    batch_iter = iter(editor.data_generator)
    a_batch_organized, samp_n = next(batch_iter)

    log(f"Batch: {samp_n} samples, tuple length = {len(a_batch_organized)}")
    assert len(a_batch_organized) == 12, f"Expected 12 elements with ar_mode=True, got {len(a_batch_organized)}"

    log("Running train_a_batch (AR path)...")
    loss, log_dict = editor.train_a_batch(a_batch_organized)

    # Validation: loss is scalar float, finite, no NaN
    assert isinstance(loss, (float,)), f"Loss should be float, got {type(loss)}"
    assert math.isfinite(loss), f"Loss must be finite, got {loss}"
    assert not math.isnan(loss), "Loss must not be NaN"
    for k, v in log_dict.items():
        if isinstance(v, (int, float)):
            assert math.isfinite(v), f"log_dict['{k}'] must be finite, got {v}"
            assert not math.isnan(v), f"log_dict['{k}'] must not be NaN"

    log(f"  Loss = {loss:.6f}")
    log("  [PASS] Loss is finite, no NaNs.")
    log("\nValidation passed: AR training step (Task 4) completes with finite loss.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("FAILED:\n" + tb)
        except Exception:
            pass
        sys.exit(1)
