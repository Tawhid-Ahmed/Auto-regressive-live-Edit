"""One-off diagnostic: load BLIP2 and print stages. Safe to delete after debugging."""
import sys
import time


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    model_path = r"E:\MSC\MSc thesis project\LiveEdit\models\blip2-opt-2.7b"
    print(f"[diag] device={device}", flush=True)
    print("[diag] importing BLIP2OPTForEdit...", flush=True)
    t0 = time.perf_counter()
    from editor.vllms_for_edit.blip2.blip2 import BLIP2OPTForEdit

    print(f"[diag] import done in {time.perf_counter() - t0:.1f}s", flush=True)
    print("[diag] constructing model (load weights + move to device)...", flush=True)
    t1 = time.perf_counter()
    BLIP2OPTForEdit(model_path, device)
    print(f"[diag] construct done in {time.perf_counter() - t1:.1f}s", flush=True)
    print("[diag] OK — full load finished.", flush=True)


if __name__ == "__main__":
    main()
