"""
PRE_IMPLEMENTATION_CHECKLIST.md — Check 1: long-target GPU smoke test.

Run from repo root (with conda env that has the project deps), e.g.:
  conda run -n liveedit python test_long_target.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_vllm_editor


def _run_nvidia_smi_summary() -> None:
    import subprocess

    print("\n[CHECK 1.1] nvidia-smi (first 20 lines)")
    try:
        out = subprocess.check_output(["nvidia-smi", "-q"], text=True, stderr=subprocess.STDOUT, timeout=15)
        lines = out.splitlines()
        for line in lines[:25]:
            print(line)
    except Exception as e:
        print(f"  nvidia-smi not available: {e}")


def _max_batch_long_target_surrogate(editor, vdp, vllm, train_prompt: str, dummy_img, long_target: str, max_b: int = 12):
    """
    Find largest B such that batched long-target xym + LM forward + backward
    with grad only on inputs_embeds succeeds (VRAM proxy for long packed batch).
    """
    last_ok = 0
    for B in range(1, max_b + 1):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            prompts = [train_prompt] * B
            imgs = [dummy_img] * B
            targets = [long_target] * B
            (x, vt), y, m = vdp.prompts_imgs_target_to_xym(prompts, imgs, targets)
            emb = x["inputs_embeds"].detach().clone().requires_grad_(True)
            x2 = {"attention_mask": x["attention_mask"], "inputs_embeds": emb}
            logits = vllm.get_llm_outpt(x2, vt).logits
            loss = vllm.label_loss(logits, y, m, True)
            loss.backward()
            del loss, logits, x2, emb, x, y, m
            torch.cuda.synchronize()
            last_ok = B
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  Batch surrogate OOM at B={B}: {e}")
                break
            raise
    return last_ok


def _max_batch_xym_only(editor, vdp, train_prompt: str, dummy_img, long_target: str, max_b: int = 16):
    last_ok = 0
    for B in range(1, max_b + 1):
        try:
            torch.cuda.empty_cache()
            prompts = [train_prompt] * B
            imgs = [dummy_img] * B
            targets = [long_target] * B
            (x, vt), y, m = vdp.prompts_imgs_target_to_xym(prompts, imgs, targets)
            del x, vt, y, m
            torch.cuda.synchronize()
            last_ok = B
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batched xym OOM at B={B}: {e}")
                break
            raise
    return last_ok


def _token_len_approx(editor, text: str) -> int:
    tok = editor.vllm.get_llm_tokenizer()
    return int(tok(text, return_tensors="pt")["input_ids"].shape[1])


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; this checklist item requires a GPU.")
        return 1

    _run_nvidia_smi_summary()

    device = "cuda:0"
    extra_devices = [0]
    print("Loading LiveEdit + BLIP2 (for_train=True, single-GPU data proc)...")
    editor = load_vllm_editor("liveedit", "blip2", device, extra_devices, None, True)
    editor.set_train(True)
    vllm = editor.vllm
    vdp = editor.vllm_data_proc if editor.vllm_data_proc is not None else vllm

    long_target = "The image depicts " + " ".join(["word"] * 140)
    short_target = "A cat"
    prompt = "What is in this image? The answer is:"
    print(f"Approx tokenizer length (long_target): {_token_len_approx(editor, long_target)}")

    def try_xym(label: str, target: str) -> bool:
        print(f"\n[{label}] prompts_imgs_target_to_xym...")
        try:
            (x, vt), y, m = vdp.prompts_imgs_target_to_xym([prompt], [None], [target])
            emb = x["inputs_embeds"]
            print(f"  inputs_embeds: {tuple(emb.shape)}, labels: {tuple(y.shape)}, mask tokens: {int(m.sum().item())}")
            return True
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False

    ok_short = try_xym("SHORT", short_target)
    ok_long = try_xym("LONG", long_target)

    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\nGPU memory after xym tests: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    train_prompt = "Describe this image in detail:"
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    print("\n[CHECK 1.2 extension] Max batch size (long ~144 target tokens, dummy image)...")
    max_xym = _max_batch_xym_only(editor, vdp, train_prompt, dummy_img, long_target, max_b=16)
    print(f"  Max B for batched prompts_imgs_target_to_xym only: {max_xym}")
    max_sur = _max_batch_long_target_surrogate(editor, vdp, vllm, train_prompt, dummy_img, long_target, max_b=12)
    print(f"  Max B for batched xym + LM forward/backward (grad on inputs_embeds): {max_sur}")
    print(
        "  Gradient accumulation likely needed:"
        f" {'yes' if max_sur < 2 else 'no (for this surrogate; full LiveEdit may differ)'}"
    )

    # 1.3 — one reliability-style step (same graph as train_a_batch non-AR reliability head)
    print("\n[TRAIN STEP] reliability loss + backward + optimizer.step (long target + dummy image)...")
    editor.reinit_train_parameters()
    opt_bundle = editor.get_a_new_optimizer()
    optimizer = opt_bundle[0] if isinstance(opt_bundle, tuple) else opt_bundle

    try:
        torch.cuda.reset_peak_memory_stats()
        (input_embeds, vt_range), label_ids, label_masks = vdp.prompts_imgs_target_to_xym(
            [train_prompt], [dummy_img], [long_target]
        )
        mr = vllm.get_mid_module_outpt(input_embeds, vt_range, editor.edit_layer_path)
        er = editor.get_reps_for_edit(
            vdp, {"prompt": train_prompt, "image": dummy_img, "target": long_target}
        )
        pre_v, vision, query, ans = er
        eqr, evr, moe_c, moe_r = editor.get_new_edit(vision, query, ans)
        iqr = editor.inpt_extractor.extract_query(query)
        mm = torch.ones(len(eqr), dtype=torch.bool, device=editor.device)
        eqrs, moe_cs, moe_rs = eqr, moe_c, moe_r
        fuse_coe = editor.get_moe_fuse_coe(iqr, eqrs[mm])
        editor.train_edit_residual = editor.get_edit_residual(
            torch.cat(er, 1), moe_cs[mm], moe_rs[mm], fuse_coe
        )
        logits = vllm.forward_from_mid_layer(
            input_embeds, vt_range, mr, editor.cfg.llm_layer_tmp, editor.cfg.edit_layer_i
        ).logits
        loss = vllm.label_loss(logits, label_ids, label_masks, True)
        print(f"  label_loss: {loss.detach().float().item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  backward + step: OK | peak GPU memory: {peak:.2f} GB")
    except Exception as e:
        print(f"  TRAIN STEP FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if not (ok_short and ok_long):
        return 1

    # Append one-line summary for merged checklist log
    try:
        rpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHECKLIST_RUN_RESULTS.txt")
        with open(rpath, "a", encoding="utf-8") as f:
            f.write(
                f"\n[Check 1 test_long_target.py] short_ok={ok_short} long_ok={ok_long} "
                f"max_bat_xym={max_xym} max_bat_surrogate={max_sur} train_peak_GB={peak:.2f}\n"
            )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
