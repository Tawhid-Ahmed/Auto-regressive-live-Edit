"""Quick smoke test: ViT patch count grows with image_size; Q-Former stays 32."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from PIL import Image
import torch
from utils import get_model_path, get_full_model_name
from editor.vllms_for_edit.blip2.blip2 import BLIP2OPTForEdit

img = Image.new("RGB", (800, 600), color="red")
vllm = BLIP2OPTForEdit(get_model_path(get_full_model_name("blip2")), "cuda:0")
for sz in [224, 336, 448]:
    vllm.set_eval_image_size(sz)
    inpt = vllm.processor([img], ["describe"], return_tensors="pt")
    pv = inpt["pixel_values"]
    with torch.no_grad():
        interpolate = sz != vllm.native_image_size
        vo = vllm.model.vision_model(
            pixel_values=pv.to(vllm.device),
            interpolate_pos_encoding=interpolate,
        )
    patches = vo[0].shape[1]
    qn = vllm.get_img_token_n()
    print(f"size={sz} pixel={tuple(pv.shape)} vit_patches={patches} qformer_tokens={qn}")
print("OK")
