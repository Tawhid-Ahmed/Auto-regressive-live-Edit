from typing import List, Union, Optional
import os
from pathlib import Path
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
import torch

def _load_blip2_from_local_dir(model_path: str, device: str):
    """Load BLIP2 model and processor from a local directory without using the Hub (avoids HFValidationError on Windows paths)."""
    from transformers import Blip2Config, Blip2ForConditionalGeneration, Blip2Processor
    try:
        from huggingface_hub.errors import HFValidationError
    except ImportError:
        from huggingface_hub import HFValidationError

    model_path = str(Path(model_path).resolve())
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"BLIP2 config not found: {config_path}")

    # Load config from file (no Hub call)
    config = Blip2Config.from_json_file(config_path)
    # Create model and load weights from local file(s)
    model = Blip2ForConditionalGeneration(config)
    # Weights: try safetensors first, then pytorch_model.bin
    weights_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            weights_path = p
            break
    if weights_path is None:
        # Sharded safetensors
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            import json
            with open(idx_path) as f:
                index = json.load(f)
            state_dict = {}
            for fn in index.get("weight_map", {}).values():
                fp = os.path.join(model_path, fn)
                if os.path.isfile(fp):
                    try:
                        from safetensors.torch import load_file
                        state_dict.update(load_file(fp))
                    except Exception:
                        try:
                            state_dict.update(torch.load(fp, map_location="cpu", weights_only=True))
                        except TypeError:
                            state_dict.update(torch.load(fp, map_location="cpu"))
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
                weights_path = "(sharded)"
    if weights_path and weights_path != "(sharded)":
        if weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # Processor: try from_pretrained first; on HFValidationError load from local files
    try:
        processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
    except HFValidationError:
        # Load processor from local files without passing directory path to Hub
        image_processor_path = os.path.join(model_path, "preprocessor_config.json")
        if os.path.isfile(image_processor_path):
            from transformers import Blip2ImageProcessor
            image_processor = Blip2ImageProcessor.from_json_file(image_processor_path)
        else:
            from transformers import Blip2ImageProcessor
            image_processor = Blip2ImageProcessor()
        # Tokenizer from tokenizer.json file (no Hub path)
        tokenizer_json = os.path.join(model_path, "tokenizer.json")
        if not os.path.isfile(tokenizer_json):
            raise FileNotFoundError(f"Processor fallback requires {tokenizer_json}")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
        processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    return model, processor


def _load_blip2_fp16_to_device(model_path: str, device_obj):
    """Load sharded safetensors into an empty model and materialize fp16 on device.

    Avoids transformers from_pretrained meta/half paths that OOM or AV on some
    Windows + torch stacks (~32GB RAM / WDDM).
    """
    import json
    import gc
    from safetensors.torch import load_file
    from transformers import Blip2Config, Blip2ForConditionalGeneration
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    model_path = str(Path(model_path).resolve())
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError("Missing safetensors index: %s" % idx_path)

    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    state = {}
    for fn in shard_files:
        fp = os.path.join(model_path, fn)
        state.update(load_file(fp))

    config = Blip2Config.from_pretrained(model_path, local_files_only=True)
    with init_empty_weights():
        model = Blip2ForConditionalGeneration(config)

    for name in list(state.keys()):
        tensor = state.pop(name)
        dtype = torch.float16 if tensor.is_floating_point() else tensor.dtype
        set_module_tensor_to_device(model, name, device_obj, value=tensor, dtype=dtype)
        del tensor

    # OPT lm_head is tied to embed_tokens and often absent from shards.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    # Any leftover meta params/buffers (should be rare): materialize empty then error clearly.
    leftover = [
        n for n, p in model.named_parameters()
        if getattr(p, "is_meta", False) or str(p.device) == "meta"
    ] + [
        n for n, b in model.named_buffers()
        if getattr(b, "is_meta", False) or str(b.device) == "meta"
    ]
    if leftover:
        raise RuntimeError("fp16-direct left meta tensors: %s" % leftover[:20])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


class BLIP2OPTForEdit(BaseVLLMForEdit):
    '''For blip2-opt 2.7b'''
    def __init__(self, model_path:str, device = 'cuda', image_size: Optional[int] = None) -> None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        try:
            from huggingface_hub.errors import HFValidationError
        except ImportError:
            from huggingface_hub import HFValidationError
        import torch
        import gc
        model_path = str(Path(model_path).resolve())
        print(f"Loading BLIP2 model from {model_path}...")
        config_json = os.path.join(model_path, "config.json")
        _p = Path(model_path)
        is_local_path = os.path.sep in model_path or (getattr(_p, "drive", "") != "")
        device_obj = (
            torch.device(device)
            if isinstance(device, str) and ":" in device
            else torch.device(device if isinstance(device, str) else f"cuda:{device}")
        )
        try:
            device_map_loaded = False
            staged_gpu_loaded = False
            # Local directory: use HF from_pretrained (avoids manual shard merge + .to(cuda) access violations on some Windows/GPU stacks).
            if os.path.isfile(config_json):
                try:
                    if device != "cpu" and torch.cuda.is_available():
                        use_device_map = os.environ.get(
                            "LIVEEDIT_BLIP2_DEVICE_MAP", ""
                        ).lower() in ("1", "true", "yes")
                        if use_device_map:
                            try:
                                try:
                                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                                        model_path,
                                        local_files_only=True,
                                        device_map=device,
                                        dtype=torch.float16,
                                        low_cpu_mem_usage=True,
                                    )
                                except TypeError:
                                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                                        model_path,
                                        local_files_only=True,
                                        device_map=device,
                                        torch_dtype=torch.float16,
                                        low_cpu_mem_usage=True,
                                    )
                                device_map_loaded = True
                            except Exception as e:
                                raise RuntimeError(
                                    "BLIP2 device_map load failed (%s: %s). "
                                    "Unset LIVEEDIT_BLIP2_DEVICE_MAP to use staged load."
                                    % (type(e).__name__, e)
                                ) from e
                        else:
                            # Direct fp16 materialization (safe on low-RAM Windows). Avoids
                            # HF from_pretrained meta/half paths that OOM or access-violate.
                            gc.collect()
                            torch.cuda.empty_cache()
                            self.model = _load_blip2_fp16_to_device(model_path, device_obj)
                            staged_gpu_loaded = True
                            print(f"Model on {device_obj} (fp16 direct)")
                    else:
                        self.model = Blip2ForConditionalGeneration.from_pretrained(
                            model_path, local_files_only=True
                        )
                    self.processor = Blip2Processor.from_pretrained(
                        model_path, local_files_only=True
                    )
                except HFValidationError:
                    self.model, self.processor = _load_blip2_from_local_dir(
                        model_path, device
                    )
                    device_map_loaded = False
            elif is_local_path or os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Model directory not found or missing config.json: {model_path}. "
                    "Download the BLIP2 model (e.g. Salesforce/blip2-opt-2.7b) into this path."
                )
            else:
                # Hub repo id
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path,
                    local_files_only=False,
                    device_map=None,
                    torch_dtype=torch.float32
                )
                self.processor = Blip2Processor.from_pretrained(model_path, local_files_only=False)
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device_map_loaded:
                print(f"Model on {device_obj} (device_map, fp16)")
            elif staged_gpu_loaded:
                pass
            elif device == "cpu":
                self.model = self.model.to(torch.device("cpu"))
                print("Model moved to cpu")
            elif not device_map_loaded:
                self.model = self.model.to(device_obj)
                print(f"Model moved to {device_obj}")
        except Exception as e:
            print(f"ERROR loading BLIP2 model: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.model = self.model.eval().requires_grad_(False)
        super().__init__(self.model, device, False)
        ip = self.processor.image_processor
        native = ip.size
        if isinstance(native, dict):
            self.native_image_size = int(native.get("height", native.get("width", 224)))
        else:
            self.native_image_size = int(native)
        self.eval_image_size = None
        self.expose_vit_grid = False
        self._last_vit_patches = None
        self._last_qformer_cross_attn = None
        if image_size is not None:
            self.set_eval_image_size(image_size)

    def set_expose_vit_grid(self, enabled: bool = True) -> None:
        """Opt-in: cache ViT patch grid + Q-Former cross-attention during encode."""
        self.expose_vit_grid = bool(enabled)
        if not self.expose_vit_grid:
            self._last_vit_patches = None
            self._last_qformer_cross_attn = None

    def set_eval_image_size(self, size: int) -> int:
        """Eval-only override of processor resize (default legacy 224). Training unchanged."""
        size = int(size)
        ip = self.processor.image_processor
        ip.size = {"height": size, "width": size}
        if hasattr(ip, "crop_size") and ip.crop_size is not None:
            ip.crop_size = {"height": size, "width": size}
        self.eval_image_size = size
        print(f"BLIP2 eval image_size set to {size}x{size} (Q-Former still outputs {self.get_img_token_n()} tokens)")
        return size

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    @torch.no_grad()
    def _encode_vit_and_qformer(self, pixel_values, return_cross_attn: bool = False):
        interpolate_pos = (
            self.eval_image_size is not None
            and self.eval_image_size != self.native_image_size
        )
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos,
            return_dict=True,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=self.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_kwargs = dict(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        if return_cross_attn:
            qformer_kwargs["output_attentions"] = True
        query_outputs = self.model.qformer(**qformer_kwargs)
        cross_attn = None
        if return_cross_attn and getattr(query_outputs, "cross_attentions", None):
            layers = [a.float() for a in query_outputs.cross_attentions if a is not None]
            if layers:
                cross_attn = torch.stack(layers, dim=0).mean(dim=0).mean(dim=1)[0]
        return image_embeds, query_outputs[0], cross_attn

    @torch.no_grad()
    def get_vit_patch_embeds(self, image: ImageClass) -> torch.Tensor:
        """Pre-Q-Former EVA-ViT patch grid [1, P, D] for patch reranking."""
        inpt = self.processor(images=image, return_tensors="pt")
        pixel_values = inpt["pixel_values"].to(self.device)
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype)
        image_embeds, _, _ = self._encode_vit_and_qformer(pixel_values, return_cross_attn=False)
        return image_embeds

    @torch.no_grad()
    def get_qformer_cross_attention(self, image: ImageClass) -> Optional[torch.Tensor]:
        """Mean Q-Former cross-attention query×patch [Q, P] for vision-gated chunks."""
        try:
            inpt = self.processor(images=image, return_tensors="pt")
            pixel_values = inpt["pixel_values"].to(self.device)
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=model_dtype)
            _, _, cross_attn = self._encode_vit_and_qformer(pixel_values, return_cross_attn=True)
            return cross_attn
        except RecursionError:
            # Guard against transformers output_attentions recursion on some stacks.
            print("[BLIP2] get_qformer_cross_attention recursion; falling back to None (positional AR chunks)")
            return None

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_blip2_llm_inpt(pixel_values, input_ids, attention_mask):
            # Always encode without output_attentions. On some Windows/transformers
            # stacks, output_attentions=True triggers recursive forward wrapping
            # (RecursionError). Vision-gated AR can lazily call
            # get_qformer_cross_attention() when needed.
            want_grid = getattr(self, "expose_vit_grid", False)
            image_embeds, query_output, _ = self._encode_vit_and_qformer(
                pixel_values, return_cross_attn=False
            )
            if want_grid:
                self._last_vit_patches = image_embeds.detach()
                self._last_qformer_cross_attn = None
            else:
                self._last_vit_patches = None
                self._last_qformer_cross_attn = None
            # step 3: use the language model, conditioned on the query outputs and the prompt
            proj = self.model.language_projection
            qdtype = proj.weight.dtype
            language_model_inputs = proj(query_output.to(dtype=qdtype))
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=self.device)
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds.to(dtype=language_model_inputs.dtype)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(self.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(self.device)], dim=1)
            inpt = {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds}
            return inpt
        if imgs != None:
            inpt = self.processor(imgs, texts, return_tensors = 'pt', padding = True)
            inpt = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inpt.items()}
            if 'pixel_values' in inpt:
                model_dtype = next(self.model.parameters()).dtype
                inpt['pixel_values'] = inpt['pixel_values'].to(dtype=model_dtype)
            llm_inpt = get_blip2_llm_inpt(inpt['pixel_values'], inpt['input_ids'], inpt['attention_mask'])
        else:
            inpt = self.get_llm_tokenizer()(texts, return_tensors = 'pt', padding = True).to(self.device)
            inputs_embeds = self.model.language_model.get_input_embeddings()(inpt.input_ids)
            llm_inpt = {'attention_mask': inpt.attention_mask, 'inputs_embeds': inputs_embeds}
        vt_range = None if imgs == None else [0, self.get_img_token_n()]
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        outpt = self.model.language_model(
            inputs_embeds=llm_inpt['inputs_embeds'],
            attention_mask=llm_inpt['attention_mask'],
            output_attentions=None, output_hidden_states=None,
            return_dict=True, use_cache = False
        )
        return outpt

    def get_img_special_token_str(self):
        return None

    def get_img_special_token_id(self):
        return None
        
    def get_img_token_n(self):
        return self.model.config.num_query_tokens

    def is_q_former_based(self):
        return True

