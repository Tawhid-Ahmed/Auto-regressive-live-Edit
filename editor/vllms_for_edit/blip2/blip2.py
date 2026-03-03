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


class BLIP2OPTForEdit(BaseVLLMForEdit):
    '''For blip2-opt 2.7b'''
    def __init__(self, model_path:str, device = 'cuda') -> None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        try:
            from huggingface_hub.errors import HFValidationError
        except ImportError:
            from huggingface_hub import HFValidationError
        import torch
        model_path = str(Path(model_path).resolve())
        print(f"Loading BLIP2 model from {model_path}...")
        config_json = os.path.join(model_path, "config.json")
        _p = Path(model_path)
        is_local_path = os.path.sep in model_path or (getattr(_p, "drive", "") != "")
        try:
            # Local path (e.g. Windows): load from files to avoid Hub repo_id validation
            if os.path.isfile(config_json):
                self.model, self.processor = _load_blip2_from_local_dir(model_path, device)
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
            # Move model to device
            if device == 'cpu':
                device_obj = torch.device('cpu')
                self.model = self.model.to(device_obj)
            else:
                device_obj = torch.device(device) if isinstance(device, str) and ':' in device else torch.device(device if isinstance(device, str) else f'cuda:{device}')
                self.model = self.model.to(device_obj)
            print(f"Model moved to {device_obj}")
        except Exception as e:
            print(f"ERROR loading BLIP2 model: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.model = self.model.eval().requires_grad_(False)
        super().__init__(self.model, device, False)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_blip2_llm_inpt(pixel_values, input_ids, attention_mask):
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            )
            image_embeds = vision_outputs[0]
            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            )
            query_output = query_outputs[0]
            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.model.language_projection(query_output)
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=self.device)
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(self.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(self.device)], dim=1)
            inpt = {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds}
            return inpt
        if imgs != None:
            inpt = self.processor(imgs, texts, return_tensors = 'pt', padding = True)
            inpt = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inpt.items()}
            # Ensure pixel_values match model dtype (float32)
            if 'pixel_values' in inpt:
                inpt['pixel_values'] = inpt['pixel_values'].to(torch.float32)
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

