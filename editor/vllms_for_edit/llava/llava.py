from typing import List, Union, Optional
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
import torch, inspect


class LlavaForEdit(BaseVLLMForEdit):
    '''For llava-v1.5-7b-hf'''
    def __init__(self, model_path:str, device = 'cuda', 
                 auto_add_img_special_token = True) -> None:
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        # float16 required on ~16GB GPUs (fp32 7B OOMs during shard load).
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.processor = LlavaProcessor.from_pretrained(model_path)
        # self.processor.patch_size = self.model.config.vision_config.patch_size
        # self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.model = self.model.eval().requires_grad_(False)
        # Cut activation memory for AVBL multi-forward on ~24GB (LLaVA-7B).
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "language_model") and hasattr(
                self.model.language_model, "gradient_checkpointing_enable"
            ):
                self.model.language_model.gradient_checkpointing_enable()
        except Exception as e:
            print("[LlavaForEdit] gradient_checkpointing_enable skipped:", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().__init__(self.model, device, auto_add_img_special_token)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_llava_llm_inpt(mllm, input_ids, attention_mask, pixel_values):
            # transformers≥4.45: processor expands <image> to N patch tokens; merge via masked_scatter.
            inputs_embeds = mllm.get_input_embeddings()(input_ids)
            if pixel_values is not None and input_ids.shape[1] != 1:
                vision_feature_layer = mllm.config.vision_feature_layer
                vision_feature_select_strategy = mllm.config.vision_feature_select_strategy
                image_features = mllm.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )
                image_features = torch.cat(image_features, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                special_image_mask = mllm.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            return {
                'attention_mask': attention_mask,
                'inputs_embeds': inputs_embeds,
                'position_ids': None,
            }
        # LlavaProcessor signature is (images=None, text=None, ...); positional args swap and break.
        inpt = self.processor(text=texts, images=imgs, return_tensors='pt', padding=True)
        for k, v in inpt.items():
            inpt[k] = v.to(self.device) if hasattr(v, 'to') else v
        llm_inpt = get_llava_llm_inpt(
            self.model, inpt.input_ids, inpt.attention_mask,
            inpt.pixel_values if imgs is not None else None,
        )
        if imgs is not None:
            img_pos = torch.where(inpt['input_ids'][0] == self.get_img_special_token_id())[0]
            img_begin = int(img_pos[0])
            img_end = int(img_pos[-1]) + 1
            vt_range = [img_begin, img_end]
        else:
            vt_range = None
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        assert 'inputs_embeds' in llm_inpt.keys()
        # language_model is LlamaModel (no lm_head); logits come from LlavaForConditionalGeneration.lm_head.
        sig = inspect.signature(self.model.language_model.forward)
        filtered = {k: v for k, v in llm_inpt.items() if k in sig.parameters and v is not None}
        outpt = self.model.language_model(**filtered, use_cache=False)
        hidden = outpt.last_hidden_state if hasattr(outpt, 'last_hidden_state') else outpt[0]
        logits = self.model.lm_head(hidden)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, last_hidden_state=hidden)

    def get_img_special_token_str(self):
        return '<image>'

    def get_img_special_token_id(self):
        return self.model.config.image_token_index
        
    def get_img_token_n(self):
        return (self.model.config.vision_config.image_size//self.model.config.vision_config.patch_size)**2 

    def is_q_former_based(self):
        return False

