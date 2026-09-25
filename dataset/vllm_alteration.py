"""Step 3 patch: VLKEB loader with same-image alteration generality probes.

Edit uses the original image; test probes apply on-the-fly transforms
(low-res, rotate, mirror, crop, mild degradation) to measure whether
stored edits survive appearance change.

Opt-in only — use with ``test_alteration_eval.py``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Union

from tqdm import tqdm

from .image_alterations import ALTERATION_NAMES, AlteredImageRef
from .vllm_grounding import GroundingVLKEB


class AlterationVLKEB(GroundingVLKEB):
    """VLKEB eval rows plus Step-3 same-image alteration probes."""

    PROBE_PREFIX = "alter_"

    def __init__(
        self,
        data_path: str = "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json",
        img_root_dir: str = "data/VLKEB/VLKEB_images/mmkb_images",
        data_n=None,
        preload_images: bool = False,
        alterations: tuple = ALTERATION_NAMES,
    ) -> None:
        self._alteration_keys = tuple(alterations)
        print("Load AlterationVLKEB from: %s " % data_path)
        raw_rows, data_with_img_path = self._load_rows(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d["locality"]["text_loc"][0]["prompt"] = "%s?" % d["locality"]["text_loc"][0]["prompt"]
            d["locality"]["image_loc"][0]["prompt"] = "%s The answer is:" % d["locality"]["image_loc"][0]["prompt"]
        self._attach_alteration_probes(raw_rows, data_with_img_path)
        data_with_img = deepcopy(data_with_img_path)
        for d in data_with_img:
            self._promote_alteration_refs(d)
        if preload_images:
            for d in tqdm(data_with_img, "Loading images"):
                self.__load_imgs_for_data_with_img_path__(d)
        from .vllm import BaseVLLMEditData

        BaseVLLMEditData.__init__(self, data_with_img, data_with_img_path)

    def _attach_grounding_probes(self, raw_rows, data_with_img_path, img_root_dir):
        """Step 1 adversarial probes omitted — alteration-only eval."""

    def _attach_alteration_probes(self, raw_rows, data_with_img_path):
        for raw, structured in zip(raw_rows, data_with_img_path):
            src = raw["src"]
            alt = raw["alt"]
            img_path = structured["requests"][0]["image"]
            for alt_key in self._alteration_keys:
                probe_name = "%s%s" % (self.PROBE_PREFIX, alt_key)
                structured["generality"][probe_name] = [
                    {
                        "image": img_path,
                        "prompt": src,
                        "target": alt,
                        "_alteration": alt_key,
                    }
                ]

    def _promote_alteration_refs(self, d: Union[List, Dict]):
        if isinstance(d, dict):
            if d.get("_alteration") and isinstance(d.get("image"), str):
                d["image"] = AlteredImageRef(d["image"], d["_alteration"])
                del d["_alteration"]
            for v in d.values():
                self._promote_alteration_refs(v)
        elif isinstance(d, list):
            for item in d:
                self._promote_alteration_refs(item)

    def dataset_name(self):
        return "VLKEB_alteration"
