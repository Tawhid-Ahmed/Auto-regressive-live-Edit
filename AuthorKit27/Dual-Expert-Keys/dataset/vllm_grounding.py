"""VLKEB / EVQA loaders with adversarial grounding probes.

Paper (Experiments): attaches image-swap and hard-negative generality keys so the
evaluator can measure false-positive retrieval (FPR) under a fixed prompt and a
mismatched image. Opt-in — does not modify ``dataset.vllm.VLKEB``.

Used by:
  - ``test_grounding_eval.py`` (token metrics + optional routing FPR logs)
  - ``tools/log_grounding_diagnostics.py`` (routing false_cross / false_activation)
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

from tqdm import tqdm

from .vllm import BaseVLLMEditData


class GroundingVLKEB(BaseVLLMEditData):
    """VLKEB rows plus paper adversarial grounding generality probes.

    Probe keys under ``generality`` (same target as reliability unless noted):
      image_swap, hard_negative, modality_consistency, prompt_conflict.
    """

    def __init__(
        self,
        data_path: str = "data/VLKEB/grounding/grounding_eval_short.json",
        img_root_dir: str = "data/VLKEB/VLKEB_images/mmkb_images",
        data_n=None,
        preload_images: bool = False,
        sample_seed=None,
    ) -> None:
        print("Load GroundingVLKEB from: %s " % data_path)
        raw_rows, data_with_img_path = self._load_rows(
            data_path, img_root_dir, data_n, sample_seed=sample_seed
        )
        for d in data_with_img_path:
            d["locality"]["text_loc"][0]["prompt"] = "%s?" % d["locality"]["text_loc"][0]["prompt"]
            d["locality"]["image_loc"][0]["prompt"] = "%s The answer is:" % d["locality"]["image_loc"][0]["prompt"]
        self._attach_grounding_probes(raw_rows, data_with_img_path, img_root_dir)
        data_with_img = deepcopy(data_with_img_path)
        if preload_images:
            for d in tqdm(data_with_img, "Loading images"):
                self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def _read_payload_rows(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"], payload.get("meta")
        return payload, None

    def _rows_to_structured(self, raw_rows, img_root_dir: str):
        data_with_img_path = []
        for d in raw_rows:
            new_d = {
                "requests": [{}],
                "generality": {"text_rephrase": [], "image_rephrase": []},
                "locality": {"text_loc": [], "image_loc": []},
            }
            new_d["requests"][0]["image"] = os.path.join(img_root_dir, d["image"])
            new_d["requests"][0]["prompt"] = d["src"]
            new_d["requests"][0]["target_new"] = d["alt"]
            new_d["generality"]["text_rephrase"].append(
                {"image": new_d["requests"][0]["image"], "prompt": d["rephrase"], "target": d["alt"]}
            )
            new_d["generality"]["image_rephrase"].append(
                {
                    "image": os.path.join(img_root_dir, d["image_rephrase"]),
                    "prompt": d["src"],
                    "target": d["alt"],
                }
            )
            new_d["locality"]["text_loc"].append(
                {"image": None, "prompt": d["loc"], "target": d["loc_ans"]}
            )
            new_d["locality"]["image_loc"].append(
                {
                    "image": os.path.join(img_root_dir, d["m_loc"]),
                    "prompt": d["m_loc_q"],
                    "target": d["m_loc_a"],
                }
            )
            data_with_img_path.append(new_d)
        return data_with_img_path

    def _load_rows(self, data_path: str, img_root_dir: str, data_n, sample_seed=None):
        if data_n is None:
            data_n = 99999999
        rows, _ = self._read_payload_rows(data_path)
        if data_n < len(rows) and sample_seed is not None:
            import numpy as np

            rng = np.random.default_rng(int(sample_seed))
            idx = rng.permutation(len(rows))[: int(data_n)]
            raw_rows = [rows[int(i)] for i in idx]
            print(
                "Subsampled %d / %d rows with sample_seed=%s"
                % (len(raw_rows), len(rows), sample_seed)
            )
        else:
            data_n = min(len(rows), data_n)
            raw_rows = rows[:data_n]
        data_with_img_path = self._rows_to_structured(raw_rows, img_root_dir)
        return raw_rows, data_with_img_path

    @classmethod
    def from_source_index(
        cls,
        data_path: str,
        img_root_dir: str,
        source_index: int,
        preload_images: bool = True,
    ) -> "GroundingVLKEB":
        """Load exactly one sample by ``source_index`` from the grounding JSON."""
        inst = cls.__new__(cls)
        rows, meta = inst._read_payload_rows(data_path)
        matches = [r for r in rows if r.get("source_index") == source_index]
        if not matches:
            matches = [rows[source_index]] if 0 <= source_index < len(rows) else []
        if not matches:
            raise IndexError("No sample with source_index=%s in %s" % (source_index, data_path))
        raw_rows = matches[:1]
        print("Load GroundingVLKEB single sample source_index=%s from: %s" % (source_index, data_path))
        data_with_img_path = inst._rows_to_structured(raw_rows, img_root_dir)
        for d in data_with_img_path:
            d["locality"]["text_loc"][0]["prompt"] = "%s?" % d["locality"]["text_loc"][0]["prompt"]
            d["locality"]["image_loc"][0]["prompt"] = "%s The answer is:" % d["locality"]["image_loc"][0]["prompt"]
        inst._attach_grounding_probes(raw_rows, data_with_img_path, img_root_dir)
        data_with_img = deepcopy(data_with_img_path)
        if preload_images:
            for d in data_with_img:
                inst.__load_imgs_for_data_with_img_path__(d)
        BaseVLLMEditData.__init__(inst, data_with_img, data_with_img_path)
        inst._raw_row = raw_rows[0]
        inst._meta = meta
        return inst

    @classmethod
    def from_row_indices(
        cls,
        data_path: str,
        img_root_dir: str,
        row_indices: list,
        preload_images: bool = True,
    ) -> "GroundingVLKEB":
        """Load selected 0-based rows (for pooled single-target eval)."""
        inst = cls.__new__(cls)
        rows, meta = inst._read_payload_rows(data_path)
        raw_rows = [rows[i] for i in row_indices if 0 <= i < len(rows)]
        if not raw_rows:
            raise IndexError("No valid rows in %s" % row_indices)
        print("Load GroundingVLKEB rows %s from: %s" % (row_indices, data_path))
        data_with_img_path = inst._rows_to_structured(raw_rows, img_root_dir)
        for d in data_with_img_path:
            d["locality"]["text_loc"][0]["prompt"] = "%s?" % d["locality"]["text_loc"][0]["prompt"]
            d["locality"]["image_loc"][0]["prompt"] = "%s The answer is:" % d["locality"]["image_loc"][0]["prompt"]
        inst._attach_grounding_probes(raw_rows, data_with_img_path, img_root_dir)
        data_with_img = deepcopy(data_with_img_path)
        if preload_images:
            for d in tqdm(data_with_img, "Loading images"):
                inst.__load_imgs_for_data_with_img_path__(d)
        BaseVLLMEditData.__init__(inst, data_with_img, data_with_img_path)
        inst._raw_rows = raw_rows
        inst._meta = meta
        return inst

    def _attach_grounding_probes(self, raw_rows, data_with_img_path, img_root_dir):
        for raw, structured in zip(raw_rows, data_with_img_path):
            gp = raw.get("grounding_probes")
            if not gp:
                continue
            alt = raw["alt"]
            src = raw["src"]

            def _probe(image_rel, prompt, target):
                return {
                    "image": os.path.join(img_root_dir, image_rel),
                    "prompt": prompt,
                    "target": target,
                }

            if gp.get("image_swap"):
                structured["generality"]["image_swap"] = [_probe(gp["image_swap"], src, alt)]
            if gp.get("hard_neg_image"):
                structured["generality"]["hard_negative"] = [_probe(gp["hard_neg_image"], src, alt)]
            consistency_img = gp.get("modality_consistency_image") or raw.get("image_rephrase")
            if consistency_img:
                structured["generality"]["modality_consistency"] = [_probe(consistency_img, src, alt)]
            if gp.get("conflict_prompt"):
                structured["generality"]["prompt_conflict"] = [
                    _probe(raw["image"], gp["conflict_prompt"], alt)
                ]

    def dataset_name(self):
        return "VLKEB_grounding"


class GroundingEVQA(GroundingVLKEB):
    """EVQA rows plus VLKEB-style grounding probes (swap / hard-neg / conflict).

    Uses EVQA prompt templates (``The answer is:``) so it matches LiveEdit EVQA training.
    """

    def __init__(
        self,
        data_path: str = "data/easy-edit-mm/grounding/grounding_train_short.json",
        img_root_dir: str = "data/easy-edit-mm/images",
        data_n=None,
        preload_images: bool = False,
        sample_seed=None,
    ) -> None:
        print("Load GroundingEVQA from: %s " % data_path)
        raw_rows, data_with_img_path = self._load_rows(
            data_path, img_root_dir, data_n, sample_seed=sample_seed
        )
        for d in data_with_img_path:
            d["requests"][0]["prompt"] = "%s The answer is:" % d["requests"][0]["prompt"]
            d["generality"]["text_rephrase"][0]["prompt"] = (
                "%s The answer is:" % d["generality"]["text_rephrase"][0]["prompt"]
            )
            d["generality"]["image_rephrase"][0]["prompt"] = (
                "%s The answer is:" % d["generality"]["image_rephrase"][0]["prompt"]
            )
            d["locality"]["text_loc"][0]["prompt"] = "%s?" % d["locality"]["text_loc"][0]["prompt"]
            d["locality"]["image_loc"][0]["prompt"] = (
                "%s The answer is:" % d["locality"]["image_loc"][0]["prompt"]
            )
        self._attach_grounding_probes(raw_rows, data_with_img_path, img_root_dir)
        # EVQA probe prompts also need the answer suffix where applicable
        for d in data_with_img_path:
            for key in ("image_swap", "hard_negative", "modality_consistency", "prompt_conflict"):
                if key in d.get("generality", {}) and d["generality"][key]:
                    p = d["generality"][key][0].get("prompt", "")
                    if p and not p.rstrip().endswith(":"):
                        d["generality"][key][0]["prompt"] = "%s The answer is:" % p
        data_with_img = deepcopy(data_with_img_path)
        if preload_images:
            for d in tqdm(data_with_img, "Loading images"):
                self.__load_imgs_for_data_with_img_path__(d)
        BaseVLLMEditData.__init__(self, data_with_img, data_with_img_path)

    def dataset_name(self):
        return "EVQA_grounding"
