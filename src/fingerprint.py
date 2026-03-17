from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .params import PipelineParams
from .utils import resize_keep_aspect
from .pipeline import run_pipeline, evaluate_pipeline

from .binarization import binarize_image, apply_mask_to_binary
from .morphology import apply_morphology
from .skeleton import get_skeleton, prune_skeleton_topology
from .minutiae import extract_minutiae_crossing_number
from .debugger import PipelineDebugger


class FingerprintImage:

    def __init__(self, name: str, kind: str, path: Optional[str] = None):

        self.name = name
        self.kind = kind
        self.path = path

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------
        self.input: Dict[str, Any] = {
            "rgb": None,
            "gray": None,
            "shape": None,
        }

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------
        self.crop: Dict[str, Any] = {
            "rgb": None,
            "gray": None,
            "mask": None,
            "bbox": None,
        }

        # ----------------------------------------------------
        # VARIANTS
        # ----------------------------------------------------
        self.variants: Dict[str, Any] = {}

        # ----------------------------------------------------
        # PIPELINE STORAGE
        # ----------------------------------------------------
        self.pipeline: Dict[str, Any] = {}

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------
        self.evaluation: Dict[str, Any] = {}

        # attach debugger
        self.debug = PipelineDebugger(self)

    # ========================================================
    # LOADING
    # ========================================================

    def load_photo(self, path: Optional[str] = None):

        if path is not None:
            self.path = path

        img_bgr = cv2.imread(self.path)

        if img_bgr is None:
            raise ValueError(f"Could not load image: {self.path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        self.input["rgb"] = img_rgb
        self.input["gray"] = gray
        self.input["shape"] = img_rgb.shape

    def load_scanned(self, path: Optional[str] = None, ridges_white: bool = True):

        if path is not None:
            self.path = path

        gray = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise ValueError(f"Could not load image: {self.path}")

        gray = gray.astype(np.uint8)

        if not ridges_white:
            gray = 255 - gray

        self.input["gray"] = gray
        self.input["shape"] = gray.shape

    # ========================================================
    # CROP
    # ========================================================

    def set_crop(self, x: int, y: int, w: int, h: int, mask: Optional[np.ndarray] = None):

        self.crop["bbox"] = (x, y, w, h)

        if self.input["rgb"] is not None:
            self.crop["rgb"] = self.input["rgb"][y:y + h, x:x + w].copy()

        if self.input["gray"] is not None:
            self.crop["gray"] = self.input["gray"][y:y + h, x:x + w].copy()

        if mask is not None:
            self.crop["mask"] = mask[y:y + h, x:x + w].copy()

    def use_full_image_as_crop(self):

        gray = self.input["gray"]

        if gray is None:
            raise ValueError("No grayscale image loaded")

        h, w = gray.shape[:2]

        self.set_crop(0, 0, w, h)

    # ========================================================
    # VARIANTS
    # ========================================================

    def compute_crop_variants(self):

        if self.crop["gray"] is None:
            raise ValueError("Crop must exist before computing variants")

        gray = self.crop["gray"]

        self.variants = {
            "gray_default": gray.copy()
        }

    # ========================================================
    # PIPELINE INPUT
    # ========================================================

    def get_processing_input(self, source_name="gray_default"):

        if self.variants and source_name in self.variants:
            return self.variants[source_name]

        if self.crop["gray"] is not None:
            return self.crop["gray"]

        if self.input["gray"] is not None:
            return self.input["gray"]

        raise ValueError("No grayscale image available")

    # ========================================================
    # PIPELINE (delegates to module)
    # ========================================================

    def run_pipeline(self, params: PipelineParams, source_name: str = "gray_default"):
        return run_pipeline(self, params, source_name)

    def evaluate(self):
        return evaluate_pipeline(self)

    # ========================================================
    # STEP HELPERS (for notebooks / debugging)
    # ========================================================

    def step_binarize(self, img, method="adaptive", use_mask=True, **kwargs):

        out = binarize_image(img, method=method, **kwargs)

        if use_mask and self.crop.get("mask") is not None:
            out = apply_mask_to_binary(out, self.crop["mask"])

        return out

    def step_morphology(
        self,
        binary,
        close_k=3,
        open_k=1,
        min_area=0,
        fill=False,
        dilate_k=0,
        use_mask=True
    ):

        out = apply_morphology(binary, close_k, open_k, min_area, fill, dilate_k)

        if use_mask and self.crop.get("mask") is not None:
            out = apply_mask_to_binary(out, self.crop["mask"])

        return out

    def step_skeleton(self, binary):
        return get_skeleton(binary)

    def step_prune(self, skeleton, spur_length=8, min_component_size=10, passes=2):
        return prune_skeleton_topology(skeleton, spur_length, min_component_size, passes)

    def step_minutiae(self, skeleton, border_margin=10, cluster_dist=6):

        return extract_minutiae_crossing_number(
            skeleton,
            mask=self.crop.get("mask"),
            border_margin=border_margin,
            cluster_dist=cluster_dist
        )

    # ========================================================
    # EXPORT
    # ========================================================

    def to_summary_dict(self):

        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "input_shape": self.input["shape"],
            "crop_bbox": self.crop["bbox"],
            "pipeline_params": self.pipeline.get("params"),
            "evaluation": self.evaluation,
        }

    def save_summary_json(self, path: str):

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_summary_dict(), f, indent=2)

if __name__ == "__main__":
    from pathlib import Path
    from .params import PipelineParams

    img_path = Path("images/scanned/left2.png")

    fp = FingerprintImage("left2", "scanned", str(img_path))
    fp.load_scanned()
    fp.use_full_image_as_crop()
    fp.compute_crop_variants()

    params = PipelineParams(
        bin_method="adaptive_gaussian_detector",
        bin_block=21,
        bin_C=2,
        bin_blur_ksize=3,
        bin_invert_before_check=True,
        want_ridges_white=True,
        apply_mask=True,
        auto_mask=True,
        mask_from_black_background=True,
    )

    fp.run_pipeline(params)
    fp.evaluate()

    print("Score:", fp.evaluation.get("total_score"))
    print("Minutiae summary:", fp.evaluation.get("minutiae_summary"))

    fp.debug.show_pipeline_core()
    fp.debug.show_minutiae()