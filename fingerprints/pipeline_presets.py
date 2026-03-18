from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import ipywidgets as widgets
import numpy as np

from .interactive import InteractivePipeline, Step


# ============================================================
# OPTIONAL IMPORTS
# The module stays importable even if some functions are absent.
# ============================================================

def _optional_attr(module_name: str, attr_name: str):
    try:
        module = __import__(f"{__package__}.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name, None)
    except Exception:
        return None


# Enhancement / preprocessing
apply_clahe = _optional_attr("enhancement", "apply_clahe")
difference_of_gaussians = _optional_attr("enhancement", "difference_of_gaussians")
local_contrast_enhancement = _optional_attr("enhancement", "local_contrast_enhancement")
contrast_stretch_saturated = _optional_attr("enhancement", "contrast_stretch_saturated")
gamma_correction = _optional_attr("enhancement", "gamma_correction")
dynamic_gabor_pipeline = _optional_attr("enhancement", "dynamic_gabor_pipeline")
apply_ridge_filter = _optional_attr("enhancement", "apply_ridge_filter")
fft_gaussian_bandpass = _optional_attr("enhancement", "fft_gaussian_bandpass")
coherence_enhancing_diffusion = _optional_attr("enhancement", "coherence_enhancing_diffusion")
gabor_rebuild = _optional_attr("enhancement", "gabor_rebuild")

# Mask / ROI
build_foreground_mask = _optional_attr("mask", "build_foreground_mask")
mask_from_black_background = _optional_attr("mask", "mask_from_black_background")
apply_mask_outside_zero = _optional_attr("mask", "apply_mask_outside_zero")

# Binarization / morphology / skeleton / minutiae
binarize_image = _optional_attr("binarization", "binarize_image")
apply_mask_to_binary = _optional_attr("binarization", "apply_mask_to_binary")
apply_morphology = _optional_attr("morphology", "apply_morphology")
get_skeleton = _optional_attr("skeleton", "get_skeleton")
prune_skeleton_topology = _optional_attr("skeleton", "prune_skeleton_topology")
extract_minutiae_crossing_number = _optional_attr("minutiae", "extract_minutiae_crossing_number")


# ============================================================
# HELPERS
# ============================================================

@dataclass
class OperationSpec:
    key: str
    name: str
    func: Callable
    input_map: Dict[str, str]
    output_keys: List[str]
    params_spec: Dict[str, Dict[str, Any]]
    description: str = ""
    display_key: Optional[str] = None


def _slider_from_spec(name: str, spec: Dict[str, Any]):
    kind = spec.get("kind", "int")
    label = spec.get("label", name)

    if kind == "int":
        return widgets.IntSlider(
            value=spec["value"],
            min=spec["min"],
            max=spec["max"],
            step=spec.get("step", 1),
            description=label,
            continuous_update=spec.get("continuous_update", False),
        )

    if kind == "float":
        return widgets.FloatSlider(
            value=spec["value"],
            min=spec["min"],
            max=spec["max"],
            step=spec.get("step", 0.1),
            description=label,
            continuous_update=spec.get("continuous_update", False),
        )

    if kind == "bool":
        return widgets.Checkbox(
            value=spec["value"],
            description=label,
            indent=False,
        )

    if kind == "choice":
        return widgets.Dropdown(
            options=spec["options"],
            value=spec["value"],
            description=label,
        )

    raise ValueError(f"Unknown widget kind: {kind}")


def _build_params_widgets(params_spec: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in params_spec.items():
        if spec.get("kind") == "fixed":
            out[name] = spec["value"]
        else:
            out[name] = _slider_from_spec(name, spec)
    return out


def _make_step(spec: OperationSpec, enabled: bool = True, override_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Step:
    params_spec = dict(spec.params_spec)
    if override_params:
        for key, value in override_params.items():
            params_spec[key] = value

    return Step(
        name=spec.name,
        func=spec.func,
        input_map=spec.input_map,
        output_keys=spec.output_keys,
        params=_build_params_widgets(params_spec),
        description=spec.description,
        enabled=enabled,
        display_key=spec.display_key,
    )


def _available_operation_specs() -> Dict[str, OperationSpec]:
    ops: Dict[str, OperationSpec] = {}

    if mask_from_black_background is not None:
        ops["mask_black_bg"] = OperationSpec(
            key="mask_black_bg",
            name="Mask from Black Background",
            func=mask_from_black_background,
            input_map={"gray": "image"},
            output_keys=["mask"],
            params_spec={
                "threshold": {"kind": "int", "value": 20, "min": 0, "max": 120, "step": 1, "label": "Thresh"},
            },
            description="Estimate ROI mask from dark background around the fingerprint.",
            display_key="mask",
        )

    if build_foreground_mask is not None:
        ops["foreground_mask"] = OperationSpec(
            key="foreground_mask",
            name="Foreground Mask",
            func=build_foreground_mask,
            input_map={"gray": "image"},
            output_keys=["mask"],
            params_spec={
                "blur_ksize": {"kind": "int", "value": 5, "min": 1, "max": 21, "step": 2, "label": "Blur"},
                "threshold": {"kind": "int", "value": 15, "min": 0, "max": 100, "step": 1, "label": "Thresh"},
            },
            description="Find foreground region containing usable ridge information.",
            display_key="mask",
        )

    if apply_mask_outside_zero is not None:
        ops["apply_mask"] = OperationSpec(
            key="apply_mask",
            name="Apply Mask",
            func=apply_mask_outside_zero,
            input_map={"img_gray": "image", "mask": "mask"},
            output_keys=["image"],
            params_spec={},
            description="Zero pixels outside mask.",
        )

    if contrast_stretch_saturated is not None:
        ops["stretch"] = OperationSpec(
            key="stretch",
            name="Contrast Stretch",
            func=contrast_stretch_saturated,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "low_pct": {"kind": "float", "value": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "label": "Low %"},
                "high_pct": {"kind": "float", "value": 99.0, "min": 90.0, "max": 100.0, "step": 0.1, "label": "High %"},
            },
            description="Simple contrast rescaling by saturated percentiles.",
        )

    if gamma_correction is not None:
        ops["gamma"] = OperationSpec(
            key="gamma",
            name="Gamma Correction",
            func=gamma_correction,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "gamma": {"kind": "float", "value": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "label": "Gamma"},
            },
            description="Adjust global brightness / ridge contrast.",
        )

    if apply_clahe is not None:
        ops["clahe"] = OperationSpec(
            key="clahe",
            name="CLAHE",
            func=apply_clahe,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "clip": {"kind": "float", "value": 2.5, "min": 1.0, "max": 8.0, "step": 0.1, "label": "Clip"},
                "tile": {"kind": "int", "value": 8, "min": 2, "max": 32, "step": 2, "label": "Tile"},
            },
            description="Local histogram equalization.",
        )

    if local_contrast_enhancement is not None:
        ops["local_contrast"] = OperationSpec(
            key="local_contrast",
            name="Local Contrast",
            func=local_contrast_enhancement,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "ksize": {"kind": "int", "value": 15, "min": 3, "max": 61, "step": 2, "label": "Kernel"},
                "amount": {"kind": "float", "value": 1.0, "min": 0.2, "max": 3.0, "step": 0.1, "label": "Amount"},
            },
            description="Local contrast boosting for uneven illumination.",
        )

    if difference_of_gaussians is not None:
        ops["dog"] = OperationSpec(
            key="dog",
            name="Difference of Gaussians",
            func=difference_of_gaussians,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "sigma1": {"kind": "float", "value": 1.0, "min": 0.2, "max": 8.0, "step": 0.1, "label": "Sigma 1"},
                "sigma2": {"kind": "float", "value": 2.5, "min": 0.5, "max": 20.0, "step": 0.1, "label": "Sigma 2"},
            },
            description="Band-pass like enhancement.",
        )

    if fft_gaussian_bandpass is not None:
        ops["fft_bandpass"] = OperationSpec(
            key="fft_bandpass",
            name="FFT Bandpass",
            func=fft_gaussian_bandpass,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "sigma_low": {"kind": "float", "value": 3.0, "min": 0.5, "max": 30.0, "step": 0.5, "label": "Sigma Low"},
                "sigma_high": {"kind": "float", "value": 20.0, "min": 2.0, "max": 100.0, "step": 1.0, "label": "Sigma High"},
            },
            description="Frequency-domain band-pass enhancement.",
        )

    if coherence_enhancing_diffusion is not None:
        ops["ced"] = OperationSpec(
            key="ced",
            name="CED",
            func=coherence_enhancing_diffusion,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "iterations": {"kind": "int", "value": 5, "min": 1, "max": 30, "step": 1, "label": "Iter"},
            },
            description="Coherence-enhancing diffusion.",
        )

    if apply_ridge_filter is not None:
        ops["ridge_filter"] = OperationSpec(
            key="ridge_filter",
            name="Ridge Filter",
            func=apply_ridge_filter,
            input_map={"img_gray": "image"},
            output_keys=["image"],
            params_spec={
                "sigma": {"kind": "float", "value": 2.0, "min": 0.5, "max": 10.0, "step": 0.1, "label": "Sigma"},
            },
            description="Ridge-like structure enhancement.",
        )

    if dynamic_gabor_pipeline is not None:
        ops["dynamic_gabor"] = OperationSpec(
            key="dynamic_gabor",
            name="Dynamic Gabor",
            func=dynamic_gabor_pipeline,
            input_map={"gray": "image"},
            output_keys=["image", "orientation", "frequency"],
            params_spec={
                "orient_block": {"kind": "int", "value": 16, "min": 4, "max": 64, "step": 2, "label": "Ori blk"},
                "orient_smooth": {"kind": "int", "value": 5, "min": 1, "max": 21, "step": 2, "label": "Ori sm"},
                "freq_block": {"kind": "int", "value": 32, "min": 8, "max": 96, "step": 2, "label": "Freq blk"},
                "min_wave": {"kind": "int", "value": 4, "min": 2, "max": 20, "step": 1, "label": "Min wave"},
                "max_wave": {"kind": "int", "value": 20, "min": 4, "max": 40, "step": 1, "label": "Max wave"},
                "default_freq": {"kind": "float", "value": 0.08, "min": 0.02, "max": 0.25, "step": 0.01, "label": "Def freq"},
                "sigma_base": {"kind": "float", "value": 4.0, "min": 1.0, "max": 12.0, "step": 0.1, "label": "Sigma"},
                "gamma": {"kind": "float", "value": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Gamma"},
                "kernel_size": {"kind": "int", "value": 21, "min": 7, "max": 51, "step": 2, "label": "Kernel"},
            },
            description="Estimate orientation/frequency and apply contextual Gabor enhancement.",
            display_key="image",
        )

    if gabor_rebuild is not None:
        ops["gabor_rebuild"] = OperationSpec(
            key="gabor_rebuild",
            name="Gabor Rebuild",
            func=gabor_rebuild,
            input_map={"gray": "image", "theta_map": "orientation", "freq_map": "frequency"},
            output_keys=["image"],
            params_spec={
                "sigma_base": {"kind": "float", "value": 4.0, "min": 1.0, "max": 12.0, "step": 0.1, "label": "Sigma"},
                "gamma": {"kind": "float", "value": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Gamma"},
                "kernel_size": {"kind": "int", "value": 21, "min": 7, "max": 51, "step": 2, "label": "Kernel"},
            },
            description="Rebuild enhanced image from precomputed orientation/frequency maps.",
        )

    if binarize_image is not None:
        ops["binarize"] = OperationSpec(
            key="binarize",
            name="Binarization",
            func=binarize_image,
            input_map={"img_gray": "image"},
            output_keys=["binary"],
            params_spec={
                "method": {"kind": "choice", "value": "adaptive", "options": ["adaptive", "otsu"], "label": "Method"},
                "block_size": {"kind": "int", "value": 31, "min": 3, "max": 101, "step": 2, "label": "Block"},
                "C": {"kind": "int", "value": 5, "min": -20, "max": 20, "step": 1, "label": "C"},
                "invert": {"kind": "bool", "value": False, "label": "Invert"},
            },
            description="Convert grayscale ridge image into binary mask.",
            display_key="binary",
        )

    if apply_mask_to_binary is not None:
        ops["mask_binary"] = OperationSpec(
            key="mask_binary",
            name="Mask Binary",
            func=apply_mask_to_binary,
            input_map={"binary": "binary", "mask": "mask"},
            output_keys=["binary"],
            params_spec={},
            description="Apply ROI mask to binary result.",
            display_key="binary",
        )

    if apply_morphology is not None:
        ops["morphology"] = OperationSpec(
            key="morphology",
            name="Morphology",
            func=apply_morphology,
            input_map={"binary": "binary"},
            output_keys=["binary"],
            params_spec={
                "open_ksize": {"kind": "int", "value": 0, "min": 0, "max": 11, "step": 1, "label": "Open"},
                "close_ksize": {"kind": "int", "value": 0, "min": 0, "max": 11, "step": 1, "label": "Close"},
                "dilate_ksize": {"kind": "int", "value": 0, "min": 0, "max": 11, "step": 1, "label": "Dilate"},
                "erode_ksize": {"kind": "int", "value": 0, "min": 0, "max": 11, "step": 1, "label": "Erode"},
            },
            description="Clean small noise and repair binary mask.",
            display_key="binary",
        )

    if get_skeleton is not None:
        ops["skeleton"] = OperationSpec(
            key="skeleton",
            name="Skeletonization",
            func=get_skeleton,
            input_map={"binary": "binary"},
            output_keys=["skeleton"],
            params_spec={},
            description="Thin binary ridges to 1-pixel skeleton.",
            display_key="skeleton",
        )

    if prune_skeleton_topology is not None:
        ops["prune_skeleton"] = OperationSpec(
            key="prune_skeleton",
            name="Prune Skeleton",
            func=prune_skeleton_topology,
            input_map={"skeleton": "skeleton"},
            output_keys=["skeleton"],
            params_spec={
                "min_branch_length": {"kind": "int", "value": 5, "min": 1, "max": 30, "step": 1, "label": "Min br"},
            },
            description="Remove short spurs / unstable branches from skeleton.",
            display_key="skeleton",
        )

    if extract_minutiae_crossing_number is not None:
        ops["minutiae"] = OperationSpec(
            key="minutiae",
            name="Minutiae Extraction",
            func=extract_minutiae_crossing_number,
            input_map={"skeleton": "skeleton", "mask": "mask"},
            output_keys=["minutiae_map", "minutiae_list"],
            params_spec={
                "border_margin": {"kind": "int", "value": 10, "min": 0, "max": 50, "step": 1, "label": "Margin"},
            },
            description="Detect ridge endings / bifurcations from skeleton.",
            display_key="minutiae_map",
        )

    return ops


AVAILABLE_OPERATIONS = _available_operation_specs()


# ============================================================
# PRESET DEFINITIONS
# ============================================================

SCAN_STUDENT_ORDER = [
    "foreground_mask",
    "apply_mask",
    "clahe",
    "binarize",
    "mask_binary",
    "morphology",
    "skeleton",
    "prune_skeleton",
    "minutiae",
]

SCAN_STUDENT_ENABLED = {
    "foreground_mask": True,
    "apply_mask": True,
    "clahe": True,
    "binarize": True,
    "mask_binary": True,
    "morphology": False,
    "skeleton": True,
    "prune_skeleton": False,
    "minutiae": True,
}

PHOTO_HOME_ORDER = [
    "mask_black_bg",
    "apply_mask",
    "stretch",
    "gamma",
    "clahe",
    "local_contrast",
    "dog",
    "fft_bandpass",
    "ced",
    "ridge_filter",
    "dynamic_gabor",
    "binarize",
    "mask_binary",
    "morphology",
    "skeleton",
    "prune_skeleton",
    "minutiae",
]

PHOTO_HOME_ENABLED = {
    "mask_black_bg": True,
    "apply_mask": True,
    "stretch": False,
    "gamma": False,
    "clahe": True,
    "local_contrast": False,
    "dog": False,
    "fft_bandpass": False,
    "ced": False,
    "ridge_filter": False,
    "dynamic_gabor": False,
    "binarize": True,
    "mask_binary": True,
    "morphology": True,
    "skeleton": False,
    "prune_skeleton": False,
    "minutiae": False,
}


# ============================================================
# BUILDERS
# ============================================================

def available_operation_keys() -> List[str]:
    return sorted(AVAILABLE_OPERATIONS.keys())


def build_pipeline_steps(
    order: List[str],
    enabled_map: Optional[Dict[str, bool]] = None,
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> List[Step]:
    enabled_map = enabled_map or {}
    per_step_param_overrides = per_step_param_overrides or {}

    steps: List[Step] = []
    for key in order:
        spec = AVAILABLE_OPERATIONS.get(key)
        if spec is None:
            continue
        enabled = enabled_map.get(key, True)
        step = _make_step(
            spec,
            enabled=enabled,
            override_params=per_step_param_overrides.get(key),
        )
        steps.append(step)
    return steps


def make_scan_student_pipeline(
    image: np.ndarray,
    *,
    preset_name: str = "scan_student",
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=SCAN_STUDENT_ORDER,
        enabled_map=SCAN_STUDENT_ENABLED,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)


def make_photo_home_pipeline(
    image: np.ndarray,
    *,
    preset_name: str = "photo_home",
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=PHOTO_HOME_ORDER,
        enabled_map=PHOTO_HOME_ENABLED,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)


# ============================================================
# GENERIC CUSTOM PRESET
# ============================================================

def make_custom_pipeline(
    image: np.ndarray,
    order: List[str],
    enabled_map: Optional[Dict[str, bool]] = None,
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    *,
    preset_name: str = "custom",
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=order,
        enabled_map=enabled_map,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)
