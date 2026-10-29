"""
Sen1Floods11 Core Toolkit & Deep Learning Flood Detection Package
"""

from .models import build_fcn_resnet50, convert_bn_to_gn
from .preprocessing import preprocess_s1, preprocess_s2
from .inference import (
    predict_otsu,
    predict_vh_threshold,
    predict_optical_ndwi,
    predict_fcn
)
from .metrics import evaluate_segmentation, compute_flood_area
from .visualizer import (
    create_s1_composite,
    create_s2_composite,
    render_mask_overlay,
    render_error_map,
    array_to_base64_png
)

__all__ = [
    "build_fcn_resnet50",
    "convert_bn_to_gn",
    "preprocess_s1",
    "preprocess_s2",
    "predict_otsu",
    "predict_vh_threshold",
    "predict_optical_ndwi",
    "predict_fcn",
    "evaluate_segmentation",
    "compute_flood_area",
    "create_s1_composite",
    "create_s2_composite",
    "render_mask_overlay",
    "render_error_map",
    "array_to_base64_png",
]
