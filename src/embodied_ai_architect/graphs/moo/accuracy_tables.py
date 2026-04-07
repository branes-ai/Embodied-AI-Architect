"""Model accuracy lookup tables and quantization impact curves.

Provides published/measured accuracy data for common model families used in
embodied AI pipelines. Enables the MOO evaluator to estimate accuracy without
running inference, closing the loop from hardware design → capability per watt.

Data sources: published papers, MLPerf Edge, torchvision model zoo, Ultralytics docs.

Usage:
    from embodied_ai_architect.graphs.moo.accuracy_tables import (
        get_model_accuracy,
        get_quantization_impact,
        estimate_effective_accuracy,
        list_model_families,
    )

    acc = get_model_accuracy("yolov8", "s")            # 44.9 mAP@50
    impact = get_quantization_impact("int8")            # ~1.5% drop
    effective = estimate_effective_accuracy("yolov8", "s", "int8")  # ~43.4
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ModelVariantSpec(BaseModel):
    """Accuracy and compute characteristics for a single model variant."""

    family: str = Field(..., description="Model family (e.g. 'yolov8')")
    variant: str = Field(..., description="Variant name (e.g. 'n', 's', 'm', 'l', 'x')")
    task: str = Field(..., description="Task type: detection, classification, segmentation")
    primary_metric: str = Field(
        ..., description="Primary accuracy metric name (e.g. 'mAP50', 'top1')"
    )
    accuracy: float = Field(..., description="Primary accuracy value (0-100 scale)")
    params_m: float = Field(..., description="Parameter count in millions")
    gflops: float = Field(..., description="Compute cost in GFLOPS")
    input_resolution: int = Field(default=640, description="Default input resolution (pixels)")
    secondary_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional metrics (e.g. mAP50-95, top5)",
    )


class QuantizationImpact(BaseModel):
    """Accuracy and performance impact of a quantization scheme."""

    dtype: str = Field(..., description="Data type (e.g. 'fp16', 'int8', 'int4')")
    accuracy_drop_pct: float = Field(..., description="Typical accuracy drop in percentage points")
    accuracy_drop_range: tuple[float, float] = Field(
        ..., description="(min, max) accuracy drop range across model families"
    )
    speedup_factor: float = Field(..., description="Typical inference speedup vs FP32")
    memory_reduction: float = Field(..., description="Memory reduction factor vs FP32")
    notes: str = Field(default="", description="Additional notes about this scheme")


# ---------------------------------------------------------------------------
# Model accuracy tables — published/measured data
# ---------------------------------------------------------------------------

# Detection models (COCO val2017, mAP@50)
_DETECTION_MODELS: list[dict[str, Any]] = [
    # YOLOv8 family (Ultralytics, 640px input)
    {
        "family": "yolov8",
        "variant": "n",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 37.3,
        "params_m": 3.2,
        "gflops": 8.7,
        "secondary_metrics": {"mAP50_95": 18.4},
    },
    {
        "family": "yolov8",
        "variant": "s",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 44.9,
        "params_m": 11.2,
        "gflops": 28.6,
        "secondary_metrics": {"mAP50_95": 27.7},
    },
    {
        "family": "yolov8",
        "variant": "m",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 50.2,
        "params_m": 25.9,
        "gflops": 78.9,
        "secondary_metrics": {"mAP50_95": 33.3},
    },
    {
        "family": "yolov8",
        "variant": "l",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 52.9,
        "params_m": 43.7,
        "gflops": 165.2,
        "secondary_metrics": {"mAP50_95": 35.5},
    },
    {
        "family": "yolov8",
        "variant": "x",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 53.9,
        "params_m": 68.2,
        "gflops": 257.8,
        "secondary_metrics": {"mAP50_95": 36.5},
    },
    # YOLOv5 family (Ultralytics, 640px input)
    {
        "family": "yolov5",
        "variant": "n",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 45.7,
        "params_m": 1.9,
        "gflops": 4.5,
        "secondary_metrics": {"mAP50_95": 28.0},
    },
    {
        "family": "yolov5",
        "variant": "s",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 56.8,
        "params_m": 7.2,
        "gflops": 16.5,
        "secondary_metrics": {"mAP50_95": 37.4},
    },
    {
        "family": "yolov5",
        "variant": "m",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 64.1,
        "params_m": 21.2,
        "gflops": 49.0,
        "secondary_metrics": {"mAP50_95": 45.4},
    },
    {
        "family": "yolov5",
        "variant": "l",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 67.3,
        "params_m": 46.5,
        "gflops": 109.1,
        "secondary_metrics": {"mAP50_95": 49.0},
    },
    # SSD MobileNet (300px input)
    {
        "family": "ssd_mobilenet",
        "variant": "v2",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 22.1,
        "params_m": 4.3,
        "gflops": 0.8,
        "input_resolution": 300,
        "secondary_metrics": {"mAP50_95": 13.7},
    },
    # EfficientDet
    {
        "family": "efficientdet",
        "variant": "d0",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 55.2,
        "params_m": 3.9,
        "gflops": 2.5,
        "input_resolution": 512,
        "secondary_metrics": {"mAP50_95": 33.8},
    },
    {
        "family": "efficientdet",
        "variant": "d1",
        "task": "detection",
        "primary_metric": "mAP50",
        "accuracy": 59.1,
        "params_m": 6.6,
        "gflops": 6.1,
        "input_resolution": 640,
        "secondary_metrics": {"mAP50_95": 39.6},
    },
]

# Classification models (ImageNet top-1 accuracy)
_CLASSIFICATION_MODELS: list[dict[str, Any]] = [
    # ResNet family (224px input)
    {
        "family": "resnet",
        "variant": "18",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 69.8,
        "params_m": 11.7,
        "gflops": 1.8,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 89.1},
    },
    {
        "family": "resnet",
        "variant": "34",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 73.3,
        "params_m": 21.8,
        "gflops": 3.7,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 91.4},
    },
    {
        "family": "resnet",
        "variant": "50",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 76.1,
        "params_m": 25.6,
        "gflops": 4.1,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 92.9},
    },
    {
        "family": "resnet",
        "variant": "101",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 77.4,
        "params_m": 44.5,
        "gflops": 7.8,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 93.5},
    },
    # MobileNet family (224px input)
    {
        "family": "mobilenet",
        "variant": "v2",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 71.9,
        "params_m": 3.4,
        "gflops": 0.3,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 90.3},
    },
    {
        "family": "mobilenet",
        "variant": "v3_small",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 67.7,
        "params_m": 2.5,
        "gflops": 0.06,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 87.4},
    },
    {
        "family": "mobilenet",
        "variant": "v3_large",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 74.0,
        "params_m": 5.4,
        "gflops": 0.22,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 91.3},
    },
    # EfficientNet family (variable input)
    {
        "family": "efficientnet",
        "variant": "b0",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 77.1,
        "params_m": 5.3,
        "gflops": 0.4,
        "input_resolution": 224,
        "secondary_metrics": {"top5": 93.3},
    },
    {
        "family": "efficientnet",
        "variant": "b1",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 78.8,
        "params_m": 7.8,
        "gflops": 0.7,
        "input_resolution": 240,
        "secondary_metrics": {"top5": 94.4},
    },
    {
        "family": "efficientnet",
        "variant": "b2",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 80.1,
        "params_m": 9.2,
        "gflops": 1.0,
        "input_resolution": 260,
        "secondary_metrics": {"top5": 94.9},
    },
    {
        "family": "efficientnet",
        "variant": "b3",
        "task": "classification",
        "primary_metric": "top1",
        "accuracy": 81.6,
        "params_m": 12.0,
        "gflops": 1.8,
        "input_resolution": 300,
        "secondary_metrics": {"top5": 95.7},
    },
]

# Segmentation models (COCO val2017 mIoU for panoptic, ADE20K for semantic)
_SEGMENTATION_MODELS: list[dict[str, Any]] = [
    {
        "family": "deeplabv3",
        "variant": "resnet50",
        "task": "segmentation",
        "primary_metric": "mIoU",
        "accuracy": 73.5,
        "params_m": 39.6,
        "gflops": 178.7,
        "input_resolution": 520,
    },
    {
        "family": "deeplabv3",
        "variant": "mobilenet_v3",
        "task": "segmentation",
        "primary_metric": "mIoU",
        "accuracy": 60.3,
        "params_m": 11.0,
        "gflops": 10.5,
        "input_resolution": 520,
    },
    {
        "family": "fcn",
        "variant": "resnet50",
        "task": "segmentation",
        "primary_metric": "mIoU",
        "accuracy": 71.6,
        "params_m": 35.3,
        "gflops": 152.0,
        "input_resolution": 520,
    },
]

# Combined registry
_ALL_MODELS: list[dict[str, Any]] = (
    _DETECTION_MODELS + _CLASSIFICATION_MODELS + _SEGMENTATION_MODELS
)

# ---------------------------------------------------------------------------
# Quantization impact data
# ---------------------------------------------------------------------------

_QUANTIZATION_IMPACTS: dict[str, dict[str, Any]] = {
    "fp32": {
        "dtype": "fp32",
        "accuracy_drop_pct": 0.0,
        "accuracy_drop_range": (0.0, 0.0),
        "speedup_factor": 1.0,
        "memory_reduction": 1.0,
        "notes": "Baseline full precision",
    },
    "fp16": {
        "dtype": "fp16",
        "accuracy_drop_pct": 0.1,
        "accuracy_drop_range": (0.0, 0.3),
        "speedup_factor": 1.8,
        "memory_reduction": 2.0,
        "notes": "Near-lossless on most architectures",
    },
    "bf16": {
        "dtype": "bf16",
        "accuracy_drop_pct": 0.15,
        "accuracy_drop_range": (0.0, 0.5),
        "speedup_factor": 1.7,
        "memory_reduction": 2.0,
        "notes": "Better training stability than FP16, slightly less inference precision",
    },
    "int8": {
        "dtype": "int8",
        "accuracy_drop_pct": 1.5,
        "accuracy_drop_range": (0.3, 4.0),
        "speedup_factor": 3.2,
        "memory_reduction": 4.0,
        "notes": "Post-training quantization; QAT can reduce drop to <0.5%",
    },
    "int4": {
        "dtype": "int4",
        "accuracy_drop_pct": 5.0,
        "accuracy_drop_range": (2.0, 12.0),
        "speedup_factor": 5.5,
        "memory_reduction": 8.0,
        "notes": "Aggressive quantization; best with QAT and mixed-precision",
    },
    "mixed_int8_fp16": {
        "dtype": "mixed_int8_fp16",
        "accuracy_drop_pct": 0.8,
        "accuracy_drop_range": (0.2, 2.0),
        "speedup_factor": 2.5,
        "memory_reduction": 3.0,
        "notes": "INT8 for compute-heavy layers, FP16 for sensitive layers",
    },
}

# Build parsed model specs at import time
_MODEL_SPECS: list[ModelVariantSpec] = []
for _m in _ALL_MODELS:
    _MODEL_SPECS.append(
        ModelVariantSpec(
            family=_m["family"],
            variant=_m["variant"],
            task=_m["task"],
            primary_metric=_m["primary_metric"],
            accuracy=_m["accuracy"],
            params_m=_m["params_m"],
            gflops=_m["gflops"],
            input_resolution=_m.get("input_resolution", 640),
            secondary_metrics=_m.get("secondary_metrics", {}),
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_model_families() -> list[str]:
    """List all available model families."""
    return sorted({m.family for m in _MODEL_SPECS})


def list_variants(family: str) -> list[str]:
    """List available variants for a model family."""
    return [m.variant for m in _MODEL_SPECS if m.family == family]


def get_model_spec(family: str, variant: str) -> Optional[ModelVariantSpec]:
    """Get full spec for a specific model variant.

    Args:
        family: Model family name (e.g. 'yolov8', 'resnet', 'mobilenet').
        variant: Variant name (e.g. 'n', 's', '50').

    Returns:
        ModelVariantSpec or None if not found.
    """
    for m in _MODEL_SPECS:
        if m.family == family and m.variant == variant:
            return m
    return None


def get_model_accuracy(family: str, variant: str) -> Optional[float]:
    """Get the primary accuracy metric for a model variant.

    Returns accuracy on a 0-100 scale (e.g. 44.9 for YOLOv8s mAP@50).
    """
    spec = get_model_spec(family, variant)
    return spec.accuracy if spec else None


def get_model_gflops(family: str, variant: str) -> Optional[float]:
    """Get the compute cost in GFLOPS for a model variant."""
    spec = get_model_spec(family, variant)
    return spec.gflops if spec else None


def get_quantization_impact(dtype: str) -> Optional[QuantizationImpact]:
    """Get accuracy and performance impact for a quantization scheme.

    Args:
        dtype: Data type string ('fp32', 'fp16', 'bf16', 'int8', 'int4',
               'mixed_int8_fp16').

    Returns:
        QuantizationImpact or None if dtype not recognized.
    """
    if not dtype:
        return None
    data = _QUANTIZATION_IMPACTS.get(dtype.lower())
    if data is None:
        return None
    return QuantizationImpact(**data)


def estimate_effective_accuracy(
    family: str,
    variant: str,
    dtype: str = "fp32",
    pruning_ratio: float = 0.0,
) -> Optional[float]:
    """Estimate effective accuracy after quantization and pruning.

    Applies quantization degradation and pruning degradation curves to the
    baseline FP32 accuracy.

    Args:
        family: Model family name.
        variant: Variant name.
        dtype: Quantization data type.
        pruning_ratio: Fraction of weights pruned (0.0 to 0.8).

    Returns:
        Estimated accuracy (0-100 scale) or None if model not found.
    """
    base_accuracy = get_model_accuracy(family, variant)
    if base_accuracy is None:
        return None

    # Apply quantization drop
    quant_impact = get_quantization_impact(dtype)
    quant_drop = quant_impact.accuracy_drop_pct if quant_impact else 0.0

    # Apply pruning degradation (empirical: roughly quadratic in pruning ratio)
    # At 50% pruning, ~2% accuracy drop; at 80% pruning, ~8% drop
    pruning_drop = 0.0
    if pruning_ratio > 0:
        pruning_drop = 20.0 * pruning_ratio**2

    effective = max(0.0, base_accuracy - quant_drop - pruning_drop)
    return round(effective, 2)


def estimate_effective_gflops(
    family: str,
    variant: str,
    pruning_ratio: float = 0.0,
) -> Optional[float]:
    """Estimate effective GFLOPS after pruning.

    Structured pruning reduces FLOPS roughly proportionally to pruning ratio.

    Args:
        family: Model family name.
        variant: Variant name.
        pruning_ratio: Fraction of weights pruned (0.0 to 0.8).

    Returns:
        Estimated GFLOPS or None if model not found.
    """
    base_gflops = get_model_gflops(family, variant)
    if base_gflops is None:
        return None

    # Structured pruning reduces FLOPS roughly proportionally
    effective = base_gflops * (1.0 - pruning_ratio * 0.85)
    return round(effective, 2)


def get_models_for_task(task: str) -> list[ModelVariantSpec]:
    """Get all model variants for a given task type.

    Args:
        task: Task type ('detection', 'classification', 'segmentation').

    Returns:
        List of ModelVariantSpec sorted by accuracy (descending).
    """
    models = [m for m in _MODEL_SPECS if m.task == task]
    return sorted(models, key=lambda m: m.accuracy, reverse=True)


def get_accuracy_gflops_frontier(task: str) -> list[ModelVariantSpec]:
    """Get the Pareto frontier of accuracy vs GFLOPS for a task.

    Returns models where no other model is both more accurate AND less
    compute-intensive.

    Args:
        task: Task type.

    Returns:
        Pareto-optimal models sorted by GFLOPS (ascending).
    """
    models = get_models_for_task(task)
    if not models:
        return []

    # Sort by GFLOPS ascending
    models.sort(key=lambda m: m.gflops)

    # Compute Pareto front (maximize accuracy, minimize GFLOPS)
    frontier: list[ModelVariantSpec] = []
    best_accuracy = -1.0
    for m in models:
        if m.accuracy > best_accuracy:
            frontier.append(m)
            best_accuracy = m.accuracy

    return frontier


def lookup_accuracy_for_workload(
    workload_type: str,
    estimated_gflops: float,
    dtype: str = "fp32",
) -> dict[str, Any]:
    """Estimate accuracy for a workload based on its compute budget.

    Maps a workload type and GFLOPS budget to the best-fit model variant,
    returning the estimated accuracy. Used by the evaluator when the workload
    profile doesn't specify an explicit model.

    Args:
        workload_type: Workload type string ('ml_inference', 'detection',
                      'classification', 'segmentation').
        estimated_gflops: Compute budget in GFLOPS.
        dtype: Quantization data type for accuracy adjustment.

    Returns:
        Dict with 'accuracy', 'model_family', 'model_variant', 'gflops',
        'primary_metric', 'dtype'.
    """
    # Map workload types to task types
    task_map = {
        "ml_inference": "detection",
        "detection": "detection",
        "object_detection": "detection",
        "classification": "classification",
        "image_classification": "classification",
        "segmentation": "segmentation",
        "semantic_segmentation": "segmentation",
    }
    task = task_map.get(workload_type, "detection")

    models = get_models_for_task(task)
    if not models:
        return {
            "accuracy": 0.0,
            "model_family": "unknown",
            "model_variant": "unknown",
            "gflops": estimated_gflops,
            "primary_metric": "unknown",
            "dtype": dtype,
        }

    # Find the best model that fits within the GFLOPS budget
    best_fit: Optional[ModelVariantSpec] = None
    for m in sorted(models, key=lambda m: m.accuracy, reverse=True):
        if m.gflops <= estimated_gflops:
            best_fit = m
            break

    # If nothing fits, use the smallest model
    if best_fit is None:
        best_fit = min(models, key=lambda m: m.gflops)

    effective_acc = estimate_effective_accuracy(best_fit.family, best_fit.variant, dtype)

    return {
        "accuracy": effective_acc or best_fit.accuracy,
        "model_family": best_fit.family,
        "model_variant": best_fit.variant,
        "gflops": best_fit.gflops,
        "primary_metric": best_fit.primary_metric,
        "dtype": dtype,
    }
