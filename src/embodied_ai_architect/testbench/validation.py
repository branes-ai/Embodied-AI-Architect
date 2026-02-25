"""Model validation against ground truth datasets.

Provides tools for validating model accuracy against standard
datasets and custom validation sets.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .metrics import (
    MetricResult,
    MetricType,
    ValidationResult,
    compute_accuracy,
    compute_map,
    compute_miou,
    compute_top_k_accuracy,
)

console = Console()


class ModelValidator:
    """Validates model accuracy against ground truth.

    Supports detection, classification, and segmentation tasks
    with configurable metrics and thresholds.
    """

    def __init__(
        self,
        model_id: str,
        task: str = "detection",
        thresholds: Optional[dict[MetricType, float]] = None,
    ):
        """Initialize validator.

        Args:
            model_id: Model identifier
            task: Task type (detection, classification, segmentation)
            thresholds: Metric thresholds for pass/fail
        """
        self.model_id = model_id
        self.task = task
        self.thresholds = thresholds or {}

        # Default thresholds by task
        if not thresholds:
            if task == "detection":
                self.thresholds = {MetricType.MAP50: 0.5}
            elif task == "classification":
                self.thresholds = {MetricType.TOP1: 0.7}
            elif task == "segmentation":
                self.thresholds = {MetricType.MIOU: 0.5}

    def validate(
        self,
        inference_fn: Callable[[Any], Any],
        dataset: list[dict],
        dataset_name: str = "custom",
    ) -> ValidationResult:
        """Run validation on a dataset.

        Args:
            inference_fn: Function that takes input and returns predictions
            dataset: List of {input, ground_truth} dicts
            dataset_name: Name of the dataset

        Returns:
            ValidationResult with metrics
        """
        predictions = []
        ground_truths = []
        latencies = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Validating {self.model_id}...",
                total=len(dataset),
            )

            for sample in dataset:
                input_data = sample["input"]
                gt = sample["ground_truth"]

                # Time inference
                start = time.perf_counter()
                pred = inference_fn(input_data)
                latency = (time.perf_counter() - start) * 1000

                predictions.append(pred)
                ground_truths.append(gt)
                latencies.append(latency)

                progress.advance(task)

        # Compute metrics based on task
        metrics = self._compute_metrics(predictions, ground_truths, latencies)

        return ValidationResult(
            model_id=self.model_id,
            dataset=dataset_name,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            metadata={
                "task": self.task,
                "num_samples": len(dataset),
            },
        )

    def _compute_metrics(
        self,
        predictions: list,
        ground_truths: list,
        latencies: list[float],
    ) -> list[MetricResult]:
        """Compute metrics based on task type."""
        metrics = []

        if self.task == "detection":
            metrics.extend(self._compute_detection_metrics(predictions, ground_truths))
        elif self.task == "classification":
            metrics.extend(self._compute_classification_metrics(predictions, ground_truths))
        elif self.task == "segmentation":
            metrics.extend(self._compute_segmentation_metrics(predictions, ground_truths))

        # Always add latency metrics
        avg_latency = np.mean(latencies)
        metrics.append(
            MetricResult(
                metric=MetricType.LATENCY_MS,
                value=avg_latency,
            )
        )

        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0
        metrics.append(
            MetricResult(
                metric=MetricType.THROUGHPUT_FPS,
                value=throughput,
            )
        )

        return metrics

    def _compute_detection_metrics(
        self,
        predictions: list,
        ground_truths: list,
    ) -> list[MetricResult]:
        """Compute detection metrics."""
        metrics = []

        # mAP@50
        map50 = compute_map(predictions, ground_truths, iou_threshold=0.5)
        metrics.append(
            MetricResult(
                metric=MetricType.MAP50,
                value=map50,
                threshold=self.thresholds.get(MetricType.MAP50),
            )
        )

        # mAP@75
        map75 = compute_map(predictions, ground_truths, iou_threshold=0.75)
        metrics.append(
            MetricResult(
                metric=MetricType.MAP75,
                value=map75,
                threshold=self.thresholds.get(MetricType.MAP75),
            )
        )

        return metrics

    def _compute_classification_metrics(
        self,
        predictions: list,
        ground_truths: list,
    ) -> list[MetricResult]:
        """Compute classification metrics."""
        metrics = []

        # Handle both logits and class predictions
        if predictions and isinstance(predictions[0], dict):
            logits = np.array([p.get("logits", p.get("scores", [])) for p in predictions])
            pred_classes = np.argmax(logits, axis=1)
        else:
            logits = np.array(predictions)
            pred_classes = np.argmax(logits, axis=1) if logits.ndim > 1 else logits

        labels = np.array(
            [gt.get("label", gt) if isinstance(gt, dict) else gt for gt in ground_truths]
        )

        # Top-1 accuracy
        top1 = compute_accuracy(pred_classes, labels)
        metrics.append(
            MetricResult(
                metric=MetricType.TOP1,
                value=top1,
                threshold=self.thresholds.get(MetricType.TOP1),
            )
        )

        # Top-5 accuracy (if logits available)
        if logits.ndim > 1 and logits.shape[1] >= 5:
            top5 = compute_top_k_accuracy(logits, labels, k=5)
            metrics.append(
                MetricResult(
                    metric=MetricType.TOP5,
                    value=top5,
                    threshold=self.thresholds.get(MetricType.TOP5),
                )
            )

        return metrics

    def _compute_segmentation_metrics(
        self,
        predictions: list,
        ground_truths: list,
    ) -> list[MetricResult]:
        """Compute segmentation metrics."""
        metrics = []

        # Stack predictions and ground truths
        pred_masks = np.array([p.get("mask", p) if isinstance(p, dict) else p for p in predictions])
        gt_masks = np.array(
            [gt.get("mask", gt) if isinstance(gt, dict) else gt for gt in ground_truths]
        )

        # Determine number of classes
        num_classes = max(pred_masks.max(), gt_masks.max()) + 1

        # mIoU
        miou = compute_miou(pred_masks, gt_masks, num_classes)
        metrics.append(
            MetricResult(
                metric=MetricType.MIOU,
                value=miou,
                threshold=self.thresholds.get(MetricType.MIOU),
            )
        )

        # Pixel accuracy
        pixel_acc = compute_accuracy(pred_masks.flatten(), gt_masks.flatten())
        metrics.append(
            MetricResult(
                metric=MetricType.PIXEL_ACCURACY,
                value=pixel_acc,
                threshold=self.thresholds.get(MetricType.PIXEL_ACCURACY),
            )
        )

        return metrics


def validate_model(
    model_path: Path,
    dataset_path: Path,
    task: str = "detection",
    thresholds: Optional[dict] = None,
) -> ValidationResult:
    """Convenience function to validate a model.

    Args:
        model_path: Path to model file
        dataset_path: Path to validation dataset
        task: Task type
        thresholds: Metric thresholds

    Returns:
        ValidationResult
    """
    # Load model
    model = _load_model(model_path, task)

    # Load dataset
    dataset = _load_dataset(dataset_path, task)

    # Create validator
    validator = ModelValidator(
        model_id=model_path.stem,
        task=task,
        thresholds=thresholds,
    )

    # Run validation
    return validator.validate(
        inference_fn=model,
        dataset=dataset,
        dataset_name=dataset_path.stem,
    )


def _load_model(model_path: Path, task: str) -> Callable:
    """Load model for inference."""
    suffix = model_path.suffix.lower()

    if suffix == ".onnx":
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            def infer(x):
                return session.run(None, {input_name: x})[0]

            return infer
        except ImportError:
            raise RuntimeError("onnxruntime not installed")

    elif suffix in (".pt", ".pth"):
        try:
            import torch

            model = torch.load(model_path, map_location="cpu")
            model.eval()

            def infer(x):
                with torch.no_grad():
                    return model(torch.from_numpy(x)).numpy()

            return infer
        except ImportError:
            raise RuntimeError("PyTorch not installed")

    else:
        raise ValueError(f"Unsupported model format: {suffix}")


def _load_dataset(dataset_path: Path, task: str) -> list[dict]:
    """Load validation dataset."""
    import json

    if dataset_path.suffix == ".json":
        with open(dataset_path) as f:
            return json.load(f)

    elif dataset_path.is_dir():
        # Load from directory structure
        samples = []
        annotations_file = dataset_path / "annotations.json"

        if annotations_file.exists():
            with open(annotations_file) as f:
                annotations = json.load(f)

            for ann in annotations.get("samples", []):
                samples.append(
                    {
                        "input": ann.get("input"),
                        "ground_truth": ann.get("ground_truth"),
                    }
                )

        return samples

    else:
        raise ValueError(f"Cannot load dataset from: {dataset_path}")
