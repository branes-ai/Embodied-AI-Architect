"""Ultralytics provider for YOLO models.

Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11, and YOLO-World models
in PyTorch, ONNX, TensorRT, and other formats.
"""

from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# YOLO model catalog with known specifications
# Source: https://docs.ultralytics.com/models/
YOLO_MODELS = {
    # YOLOv8 Detection
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "task": "detection",
        "parameters": 3_200_000,
        "flops": 8_700_000_000,
        "map50": 0.52,
        "map50_95": 0.37,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8s": {
        "name": "YOLOv8 Small",
        "task": "detection",
        "parameters": 11_200_000,
        "flops": 28_600_000_000,
        "map50": 0.61,
        "map50_95": 0.45,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8m": {
        "name": "YOLOv8 Medium",
        "task": "detection",
        "parameters": 25_900_000,
        "flops": 78_900_000_000,
        "map50": 0.67,
        "map50_95": 0.50,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8l": {
        "name": "YOLOv8 Large",
        "task": "detection",
        "parameters": 43_700_000,
        "flops": 165_200_000_000,
        "map50": 0.69,
        "map50_95": 0.53,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8x": {
        "name": "YOLOv8 Extra-Large",
        "task": "detection",
        "parameters": 68_200_000,
        "flops": 257_800_000_000,
        "map50": 0.71,
        "map50_95": 0.54,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    # YOLOv8 Segmentation
    "yolov8n-seg": {
        "name": "YOLOv8 Nano Segmentation",
        "task": "segmentation",
        "parameters": 3_400_000,
        "flops": 12_600_000_000,
        "map50": 0.50,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8s-seg": {
        "name": "YOLOv8 Small Segmentation",
        "task": "segmentation",
        "parameters": 11_800_000,
        "flops": 42_600_000_000,
        "map50": 0.58,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    # YOLOv8 Pose
    "yolov8n-pose": {
        "name": "YOLOv8 Nano Pose",
        "task": "pose",
        "parameters": 3_300_000,
        "flops": 9_200_000_000,
        "map50": 0.65,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8s-pose": {
        "name": "YOLOv8 Small Pose",
        "task": "pose",
        "parameters": 11_600_000,
        "flops": 30_200_000_000,
        "map50": 0.72,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v8",
    },
    # YOLOv8 Classification
    "yolov8n-cls": {
        "name": "YOLOv8 Nano Classification",
        "task": "classification",
        "parameters": 2_700_000,
        "flops": 4_400_000_000,
        "top1": 0.69,
        "input_shape": (1, 3, 224, 224),
        "family": "yolo",
        "version": "v8",
    },
    "yolov8s-cls": {
        "name": "YOLOv8 Small Classification",
        "task": "classification",
        "parameters": 6_400_000,
        "flops": 13_500_000_000,
        "top1": 0.73,
        "input_shape": (1, 3, 224, 224),
        "family": "yolo",
        "version": "v8",
    },
    # YOLOv5 (legacy but still popular)
    "yolov5n": {
        "name": "YOLOv5 Nano",
        "task": "detection",
        "parameters": 1_900_000,
        "flops": 4_500_000_000,
        "map50": 0.46,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v5",
    },
    "yolov5s": {
        "name": "YOLOv5 Small",
        "task": "detection",
        "parameters": 7_200_000,
        "flops": 16_500_000_000,
        "map50": 0.56,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v5",
    },
    # YOLO11 (latest)
    "yolo11n": {
        "name": "YOLO11 Nano",
        "task": "detection",
        "parameters": 2_600_000,
        "flops": 6_500_000_000,
        "map50": 0.56,
        "map50_95": 0.39,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v11",
    },
    "yolo11s": {
        "name": "YOLO11 Small",
        "task": "detection",
        "parameters": 9_400_000,
        "flops": 21_500_000_000,
        "map50": 0.63,
        "map50_95": 0.47,
        "input_shape": (1, 3, 640, 640),
        "family": "yolo",
        "version": "v11",
    },
}


class UltralyticsProvider(ModelProvider):
    """Provider for Ultralytics YOLO models.

    Downloads and exports YOLO models using the ultralytics package.
    Supports export to ONNX, TensorRT, CoreML, and other formats.
    """

    @property
    def name(self) -> str:
        return "ultralytics"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        return [
            ModelFormat.PYTORCH,
            ModelFormat.TORCHSCRIPT,
            ModelFormat.ONNX,
            ModelFormat.TENSORRT,
            ModelFormat.COREML,
            ModelFormat.OPENVINO,
        ]

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available YOLO models."""
        models = []
        for model_id, info in YOLO_MODELS.items():
            model_info = {
                "id": model_id,
                "provider": self.name,
                "benchmarked": True,  # All have published metrics
                **info,
            }

            if query is None or query.matches(model_info):
                models.append(model_info)

        return sorted(models, key=lambda m: m.get("parameters", 0))

    def download(
        self,
        model_id: str,
        format: ModelFormat,
        cache_dir: Path,
    ) -> ModelArtifact:
        """Download and export a YOLO model.

        Args:
            model_id: YOLO model identifier (e.g., 'yolov8n', 'yolov8s-seg')
            format: Target export format
            cache_dir: Directory to store the model

        Returns:
            ModelArtifact with path and metadata
        """
        if not self.supports_format(format):
            raise ValueError(
                f"Format {format} not supported. "
                f"Supported: {[f.value for f in self.supported_formats]}"
            )

        # Get model info
        info = YOLO_MODELS.get(model_id)
        if info is None:
            # Allow arbitrary model names for custom/new models
            info = {
                "name": model_id,
                "task": "detection",
                "family": "yolo",
            }

        # Determine file extension and export format
        ext_map = {
            ModelFormat.PYTORCH: ".pt",
            ModelFormat.TORCHSCRIPT: ".torchscript",
            ModelFormat.ONNX: ".onnx",
            ModelFormat.TENSORRT: ".engine",
            ModelFormat.COREML: ".mlpackage",
            ModelFormat.OPENVINO: "_openvino_model",
        }

        ext = ext_map[format]
        model_filename = f"{model_id}{ext}"
        model_path = cache_dir / model_filename

        # Check if already cached
        if model_path.exists():
            return self._create_artifact(model_id, format, model_path, info)

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download/export using ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        # Load the model (downloads if needed)
        model = YOLO(f"{model_id}.pt")

        if format == ModelFormat.PYTORCH:
            # Just save the PyTorch model
            import shutil

            src_path = Path(f"{model_id}.pt")
            if src_path.exists():
                shutil.move(str(src_path), str(model_path))
            else:
                # Model was cached by ultralytics, find it
                from ultralytics.utils import ASSETS
                import os

                yolo_cache = Path.home() / ".cache" / "ultralytics"
                for root, dirs, files in os.walk(yolo_cache):
                    for f in files:
                        if f == f"{model_id}.pt":
                            shutil.copy(Path(root) / f, model_path)
                            break
        else:
            # Export to target format
            export_format = {
                ModelFormat.TORCHSCRIPT: "torchscript",
                ModelFormat.ONNX: "onnx",
                ModelFormat.TENSORRT: "engine",
                ModelFormat.COREML: "coreml",
                ModelFormat.OPENVINO: "openvino",
            }[format]

            export_path = model.export(format=export_format, imgsz=640, simplify=True)

            # Move to cache location
            import shutil

            exported = Path(export_path)
            if exported.is_dir():
                # OpenVINO exports a directory
                import shutil

                if model_path.exists():
                    shutil.rmtree(model_path)
                shutil.move(str(exported), str(model_path))
            else:
                shutil.move(str(exported), str(model_path))

        if not model_path.exists():
            raise RuntimeError(f"Failed to export model to {model_path}")

        return self._create_artifact(model_id, format, model_path, info)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = YOLO_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            **info,
        }

    def _create_artifact(
        self,
        model_id: str,
        format: ModelFormat,
        path: Path,
        info: dict[str, Any],
    ) -> ModelArtifact:
        """Create a ModelArtifact from download result."""
        # Get file size
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            size = path.stat().st_size

        return ModelArtifact(
            model_id=model_id,
            provider=self.name,
            format=format,
            path=path,
            size_bytes=size,
            name=info.get("name", model_id),
            version=info.get("version"),
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("map50") or info.get("top1"),
            metadata=info,
        )
