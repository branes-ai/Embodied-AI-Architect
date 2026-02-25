"""ONNX Model Zoo provider.

Provides access to pre-trained, validated ONNX models from the official
ONNX Model Zoo (https://github.com/onnx/models).

Models are already in ONNX format - no conversion needed.
"""

import tarfile
import tempfile
import urllib.request
import shutil
from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# ONNX Model Zoo catalog
# Source: https://github.com/onnx/models
# Models are hosted on GitHub releases or external URLs
ONNX_ZOO_MODELS = {
    # ==========================================================================
    # Image Classification
    # ==========================================================================
    "resnet50-v1": {
        "name": "ResNet-50 v1",
        "task": "classification",
        "parameters": 25_600_000,
        "top1": 0.7490,
        "top5": 0.9216,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "description": "ResNet-50 v1 trained on ImageNet",
    },
    "resnet50-v2": {
        "name": "ResNet-50 v2",
        "task": "classification",
        "parameters": 25_600_000,
        "top1": 0.7560,
        "top5": 0.9290,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
        "description": "ResNet-50 v2 with pre-activation",
    },
    "resnet18-v1": {
        "name": "ResNet-18 v1",
        "task": "classification",
        "parameters": 11_700_000,
        "top1": 0.6970,
        "top5": 0.8930,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
        "description": "ResNet-18 v1 trained on ImageNet",
    },
    "resnet101-v1": {
        "name": "ResNet-101 v1",
        "task": "classification",
        "parameters": 44_500_000,
        "top1": 0.7730,
        "top5": 0.9360,
        "input_shape": (1, 3, 224, 224),
        "family": "resnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet101-v1-7.onnx",
        "description": "ResNet-101 v1 trained on ImageNet",
    },
    "mobilenetv2-7": {
        "name": "MobileNet v2",
        "task": "classification",
        "parameters": 3_500_000,
        "top1": 0.7140,
        "top5": 0.9010,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "description": "MobileNet v2 optimized for mobile",
    },
    "squeezenet1.1": {
        "name": "SqueezeNet 1.1",
        "task": "classification",
        "parameters": 1_200_000,
        "top1": 0.5750,
        "top5": 0.8030,
        "input_shape": (1, 3, 224, 224),
        "family": "squeezenet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "description": "SqueezeNet 1.1 - compact model",
    },
    "shufflenet-v2-10": {
        "name": "ShuffleNet v2",
        "task": "classification",
        "parameters": 2_300_000,
        "top1": 0.6920,
        "top5": 0.8850,
        "input_shape": (1, 3, 224, 224),
        "family": "shufflenet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
        "description": "ShuffleNet v2 for efficient inference",
    },
    "efficientnet-lite4": {
        "name": "EfficientNet Lite4",
        "task": "classification",
        "parameters": 13_000_000,
        "top1": 0.8070,
        "top5": 0.9520,
        "input_shape": (1, 3, 300, 300),
        "family": "efficientnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "description": "EfficientNet Lite4 optimized for edge",
    },
    "vgg16": {
        "name": "VGG-16",
        "task": "classification",
        "parameters": 138_000_000,
        "top1": 0.7150,
        "top5": 0.9040,
        "input_shape": (1, 3, 224, 224),
        "family": "vgg",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.onnx",
        "description": "VGG-16 trained on ImageNet",
    },
    "vgg19": {
        "name": "VGG-19",
        "task": "classification",
        "parameters": 143_000_000,
        "top1": 0.7250,
        "top5": 0.9100,
        "input_shape": (1, 3, 224, 224),
        "family": "vgg",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-7.onnx",
        "description": "VGG-19 trained on ImageNet",
    },
    "googlenet-12": {
        "name": "GoogLeNet",
        "task": "classification",
        "parameters": 6_600_000,
        "top1": 0.6890,
        "top5": 0.8890,
        "input_shape": (1, 3, 224, 224),
        "family": "inception",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx",
        "description": "GoogLeNet/Inception v1",
    },
    "caffenet-12": {
        "name": "CaffeNet",
        "task": "classification",
        "parameters": 60_000_000,
        "top1": 0.5710,
        "top5": 0.8020,
        "input_shape": (1, 3, 224, 224),
        "family": "alexnet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/alexnet/model/caffenet-12.onnx",
        "description": "CaffeNet (AlexNet variant)",
    },
    # ==========================================================================
    # Object Detection
    # ==========================================================================
    "yolov3": {
        "name": "YOLOv3",
        "task": "detection",
        "parameters": 62_000_000,
        "map50": 0.5530,
        "input_shape": (1, 3, 416, 416),
        "family": "yolo",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
        "description": "YOLOv3 object detector",
    },
    "yolov4": {
        "name": "YOLOv4",
        "task": "detection",
        "parameters": 64_000_000,
        "map50": 0.6550,
        "input_shape": (1, 3, 416, 416),
        "family": "yolo",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
        "description": "YOLOv4 object detector",
    },
    "tiny-yolov3": {
        "name": "Tiny YOLOv3",
        "task": "detection",
        "parameters": 8_700_000,
        "map50": 0.3310,
        "input_shape": (1, 3, 416, 416),
        "family": "yolo",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
        "description": "Tiny YOLOv3 for edge devices",
    },
    "ssd-mobilenetv1": {
        "name": "SSD MobileNet v1",
        "task": "detection",
        "parameters": 6_800_000,
        "map50": 0.2300,
        "input_shape": (1, 3, 300, 300),
        "family": "ssd",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
        "description": "SSD with MobileNet v1 backbone",
    },
    "faster-rcnn-r50-fpn": {
        "name": "Faster R-CNN R50-FPN",
        "task": "detection",
        "parameters": 41_000_000,
        "map50": 0.3690,
        "input_shape": (1, 3, 800, 800),
        "family": "rcnn",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx",
        "description": "Faster R-CNN with ResNet-50 FPN",
    },
    "retinanet-r101": {
        "name": "RetinaNet R101",
        "task": "detection",
        "parameters": 56_000_000,
        "map50": 0.3760,
        "input_shape": (1, 3, 480, 640),
        "family": "retinanet",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx",
        "description": "RetinaNet with ResNet-101 backbone",
    },
    # ==========================================================================
    # Semantic Segmentation
    # ==========================================================================
    "fcn-resnet50": {
        "name": "FCN ResNet-50",
        "task": "segmentation",
        "parameters": 35_000_000,
        "miou": 0.6030,
        "input_shape": (1, 3, 520, 520),
        "family": "fcn",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx",
        "description": "FCN with ResNet-50 for semantic segmentation",
    },
    "fcn-resnet101": {
        "name": "FCN ResNet-101",
        "task": "segmentation",
        "parameters": 54_000_000,
        "miou": 0.6330,
        "input_shape": (1, 3, 520, 520),
        "family": "fcn",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.onnx",
        "description": "FCN with ResNet-101 for semantic segmentation",
    },
    # ==========================================================================
    # Face Detection & Recognition
    # ==========================================================================
    "ultraface-320": {
        "name": "Ultra-Light-Fast Face 320",
        "task": "face_detection",
        "parameters": 1_200_000,
        "input_shape": (1, 3, 240, 320),
        "family": "ultraface",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
        "description": "Ultra-lightweight face detector 320px",
    },
    "ultraface-640": {
        "name": "Ultra-Light-Fast Face 640",
        "task": "face_detection",
        "parameters": 1_200_000,
        "input_shape": (1, 3, 480, 640),
        "family": "ultraface",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-640.onnx",
        "description": "Ultra-lightweight face detector 640px",
    },
    "arcface": {
        "name": "ArcFace",
        "task": "face_recognition",
        "parameters": 24_000_000,
        "input_shape": (1, 3, 112, 112),
        "family": "arcface",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        "description": "ArcFace for face recognition/embedding",
    },
    # ==========================================================================
    # Pose Estimation
    # ==========================================================================
    "hrnet-pose": {
        "name": "HRNet Pose",
        "task": "pose",
        "parameters": 28_500_000,
        "input_shape": (1, 3, 384, 288),
        "family": "hrnet",
        "url": "https://github.com/onnx/models/raw/main/Community/cv/Pose/HRNet/pose_hrnet_w32.onnx",
        "description": "HRNet for human pose estimation",
    },
    # ==========================================================================
    # Super Resolution
    # ==========================================================================
    "super-resolution-10": {
        "name": "Super Resolution",
        "task": "super_resolution",
        "parameters": 37_000,
        "input_shape": (1, 1, 224, 224),
        "family": "espcn",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
        "description": "ESPCN 3x super resolution",
    },
    # ==========================================================================
    # Image Processing
    # ==========================================================================
    "fast-neural-style-mosaic": {
        "name": "Neural Style Mosaic",
        "task": "style_transfer",
        "parameters": 1_700_000,
        "input_shape": (1, 3, 224, 224),
        "family": "style_transfer",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "description": "Fast neural style transfer - mosaic",
    },
    "fast-neural-style-candy": {
        "name": "Neural Style Candy",
        "task": "style_transfer",
        "parameters": 1_700_000,
        "input_shape": (1, 3, 224, 224),
        "family": "style_transfer",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        "description": "Fast neural style transfer - candy",
    },
    "fast-neural-style-pointilism": {
        "name": "Neural Style Pointilism",
        "task": "style_transfer",
        "parameters": 1_700_000,
        "input_shape": (1, 3, 224, 224),
        "family": "style_transfer",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        "description": "Fast neural style transfer - pointilism",
    },
    # ==========================================================================
    # Depth Estimation
    # ==========================================================================
    "midas-v2": {
        "name": "MiDaS v2",
        "task": "depth_estimation",
        "parameters": 104_000_000,
        "input_shape": (1, 3, 384, 384),
        "family": "midas",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx",
        "description": "MiDaS v2 monocular depth estimation",
    },
    "midas-v2-small": {
        "name": "MiDaS v2 Small",
        "task": "depth_estimation",
        "parameters": 21_000_000,
        "input_shape": (1, 3, 256, 256),
        "family": "midas",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "description": "MiDaS v2 small for faster inference",
    },
}


class ONNXModelZooProvider(ModelProvider):
    """Provider for ONNX Model Zoo.

    Provides access to pre-trained, validated ONNX models from the official
    ONNX Model Zoo. Models are already in ONNX format - no conversion needed.

    Source: https://github.com/onnx/models
    """

    @property
    def name(self) -> str:
        return "onnx-zoo"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        # ONNX Zoo models are already in ONNX format
        return [ModelFormat.ONNX]

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available ONNX Model Zoo models."""
        models = []
        for model_id, info in ONNX_ZOO_MODELS.items():
            model_info = {
                "id": model_id,
                "provider": self.name,
                "benchmarked": True,
                "accuracy": info.get("top1") or info.get("map50") or info.get("miou"),
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
        """Download an ONNX model from the zoo.

        Args:
            model_id: Model identifier (e.g., 'resnet50-v1', 'yolov4')
            format: Must be ONNX (only supported format)
            cache_dir: Directory to store the model

        Returns:
            ModelArtifact with path and metadata
        """
        if format != ModelFormat.ONNX:
            raise ValueError(
                f"ONNX Model Zoo only provides ONNX format. " f"Requested: {format.value}"
            )

        # Get model info
        info = ONNX_ZOO_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in ONNX Model Zoo")

        # Create filename
        model_filename = f"{model_id}.onnx"
        model_path = cache_dir / model_filename

        # Check if already cached
        if model_path.exists():
            return self._create_artifact(model_id, format, model_path, info)

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download from URL
        url = info["url"]
        print(f"[ONNX Zoo] Downloading {model_id} from {url}...")

        try:
            # Download with progress
            self._download_file(url, model_path)
        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

        if not model_path.exists():
            raise RuntimeError(f"Failed to download model to {model_path}")

        print(f"[ONNX Zoo] Saved to {model_path}")
        return self._create_artifact(model_id, format, model_path, info)

    def _download_file(self, url: str, dest_path: Path) -> None:
        """Download a file with progress indication."""
        # Handle tar.gz files (some models are compressed)
        if url.endswith(".tar.gz"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tar_path = Path(tmpdir) / "model.tar.gz"

                # Download tar file
                urllib.request.urlretrieve(url, tar_path)

                # Extract
                with tarfile.open(tar_path, "r:gz") as tar:
                    # Find the .onnx file in the archive
                    for member in tar.getmembers():
                        if member.name.endswith(".onnx"):
                            # Extract to temp and move
                            tar.extract(member, tmpdir)
                            extracted = Path(tmpdir) / member.name
                            shutil.move(str(extracted), str(dest_path))
                            return

                raise RuntimeError("No .onnx file found in archive")
        else:
            # Direct download
            urllib.request.urlretrieve(url, dest_path)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = ONNX_ZOO_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            "accuracy": info.get("top1") or info.get("map50") or info.get("miou"),
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
        size = path.stat().st_size

        return ModelArtifact(
            model_id=model_id,
            provider=self.name,
            format=format,
            path=path,
            size_bytes=size,
            name=info.get("name", model_id),
            version=None,
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("top1") or info.get("map50") or info.get("miou"),
            metadata=info,
        )
