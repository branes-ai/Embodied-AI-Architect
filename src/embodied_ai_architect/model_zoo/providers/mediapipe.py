"""MediaPipe provider for Google's on-device ML models.

Provides access to lightweight, real-time ML models from MediaPipe:
- Hand detection and landmarks
- Face detection and mesh
- Pose estimation
- Object detection
- Segmentation (selfie, hair)
- Gesture recognition

Models are optimized for mobile/edge deployment with minimal latency.
Source: https://developers.google.com/mediapipe
"""

import urllib.request
from pathlib import Path
from typing import Any, Optional

from .base import ModelProvider, ModelFormat, ModelQuery, ModelArtifact

# MediaPipe model catalog
# Models hosted at storage.googleapis.com/mediapipe-models/
MEDIAPIPE_MODELS = {
    # ==========================================================================
    # Hand Detection & Landmarks
    # ==========================================================================
    "hand_landmarker": {
        "name": "Hand Landmarker",
        "task": "hand_tracking",
        "parameters": 2_000_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "description": "Detects hands and 21 3D landmarks per hand",
    },
    "palm_detection_full": {
        "name": "Palm Detection Full",
        "task": "hand_detection",
        "parameters": 1_200_000,
        "input_shape": (1, 3, 192, 192),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite",
        "description": "Full palm detection model",
    },
    "palm_detection_lite": {
        "name": "Palm Detection Lite",
        "task": "hand_detection",
        "parameters": 800_000,
        "input_shape": (1, 3, 192, 192),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite",
        "description": "Lightweight palm detection for mobile",
    },
    "hand_landmark_full": {
        "name": "Hand Landmark Full",
        "task": "hand_tracking",
        "parameters": 2_600_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite",
        "description": "Full hand landmark model - 21 3D points",
    },
    "hand_landmark_lite": {
        "name": "Hand Landmark Lite",
        "task": "hand_tracking",
        "parameters": 1_000_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite",
        "description": "Lightweight hand landmark model",
    },
    # ==========================================================================
    # Face Detection & Mesh
    # ==========================================================================
    "face_detector": {
        "name": "Face Detector",
        "task": "face_detection",
        "parameters": 800_000,
        "input_shape": (1, 3, 128, 128),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "description": "BlazeFace short-range face detector",
    },
    "face_landmarker": {
        "name": "Face Landmarker",
        "task": "face_mesh",
        "parameters": 2_500_000,
        "input_shape": (1, 3, 192, 192),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "description": "468 face landmarks with blendshapes",
    },
    "face_detection_short": {
        "name": "Face Detection Short Range",
        "task": "face_detection",
        "parameters": 800_000,
        "input_shape": (1, 3, 128, 128),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        "description": "BlazeFace for faces within 2m",
    },
    "face_detection_full": {
        "name": "Face Detection Full Range",
        "task": "face_detection",
        "parameters": 1_200_000,
        "input_shape": (1, 3, 192, 192),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite",
        "description": "BlazeFace for faces at any distance",
    },
    "face_mesh": {
        "name": "Face Mesh",
        "task": "face_mesh",
        "parameters": 2_300_000,
        "input_shape": (1, 3, 192, 192),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite",
        "description": "468 3D face landmarks",
    },
    # ==========================================================================
    # Pose Estimation
    # ==========================================================================
    "pose_landmarker_full": {
        "name": "Pose Landmarker Full",
        "task": "pose",
        "parameters": 3_500_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "description": "Full body pose with 33 landmarks",
    },
    "pose_landmarker_lite": {
        "name": "Pose Landmarker Lite",
        "task": "pose",
        "parameters": 2_000_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "description": "Lightweight pose estimation",
    },
    "pose_landmarker_heavy": {
        "name": "Pose Landmarker Heavy",
        "task": "pose",
        "parameters": 5_000_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "description": "High accuracy pose estimation",
    },
    "pose_detection": {
        "name": "Pose Detection",
        "task": "pose",
        "parameters": 1_500_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite",
        "description": "BlazePose detector",
    },
    "pose_landmark_full": {
        "name": "Pose Landmark Full",
        "task": "pose",
        "parameters": 3_000_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite",
        "description": "Full pose landmark model",
    },
    "pose_landmark_lite": {
        "name": "Pose Landmark Lite",
        "task": "pose",
        "parameters": 1_800_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite",
        "description": "Lightweight pose landmark model",
    },
    "pose_landmark_heavy": {
        "name": "Pose Landmark Heavy",
        "task": "pose",
        "parameters": 4_500_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite",
        "description": "Heavy pose landmark model",
    },
    # ==========================================================================
    # Object Detection
    # ==========================================================================
    "object_detector_efficientdet_lite0": {
        "name": "EfficientDet Lite0",
        "task": "detection",
        "parameters": 4_400_000,
        "map50": 0.258,
        "input_shape": (1, 3, 320, 320),
        "family": "efficientdet",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
        "description": "EfficientDet Lite0 - 91 COCO classes",
    },
    "object_detector_efficientdet_lite2": {
        "name": "EfficientDet Lite2",
        "task": "detection",
        "parameters": 7_000_000,
        "map50": 0.336,
        "input_shape": (1, 3, 448, 448),
        "family": "efficientdet",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite",
        "description": "EfficientDet Lite2 - higher accuracy",
    },
    "ssd_mobilenet_v2": {
        "name": "SSD MobileNet V2",
        "task": "detection",
        "parameters": 4_300_000,
        "map50": 0.220,
        "input_shape": (1, 3, 320, 320),
        "family": "ssd",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/ssdlite_object_detection.tflite",
        "description": "SSD MobileNet V2 for object detection",
    },
    # ==========================================================================
    # Image Segmentation
    # ==========================================================================
    "selfie_segmenter": {
        "name": "Selfie Segmenter",
        "task": "segmentation",
        "parameters": 200_000,
        "input_shape": (1, 3, 256, 256),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "description": "Person/background segmentation",
    },
    "selfie_segmenter_landscape": {
        "name": "Selfie Segmenter Landscape",
        "task": "segmentation",
        "parameters": 200_000,
        "input_shape": (1, 3, 144, 256),
        "family": "mediapipe",
        "format": "tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/selfie_segmentation_landscape.tflite",
        "description": "Selfie segmentation for landscape images",
    },
    "hair_segmenter": {
        "name": "Hair Segmenter",
        "task": "segmentation",
        "parameters": 500_000,
        "input_shape": (1, 3, 512, 512),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        "description": "Hair segmentation model",
    },
    "deeplab_v3": {
        "name": "DeepLab V3",
        "task": "segmentation",
        "parameters": 2_100_000,
        "input_shape": (1, 3, 257, 257),
        "family": "deeplab",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite",
        "description": "DeepLab V3 for semantic segmentation",
    },
    # ==========================================================================
    # Gesture Recognition
    # ==========================================================================
    "gesture_recognizer": {
        "name": "Gesture Recognizer",
        "task": "gesture_recognition",
        "parameters": 3_000_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mediapipe",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        "description": "Hand gesture classification",
    },
    # ==========================================================================
    # Image Classification
    # ==========================================================================
    "image_classifier_efficientnet_lite0": {
        "name": "EfficientNet Lite0 Classifier",
        "task": "classification",
        "parameters": 4_700_000,
        "top1": 0.753,
        "input_shape": (1, 3, 224, 224),
        "family": "efficientnet",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite",
        "description": "EfficientNet Lite0 ImageNet classifier",
    },
    "image_classifier_efficientnet_lite2": {
        "name": "EfficientNet Lite2 Classifier",
        "task": "classification",
        "parameters": 6_100_000,
        "top1": 0.770,
        "input_shape": (1, 3, 260, 260),
        "family": "efficientnet",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/latest/efficientnet_lite2.tflite",
        "description": "EfficientNet Lite2 ImageNet classifier",
    },
    # ==========================================================================
    # Image Embedding
    # ==========================================================================
    "image_embedder_mobilenet_v3": {
        "name": "Image Embedder MobileNet V3",
        "task": "embedding",
        "parameters": 2_900_000,
        "input_shape": (1, 3, 224, 224),
        "family": "mobilenet",
        "format": "task",
        "url": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite",
        "description": "Image feature embedding with MobileNet V3",
    },
}


class MediaPipeProvider(ModelProvider):
    """Provider for Google MediaPipe models.

    Provides access to lightweight, real-time ML models optimized for
    mobile and edge deployment. Models are available in TFLite format.

    Source: https://developers.google.com/mediapipe
    """

    @property
    def name(self) -> str:
        return "mediapipe"

    @property
    def supported_formats(self) -> list[ModelFormat]:
        # MediaPipe uses TFLite format, we'll treat it as a special case
        # Note: We return ONNX here but actually download TFLite
        # The format field in the catalog indicates the actual format
        return [ModelFormat.ONNX]  # We use ONNX enum but download TFLite

    def list_models(self, query: Optional[ModelQuery] = None) -> list[dict[str, Any]]:
        """List available MediaPipe models."""
        models = []
        for model_id, info in MEDIAPIPE_MODELS.items():
            model_info = {
                "id": model_id,
                "provider": self.name,
                "benchmarked": True,
                "accuracy": info.get("top1") or info.get("map50"),
                "native_format": info.get("format", "tflite"),
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
        """Download a MediaPipe model.

        Args:
            model_id: Model identifier
            format: Requested format (models are in TFLite/Task format)
            cache_dir: Directory to store the model

        Returns:
            ModelArtifact with path and metadata
        """
        # Get model info
        info = MEDIAPIPE_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in MediaPipe catalog")

        # Determine actual file extension based on model format
        native_format = info.get("format", "tflite")
        ext = f".{native_format}"
        model_filename = f"{model_id}{ext}"
        model_path = cache_dir / model_filename

        # Check if already cached
        if model_path.exists():
            return self._create_artifact(model_id, format, model_path, info)

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download from URL
        url = info["url"]
        print(f"[MediaPipe] Downloading {model_id} from {url}...")

        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

        if not model_path.exists():
            raise RuntimeError(f"Failed to download model to {model_path}")

        print(f"[MediaPipe] Saved to {model_path}")
        return self._create_artifact(model_id, format, model_path, info)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information."""
        info = MEDIAPIPE_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in catalog")

        return {
            "id": model_id,
            "provider": self.name,
            "benchmarked": True,
            "accuracy": info.get("top1") or info.get("map50"),
            "native_format": info.get("format", "tflite"),
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
            format=format,  # Requested format
            path=path,
            size_bytes=size,
            name=info.get("name", model_id),
            version=None,
            task=info.get("task"),
            parameters=info.get("parameters"),
            input_shape=info.get("input_shape"),
            accuracy=info.get("top1") or info.get("map50"),
            metadata={
                **info,
                "native_format": info.get("format", "tflite"),
            },
        )
