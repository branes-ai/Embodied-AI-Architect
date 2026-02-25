"""YOLOv8 object detector using ONNX Runtime.

Supports CPU, GPU (CUDA/ROCm), and NPU (Ryzen AI, QNN, CoreML) execution.
"""

from typing import Any

import numpy as np

from ..base import Operator


class YOLOv8ONNX(Operator):
    """YOLOv8 object detection using ONNX Runtime.

    Supports multiple execution providers:
    - CPU: CPUExecutionProvider
    - GPU: CUDAExecutionProvider, ROCMExecutionProvider
    - NPU: RyzenAIExecutionProvider, QNNExecutionProvider, CoreMLExecutionProvider
    """

    # Map execution targets to ONNX Runtime providers
    PROVIDER_MAP = {
        "cpu": ["CPUExecutionProvider"],
        "gpu": ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
        "npu": [
            "RyzenAIExecutionProvider",
            "QNNExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    }

    def __init__(self, variant: str = "n"):
        """Initialize YOLO detector.

        Args:
            variant: Model variant (n, s, m, l, x)
        """
        super().__init__(operator_id=f"yolo_detector_{variant}")
        self.variant = variant
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def setup(self, config: dict[str, Any], execution_target: str = "cpu") -> None:
        """Initialize ONNX Runtime session.

        Args:
            config: Configuration with optional keys:
                - model_path: Path to ONNX model (default: auto-download)
                - conf_threshold: Confidence threshold (default: 0.25)
                - iou_threshold: IOU threshold for NMS (default: 0.45)
            execution_target: cpu, gpu, or npu
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        self._execution_target = execution_target
        self._config = config

        # Get configuration
        self.conf_threshold = config.get("conf_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)

        # Get model path - use model zoo for automatic acquisition
        model_path = config.get("model_path")
        if model_path is None:
            from embodied_ai_architect.model_zoo import acquire

            model_path = acquire(f"yolov8{self.variant}", format="onnx")

        # Select providers based on target
        providers = self.PROVIDER_MAP.get(execution_target, ["CPUExecutionProvider"])
        available = ort.get_available_providers()

        # Filter to available providers
        selected = [p for p in providers if p in available]
        if not selected:
            print(f"Warning: No providers for target={execution_target}, falling back to CPU")
            selected = ["CPUExecutionProvider"]

        print(f"[YOLOv8ONNX] Loading {model_path} with providers: {selected}")

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=selected,
        )

        # Get input info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape  # [batch, channels, height, width]

        self._is_setup = True
        print(f"[YOLOv8ONNX] Ready on {execution_target} (input: {self.input_shape})")

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run detection on input image.

        Args:
            inputs: Dictionary with 'image' key containing:
                - numpy array (H, W, 3) RGB uint8, or
                - preprocessed tensor (1, 3, H, W) float32

        Returns:
            Dictionary with 'detections' key containing list of detection dicts
        """
        image = inputs["image"]

        # Preprocess if needed
        if image.ndim == 3:
            blob = self._preprocess(image)
        else:
            blob = image

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})

        # Postprocess
        detections = self._postprocess(outputs[0], image.shape[:2] if image.ndim == 3 else None)

        return {"detections": detections}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            Preprocessed tensor (1, 3, 640, 640) float32
        """
        import cv2

        # Resize to model input size
        h, w = self.input_shape[2:4] if len(self.input_shape) >= 4 else (640, 640)
        resized = cv2.resize(image, (w, h))

        # Normalize to [0, 1]
        blob = resized.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Postprocess YOLO output.

        Args:
            output: Raw YOLO output tensor
            original_shape: Original image (H, W) for rescaling boxes

        Returns:
            List of detection dictionaries
        """
        # YOLOv8 output: (1, 84, 8400) for 80 classes
        # Transpose to (1, 8400, 84)
        if output.shape[1] == 84:
            output = np.transpose(output, (0, 2, 1))

        predictions = output[0]  # (8400, 84)

        # Extract boxes and scores
        boxes = predictions[:, :4]  # xywh
        scores = predictions[:, 4:]  # class scores

        # Get best class for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert xywh to xyxy
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        # Scale boxes if original shape provided
        if original_shape is not None:
            h_orig, w_orig = original_shape
            h_model, w_model = self.input_shape[2:4] if len(self.input_shape) >= 4 else (640, 640)
            scale_x = w_orig / w_model
            scale_y = h_orig / h_model
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y

        # Build detection list
        detections = []
        for idx in indices:
            detections.append(
                {
                    "bbox": boxes_xyxy[idx].tolist(),
                    "class_id": int(class_ids[idx]),
                    "confidence": float(confidences[idx]),
                }
            )

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Non-maximum suppression.

        Args:
            boxes: (N, 4) xyxy format
            scores: (N,) confidence scores
            iou_threshold: IOU threshold

        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def teardown(self) -> None:
        """Release ONNX session."""
        self.session = None
        self._is_setup = False
