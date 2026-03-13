"""YOLOv8 object detector."""

import os
import sys
from contextlib import contextmanager
from typing import List, Optional

import numpy as np

from ..common import Detection, BBox

# COCO class IDs relevant to drone perception
DRONE_CLASSES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
]

# Minimum bbox area in pixels^2 — reject detections smaller than this
MIN_BBOX_AREA = 100


@contextmanager
def _suppress_stderr():
    """Redirect stderr to /dev/null to suppress C++ NNPACK warnings."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull)
        os.close(saved_fd)


class YOLODetector:
    """
    YOLOv8 object detector using Ultralytics.

    Supports different model sizes:
    - yolov8n (nano): Fastest, least accurate
    - yolov8s (small): Good balance for drone use
    - yolov8m (medium): Better accuracy
    - yolov8l/x (large/xlarge): Best accuracy, slowest
    """

    def __init__(
        self,
        model_size: str = 's',
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = 'cpu',
        classes: Optional[List[int]] = None
    ):
        """
        Initialize YOLO detector.

        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: 'cpu' or 'cuda'
            classes: Class IDs to detect. None = drone-relevant subset.
                     Pass [] to detect all 80 COCO classes.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )

        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Class filtering: None → drone defaults, [] → all classes, list → custom
        if classes is not None and len(classes) == 0:
            self.filter_classes = None  # ultralytics convention: None = all
        elif classes is not None:
            self.filter_classes = classes
        else:
            self.filter_classes = DRONE_CLASSES

        # Load model
        model_name = f'yolov8{model_size}.pt'
        print(f"[YOLODetector] Loading {model_name}...")
        self.model = YOLO(model_name)

        # Warmup
        print(f"[YOLODetector] Warming up on {device}...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with _suppress_stderr():
            _ = self.model(dummy, device=device, verbose=False)

        # COCO class names (YOLOv8 default)
        self.class_names = self.model.names

        if self.filter_classes:
            filtered_names = [self.class_names[i] for i in self.filter_classes]
            print(f"[YOLODetector] Ready! Detecting: {filtered_names}")
        else:
            print(f"[YOLODetector] Ready! Detecting all {len(self.class_names)} classes")

    def detect(self, image: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Run detection on an image.

        Args:
            image: RGB image (H, W, 3)
            frame_id: Frame identifier

        Returns:
            List of Detection objects
        """
        # Run inference
        with _suppress_stderr():
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                classes=self.filter_classes
            )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                # Get bbox in xyxy format
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                bbox = BBox.from_xyxy(x1, y1, x2, y2)

                # Skip tiny detections (noise)
                if bbox.area < MIN_BBOX_AREA:
                    continue

                # Get class and confidence
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = self.class_names[class_id]

                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    frame_id=frame_id
                )
                detections.append(detection)

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run batch detection on multiple images.

        Args:
            images: List of RGB images

        Returns:
            List of detection lists (one per image)
        """
        # Run batch inference
        with _suppress_stderr():
            results = self.model(
                images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                classes=self.filter_classes
            )

        # Parse results
        all_detections = []
        for frame_id, result in enumerate(results):
            detections = []
            boxes = result.boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                bbox = BBox.from_xyxy(x1, y1, x2, y2)

                if bbox.area < MIN_BBOX_AREA:
                    continue

                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = self.class_names[class_id]

                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    frame_id=frame_id
                )
                detections.append(detection)

            all_detections.append(detections)

        return all_detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: RGB image
            detections: List of detections
            thickness: Line thickness

        Returns:
            Image with drawn boxes
        """
        import cv2

        img_draw = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Color based on class (use hash for consistency)
            color_seed = hash(det.class_name) % 256
            color = (
                (color_seed * 123) % 256,
                (color_seed * 456) % 256,
                (color_seed * 789) % 256
            )

            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_draw,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return img_draw
