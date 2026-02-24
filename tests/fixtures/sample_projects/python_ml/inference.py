"""Main inference pipeline."""

import numpy as np


def load_model(model_path: str):
    """Load ONNX model for inference."""
    import onnxruntime as ort
    return ort.InferenceSession(model_path)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model input."""
    import cv2
    resized = cv2.resize(image, (640, 640))
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def postprocess(outputs: list) -> list:
    """Post-process model outputs to detections."""
    # NMS and thresholding
    detections = []
    for output in outputs:
        scores = output[:, 4]
        mask = scores > 0.5
        detections.extend(output[mask].tolist())
    return detections


def run_pipeline(image: np.ndarray, model_path: str = "model.onnx") -> list:
    """Run full inference pipeline."""
    session = load_model(model_path)
    input_tensor = preprocess(image)
    outputs = session.run(None, {"images": input_tensor})
    return postprocess(outputs)


if __name__ == "__main__":
    import cv2
    image = cv2.imread("test.jpg")
    results = run_pipeline(image)
    print(f"Found {len(results)} detections")
