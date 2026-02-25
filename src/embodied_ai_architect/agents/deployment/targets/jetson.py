"""Jetson/TensorRT deployment target."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import DeploymentTarget
from ..models import (
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    ValidationConfig,
    ValidationResult,
)


class JetsonTarget(DeploymentTarget):
    """TensorRT-based deployment target for NVIDIA Jetson devices.

    Supports:
    - FP32, FP16, INT8 precision
    - INT8 calibration with IInt8EntropyCalibrator2
    - Dynamic batch sizes (optional)
    - Validation against PyTorch/ONNX baseline

    Requires:
    - tensorrt Python package
    - pycuda (for calibration and inference)
    """

    def __init__(self, workspace_size_gb: float = 4.0):
        """Initialize Jetson/TensorRT target.

        Args:
            workspace_size_gb: TensorRT workspace size in GB
        """
        super().__init__(name="jetson")
        self.workspace_size = int(workspace_size_gb * (1 << 30))
        self._trt = None
        self._cuda = None

    def is_available(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt

            self._trt = trt
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return TensorRT capabilities."""
        caps = {
            "name": self.name,
            "supported_precisions": ["fp32", "fp16", "int8"],
            "supports_calibration": True,
            "supports_dynamic_batch": True,
            "requires_onnx_input": True,
            "output_format": ".engine",
        }

        if self._trt:
            caps["tensorrt_version"] = self._trt.__version__

        return caps

    def deploy(
        self,
        model: Path,
        precision: DeploymentPrecision,
        output_path: Path,
        input_shape: tuple[int, ...],
        calibration: CalibrationConfig | None = None,
        **kwargs,
    ) -> DeploymentArtifact:
        """Deploy model to TensorRT engine.

        Workflow:
        1. Parse ONNX model
        2. Configure builder for precision
        3. Run INT8 calibration if needed
        4. Build and serialize engine
        """
        if not self.is_available():
            raise RuntimeError("TensorRT not available. Install with: pip install tensorrt")

        if precision == DeploymentPrecision.INT8 and calibration is None:
            raise ValueError("INT8 precision requires calibration config")

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        model_path = Path(model)
        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parsing failed: {errors}")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        if precision == DeploymentPrecision.FP16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == DeploymentPrecision.INT8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = self._create_calibrator(calibration, input_shape)
            config.int8_calibrator = calibrator

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Engine build failed")

        # Save engine
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        # Get output shape from network
        output_tensor = network.get_output(0)
        output_shape = tuple(output_tensor.shape) if output_tensor else None

        return DeploymentArtifact(
            engine_path=output_path,
            precision=precision,
            target=self.name,
            input_shape=input_shape,
            output_shape=output_shape,
            size_bytes=output_path.stat().st_size,
            metadata={
                "tensorrt_version": trt.__version__,
                "workspace_size_gb": self.workspace_size / (1 << 30),
            },
        )

    def _create_calibrator(
        self,
        config: CalibrationConfig,
        input_shape: tuple[int, ...],
    ):
        """Create INT8 calibrator with calibration dataset."""
        import tensorrt as trt

        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError:
            raise ImportError(
                "pycuda required for INT8 calibration. " "Install with: pip install pycuda"
            )

        data_path = Path(config.data_path)
        cache_file = data_path.parent / "calibration.cache"

        class Int8Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(
                calibrator_self,
                data_path: Path,
                input_shape: tuple,
                num_samples: int,
                batch_size: int,
                preprocessing: str,
            ):
                super().__init__()
                calibrator_self.data_path = data_path
                calibrator_self.input_shape = input_shape
                calibrator_self.batch_size = batch_size
                calibrator_self.num_samples = num_samples
                calibrator_self.preprocessing = preprocessing
                calibrator_self.cache_file = cache_file
                calibrator_self.current_index = 0

                # Load image paths
                calibrator_self.image_paths = calibrator_self._get_image_paths()

                # Allocate device memory
                calibrator_self.device_input = cuda.mem_alloc(
                    int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
                )

            def _get_image_paths(calibrator_self) -> list[Path]:
                """Get list of calibration image paths."""
                extensions = {".jpg", ".jpeg", ".png", ".bmp"}
                paths = []
                for ext in extensions:
                    paths.extend(calibrator_self.data_path.glob(f"*{ext}"))
                    paths.extend(calibrator_self.data_path.glob(f"*{ext.upper()}"))
                return sorted(paths)[: calibrator_self.num_samples]

            def _load_and_preprocess(calibrator_self, image_path: Path) -> np.ndarray:
                """Load and preprocess a single image."""
                try:
                    from PIL import Image
                except ImportError:
                    raise ImportError("Pillow required for image loading")

                img = Image.open(image_path).convert("RGB")

                # Resize to input shape (H, W from input_shape)
                h, w = calibrator_self.input_shape[2], calibrator_self.input_shape[3]
                img = img.resize((w, h), Image.BILINEAR)

                # Convert to numpy
                img_array = np.array(img, dtype=np.float32)

                # Preprocessing
                if calibrator_self.preprocessing == "imagenet":
                    # ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_array = img_array / 255.0
                    img_array = (img_array - mean) / std
                elif calibrator_self.preprocessing == "yolo":
                    # YOLO: 0-1 normalization
                    img_array = img_array / 255.0
                elif calibrator_self.preprocessing == "coco":
                    # COCO: same as ImageNet typically
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_array = img_array / 255.0
                    img_array = (img_array - mean) / std
                # else: no preprocessing

                # HWC -> CHW
                img_array = img_array.transpose(2, 0, 1)

                return img_array.astype(np.float32)

            def get_batch_size(calibrator_self):
                return calibrator_self.batch_size

            def get_batch(calibrator_self, names):
                if calibrator_self.current_index >= len(calibrator_self.image_paths):
                    return None

                # Load batch
                batch_images = []
                for _ in range(calibrator_self.batch_size):
                    if calibrator_self.current_index >= len(calibrator_self.image_paths):
                        break
                    img = calibrator_self._load_and_preprocess(
                        calibrator_self.image_paths[calibrator_self.current_index]
                    )
                    batch_images.append(img)
                    calibrator_self.current_index += 1

                if not batch_images:
                    return None

                batch = np.stack(batch_images, axis=0)
                cuda.memcpy_htod(calibrator_self.device_input, batch.ravel())
                return [int(calibrator_self.device_input)]

            def read_calibration_cache(calibrator_self):
                if calibrator_self.cache_file.exists():
                    return calibrator_self.cache_file.read_bytes()
                return None

            def write_calibration_cache(calibrator_self, cache):
                calibrator_self.cache_file.write_bytes(cache)

        return Int8Calibrator(
            data_path=data_path,
            input_shape=input_shape,
            num_samples=config.num_samples,
            batch_size=config.batch_size,
            preprocessing=config.preprocessing,
        )

    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate TensorRT engine against baseline."""
        if not self.is_available():
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError:
            raise ImportError("pycuda required for validation")

        errors = []

        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(deployed_artifact.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Load baseline model (ONNX Runtime)
        baseline_session = self._load_baseline(baseline_model)

        # Allocate TensorRT buffers
        input_shape = deployed_artifact.input_shape
        output_shape = deployed_artifact.output_shape or (1, 1000)  # Fallback

        d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
        d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))
        stream = cuda.Stream()

        # Run validation
        latencies_baseline = []
        latencies_deployed = []
        max_diff = 0.0
        samples_compared = 0

        test_loader = self._create_test_loader(config, input_shape)

        for i, input_data in enumerate(test_loader):
            if i >= config.num_samples:
                break

            # Baseline inference
            start = time.perf_counter()
            input_name = baseline_session.get_inputs()[0].name
            baseline_out = baseline_session.run(None, {input_name: input_data})[0]
            baseline_time = time.perf_counter() - start
            latencies_baseline.append(baseline_time)

            # TensorRT inference
            start = time.perf_counter()
            cuda.memcpy_htod_async(d_input, input_data.ravel(), stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
            )
            trt_out = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(trt_out, d_output, stream)
            stream.synchronize()
            trt_time = time.perf_counter() - start
            latencies_deployed.append(trt_time)

            # Compare outputs
            if config.compare_outputs:
                diff = np.abs(baseline_out - trt_out).max()
                max_diff = max(max_diff, float(diff))

            samples_compared += 1

        # Calculate metrics
        baseline_latency = float(np.mean(latencies_baseline) * 1000)
        deployed_latency = float(np.mean(latencies_deployed) * 1000)
        speedup = baseline_latency / deployed_latency if deployed_latency > 0 else 0.0

        # Determine pass/fail
        passed = max_diff < (config.tolerance_percent / 100.0)

        return ValidationResult(
            passed=passed,
            baseline_latency_ms=baseline_latency,
            deployed_latency_ms=deployed_latency,
            speedup=speedup,
            max_output_diff=max_diff,
            samples_compared=samples_compared,
            errors=errors,
        )

    def _load_baseline(self, model_path: Path):
        """Load baseline model for comparison."""
        try:
            import onnxruntime as ort

            return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except ImportError:
            raise ImportError(
                "onnxruntime required for validation. " "Install with: pip install onnxruntime"
            )

    def _create_test_loader(self, config: ValidationConfig, input_shape: tuple[int, ...]):
        """Create test data loader."""
        if config.test_data_path is None:
            # Generate random test data
            for _ in range(config.num_samples):
                yield np.random.randn(*input_shape).astype(np.float32)
        else:
            # Load test images
            from PIL import Image

            extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            paths = []
            for ext in extensions:
                paths.extend(config.test_data_path.glob(f"*{ext}"))
                paths.extend(config.test_data_path.glob(f"*{ext.upper()}"))

            h, w = input_shape[2], input_shape[3]

            for path in sorted(paths)[: config.num_samples]:
                img = Image.open(path).convert("RGB")
                img = img.resize((w, h), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array.transpose(2, 0, 1)
                yield img_array[np.newaxis, ...].astype(np.float32)
