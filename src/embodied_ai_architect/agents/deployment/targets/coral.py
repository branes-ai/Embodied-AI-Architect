"""Coral Edge TPU deployment target.

Supports deployment to Google Coral Edge TPU devices:
- Coral USB Accelerator
- Coral Dev Board
- Coral M.2/Mini PCIe Accelerator

The Edge TPU only supports INT8 quantized models, providing:
- Ultra-low power consumption (~0.5W per TOPS)
- Fast inference (sub-millisecond for small models)
- Ideal for battery-powered and always-on applications
"""

import subprocess
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


class CoralTarget(DeploymentTarget):
    """Edge TPU deployment target for Google Coral devices.

    Supports:
    - INT8 precision only (Edge TPU requirement)
    - Quantization-aware conversion from ONNX
    - Edge TPU compilation for hardware acceleration
    - Validation against TFLite CPU baseline

    Requires:
    - tensorflow or tflite-runtime
    - onnx2tf or onnx-tf for ONNX conversion
    - edgetpu_compiler (optional, for Edge TPU compilation)
    - pycoral (optional, for Edge TPU inference)
    """

    def __init__(self):
        """Initialize Coral Edge TPU target."""
        super().__init__(name="coral")
        self._tf = None
        self._has_edgetpu_compiler = False
        self._has_pycoral = False

    def is_available(self) -> bool:
        """Check if TensorFlow/TFLite is available."""
        try:
            import tensorflow as tf

            self._tf = tf
            return True
        except ImportError:
            pass

        # Try tflite-runtime as fallback
        try:
            import tflite_runtime.interpreter as tflite

            self._tflite_runtime = tflite
            return True
        except ImportError:
            pass

        return False

    def _check_edgetpu_compiler(self) -> bool:
        """Check if Edge TPU compiler is available."""
        try:
            result = subprocess.run(
                ["edgetpu_compiler", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._has_edgetpu_compiler = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            self._has_edgetpu_compiler = False
        return self._has_edgetpu_compiler

    def _check_pycoral(self) -> bool:
        """Check if pycoral is available."""
        try:
            from pycoral.utils import edgetpu  # noqa: F401

            self._has_pycoral = True
        except ImportError:
            self._has_pycoral = False
        return self._has_pycoral

    def get_capabilities(self) -> dict[str, Any]:
        """Return Edge TPU capabilities."""
        # Check optional components
        self._check_edgetpu_compiler()
        self._check_pycoral()

        caps = {
            "name": self.name,
            "supported_precisions": ["int8"],  # Edge TPU only supports INT8
            "supports_calibration": True,
            "supports_dynamic_batch": False,  # Edge TPU requires fixed batch
            "requires_onnx_input": True,
            "output_format": ".tflite",
            "edgetpu_compiler_available": self._has_edgetpu_compiler,
            "pycoral_available": self._has_pycoral,
        }

        if self._tf:
            caps["tensorflow_version"] = self._tf.__version__

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
        """Deploy model to Edge TPU format.

        Workflow:
        1. Convert ONNX to TensorFlow SavedModel
        2. Quantize to INT8 TFLite with representative dataset
        3. Compile for Edge TPU (if compiler available)
        """
        if not self.is_available():
            raise RuntimeError("TensorFlow not available. Install with: pip install tensorflow")

        # Edge TPU only supports INT8
        if precision != DeploymentPrecision.INT8:
            raise ValueError(
                f"Edge TPU only supports INT8 precision, got {precision.value}. "
                "The Edge TPU hardware requires fully quantized INT8 models."
            )

        if calibration is None:
            raise ValueError(
                "INT8 quantization requires calibration config with representative data"
            )

        import tensorflow as tf

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert ONNX to TensorFlow
        model_path = Path(model)
        saved_model_dir = output_path.parent / "saved_model"
        self._convert_onnx_to_tf(model_path, saved_model_dir)

        # Step 2: Convert to quantized TFLite
        tflite_path = output_path.with_suffix(".tflite")
        self._convert_to_tflite_int8(
            saved_model_dir,
            tflite_path,
            calibration,
            input_shape,
        )

        # Step 3: Compile for Edge TPU (if compiler available)
        edgetpu_path = None
        if self._check_edgetpu_compiler():
            edgetpu_path = self._compile_for_edgetpu(tflite_path)

        # Use Edge TPU model if available, otherwise standard TFLite
        final_path = edgetpu_path if edgetpu_path else tflite_path

        # Get output shape from TFLite model
        output_shape = self._get_tflite_output_shape(final_path)

        return DeploymentArtifact(
            engine_path=final_path,
            precision=precision,
            target=self.name,
            input_shape=input_shape,
            output_shape=output_shape,
            size_bytes=final_path.stat().st_size,
            metadata={
                "tensorflow_version": tf.__version__,
                "edgetpu_compiled": edgetpu_path is not None,
                "tflite_path": str(tflite_path),
                "edgetpu_path": str(edgetpu_path) if edgetpu_path else None,
            },
        )

    def _convert_onnx_to_tf(self, onnx_path: Path, output_dir: Path) -> None:
        """Convert ONNX model to TensorFlow SavedModel."""
        try:
            import onnx2tf
        except ImportError:
            try:
                # Fallback to onnx-tf
                import onnx
                from onnx_tf.backend import prepare

                onnx_model = onnx.load(str(onnx_path))
                tf_rep = prepare(onnx_model)
                tf_rep.export_graph(str(output_dir))
                return
            except ImportError:
                raise ImportError(
                    "onnx2tf or onnx-tf required for ONNX to TensorFlow conversion. "
                    "Install with: pip install onnx2tf or pip install onnx-tf"
                )

        # Use onnx2tf (preferred, better compatibility)
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(output_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )

    def _convert_to_tflite_int8(
        self,
        saved_model_dir: Path,
        output_path: Path,
        calibration: CalibrationConfig,
        input_shape: tuple[int, ...],
    ) -> None:
        """Convert SavedModel to quantized INT8 TFLite."""
        import tensorflow as tf

        # Create converter
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        # Enable full integer quantization for Edge TPU
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Create representative dataset generator
        def representative_dataset():
            for data in self._create_calibration_data(calibration, input_shape):
                yield [data]

        converter.representative_dataset = representative_dataset

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, "wb") as f:
            f.write(tflite_model)

    def _create_calibration_data(
        self,
        config: CalibrationConfig,
        input_shape: tuple[int, ...],
    ):
        """Generate calibration data for TFLite quantization."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for calibration image loading")

        data_path = Path(config.data_path)

        # Get image paths
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = []
        for ext in extensions:
            image_paths.extend(data_path.glob(f"*{ext}"))
            image_paths.extend(data_path.glob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)[: config.num_samples]

        # TFLite expects NHWC format, but input_shape might be NCHW
        if len(input_shape) == 4:
            if input_shape[1] == 3:  # NCHW format
                n, c, h, w = input_shape
                is_nchw = True
            else:  # NHWC format
                n, h, w, c = input_shape
                is_nchw = False
        else:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((w, h), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32)

            # Apply preprocessing
            if config.preprocessing == "imagenet":
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = img_array / 255.0
                img_array = (img_array - mean) / std
            elif config.preprocessing == "yolo":
                img_array = img_array / 255.0
            elif config.preprocessing == "coco":
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = img_array / 255.0
                img_array = (img_array - mean) / std
            else:
                img_array = img_array / 255.0  # Default normalization

            # TFLite expects NHWC, convert if needed
            if is_nchw:
                # Input shape is NCHW, but TFLite wants NHWC
                # img_array is already HWC from PIL
                pass

            # Add batch dimension (NHWC format)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            yield img_array

    def _compile_for_edgetpu(self, tflite_path: Path) -> Path | None:
        """Compile TFLite model for Edge TPU."""
        try:
            output_dir = tflite_path.parent

            result = subprocess.run(
                [
                    "edgetpu_compiler",
                    "-s",  # Show operations
                    "-o",
                    str(output_dir),
                    str(tflite_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                # Compilation failed, return None (will use CPU TFLite)
                return None

            # Edge TPU compiler creates file with _edgetpu suffix
            edgetpu_path = tflite_path.with_name(tflite_path.stem + "_edgetpu.tflite")

            if edgetpu_path.exists():
                return edgetpu_path

            return None

        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _get_tflite_output_shape(self, tflite_path: Path) -> tuple[int, ...] | None:
        """Get output shape from TFLite model."""
        try:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            output_details = interpreter.get_output_details()
            if output_details:
                return tuple(output_details[0]["shape"])
        except Exception:
            pass

        try:
            import tflite_runtime.interpreter as tflite

            interpreter = tflite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            output_details = interpreter.get_output_details()
            if output_details:
                return tuple(output_details[0]["shape"])
        except Exception:
            pass

        return None

    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate TFLite model against ONNX baseline."""
        errors = []

        # Load TFLite interpreter
        interpreter = self._load_tflite_interpreter(deployed_artifact.engine_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check if Edge TPU delegate is available
        using_edgetpu = False
        if self._check_pycoral() and deployed_artifact.metadata.get("edgetpu_compiled"):
            try:
                from pycoral.utils import edgetpu

                interpreter = edgetpu.make_interpreter(str(deployed_artifact.engine_path))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                using_edgetpu = True
            except Exception as e:
                errors.append(f"Edge TPU delegate failed: {e}")

        # Load baseline model (ONNX Runtime)
        baseline_session = self._load_baseline(baseline_model)

        # Get shapes
        input_shape = deployed_artifact.input_shape

        # Run validation
        latencies_baseline = []
        latencies_deployed = []
        max_diff = 0.0
        samples_compared = 0

        test_loader = self._create_test_loader(config, input_shape)

        for i, input_data in enumerate(test_loader):
            if i >= config.num_samples:
                break

            # Baseline inference (ONNX Runtime)
            start = time.perf_counter()
            input_name = baseline_session.get_inputs()[0].name
            baseline_out = baseline_session.run(None, {input_name: input_data})[0]
            baseline_time = time.perf_counter() - start
            latencies_baseline.append(baseline_time)

            # TFLite inference
            start = time.perf_counter()

            # Prepare input (may need quantization for INT8)
            tflite_input = self._prepare_tflite_input(input_data, input_details[0])

            interpreter.set_tensor(input_details[0]["index"], tflite_input)
            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]["index"])

            # Dequantize output if needed
            tflite_out = self._dequantize_output(tflite_out, output_details[0])

            tflite_time = time.perf_counter() - start
            latencies_deployed.append(tflite_time)

            # Compare outputs
            if config.compare_outputs:
                # Handle shape differences (NCHW vs NHWC)
                if baseline_out.shape != tflite_out.shape:
                    # Try to reshape for comparison
                    try:
                        tflite_out = tflite_out.reshape(baseline_out.shape)
                    except ValueError:
                        pass

                diff = np.abs(baseline_out.astype(np.float32) - tflite_out.astype(np.float32)).max()
                max_diff = max(max_diff, float(diff))

            samples_compared += 1

        # Calculate metrics
        baseline_latency = float(np.mean(latencies_baseline) * 1000)
        deployed_latency = float(np.mean(latencies_deployed) * 1000)
        speedup = baseline_latency / deployed_latency if deployed_latency > 0 else 0.0

        # INT8 quantization has larger tolerance
        passed = max_diff < (config.tolerance_percent / 100.0 * 10)  # Scale tolerance

        if using_edgetpu:
            errors.append("Using Edge TPU hardware acceleration")

        return ValidationResult(
            passed=passed,
            baseline_latency_ms=baseline_latency,
            deployed_latency_ms=deployed_latency,
            speedup=speedup,
            max_output_diff=max_diff,
            samples_compared=samples_compared,
            errors=errors,
        )

    def _load_tflite_interpreter(self, model_path: Path):
        """Load TFLite interpreter."""
        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(model_path))
        except (ImportError, AttributeError):
            pass

        try:
            import tflite_runtime.interpreter as tflite

            return tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            raise ImportError("tensorflow or tflite-runtime required for TFLite inference")

    def _prepare_tflite_input(self, input_data: np.ndarray, input_details: dict) -> np.ndarray:
        """Prepare input for TFLite (quantize if needed)."""
        dtype = input_details["dtype"]

        # Convert NCHW to NHWC if needed
        if len(input_data.shape) == 4 and input_data.shape[1] == 3:
            input_data = np.transpose(input_data, (0, 2, 3, 1))

        if dtype == np.int8:
            # Quantize
            scale = input_details["quantization"][0]
            zero_point = input_details["quantization"][1]
            quantized = (input_data / scale + zero_point).astype(np.int8)
            return quantized
        elif dtype == np.uint8:
            scale = input_details["quantization"][0]
            zero_point = input_details["quantization"][1]
            quantized = (input_data / scale + zero_point).astype(np.uint8)
            return quantized

        return input_data.astype(dtype)

    def _dequantize_output(self, output: np.ndarray, output_details: dict) -> np.ndarray:
        """Dequantize TFLite output if needed."""
        if output_details["dtype"] in (np.int8, np.uint8):
            scale = output_details["quantization"][0]
            zero_point = output_details["quantization"][1]
            return (output.astype(np.float32) - zero_point) * scale
        return output

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

            # Handle NCHW vs NHWC
            if len(input_shape) == 4:
                if input_shape[1] == 3:  # NCHW
                    h, w = input_shape[2], input_shape[3]
                else:  # NHWC
                    h, w = input_shape[1], input_shape[2]
            else:
                h, w = 224, 224  # Default

            for path in sorted(paths)[: config.num_samples]:
                img = Image.open(path).convert("RGB")
                img = img.resize((w, h), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Return in NCHW format (will be converted in prepare_input)
                img_array = img_array.transpose(2, 0, 1)
                yield img_array[np.newaxis, ...].astype(np.float32)
