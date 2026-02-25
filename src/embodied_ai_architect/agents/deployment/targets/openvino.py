"""OpenVINO deployment target for Intel/AMD devices.

Supports deployment to:
- Intel CPUs (with AVX-512, AMX)
- Intel integrated/discrete GPUs
- Intel NPUs (Meteor Lake, Lunar Lake)
- AMD Ryzen AI NPUs (via OpenVINO compatibility)
"""

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


class OpenVINOTarget(DeploymentTarget):
    """OpenVINO-based deployment target for Intel and compatible devices.

    Supports:
    - FP32, FP16, INT8 precision
    - INT8 calibration with NNCF (Neural Network Compression Framework)
    - Multiple device targets: CPU, GPU, NPU
    - Validation against ONNX baseline

    Requires:
    - openvino Python package
    - nncf (for INT8 quantization)
    """

    def __init__(self, device: str = "CPU"):
        """Initialize OpenVINO target.

        Args:
            device: Target device - "CPU", "GPU", "NPU", or "AUTO"
        """
        super().__init__(name="openvino")
        self.device = device
        self._ov = None

    def is_available(self) -> bool:
        """Check if OpenVINO is available."""
        try:
            import openvino as ov

            self._ov = ov
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return OpenVINO capabilities."""
        caps = {
            "name": self.name,
            "supported_precisions": ["fp32", "fp16", "int8"],
            "supports_calibration": True,
            "supports_dynamic_batch": True,
            "requires_onnx_input": True,
            "output_format": ".xml/.bin",
            "default_device": self.device,
        }

        # Ensure openvino is loaded
        if self._ov is None and self.is_available():
            pass  # is_available() sets self._ov

        if self._ov:
            caps["openvino_version"] = self._ov.__version__
            # List available devices
            core = self._ov.Core()
            caps["available_devices"] = core.available_devices

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
        """Deploy model to OpenVINO IR format.

        Workflow:
        1. Load ONNX model
        2. Convert to OpenVINO IR
        3. Apply quantization if INT8
        4. Save IR files (.xml, .bin)
        """
        if not self.is_available():
            raise RuntimeError("OpenVINO not available. Install with: pip install openvino")

        if precision == DeploymentPrecision.INT8 and calibration is None:
            raise ValueError("INT8 precision requires calibration config")

        import openvino as ov

        core = ov.Core()

        # Load ONNX model
        model_path = Path(model)
        ov_model = core.read_model(str(model_path))

        # Apply precision settings
        if precision == DeploymentPrecision.FP16:
            ov_model = self._convert_to_fp16(ov_model)
        elif precision == DeploymentPrecision.INT8:
            ov_model = self._quantize_int8(ov_model, calibration, input_shape)

        # Save IR format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Change extension to .xml for IR format
        ir_path = output_path.with_suffix(".xml")
        ov.save_model(ov_model, str(ir_path))

        # Calculate total size (xml + bin)
        bin_path = ir_path.with_suffix(".bin")
        total_size = ir_path.stat().st_size
        if bin_path.exists():
            total_size += bin_path.stat().st_size

        # Get output shape (handle dynamic shapes gracefully)
        output_shape = None
        if ov_model.outputs:
            partial_shape = ov_model.outputs[0].get_partial_shape()
            if partial_shape.is_static:
                output_shape = tuple(partial_shape.to_shape())
            else:
                # For dynamic shapes, get the partial shape dimensions
                output_shape = tuple(d.get_length() if d.is_static else -1 for d in partial_shape)

        return DeploymentArtifact(
            engine_path=ir_path,
            precision=precision,
            target=self.name,
            input_shape=input_shape,
            output_shape=output_shape,
            size_bytes=total_size,
            metadata={
                "openvino_version": ov.__version__,
                "device": self.device,
                "bin_path": str(bin_path),
            },
        )

    def _convert_to_fp16(self, model):
        """Convert model to FP16 precision."""

        # Use compression to FP16

        # OpenVINO automatically handles FP16 conversion during compilation
        # For explicit FP16, we can use the compress_model_transformation
        try:
            from openvino.runtime.passes import Manager, CompressQuantizeWeights

            pass_manager = Manager()
            pass_manager.register_pass(CompressQuantizeWeights())
            pass_manager.run_passes(model)
        except ImportError:
            # Fallback: FP16 will be applied during inference
            pass

        return model

    def _quantize_int8(
        self,
        model,
        calibration: CalibrationConfig,
        input_shape: tuple[int, ...],
    ):
        """Quantize model to INT8 using NNCF."""
        try:
            import nncf
        except ImportError:
            raise ImportError(
                "NNCF required for INT8 quantization. " "Install with: pip install nncf"
            )

        # Create calibration dataset
        calibration_dataset = self._create_calibration_dataset(calibration, input_shape)

        # Quantize with NNCF
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.CPU,  # Works for all devices
        )

        return quantized_model

    def _create_calibration_dataset(
        self,
        config: CalibrationConfig,
        input_shape: tuple[int, ...],
    ):
        """Create calibration dataset for NNCF."""
        import nncf

        data_path = Path(config.data_path)

        def transform_fn(image_path: Path) -> np.ndarray:
            """Load and preprocess image for calibration."""
            try:
                from PIL import Image
            except ImportError:
                raise ImportError("Pillow required for image loading")

            img = Image.open(image_path).convert("RGB")

            # Resize to input shape (H, W from input_shape)
            h, w = input_shape[2], input_shape[3]
            img = img.resize((w, h), Image.BILINEAR)

            # Convert to numpy
            img_array = np.array(img, dtype=np.float32)

            # Preprocessing
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

            # HWC -> NCHW
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)

            return img_array.astype(np.float32)

        # Get image paths
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = []
        for ext in extensions:
            image_paths.extend(data_path.glob(f"*{ext}"))
            image_paths.extend(data_path.glob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)[: config.num_samples]

        # Create NNCF dataset
        calibration_data = [transform_fn(p) for p in image_paths]

        return nncf.Dataset(calibration_data)

    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate OpenVINO IR against baseline."""
        if not self.is_available():
            raise RuntimeError("OpenVINO not available")

        import openvino as ov

        errors = []

        # Load OpenVINO model
        core = ov.Core()
        ov_model = core.read_model(str(deployed_artifact.engine_path))
        compiled_model = core.compile_model(ov_model, self.device)
        infer_request = compiled_model.create_infer_request()

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

            # OpenVINO inference
            start = time.perf_counter()
            input_tensor = ov.Tensor(input_data)
            infer_request.set_input_tensor(input_tensor)
            infer_request.infer()
            ov_out = infer_request.get_output_tensor().data.copy()
            ov_time = time.perf_counter() - start
            latencies_deployed.append(ov_time)

            # Compare outputs
            if config.compare_outputs:
                diff = np.abs(baseline_out - ov_out).max()
                max_diff = max(max_diff, float(diff))

            samples_compared += 1

        # Calculate metrics
        baseline_latency = float(np.mean(latencies_baseline) * 1000)
        deployed_latency = float(np.mean(latencies_deployed) * 1000)
        speedup = baseline_latency / deployed_latency if deployed_latency > 0 else 0.0

        # Determine pass/fail for accuracy
        accuracy_passed = max_diff < (config.tolerance_percent / 100.0)

        # Power validation if configured
        power_metrics = None
        power_passed = True
        if config.power_validation is not None:
            # Create workload function for power measurement
            test_input = np.random.randn(*input_shape).astype(np.float32)

            def inference_workload():
                input_tensor = ov.Tensor(test_input)
                infer_request.set_input_tensor(input_tensor)
                infer_request.infer()

            # Get power prediction if available
            predicted_watts = None
            try:
                from ..power.predictor import PowerPredictor, PowerProfile

                predictor = PowerPredictor()
                # Estimate model characteristics from artifact
                total_params = deployed_artifact.metadata.get("total_params", 1_000_000)
                total_macs = deployed_artifact.metadata.get("total_macs", 100_000_000)
                precision = deployed_artifact.precision.value

                prediction = predictor.predict_from_model_info(
                    total_params=total_params,
                    total_macs=total_macs,
                    hardware=PowerProfile.INTEL_CORE_I7,  # Default for OpenVINO CPU
                    precision=precision,
                )
                predicted_watts = prediction.mean_watts
            except Exception:
                pass  # Prediction not available

            power_metrics = self.measure_power(
                workload=inference_workload,
                config=config.power_validation,
                predicted_watts=predicted_watts,
            )

            if power_metrics is not None:
                power_passed = self.validate_power_result(power_metrics, config.power_validation)
                if not power_passed:
                    if config.power_validation.power_budget_watts:
                        errors.append(
                            f"Power {power_metrics.measured_watts:.1f}W exceeds budget "
                            f"{config.power_validation.power_budget_watts:.1f}W"
                        )
                    if (
                        power_metrics.deviation_percent
                        and abs(power_metrics.deviation_percent)
                        > config.power_validation.tolerance_percent
                    ):
                        errors.append(
                            f"Power deviation {power_metrics.deviation_percent:.1f}% exceeds "
                            f"tolerance {config.power_validation.tolerance_percent:.1f}%"
                        )

        return ValidationResult(
            passed=accuracy_passed and power_passed,
            baseline_latency_ms=baseline_latency,
            deployed_latency_ms=deployed_latency,
            speedup=speedup,
            max_output_diff=max_diff,
            samples_compared=samples_compared,
            power=power_metrics,
            power_validation_passed=power_passed if power_metrics else None,
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
