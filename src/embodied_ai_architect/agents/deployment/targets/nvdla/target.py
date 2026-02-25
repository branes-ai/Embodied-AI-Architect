"""NVDLA Virtual Platform deployment target implementation.

This module integrates NVDLA Virtual Platform with the deployment system.
It provides compilation, execution, and validation of models on the
NVDLA hardware simulation.

Key differences from other targets:
1. Requires ONNX → Caffe conversion (NVDLA compiler only accepts Caffe)
2. Execution on Virtual Platform is slow (~hours for large models)
3. Full system simulation with Linux kernel
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..base import DeploymentTarget
from ...models import (
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    PowerConfig,
    PowerMetrics,
    ValidationConfig,
    ValidationResult,
)
from .spec import (
    NVDLAConfig,
    NVDLALoadable,
    NVDLAPrecision,
    NVDLACompilerInterface,
    NVDLARuntimeInterface,
    NVDLAExecutionResult,
    NVDLAPerformanceMetrics,
    SimulationMode,
)


class NVDLATarget(DeploymentTarget):
    """Deployment target for NVIDIA NVDLA Virtual Platform.

    This target compiles models to NVDLA loadable format and executes
    them on the Virtual Platform (SystemC simulation) or hardware.

    Usage:
        target = NVDLATarget()

        # Check availability
        if target.is_available():
            artifact = target.deploy(
                model=onnx_path,
                precision=DeploymentPrecision.FP16,
                output_path=output_dir / "model.nvdla",
                input_shape=(1, 3, 224, 224),
            )

    Note: Execution on the Virtual Platform is slow. For ResNet-50:
    - FP16: ~2.5 hours
    - INT8: ~5 hours

    For faster iteration, use the stub runtime which estimates performance
    without full VP simulation.
    """

    def __init__(
        self,
        config: NVDLAConfig | None = None,
        compiler: NVDLACompilerInterface | None = None,
        runtime: NVDLARuntimeInterface | None = None,
    ):
        """Initialize NVDLA target.

        Args:
            config: NVDLA configuration (uses default if None)
            compiler: Custom compiler implementation
            runtime: Custom runtime implementation
        """
        super().__init__(name="nvdla")
        self.config = config or NVDLAConfig()

        self._compiler = compiler
        self._runtime = runtime

        # Lazy initialization for stubs
        self._stub_compiler: "StubNVDLACompiler | None" = None
        self._stub_runtime: "StubNVDLARuntime | None" = None

    @property
    def compiler(self) -> NVDLACompilerInterface:
        """Get compiler instance."""
        if self._compiler is not None:
            return self._compiler

        if self._stub_compiler is None:
            self._stub_compiler = StubNVDLACompiler(self.config)
        return self._stub_compiler

    @property
    def runtime(self) -> NVDLARuntimeInterface:
        """Get runtime instance."""
        if self._runtime is not None:
            return self._runtime

        if self._stub_runtime is None:
            self._stub_runtime = StubNVDLARuntime(self.config)
        return self._stub_runtime

    def is_available(self) -> bool:
        """Check if NVDLA target is available.

        Returns True if ONNX is available for model parsing.
        Full VP functionality requires additional setup.
        """
        try:
            import onnx  # noqa: F401

            return True
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return NVDLA target capabilities."""
        precision_map = {
            NVDLAPrecision.FP16: "fp16",
            NVDLAPrecision.INT8: "int8",
        }

        supported = [precision_map[p] for p in self.config.supported_precisions]

        # Check for real VP installation
        vp_available = self._check_vp_installation()

        return {
            "name": self.name,
            "variant": self.config.variant.value,
            "supported_precisions": supported,
            "supports_calibration": True,
            "native_ops": self.config.native_ops,
            "peak_tops_int8": self.config.peak_tops_int8,
            "peak_tflops_fp16": self.config.peak_tflops_fp16,
            "tdp_watts": self.config.tdp_watts,
            "output_format": ".nvdla",
            "vp_available": vp_available,
            "simulation_mode": self.config.simulation_mode.value,
            "requires_caffe_conversion": True,
        }

    def _check_vp_installation(self) -> bool:
        """Check if Virtual Platform is installed."""
        if self.config.vp_install_path is None:
            return False

        vp_binary = self.config.vp_install_path / "bin" / "aarch64_toplevel"
        return vp_binary.exists()

    def deploy(
        self,
        model: Path,
        precision: DeploymentPrecision,
        output_path: Path,
        input_shape: tuple[int, ...],
        calibration: CalibrationConfig | None = None,
        **kwargs,
    ) -> DeploymentArtifact:
        """Deploy model to NVDLA loadable format.

        Workflow:
        1. Convert ONNX to Caffe format (if needed)
        2. Run NVDLA compiler to generate loadable
        3. Return deployment artifact
        """
        if not self.is_available():
            raise RuntimeError("NVDLA target not available. Install ONNX: pip install onnx")

        # Map precision
        nvdla_precision = self._map_precision(precision)
        if nvdla_precision not in self.config.supported_precisions:
            raise ValueError(
                f"Precision {precision.value} not supported. "
                f"NVDLA supports: {[p.value for p in self.config.supported_precisions]}"
            )

        # INT8 requires calibration
        if precision == DeploymentPrecision.INT8 and calibration is None:
            raise ValueError("INT8 precision requires calibration data")

        # Validate model
        model_path = Path(model)
        issues = self.compiler.validate_model(model_path, self.config)
        if issues:
            raise ValueError(f"Model compatibility issues: {'; '.join(issues)}")

        # Compile
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        loadable = self.compiler.compile(
            model_path=model_path,
            config=self.config,
            precision=nvdla_precision,
            output_path=output_path.with_suffix(".nvdla"),
            calibration_data=Path(calibration.data_path) if calibration else None,
        )

        # Determine output shape
        output_shape = None
        if loadable.output_shapes:
            output_shape = loadable.output_shapes[0]

        return DeploymentArtifact(
            engine_path=loadable.path,
            precision=precision,
            target=self.name,
            input_shape=input_shape,
            output_shape=output_shape,
            size_bytes=loadable.file_size_bytes,
            metadata={
                "variant": self.config.variant.value,
                "compiler_version": loadable.compiler_version,
                "estimated_cycles": loadable.estimated_cycles,
                "layer_count": loadable.layer_count,
                "weight_bytes": loadable.weight_bytes,
            },
        )

    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate NVDLA deployment against baseline.

        Note: Full VP validation is very slow. The stub runtime
        provides approximate validation for development.
        """
        if not self.is_available():
            raise RuntimeError("NVDLA target not available")

        errors = []

        # Load NVDLA loadable
        loadable = self._load_loadable(deployed_artifact.engine_path)
        self.runtime.load(loadable)

        # Load baseline (ONNX Runtime)
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

            # NVDLA inference
            input_specs = self.runtime.get_input_specs()
            input_name = input_specs[0][0] if input_specs else "input"
            input_dict = {input_name: input_data}

            start = time.perf_counter()
            result = self.runtime.execute(input_dict)
            nvdla_time = time.perf_counter() - start
            latencies_deployed.append(nvdla_time)

            if not result.success:
                errors.append(f"NVDLA execution failed: {result.error_message}")
                continue

            # Get output
            if result.outputs:
                output_name = list(result.outputs.keys())[0]
                nvdla_out = result.outputs[output_name]

                # Compare outputs
                if config.compare_outputs:
                    diff = np.abs(baseline_out - nvdla_out).max()
                    max_diff = max(max_diff, float(diff))

            samples_compared += 1

        # Unload
        self.runtime.unload()

        # Calculate metrics
        baseline_latency = float(np.mean(latencies_baseline) * 1000) if latencies_baseline else 0.0
        deployed_latency = float(np.mean(latencies_deployed) * 1000) if latencies_deployed else 0.0
        speedup = baseline_latency / deployed_latency if deployed_latency > 0 else 0.0

        # Accuracy check
        accuracy_passed = max_diff < (config.tolerance_percent / 100.0)

        # Power validation
        power_metrics = None
        power_passed = True
        if config.power_validation is not None:
            power_metrics, power_passed = self._validate_power(loadable, config.power_validation)

        return ValidationResult(
            passed=accuracy_passed and power_passed and not errors,
            baseline_latency_ms=baseline_latency,
            deployed_latency_ms=deployed_latency,
            speedup=speedup,
            max_output_diff=max_diff,
            samples_compared=samples_compared,
            power=power_metrics,
            power_validation_passed=power_passed if power_metrics else None,
            errors=errors,
        )

    def _map_precision(self, precision: DeploymentPrecision) -> NVDLAPrecision:
        """Map deployment precision to NVDLA precision."""
        mapping = {
            DeploymentPrecision.FP32: NVDLAPrecision.FP16,  # NVDLA max is FP16
            DeploymentPrecision.FP16: NVDLAPrecision.FP16,
            DeploymentPrecision.INT8: NVDLAPrecision.INT8,
        }
        return mapping.get(precision, NVDLAPrecision.FP16)

    def _load_loadable(self, path: Path) -> NVDLALoadable:
        """Load NVDLA loadable from file."""
        import pickle

        # Check for metadata file
        meta_path = path.with_suffix(".meta")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                return pickle.load(f)

        # Create minimal loadable from file
        return NVDLALoadable(
            path=path,
            precision=NVDLAPrecision.FP16,
            file_size_bytes=path.stat().st_size if path.exists() else 0,
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
            for _ in range(config.num_samples):
                yield np.random.randn(*input_shape).astype(np.float32)
        else:
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

    def _validate_power(
        self,
        loadable: NVDLALoadable,
        config: PowerConfig,
    ) -> tuple[PowerMetrics, bool]:
        """Validate power consumption."""
        # Estimate power based on NVDLA variant
        utilization = 0.7
        measured_watts = (
            self.config.tdp_watts * 0.1 + self.config.tdp_watts * 0.9 * utilization  # idle
        )

        within_budget = None
        if config.power_budget_watts is not None:
            within_budget = measured_watts <= config.power_budget_watts

        metrics = PowerMetrics(
            measured_watts=round(measured_watts, 2),
            within_budget=within_budget,
            measurement_method="nvdla-estimate",
        )

        passed = True
        if config.power_budget_watts is not None:
            if measured_watts > config.power_budget_watts:
                passed = False

        return metrics, passed


# =============================================================================
# Stub Implementations
# =============================================================================


class StubNVDLACompiler(NVDLACompilerInterface):
    """Stub compiler for development and testing.

    This compiler creates a minimal NVDLALoadable from an ONNX model
    without invoking the real NVDLA compiler.
    """

    def __init__(self, config: NVDLAConfig):
        self.config = config

    def compile(
        self,
        model_path: Path,
        config: NVDLAConfig,
        precision: NVDLAPrecision,
        output_path: Path,
        calibration_data: Path | None = None,
    ) -> NVDLALoadable:
        """Create stub loadable from ONNX model."""
        import onnx
        import pickle

        model = onnx.load(str(model_path))
        graph = model.graph

        # Get input/output info
        input_names = [inp.name for inp in graph.input]
        input_shapes = [
            tuple(d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim)
            for inp in graph.input
        ]

        output_names = [out.name for out in graph.output]
        output_shapes = [
            tuple(d.dim_value if d.dim_value > 0 else 1 for d in out.type.tensor_type.shape.dim)
            for out in graph.output
        ]

        # Calculate weight size
        weight_bytes = sum(
            int(np.prod(init.dims)) * (1 if precision == NVDLAPrecision.INT8 else 2)
            for init in graph.initializer
        )

        # Create loadable
        loadable = NVDLALoadable(
            path=output_path,
            precision=precision,
            source_model=str(model_path),
            input_names=input_names,
            input_shapes=input_shapes,
            output_names=output_names,
            output_shapes=output_shapes,
            weight_bytes=weight_bytes,
            target_variant=config.variant,
            compiler_version="stub-1.0",
            layer_count=len(graph.node),
            estimated_cycles=len(graph.node) * 10000,
        )

        # Save metadata
        meta_path = output_path.with_suffix(".meta")
        with open(meta_path, "wb") as f:
            pickle.dump(loadable, f)

        # Create empty loadable file (stub)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"NVDLA_STUB_LOADABLE")
        loadable.file_size_bytes = output_path.stat().st_size

        return loadable

    def convert_onnx_to_caffe(
        self,
        onnx_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Stub ONNX to Caffe conversion."""
        # In real implementation, use onnx2caffe or similar
        prototxt = output_dir / f"{onnx_path.stem}.prototxt"
        caffemodel = output_dir / f"{onnx_path.stem}.caffemodel"

        # Create stub files
        output_dir.mkdir(parents=True, exist_ok=True)
        prototxt.write_text("# Stub prototxt")
        caffemodel.write_bytes(b"")

        return prototxt, caffemodel

    def validate_model(
        self,
        model_path: Path,
        config: NVDLAConfig,
    ) -> list[str]:
        """Validate model compatibility."""
        import onnx

        issues = []

        try:
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
        except Exception as e:
            issues.append(f"Invalid ONNX model: {e}")
            return issues

        # Check for unsupported ops
        unsupported = []
        for node in model.graph.node:
            if node.op_type not in config.native_ops:
                unsupported.append(node.op_type)

        if unsupported:
            issues.append(f"Unsupported operators: {', '.join(set(unsupported))}")

        return issues

    def get_supported_ops(self) -> list[str]:
        """Get supported operators."""
        return self.config.native_ops


class StubNVDLARuntime(NVDLARuntimeInterface):
    """Stub runtime for development and testing.

    This runtime simulates NVDLA execution using ONNX Runtime
    as the backend, without invoking the actual Virtual Platform.
    """

    def __init__(self, config: NVDLAConfig):
        self.config = config
        self._loadable: NVDLALoadable | None = None
        self._onnx_session = None

    def load(self, loadable: NVDLALoadable) -> None:
        """Load NVDLA loadable."""
        self._loadable = loadable

        # Try to load original ONNX model for execution
        if loadable.source_model:
            try:
                import onnxruntime as ort

                self._onnx_session = ort.InferenceSession(
                    loadable.source_model,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass

    def execute(
        self,
        inputs: dict[str, np.ndarray],
        mode: SimulationMode = SimulationMode.SYSTEMC,
    ) -> NVDLAExecutionResult:
        """Execute with stub runtime."""
        if self._loadable is None:
            return NVDLAExecutionResult(
                success=False,
                error_message="No loadable loaded",
            )

        start_time = time.perf_counter()

        try:
            if self._onnx_session is not None:
                # Use ONNX Runtime
                ort_inputs = {}
                for name, data in inputs.items():
                    ort_inputs[self._onnx_session.get_inputs()[0].name] = data

                outputs_list = self._onnx_session.run(None, ort_inputs)
                outputs = {
                    self._onnx_session.get_outputs()[i].name: out
                    for i, out in enumerate(outputs_list)
                }
            else:
                # Generate dummy outputs
                outputs = {}
                for name, shape in zip(self._loadable.output_names, self._loadable.output_shapes):
                    outputs[name] = np.zeros(shape, dtype=np.float32)

            elapsed_us = (time.perf_counter() - start_time) * 1e6

            # Simulate metrics
            cycles = self._loadable.estimated_cycles
            metrics = NVDLAPerformanceMetrics(
                total_cycles=cycles,
                total_time_us=elapsed_us,
                conv_cycles=int(cycles * 0.7),
                sdp_cycles=int(cycles * 0.1),
                pdp_cycles=int(cycles * 0.1),
                average_power_watts=self.config.tdp_watts * 0.7,
            )

            return NVDLAExecutionResult(
                success=True,
                outputs=outputs,
                metrics=metrics,
                mode=mode,
            )

        except Exception as e:
            return NVDLAExecutionResult(
                success=False,
                error_message=str(e),
            )

    def get_input_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get input specifications."""
        if self._loadable is None:
            return []

        precision_str = self._loadable.precision.value
        return [
            (name, shape, precision_str)
            for name, shape in zip(self._loadable.input_names, self._loadable.input_shapes)
        ]

    def get_output_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get output specifications."""
        if self._loadable is None:
            return []

        precision_str = self._loadable.precision.value
        return [
            (name, shape, precision_str)
            for name, shape in zip(self._loadable.output_names, self._loadable.output_shapes)
        ]

    def unload(self) -> None:
        """Unload current loadable."""
        self._loadable = None
        self._onnx_session = None
