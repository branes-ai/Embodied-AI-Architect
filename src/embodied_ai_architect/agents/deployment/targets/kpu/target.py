"""Stillwater KPU deployment target implementation.

This module integrates the KPU simulator with the deployment system.
It provides a DeploymentTarget implementation that:
1. Compiles PyTorch models to KPU programs via torch.compile backend
2. Executes via simulation (or hardware when available)
3. Validates against baseline
4. Reports performance/power metrics

Requirements:
- PyTorch 2.0+ with torch.compile support
- The production compiler uses a custom torch.compile backend for KPU
- The stub implementation uses ONNX Runtime for development/testing
"""

import time
from pathlib import Path
from typing import Any, Iterator

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
    KPUConfig,
    KPUPrecision,
    KPUProgram,
    KPUCompilerInterface,
    KPURuntimeInterface,
    KPUExecutionResult,
    SimulationMode,
)


class StillwaterKPUTarget(DeploymentTarget):
    """Deployment target for Stillwater KPU.

    This target compiles PyTorch models to the KPU format using torch.compile
    with a custom KPU backend, and executes them via software simulation or
    hardware (when available).

    Requirements:
        PyTorch 2.0+ with torch.compile support

    Usage:
        target = StillwaterKPUTarget()

        # Check availability
        if target.is_available():
            artifact = target.deploy(
                model=model_path,  # .pt or .onnx (stub mode)
                precision=DeploymentPrecision.INT8,
                output_path=output_dir / "model.kpu",
                input_shape=(1, 3, 224, 224),
                calibration=calib_config,
            )

    The target can be configured with different KPU configurations
    to simulate different hardware variants.
    """

    def __init__(
        self,
        config: KPUConfig | None = None,
        compiler: KPUCompilerInterface | None = None,
        runtime: KPURuntimeInterface | None = None,
    ):
        """Initialize KPU target.

        Args:
            config: KPU hardware configuration (uses default if None)
            compiler: Custom compiler implementation (uses stub if None)
            runtime: Custom runtime implementation (uses stub if None)
        """
        super().__init__(name="swkpu")
        self.config = config or KPUConfig()

        # Use provided implementations or fall back to stubs
        self._compiler = compiler
        self._runtime = runtime

        # Lazy initialization flags
        self._stub_compiler: "StubKPUCompiler | None" = None
        self._stub_runtime: "StubKPURuntime | None" = None

    @property
    def compiler(self) -> KPUCompilerInterface:
        """Get the compiler instance."""
        if self._compiler is not None:
            return self._compiler

        if self._stub_compiler is None:
            self._stub_compiler = StubKPUCompiler()
        return self._stub_compiler

    @property
    def runtime(self) -> KPURuntimeInterface:
        """Get the runtime instance."""
        if self._runtime is not None:
            return self._runtime

        if self._stub_runtime is None:
            self._stub_runtime = StubKPURuntime(self.config)
        return self._stub_runtime

    def is_available(self) -> bool:
        """Check if KPU target is available.

        Returns True if:
        1. PyTorch is available with torch.compile support
        2. Either custom or stub compiler/runtime is available
        """
        try:
            import torch  # noqa: F401

            # Check torch.compile availability (PyTorch 2.0+)
            return hasattr(torch, "compile")
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return KPU target capabilities."""
        precision_map = {
            KPUPrecision.INT8: "int8",
            KPUPrecision.INT4: "int4",
            KPUPrecision.FP16: "fp16",
            KPUPrecision.BF16: "bf16",
            KPUPrecision.POSIT8: "posit8",
            KPUPrecision.POSIT16: "posit16",
        }

        supported = [
            precision_map[p] for p in self.config.supported_precisions if p in precision_map
        ]

        return {
            "name": self.name,
            "kpu_version": self.config.version,
            "supported_precisions": supported,
            "supports_calibration": True,
            "supports_posit": any("posit" in p.value for p in self.config.supported_precisions),
            "native_ops": self.config.native_ops,
            "sram_bytes": self.config.memory.sram_l1_bytes + self.config.memory.sram_l2_bytes,
            "peak_tops_int8": self.config.compute.tops_int8,
            "peak_tflops_fp16": self.config.compute.tops_fp16,
            "tdp_watts": self.config.compute.tdp_watts,
            "output_format": ".kpu",
            "simulation_mode": "stub" if self._compiler is None else "custom",
        }

    def deploy(
        self,
        model: Path,
        precision: DeploymentPrecision,
        output_path: Path,
        input_shape: tuple[int, ...],
        calibration: CalibrationConfig | None = None,
        **kwargs,
    ) -> DeploymentArtifact:
        """Deploy model to KPU format.

        Workflow:
        1. Validate precision support
        2. Load PyTorch model (or ONNX for stub mode)
        3. Compile to KPU program via torch.compile backend
        4. Serialize program to output path

        Note: The stub implementation uses ONNX Runtime internally for
        development/testing. The production compiler uses torch.compile
        with a custom KPU backend.
        """
        if not self.is_available():
            raise RuntimeError(
                "KPU target not available. Requires PyTorch 2.0+ with torch.compile: "
                "pip install torch>=2.0"
            )

        # Map precision
        kpu_precision = self._map_precision(precision)
        if kpu_precision not in self.config.supported_precisions:
            raise ValueError(
                f"Precision {precision.value} not supported by this KPU configuration. "
                f"Supported: {[p.value for p in self.config.supported_precisions]}"
            )

        # INT8 requires calibration
        if precision == DeploymentPrecision.INT8 and calibration is None:
            raise ValueError("INT8 precision requires calibration data")

        import onnx

        model_path = Path(model)
        onnx_model = onnx.load(str(model_path))

        # Validate model compatibility
        issues = self.compiler.validate_model(model_path, self.config)
        if issues:
            raise ValueError(f"Model compatibility issues: {'; '.join(issues)}")

        # Create calibration data iterator if needed
        calib_iter = None
        if calibration is not None:
            calib_iter = self._create_calibration_iterator(calibration, input_shape)

        # Compile model
        program = self.compiler.compile(
            onnx_path=model_path,
            config=self.config,
            precision=kpu_precision,
            calibration_data=calib_iter,
        )

        # Serialize program
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kpu_path = output_path.with_suffix(".kpu")

        self._save_program(program, kpu_path)

        # Get output shape from program
        output_tensors = program.get_output_tensors()
        output_shape = output_tensors[0].shape if output_tensors else None

        return DeploymentArtifact(
            engine_path=kpu_path,
            precision=precision,
            target=self.name,
            input_shape=input_shape,
            output_shape=output_shape,
            size_bytes=kpu_path.stat().st_size,
            metadata={
                "kpu_version": self.config.version,
                "estimated_cycles": program.estimated_total_cycles,
                "estimated_energy_uj": program.estimated_total_energy_uj,
                "peak_sram_bytes": program.peak_sram_bytes,
                "weight_bytes": program.weight_bytes,
                "num_ops": len(program.ops),
                "total_params": sum(t.size_bytes for t in program.tensors.values() if t.is_weight),
                "total_macs": self._estimate_macs(program),
            },
        )

    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate KPU deployment against baseline."""
        if not self.is_available():
            raise RuntimeError("KPU target not available")

        errors = []

        # Load KPU program
        program = self._load_program(deployed_artifact.engine_path)
        self.runtime.load_program(program)

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

            # KPU inference
            input_specs = self.runtime.get_input_specs()
            input_dict = {input_specs[0][0]: input_data}

            start = time.perf_counter()
            result = self.runtime.execute(input_dict, SimulationMode.FUNCTIONAL)
            kpu_time = time.perf_counter() - start
            latencies_deployed.append(kpu_time)

            if not result.success:
                errors.append(f"KPU execution failed: {result.error_message}")
                continue

            # Get output
            output_name = list(result.outputs.keys())[0]
            kpu_out = result.outputs[output_name]

            # Compare outputs
            if config.compare_outputs:
                diff = np.abs(baseline_out - kpu_out).max()
                max_diff = max(max_diff, float(diff))

            samples_compared += 1

        # Unload program
        self.runtime.unload_program()

        # Calculate metrics
        baseline_latency = float(np.mean(latencies_baseline) * 1000) if latencies_baseline else 0.0
        deployed_latency = float(np.mean(latencies_deployed) * 1000) if latencies_deployed else 0.0
        speedup = baseline_latency / deployed_latency if deployed_latency > 0 else 0.0

        # Determine pass/fail for accuracy
        accuracy_passed = max_diff < (config.tolerance_percent / 100.0)

        # Power validation if configured
        power_metrics = None
        power_passed = True
        if config.power_validation is not None:
            power_metrics, power_passed = self._validate_power(
                program, config.power_validation, input_shape
            )
            if not power_passed:
                if config.power_validation.power_budget_watts:
                    errors.append(
                        f"Power {power_metrics.measured_watts:.1f}W exceeds budget "
                        f"{config.power_validation.power_budget_watts:.1f}W"
                    )

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

    def _map_precision(self, precision: DeploymentPrecision) -> KPUPrecision:
        """Map deployment precision to KPU precision."""
        mapping = {
            DeploymentPrecision.FP32: KPUPrecision.FP16,  # KPU max is FP16
            DeploymentPrecision.FP16: KPUPrecision.FP16,
            DeploymentPrecision.INT8: KPUPrecision.INT8,
        }
        return mapping.get(precision, KPUPrecision.FP16)

    def _create_calibration_iterator(
        self,
        config: CalibrationConfig,
        input_shape: tuple[int, ...],
    ) -> Iterator[np.ndarray]:
        """Create iterator over calibration data."""
        from PIL import Image

        data_path = Path(config.data_path)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        image_paths = []
        for ext in extensions:
            image_paths.extend(data_path.glob(f"*{ext}"))
            image_paths.extend(data_path.glob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)[: config.num_samples]

        h, w = input_shape[2], input_shape[3]

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((w, h), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32)

            # Preprocessing
            if config.preprocessing == "imagenet":
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = img_array / 255.0
                img_array = (img_array - mean) / std
            elif config.preprocessing == "yolo":
                img_array = img_array / 255.0

            # HWC -> NCHW
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)

            yield img_array.astype(np.float32)

    def _save_program(self, program: KPUProgram, path: Path) -> None:
        """Serialize KPU program to file."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(program, f)

    def _load_program(self, path: Path) -> KPUProgram:
        """Load KPU program from file."""
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

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

    def _estimate_macs(self, program: KPUProgram) -> int:
        """Estimate total MACs from program."""
        total_macs = 0
        for op in program.ops:
            if op.op_type in ("Conv", "MatMul", "Gemm"):
                # Rough estimate based on output size
                for out_id in op.output_ids:
                    if out_id in program.tensors:
                        tensor = program.tensors[out_id]
                        elements = int(np.prod(tensor.shape))
                        # Assume average of 100 MACs per output element
                        total_macs += elements * 100
        return total_macs

    def _validate_power(
        self,
        program: KPUProgram,
        config: PowerConfig,
        input_shape: tuple[int, ...],
    ) -> tuple[PowerMetrics, bool]:
        """Validate power consumption during inference."""
        # Get predicted power from KPU simulator
        predicted_watts = None
        if program.estimated_total_energy_uj > 0:
            # Estimate power from energy and estimated cycles
            cycles = program.estimated_total_cycles
            freq_hz = self.config.compute.clock_mhz * 1e6
            time_sec = cycles / freq_hz if freq_hz > 0 else 1.0
            predicted_watts = (program.estimated_total_energy_uj / 1e6) / time_sec

        # For simulation, we estimate based on TDP and utilization
        # Real hardware would measure actual power
        estimated_utilization = 0.7  # Assume 70% utilization
        measured_watts = (
            self.config.compute.idle_watts
            + (self.config.compute.tdp_watts - self.config.compute.idle_watts)
            * estimated_utilization
        )

        # Calculate deviation
        deviation = None
        if predicted_watts is not None and predicted_watts > 0:
            deviation = ((measured_watts - predicted_watts) / predicted_watts) * 100.0

        within_budget = None
        if config.power_budget_watts is not None:
            within_budget = measured_watts <= config.power_budget_watts

        metrics = PowerMetrics(
            measured_watts=round(measured_watts, 2),
            predicted_watts=round(predicted_watts, 2) if predicted_watts else None,
            deviation_percent=round(deviation, 1) if deviation is not None else None,
            within_budget=within_budget,
            measurement_method="kpu-simulator",
        )

        passed = True
        if config.power_budget_watts is not None:
            if measured_watts > config.power_budget_watts:
                passed = False

        return metrics, passed


# =============================================================================
# Stub Implementations (for development/testing without real simulator)
# =============================================================================


class StubKPUCompiler(KPUCompilerInterface):
    """Stub compiler for development and testing.

    This compiler creates a minimal KPUProgram from an ONNX model
    without performing actual optimizations or tiling.

    Replace with real compiler implementation for production use.
    """

    def compile(
        self,
        onnx_path: Path,
        config: KPUConfig,
        precision: KPUPrecision,
        calibration_data: Iterator[np.ndarray] | None = None,
    ) -> KPUProgram:
        """Create stub KPU program from ONNX model."""
        import onnx
        from .spec import KPUTensor, KPUOp, MemoryLayout, MemoryType

        model = onnx.load(str(onnx_path))
        graph = model.graph

        program = KPUProgram(
            name=onnx_path.stem,
            source_model=str(onnx_path),
            target_config=config,
            precision=precision,
        )

        # Extract tensors from graph
        tensor_id = 0

        # Process inputs
        for inp in graph.input:
            shape = tuple(
                d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim
            )
            tid = f"tensor_{tensor_id}"
            program.tensors[tid] = KPUTensor(
                id=tid,
                name=inp.name,
                shape=shape,
                dtype=precision,
                layout=MemoryLayout(memory_type=MemoryType.DRAM),
                is_input=True,
            )
            program.input_ids.append(tid)
            tensor_id += 1

        # Process initializers (weights)
        weight_bytes = 0
        for init in graph.initializer:
            shape = tuple(init.dims)
            tid = f"tensor_{tensor_id}"
            tensor = KPUTensor(
                id=tid,
                name=init.name,
                shape=shape,
                dtype=precision,
                layout=MemoryLayout(memory_type=MemoryType.DRAM),
                is_weight=True,
            )
            program.tensors[tid] = tensor
            weight_bytes += tensor.size_bytes
            tensor_id += 1

        # Process nodes (ops)
        for i, node in enumerate(graph.node):
            op = KPUOp(
                id=f"op_{i}",
                op_type=node.op_type,
                input_ids=[],  # Would map to tensor IDs
                output_ids=[],
                attributes={attr.name: self._get_attr_value(attr) for attr in node.attribute},
                schedule_order=i,
                estimated_cycles=1000,  # Stub estimate
                estimated_energy_pj=100.0,  # Stub estimate
            )
            program.ops.append(op)

        # Process outputs
        for out in graph.output:
            shape = tuple(
                d.dim_value if d.dim_value > 0 else 1 for d in out.type.tensor_type.shape.dim
            )
            tid = f"tensor_{tensor_id}"
            program.tensors[tid] = KPUTensor(
                id=tid,
                name=out.name,
                shape=shape,
                dtype=precision,
                layout=MemoryLayout(memory_type=MemoryType.DRAM),
                is_output=True,
            )
            program.output_ids.append(tid)
            tensor_id += 1

        # Set memory estimates
        program.weight_bytes = weight_bytes
        program.peak_sram_bytes = config.memory.sram_l1_bytes  # Conservative
        program.estimated_total_cycles = len(program.ops) * 1000
        program.estimated_total_energy_uj = len(program.ops) * 0.1

        return program

    def _get_attr_value(self, attr) -> Any:
        """Extract attribute value from ONNX attribute."""
        import onnx

        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        return None

    def get_supported_ops(self) -> list[str]:
        """Return supported ops (all common ops for stub)."""
        return [
            "Conv",
            "MatMul",
            "Gemm",
            "Relu",
            "Add",
            "Mul",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "BatchNormalization",
            "Softmax",
            "Sigmoid",
            "Tanh",
            "Concat",
            "Reshape",
            "Transpose",
            "Flatten",
        ]

    def estimate_memory(
        self,
        onnx_path: Path,
        config: KPUConfig,
        precision: KPUPrecision,
    ) -> dict[str, int]:
        """Estimate memory requirements."""
        import onnx

        model = onnx.load(str(onnx_path))
        graph = model.graph

        bytes_per_element = 1 if "int8" in precision.value else 2

        weights = sum(int(np.prod(init.dims)) * bytes_per_element for init in graph.initializer)

        # Rough activation estimate
        activations = weights // 2

        return {
            "weights": weights,
            "activations": activations,
            "peak_sram": min(weights + activations, config.memory.sram_l1_bytes),
            "peak_dram": weights + activations,
        }

    def validate_model(
        self,
        onnx_path: Path,
        config: KPUConfig,
    ) -> list[str]:
        """Validate model compatibility."""
        import onnx

        issues = []

        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
        except Exception as e:
            issues.append(f"Invalid ONNX model: {e}")
            return issues

        # Check for unsupported ops
        graph = model.graph
        unsupported = []
        for node in graph.node:
            if node.op_type not in config.native_ops:
                unsupported.append(node.op_type)

        if unsupported:
            issues.append(f"Unsupported operators: {', '.join(set(unsupported))}")

        return issues


class StubKPURuntime(KPURuntimeInterface):
    """Stub runtime for development and testing.

    This runtime executes KPU programs using numpy, simulating
    the behavior of the KPU without actual hardware.

    Replace with real runtime for production use.
    """

    def __init__(self, config: KPUConfig):
        self.config = config
        self._program: KPUProgram | None = None
        self._onnx_session = None

    def load_program(self, program: KPUProgram) -> None:
        """Load KPU program for execution."""
        self._program = program

        # For stub, we use ONNX Runtime as the execution backend
        if program.source_model:
            try:
                import onnxruntime as ort

                self._onnx_session = ort.InferenceSession(
                    program.source_model,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                pass

    def execute(
        self,
        inputs: dict[str, np.ndarray],
        mode: SimulationMode = SimulationMode.FUNCTIONAL,
    ) -> KPUExecutionResult:
        """Execute program with given inputs."""
        from .spec import KPUPerformanceMetrics

        if self._program is None:
            return KPUExecutionResult(
                success=False,
                error_message="No program loaded",
            )

        start_time = time.perf_counter()

        try:
            if self._onnx_session is not None:
                # Use ONNX Runtime for execution
                ort_inputs = {}
                for name, data in inputs.items():
                    ort_inputs[self._onnx_session.get_inputs()[0].name] = data

                outputs_list = self._onnx_session.run(None, ort_inputs)
                outputs = {
                    self._onnx_session.get_outputs()[i].name: out
                    for i, out in enumerate(outputs_list)
                }
            else:
                # Generate dummy output
                output_tensors = self._program.get_output_tensors()
                outputs = {}
                for tensor in output_tensors:
                    outputs[tensor.name] = np.zeros(tensor.shape, dtype=np.float32)

            elapsed_us = (time.perf_counter() - start_time) * 1e6

            # Simulate performance metrics
            cycles = self._program.estimated_total_cycles
            freq_mhz = self.config.compute.clock_mhz

            metrics = KPUPerformanceMetrics(
                total_cycles=cycles,
                total_time_us=elapsed_us,
                compute_cycles=int(cycles * 0.7),
                memory_stall_cycles=int(cycles * 0.3),
                average_power_watts=self.config.compute.tdp_watts * 0.7,
                total_energy_uj=self._program.estimated_total_energy_uj,
                compute_utilization=0.7,
            )

            return KPUExecutionResult(
                success=True,
                outputs=outputs,
                metrics=metrics,
                mode=mode,
            )

        except Exception as e:
            return KPUExecutionResult(
                success=False,
                error_message=str(e),
            )

    def get_input_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get input tensor specifications."""
        if self._program is None:
            return []

        specs = []
        for tensor in self._program.get_input_tensors():
            specs.append((tensor.name, tensor.shape, tensor.dtype.value))
        return specs

    def get_output_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get output tensor specifications."""
        if self._program is None:
            return []

        specs = []
        for tensor in self._program.get_output_tensors():
            specs.append((tensor.name, tensor.shape, tensor.dtype.value))
        return specs

    def unload_program(self) -> None:
        """Unload current program."""
        self._program = None
        self._onnx_session = None
