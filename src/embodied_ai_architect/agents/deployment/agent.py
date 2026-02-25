"""Deployment agent for edge/embedded model deployment."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..base import AgentResult, BaseAgent
from .models import (
    CalibrationConfig,
    DeploymentPrecision,
    DeploymentResult,
    DeploymentStatus,
    ValidationConfig,
)
from .targets.base import DeploymentTarget


class DeploymentAgent(BaseAgent):
    """Agent that deploys models to edge/embedded targets.

    The DeploymentAgent orchestrates:
    1. Model export (PyTorch -> ONNX)
    2. Quantization/optimization for target
    3. Validation against baseline

    Supports multiple deployment targets (Jetson, Coral, etc.)
    through a plugin architecture.
    """

    def __init__(self, targets: list[DeploymentTarget] | None = None):
        """Initialize deployment agent.

        Args:
            targets: List of deployment targets. If None, auto-discovers available.
        """
        super().__init__(name="Deployment")
        self.targets: dict[str, DeploymentTarget] = {}

        if targets is None:
            self._auto_discover_targets()
        else:
            for target in targets:
                self._register_target(target)

    def _auto_discover_targets(self) -> None:
        """Discover and register available deployment targets."""
        # Import targets with optional dependencies
        try:
            from .targets.jetson import JetsonTarget

            target = JetsonTarget()
            if target.is_available():
                self._register_target(target)
        except ImportError:
            pass

        try:
            from .targets.openvino import OpenVINOTarget

            target = OpenVINOTarget()
            if target.is_available():
                self._register_target(target)
        except ImportError:
            pass

        try:
            from .targets.coral import CoralTarget

            target = CoralTarget()
            if target.is_available():
                self._register_target(target)
        except ImportError:
            pass

        try:
            from .targets.kpu import StillwaterKPUTarget

            target = StillwaterKPUTarget()
            if target.is_available():
                self._register_target(target)
        except ImportError:
            pass

        try:
            from .targets.nvdla import NVDLATarget

            target = NVDLATarget()
            if target.is_available():
                self._register_target(target)
        except ImportError:
            pass

    def _register_target(self, target: DeploymentTarget) -> None:
        """Register a deployment target."""
        if target.is_available():
            self.targets[target.name] = target
            print(f"  Registered deployment target: {target.name}")
        else:
            print(f"  Skipped unavailable target: {target.name}")

    def execute(self, input_data: dict[str, Any]) -> AgentResult:
        """Execute model deployment.

        Args:
            input_data: Dictionary with keys:
                - 'model': PyTorch model or path to model file
                - 'target': Target name (e.g., 'jetson')
                - 'precision': Precision level ('fp32', 'fp16', 'int8')
                - 'input_shape': Model input shape tuple
                - 'calibration_data': Path to calibration dataset (for INT8)
                - 'calibration_samples': Number of calibration samples
                - 'calibration_preprocessing': Preprocessing type ('imagenet', 'yolo', etc.)
                - 'test_data': Path to test dataset for validation
                - 'validation_samples': Number of validation samples
                - 'accuracy_tolerance': Max accuracy drop tolerance (percent)
                - 'output_dir': Output directory for artifacts

        Returns:
            AgentResult with deployment results
        """
        logs: list[str] = []

        try:
            # Extract inputs
            model = input_data.get("model")
            target_name = input_data.get("target", "jetson")
            precision_str = input_data.get("precision", "int8")
            input_shape = input_data.get("input_shape")

            # Validate inputs
            if model is None:
                return AgentResult(success=False, data={}, error="No model provided")

            if input_shape is None:
                return AgentResult(success=False, data={}, error="No input_shape provided")

            if target_name not in self.targets:
                available = list(self.targets.keys())
                return AgentResult(
                    success=False,
                    data={},
                    error=f"Target '{target_name}' not available. Available: {available}",
                )

            target = self.targets[target_name]
            precision = DeploymentPrecision(precision_str)

            # Build calibration config
            calibration = None
            if precision == DeploymentPrecision.INT8:
                cal_data = input_data.get("calibration_data")
                if cal_data is None:
                    return AgentResult(
                        success=False,
                        data={},
                        error="INT8 precision requires calibration_data",
                    )
                calibration = CalibrationConfig(
                    data_path=Path(cal_data),
                    num_samples=input_data.get("calibration_samples", 100),
                    batch_size=input_data.get("calibration_batch_size", 1),
                    input_shape=tuple(input_shape),
                    preprocessing=input_data.get("calibration_preprocessing", "imagenet"),
                )

            # Build validation config
            validation_config = None
            test_data = input_data.get("test_data")
            if test_data:
                validation_config = ValidationConfig(
                    test_data_path=Path(test_data),
                    num_samples=input_data.get("validation_samples", 100),
                    tolerance_percent=input_data.get("accuracy_tolerance", 1.0),
                )

            # Determine output path
            output_dir = Path(input_data.get("output_dir", "./deployments"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Export to ONNX if needed
            logs.append("Step 1: Preparing ONNX model...")
            onnx_path = self._ensure_onnx(model, tuple(input_shape), output_dir, logs)

            # Step 2: Deploy to target
            logs.append(f"Step 2: Deploying to {target_name} with {precision.value} precision...")
            engine_path = output_dir / f"model_{precision.value}.engine"

            artifact = target.deploy(
                model=onnx_path,
                precision=precision,
                output_path=engine_path,
                input_shape=tuple(input_shape),
                calibration=calibration,
            )
            logs.append(f"  Engine saved: {artifact.engine_path}")
            logs.append(f"  Engine size: {artifact.size_bytes / 1024 / 1024:.2f} MB")

            # Step 3: Validate if requested
            validation_result = None
            if validation_config:
                logs.append("Step 3: Running validation...")
                validation_result = target.validate(
                    deployed_artifact=artifact,
                    baseline_model=onnx_path,
                    config=validation_config,
                )
                logs.append(f"  Validation passed: {validation_result.passed}")
                if validation_result.speedup:
                    logs.append(f"  Speedup: {validation_result.speedup:.2f}x")
                if validation_result.max_output_diff is not None:
                    logs.append(f"  Max output diff: {validation_result.max_output_diff:.6f}")

            # Build result
            result = DeploymentResult(
                success=True,
                status=DeploymentStatus.COMPLETED,
                artifact=artifact,
                validation=validation_result,
                logs=logs,
            )

            return AgentResult(
                success=True,
                data=result.model_dump(mode="json"),
                metadata={
                    "agent": self.name,
                    "target": target_name,
                    "precision": precision.value,
                },
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={"logs": logs},
                error=f"Deployment failed: {str(e)}",
            )

    def _ensure_onnx(
        self,
        model: nn.Module | str | Path,
        input_shape: tuple,
        output_dir: Path,
        logs: list[str],
    ) -> Path:
        """Ensure model is in ONNX format, export if needed."""
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if model_path.suffix == ".onnx":
                logs.append(f"  Using existing ONNX model: {model_path}")
                return model_path

            # Load PyTorch model
            logs.append(f"  Loading PyTorch model: {model_path}")
            model = torch.load(model_path, map_location="cpu", weights_only=False)

        # Handle state_dict vs full model
        if isinstance(model, dict):
            raise ValueError("Model is a state_dict. Please provide a full model or ONNX file.")

        # Export to ONNX
        logs.append("  Exporting to ONNX...")
        onnx_path = output_dir / "model.onnx"

        model.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

        logs.append(f"  ONNX exported: {onnx_path}")
        return onnx_path

    def list_targets(self) -> list[str]:
        """List available deployment targets."""
        return list(self.targets.keys())

    def get_target_capabilities(self, target_name: str) -> dict[str, Any] | None:
        """Get capabilities of a specific target."""
        target = self.targets.get(target_name)
        if target:
            return target.get_capabilities()
        return None
