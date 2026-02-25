"""Base class for deployment targets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn

from ..models import (
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    PowerConfig,
    PowerMetrics,
    ValidationConfig,
    ValidationResult,
)
from ..power import get_power_monitor


class DeploymentTarget(ABC):
    """Abstract base class for deployment targets.

    Implementations handle specific deployment platforms like
    Jetson (TensorRT), Coral (Edge TPU), OpenVINO, etc.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this target is available on the current system.

        Returns:
            True if deployment tools are installed and accessible
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return information about target capabilities.

        Returns:
            Dictionary with supported precisions, features, etc.
        """
        pass

    @abstractmethod
    def deploy(
        self,
        model: nn.Module | Path,
        precision: DeploymentPrecision,
        output_path: Path,
        input_shape: tuple[int, ...],
        calibration: CalibrationConfig | None = None,
        **kwargs,
    ) -> DeploymentArtifact:
        """Deploy a model to this target.

        Args:
            model: PyTorch model or path to ONNX model
            precision: Target precision (fp32, fp16, int8)
            output_path: Path for output engine file
            input_shape: Model input shape
            calibration: Calibration config for INT8
            **kwargs: Target-specific options

        Returns:
            DeploymentArtifact with deployed model info

        Raises:
            ValueError: If precision not supported or calibration missing for INT8
            RuntimeError: If deployment fails
        """
        pass

    @abstractmethod
    def validate(
        self,
        deployed_artifact: DeploymentArtifact,
        baseline_model: nn.Module | Path,
        config: ValidationConfig,
    ) -> ValidationResult:
        """Validate deployed model against baseline.

        Args:
            deployed_artifact: The deployed model artifact
            baseline_model: Original model for comparison
            config: Validation configuration

        Returns:
            ValidationResult with accuracy/performance comparison
        """
        pass

    def supports_precision(self, precision: DeploymentPrecision) -> bool:
        """Check if target supports a precision level."""
        caps = self.get_capabilities()
        return precision.value in caps.get("supported_precisions", [])

    def measure_power(
        self,
        workload: Callable[[], None],
        config: PowerConfig,
        predicted_watts: float | None = None,
    ) -> PowerMetrics | None:
        """Measure power during workload execution.

        Args:
            workload: Function to call repeatedly during measurement
            config: Power measurement configuration
            predicted_watts: Optional predicted power for comparison

        Returns:
            PowerMetrics or None if power monitoring unavailable
        """
        if not config.enabled:
            return None

        monitor = get_power_monitor()
        if monitor is None:
            return None

        measurement = monitor.measure_during(
            workload=workload,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
        )

        mean_watts = measurement.mean_watts
        if mean_watts == 0.0:
            return None

        # Calculate metrics
        deviation = None
        if predicted_watts is not None and predicted_watts > 0:
            deviation = ((mean_watts - predicted_watts) / predicted_watts) * 100.0

        within_budget = None
        if config.power_budget_watts is not None:
            within_budget = mean_watts <= config.power_budget_watts

        # Energy per inference
        energy_per_inference = None
        inferences_per_joule = None
        if measurement.duration_sec > 0 and config.measurement_iterations > 0:
            total_energy_joules = mean_watts * measurement.duration_sec
            energy_per_inference = (
                total_energy_joules / config.measurement_iterations
            ) * 1000  # mJ
            if total_energy_joules > 0:
                inferences_per_joule = config.measurement_iterations / total_energy_joules

        return PowerMetrics(
            measured_watts=round(mean_watts, 2),
            predicted_watts=round(predicted_watts, 2) if predicted_watts else None,
            deviation_percent=round(deviation, 1) if deviation is not None else None,
            within_budget=within_budget,
            energy_per_inference_mj=(
                round(energy_per_inference, 3) if energy_per_inference else None
            ),
            inferences_per_joule=round(inferences_per_joule, 2) if inferences_per_joule else None,
            measurement_method=monitor.name,
            gpu_power_watts=measurement.mean_gpu_watts,
            cpu_power_watts=measurement.mean_cpu_watts,
            total_power_watts=mean_watts,
        )

    def validate_power_result(
        self,
        power_metrics: PowerMetrics | None,
        config: PowerConfig,
    ) -> bool:
        """Check if power validation passed.

        Args:
            power_metrics: Measured power metrics
            config: Power validation configuration

        Returns:
            True if power validation passed or was not required
        """
        if power_metrics is None:
            return True  # No measurement = not required

        # Check budget constraint
        if config.power_budget_watts is not None:
            if power_metrics.measured_watts > config.power_budget_watts:
                return False

        # Check prediction tolerance
        if power_metrics.predicted_watts is not None:
            if abs(power_metrics.deviation_percent or 0) > config.tolerance_percent:
                return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
