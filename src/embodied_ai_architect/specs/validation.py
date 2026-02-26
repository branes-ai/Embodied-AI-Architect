"""Cross-subsystem consistency validation for specs.

Checks for logical inconsistencies between subsystems, such as
power budget vs compute TDP, perception latency vs constraints, etc.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .models import PlatformType, SafetyLevel, SystemSpec


class Severity(str, Enum):
    """Validation issue severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """A single validation issue found in a spec."""

    severity: Severity = Field(description="Issue severity")
    path: str = Field(description="Relevant spec path(s)")
    message: str = Field(description="Human-readable description of the issue")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")


def validate_spec(spec: SystemSpec) -> list[ValidationIssue]:
    """Run all cross-subsystem consistency checks.

    Returns a list of ValidationIssue objects, sorted by severity.
    """
    issues: list[ValidationIssue] = []

    _check_power_vs_compute(spec, issues)
    _check_perception_latency(spec, issues)
    _check_platform_implications(spec, issues)
    _check_safety_redundancy(spec, issues)
    _check_mission_duration(spec, issues)
    _check_comms_bandwidth(spec, issues)
    _check_autonomy_sensors(spec, issues)

    # Sort: errors first, then warnings, then info
    severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    issues.sort(key=lambda i: severity_order[i.severity])
    return issues


def _check_power_vs_compute(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check that compute TDP fits within power budget."""
    if spec.compute and spec.power:
        tdp = spec.compute.max_tdp_watts
        compute_budget = spec.power.compute_power_watts
        total_budget = spec.power.power_budget_watts

        if tdp and compute_budget and tdp > compute_budget:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path="/compute/max_tdp_watts, /power/compute_power_watts",
                    message=(
                        f"Compute TDP ({tdp}W) exceeds compute power budget ({compute_budget}W)"
                    ),
                    suggestion=(
                        f"Increase /power/compute_power_watts to >= {tdp} "
                        "or choose a lower-TDP compute option"
                    ),
                )
            )

        if tdp and total_budget and tdp > total_budget:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path="/compute/max_tdp_watts, /power/power_budget_watts",
                    message=(
                        f"Compute TDP ({tdp}W) exceeds total power budget ({total_budget}W)"
                    ),
                    suggestion="Choose a lower-TDP compute platform or increase power budget",
                )
            )


def _check_perception_latency(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check perception latency vs FPS consistency."""
    if spec.perception:
        latency = spec.perception.max_latency_ms
        fps = spec.perception.min_fps

        if latency and fps:
            frame_time = 1000.0 / fps
            if latency > frame_time:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        path="/perception/max_latency_ms, /perception/min_fps",
                        message=(
                            f"Max latency ({latency}ms) exceeds frame time "
                            f"({frame_time:.1f}ms at {fps}fps). "
                            "Pipeline will drop frames or lag."
                        ),
                        suggestion=(
                            f"Reduce max_latency_ms to <= {frame_time:.1f} "
                            f"or lower min_fps"
                        ),
                    )
                )


def _check_platform_implications(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check platform type implies reasonable constraints."""
    if not spec.platform_type:
        return

    if spec.platform_type == PlatformType.DRONE:
        if spec.power and spec.power.power_budget_watts and spec.power.power_budget_watts > 100:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/power/power_budget_watts",
                    message=(
                        f"Drone power budget ({spec.power.power_budget_watts}W) "
                        "is unusually high for a drone platform"
                    ),
                    suggestion="Typical drone compute budgets are 10-30W",
                )
            )
        if spec.actuators and spec.actuators.payload_kg and spec.actuators.payload_kg > 10:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/actuators/payload_kg",
                    message=(
                        f"Drone payload ({spec.actuators.payload_kg}kg) "
                        "is very high for a drone"
                    ),
                    suggestion="Typical drone payloads are 0.1-5kg",
                )
            )
        if spec.safety and not spec.safety.geofencing:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    path="/safety/geofencing",
                    message="Geofencing is recommended for drone platforms",
                    suggestion="Set /safety/geofencing to true",
                )
            )

    if spec.platform_type == PlatformType.FIXED_CAMERA:
        if spec.actuators and spec.actuators.dof and spec.actuators.dof > 2:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    path="/actuators/dof",
                    message="Fixed camera typically has 0-2 DOF (pan/tilt)",
                )
            )


def _check_safety_redundancy(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check that safety level implies appropriate redundancy."""
    if not spec.safety or not spec.safety.level:
        return

    high_safety = spec.safety.level in (SafetyLevel.SIL_2, SafetyLevel.SIL_3, SafetyLevel.SIL_4)

    if high_safety and not spec.safety.redundancy:
        issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                path="/safety/redundancy",
                message=(
                    f"Safety level {spec.safety.level.value} typically requires "
                    "hardware redundancy"
                ),
                suggestion="Set /safety/redundancy to 'dual' or 'triple'",
            )
        )

    if high_safety and not spec.safety.watchdog_timeout_ms:
        issues.append(
            ValidationIssue(
                severity=Severity.INFO,
                path="/safety/watchdog_timeout_ms",
                message="High safety levels benefit from watchdog timers",
                suggestion="Set /safety/watchdog_timeout_ms (e.g., 100ms)",
            )
        )


def _check_mission_duration(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check battery capacity vs power budget vs mission duration."""
    if not spec.power:
        return

    battery = spec.power.battery_wh
    budget = spec.power.power_budget_watts
    duration = spec.power.mission_duration_min

    if battery and budget and duration:
        max_duration_min = (battery / budget) * 60
        if duration > max_duration_min:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    path="/power/battery_wh, /power/power_budget_watts, /power/mission_duration_min",
                    message=(
                        f"Mission duration ({duration}min) exceeds battery life "
                        f"({max_duration_min:.0f}min at {budget}W from {battery}Wh)"
                    ),
                    suggestion=(
                        "Increase battery capacity, reduce power budget, "
                        "or shorten mission duration"
                    ),
                )
            )
        elif duration > max_duration_min * 0.8:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/power/mission_duration_min",
                    message=(
                        f"Mission duration ({duration}min) uses >{80}% of estimated battery life "
                        f"({max_duration_min:.0f}min). Little margin for safety."
                    ),
                    suggestion="Consider adding 20% battery margin",
                )
            )


def _check_comms_bandwidth(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check if sensor data rate exceeds comms bandwidth."""
    if spec.sensors and spec.comms:
        data_rate = spec.sensors.data_rate_mbps
        bandwidth = spec.comms.bandwidth_mbps

        if data_rate and bandwidth and data_rate > bandwidth:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/sensors/data_rate_mbps, /comms/bandwidth_mbps",
                    message=(
                        f"Sensor data rate ({data_rate}Mbps) exceeds "
                        f"comms bandwidth ({bandwidth}Mbps). "
                        "Edge processing or compression needed."
                    ),
                    suggestion="Add edge compute for preprocessing or increase bandwidth",
                )
            )


def _check_autonomy_sensors(spec: SystemSpec, issues: list[ValidationIssue]) -> None:
    """Check that autonomy requirements have supporting sensors."""
    if not spec.autonomy:
        return

    if spec.autonomy.navigation == "slam":
        has_depth = False
        if spec.perception and spec.perception.camera_types:
            has_depth = any(
                t in spec.perception.camera_types for t in ["stereo", "depth", "rgbd"]
            )
        has_lidar = spec.sensors and "lidar" in (spec.sensors.modalities or [])
        if not has_depth and not has_lidar:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/autonomy/navigation",
                    message=(
                        "SLAM navigation requires depth sensing "
                        "(stereo/depth camera or LiDAR)"
                    ),
                    suggestion=(
                        "Add a depth camera to /perception/camera_types "
                        "or 'lidar' to /sensors/modalities"
                    ),
                )
            )

    if spec.autonomy.navigation == "vio":
        has_imu = spec.sensors and "imu" in (spec.sensors.modalities or [])
        if not has_imu:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    path="/autonomy/navigation, /sensors/modalities",
                    message="VIO navigation requires an IMU sensor",
                    suggestion="Add 'imu' to /sensors/modalities",
                )
            )
