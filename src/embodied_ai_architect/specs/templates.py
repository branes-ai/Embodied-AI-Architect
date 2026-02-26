"""Predefined spec templates (archetypes) for common embodied AI platforms.

Each template returns a SystemSpec with sensible defaults that users can
refine incrementally.
"""

from __future__ import annotations

from .models import (
    ActuatorSpec,
    AutonomyLevel,
    AutonomySpec,
    CommsSpec,
    ComputeSpec,
    CoolingType,
    EnvironmentalRating,
    PerceptionSpec,
    PlatformType,
    PowerSpec,
    SafetyLevel,
    SafetySpec,
    SensorSpec,
    SystemSpec,
)


def get_templates() -> dict[str, str]:
    """Return a dict of template name → description."""
    return {
        "drone-perception": "Drone with real-time perception pipeline",
        "quadruped-nav": "Quadruped robot with autonomous navigation",
        "industrial-inspection": "Industrial inspection platform",
        "amr-warehouse": "Autonomous mobile robot for warehouse operations",
        "edge-camera": "Fixed edge camera for monitoring/analytics",
        "biped-humanoid": "Bipedal humanoid robot",
    }


def get_template(template_name: str, name: str = "unnamed") -> SystemSpec:
    """Get a SystemSpec from a named template.

    Args:
        template_name: One of the template names from get_templates().
        name: Spec name to assign.

    Raises:
        ValueError: If template_name is not recognized.
    """
    builders = {
        "drone-perception": _drone_perception,
        "quadruped-nav": _quadruped_nav,
        "industrial-inspection": _industrial_inspection,
        "amr-warehouse": _amr_warehouse,
        "edge-camera": _edge_camera,
        "biped-humanoid": _biped_humanoid,
    }
    if template_name not in builders:
        available = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    return builders[template_name](name)


def _drone_perception(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Drone with real-time perception for obstacle avoidance and tracking",
        platform_type=PlatformType.DRONE,
        perception=PerceptionSpec(
            cameras=2,
            camera_types=["stereo", "depth"],
            detection_classes=["person", "vehicle", "obstacle"],
            tracking=True,
            min_accuracy=0.7,
            max_latency_ms=33.0,
            min_fps=30.0,
            resolution="640x480",
            model_family="YOLOv8",
        ),
        compute=ComputeSpec(
            soc="Jetson Orin NX",
            memory_gb=8.0,
            quantization="FP16",
            max_tdp_watts=15.0,
        ),
        power=PowerSpec(
            battery_wh=99.0,
            power_budget_watts=25.0,
            compute_power_watts=15.0,
            thermal_limit_c=85.0,
            cooling=CoolingType.PASSIVE,
            mission_duration_min=25.0,
        ),
        sensors=SensorSpec(
            modalities=["imu", "gps", "barometer"],
            imu_rate_hz=200.0,
            environmental_rating=EnvironmentalRating.OUTDOOR,
        ),
        actuators=ActuatorSpec(
            dof=4,
            control_rate_hz=400.0,
            max_speed_mps=15.0,
            payload_kg=0.5,
            actuator_types=["brushless_motor"],
        ),
        comms=CommsSpec(
            protocols=["wifi", "mavlink"],
            bandwidth_mbps=10.0,
            max_latency_ms=50.0,
            range_m=1000.0,
        ),
        autonomy=AutonomySpec(
            level=AutonomyLevel.CONDITIONAL,
            planning="waypoint",
            navigation="vio",
            decision_rate_hz=10.0,
            obstacle_avoidance=True,
        ),
        safety=SafetySpec(
            level=SafetyLevel.BASIC,
            failsafe_action="land",
            geofencing=True,
        ),
        tags=["drone", "perception", "outdoor"],
    )


def _quadruped_nav(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Quadruped robot with autonomous navigation and terrain adaptation",
        platform_type=PlatformType.QUADRUPED,
        perception=PerceptionSpec(
            cameras=3,
            camera_types=["stereo", "depth", "wide_angle"],
            detection_classes=["obstacle", "stairs", "terrain"],
            tracking=True,
            min_accuracy=0.75,
            max_latency_ms=50.0,
            min_fps=15.0,
            resolution="640x480",
        ),
        compute=ComputeSpec(
            soc="Jetson Orin NX",
            memory_gb=16.0,
            quantization="FP16",
            max_tdp_watts=25.0,
        ),
        power=PowerSpec(
            battery_wh=500.0,
            power_budget_watts=150.0,
            compute_power_watts=25.0,
            thermal_limit_c=75.0,
            cooling=CoolingType.FAN,
            mission_duration_min=60.0,
        ),
        sensors=SensorSpec(
            modalities=["imu", "lidar", "force_torque"],
            imu_rate_hz=400.0,
            lidar_points_per_sec=300000,
            environmental_rating=EnvironmentalRating.OUTDOOR,
        ),
        actuators=ActuatorSpec(
            dof=12,
            control_rate_hz=500.0,
            max_speed_mps=3.0,
            payload_kg=5.0,
            actuator_types=["servo"],
        ),
        comms=CommsSpec(
            protocols=["wifi", "ethernet"],
            bandwidth_mbps=50.0,
            max_latency_ms=20.0,
            range_m=200.0,
        ),
        autonomy=AutonomySpec(
            level=AutonomyLevel.HIGH,
            planning="behavior_tree",
            navigation="slam",
            decision_rate_hz=10.0,
            obstacle_avoidance=True,
        ),
        safety=SafetySpec(
            level=SafetyLevel.SIL_1,
            redundancy="dual",
            failsafe_action="stop",
        ),
        tags=["quadruped", "navigation", "outdoor"],
    )


def _industrial_inspection(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Industrial inspection platform for anomaly detection",
        platform_type=PlatformType.INDUSTRIAL_ARM,
        perception=PerceptionSpec(
            cameras=2,
            camera_types=["monocular", "thermal"],
            detection_classes=["defect", "crack", "corrosion", "leak"],
            tracking=False,
            min_accuracy=0.9,
            max_latency_ms=100.0,
            min_fps=5.0,
            resolution="1920x1080",
        ),
        compute=ComputeSpec(
            memory_gb=8.0,
            quantization="INT8",
            max_tdp_watts=30.0,
        ),
        power=PowerSpec(
            power_budget_watts=100.0,
            thermal_limit_c=70.0,
            cooling=CoolingType.FAN,
        ),
        sensors=SensorSpec(
            modalities=["ultrasonic", "thermal"],
            environmental_rating=EnvironmentalRating.IP54,
        ),
        safety=SafetySpec(
            level=SafetyLevel.SIL_2,
            redundancy="dual",
            failsafe_action="stop",
            certifications=["IEC-61508"],
        ),
        tags=["industrial", "inspection", "indoor"],
    )


def _amr_warehouse(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Autonomous mobile robot for warehouse pick-and-place",
        platform_type=PlatformType.AMR,
        perception=PerceptionSpec(
            cameras=4,
            camera_types=["depth", "monocular"],
            detection_classes=["person", "obstacle", "pallet", "shelf"],
            tracking=True,
            min_accuracy=0.85,
            max_latency_ms=50.0,
            min_fps=15.0,
            resolution="640x480",
        ),
        compute=ComputeSpec(
            memory_gb=8.0,
            quantization="INT8",
            max_tdp_watts=15.0,
        ),
        power=PowerSpec(
            battery_wh=1000.0,
            power_budget_watts=200.0,
            compute_power_watts=15.0,
            cooling=CoolingType.PASSIVE,
            mission_duration_min=480.0,
        ),
        sensors=SensorSpec(
            modalities=["lidar", "imu", "ultrasonic"],
            lidar_points_per_sec=600000,
            imu_rate_hz=100.0,
            environmental_rating=EnvironmentalRating.INDOOR,
        ),
        actuators=ActuatorSpec(
            dof=2,
            control_rate_hz=50.0,
            max_speed_mps=2.0,
            payload_kg=50.0,
            actuator_types=["brushless_motor"],
        ),
        comms=CommsSpec(
            protocols=["wifi", "5g"],
            bandwidth_mbps=100.0,
            max_latency_ms=10.0,
            mesh_networking=True,
        ),
        autonomy=AutonomySpec(
            level=AutonomyLevel.HIGH,
            planning="path_planning",
            navigation="slam",
            decision_rate_hz=5.0,
            obstacle_avoidance=True,
            multi_agent=True,
        ),
        safety=SafetySpec(
            level=SafetyLevel.SIL_1,
            failsafe_action="stop",
        ),
        tags=["amr", "warehouse", "indoor"],
    )


def _edge_camera(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Fixed edge camera for monitoring and video analytics",
        platform_type=PlatformType.FIXED_CAMERA,
        perception=PerceptionSpec(
            cameras=1,
            camera_types=["monocular"],
            detection_classes=["person", "vehicle"],
            tracking=True,
            min_accuracy=0.8,
            max_latency_ms=100.0,
            min_fps=10.0,
            resolution="1920x1080",
            model_family="YOLOv8",
        ),
        compute=ComputeSpec(
            memory_gb=4.0,
            quantization="INT8",
            max_tdp_watts=10.0,
        ),
        power=PowerSpec(
            power_budget_watts=15.0,
            thermal_limit_c=70.0,
            cooling=CoolingType.PASSIVE,
        ),
        comms=CommsSpec(
            protocols=["ethernet", "wifi"],
            bandwidth_mbps=20.0,
        ),
        tags=["edge", "camera", "monitoring"],
    )


def _biped_humanoid(name: str) -> SystemSpec:
    return SystemSpec(
        name=name,
        description="Bipedal humanoid robot for human-environment interaction",
        platform_type=PlatformType.BIPED,
        perception=PerceptionSpec(
            cameras=4,
            camera_types=["stereo", "depth", "wide_angle"],
            detection_classes=["person", "face", "hand", "object", "obstacle"],
            tracking=True,
            min_accuracy=0.8,
            max_latency_ms=33.0,
            min_fps=30.0,
            resolution="1280x720",
        ),
        compute=ComputeSpec(
            cpu_cores=8,
            memory_gb=32.0,
            gpu="integrated",
            quantization="FP16",
            max_tdp_watts=60.0,
        ),
        power=PowerSpec(
            battery_wh=2000.0,
            power_budget_watts=500.0,
            compute_power_watts=60.0,
            thermal_limit_c=80.0,
            cooling=CoolingType.LIQUID,
            mission_duration_min=120.0,
        ),
        sensors=SensorSpec(
            modalities=["imu", "lidar", "force_torque", "tactile"],
            imu_rate_hz=1000.0,
            lidar_points_per_sec=1000000,
            environmental_rating=EnvironmentalRating.INDOOR,
        ),
        actuators=ActuatorSpec(
            dof=30,
            control_rate_hz=1000.0,
            max_speed_mps=2.0,
            payload_kg=10.0,
            actuator_types=["servo", "linear"],
        ),
        comms=CommsSpec(
            protocols=["wifi", "5g", "ethernet"],
            bandwidth_mbps=100.0,
            max_latency_ms=5.0,
        ),
        autonomy=AutonomySpec(
            level=AutonomyLevel.HIGH,
            planning="behavior_tree",
            navigation="slam",
            decision_rate_hz=20.0,
            obstacle_avoidance=True,
        ),
        safety=SafetySpec(
            level=SafetyLevel.SIL_2,
            redundancy="triple",
            failsafe_action="stop",
            watchdog_timeout_ms=100.0,
        ),
        tags=["biped", "humanoid", "indoor"],
    )
