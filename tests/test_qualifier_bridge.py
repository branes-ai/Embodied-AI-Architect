"""Tests for the codebase → qualifier bridge."""

from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    ScanResult,
)
from embodied_ai_architect.codebase.qualifier_bridge import (
    codebase_to_qualification,
    _normalize_deps,
)
from embodied_ai_architect.qualification.qualifier import GoalQualifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scan(
    name: str = "test_project",
    deps: list[str] | None = None,
    languages: list[str] | None = None,
    ml_models: list[dict] | None = None,
) -> ScanResult:
    return ScanResult(
        project_name=name,
        project_path=f"/tmp/{name}",
        languages=languages or ["python"],
        dependencies=deps or [],
        ml_models=ml_models or [],
    )


# ---------------------------------------------------------------------------
# _normalize_deps
# ---------------------------------------------------------------------------


class TestNormalizeDeps:
    def test_lowercases(self):
        assert "torch" in _normalize_deps(["Torch"])

    def test_strips_version_specifiers(self):
        assert "torch" in _normalize_deps(["torch>=2.0"])
        assert "torch" in _normalize_deps(["torch==1.13.0"])
        assert "torch" in _normalize_deps(["torch~=2.0"])

    def test_strips_extras(self):
        assert "torch" in _normalize_deps(["torch[cuda]"])

    def test_strips_whitespace(self):
        assert "torch" in _normalize_deps(["  torch  "])


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------


class TestDomainDetection:
    def test_drone_via_mavros(self):
        scan = _make_scan(deps=["mavros", "rclpy", "numpy"])
        result = codebase_to_qualification(scan)
        assert result.domain == "drone"
        assert "mavros" in result.report.domain_evidence

    def test_drone_via_dronekit(self):
        scan = _make_scan(deps=["dronekit", "pymavlink"])
        result = codebase_to_qualification(scan)
        assert result.domain == "drone"

    def test_robot_arm_via_moveit(self):
        scan = _make_scan(deps=["moveit", "rclpy"])
        result = codebase_to_qualification(scan)
        assert result.domain == "robot_arm"

    def test_robot_arm_via_franka(self):
        scan = _make_scan(deps=["franka_ros", "panda_py"])
        result = codebase_to_qualification(scan)
        assert result.domain == "robot_arm"

    def test_ugv_via_nav2(self):
        scan = _make_scan(deps=["nav2", "rclpy", "slam_toolbox"])
        result = codebase_to_qualification(scan)
        assert result.domain == "ugv"

    def test_ugv_via_turtlebot(self):
        scan = _make_scan(deps=["turtlebot3", "amcl"])
        result = codebase_to_qualification(scan)
        assert result.domain == "ugv"

    def test_no_domain_for_generic_python(self):
        scan = _make_scan(deps=["numpy", "matplotlib", "pandas"])
        result = codebase_to_qualification(scan)
        assert result.domain is None

    def test_picks_domain_with_more_evidence(self):
        # Both robot_arm and ugv hints — robot_arm has more
        scan = _make_scan(deps=["moveit", "franka_ros", "panda_py", "nav2"])
        result = codebase_to_qualification(scan)
        assert result.domain == "robot_arm"


# ---------------------------------------------------------------------------
# Perception task detection
# ---------------------------------------------------------------------------


class TestPerceptionDetection:
    def test_yolo_to_object_detection(self):
        # Without a domain hint, the generic tag is detected but not projected
        scan = _make_scan(deps=["ultralytics", "torch"])
        result = codebase_to_qualification(scan)
        assert "object_detection" in result.report.perception_tasks
        assert result.prefilled_answers == {}  # no domain → no projection

    def test_yolo_with_drone_projects_to_object_detection(self):
        scan = _make_scan(deps=["mavros", "ultralytics", "torch"])
        result = codebase_to_qualification(scan)
        assert result.domain == "drone"
        # drone template uses "object_detection" as canonical option
        assert "object_detection" in result.prefilled_answers["perception_tasks"]

    def test_yolo_with_ugv_projects_to_object_recognition(self):
        scan = _make_scan(deps=["nav2", "ultralytics"])
        result = codebase_to_qualification(scan)
        assert result.domain == "ugv"
        # UGV template uses "object_recognition" (different vocabulary)
        assert "object_recognition" in result.prefilled_answers["perception_tasks"]

    def test_mediapipe_with_ugv_projects_to_pedestrian_detection(self):
        scan = _make_scan(deps=["nav2", "mediapipe"])
        result = codebase_to_qualification(scan)
        assert result.domain == "ugv"
        # UGV uses "pedestrian_detection" not "person_detection"
        assert "pedestrian_detection" in result.prefilled_answers["perception_tasks"]

    def test_orb_slam_to_slam(self):
        scan = _make_scan(deps=["orb_slam3", "rtabmap"])
        result = codebase_to_qualification(scan)
        assert "slam" in result.report.perception_tasks

    def test_vins_to_visual_odometry(self):
        scan = _make_scan(deps=["vins_fusion"])
        result = codebase_to_qualification(scan)
        assert "visual_odometry" in result.report.perception_tasks

    def test_mediapipe_to_person_detection(self):
        scan = _make_scan(deps=["mediapipe"])
        result = codebase_to_qualification(scan)
        assert "person_detection" in result.report.perception_tasks

    def test_ml_models_default_to_object_detection(self):
        scan = _make_scan(
            deps=["torch"],
            ml_models=[{"path": "yolov8n.pt", "format": "pt", "size_bytes": 6_000_000}],
        )
        result = codebase_to_qualification(scan)
        assert "object_detection" in result.report.perception_tasks

    def test_no_perception_for_generic_project(self):
        scan = _make_scan(deps=["numpy", "matplotlib"])
        result = codebase_to_qualification(scan)
        assert result.report.perception_tasks == []


# ---------------------------------------------------------------------------
# Control output detection
# ---------------------------------------------------------------------------


class TestControlDetection:
    def test_simple_pid_with_drone_to_flight_controller(self):
        scan = _make_scan(deps=["simple_pid", "mavros"])
        result = codebase_to_qualification(scan)
        # Generic capability tag in the report
        assert "low_level_control" in result.report.control_output
        # Domain-projected canonical option in prefilled answers
        assert result.prefilled_answers.get("control_output") == "flight_controller"

    def test_simple_pid_with_ugv_to_motor_controller(self):
        scan = _make_scan(deps=["simple_pid", "nav2"])
        result = codebase_to_qualification(scan)
        assert result.domain == "ugv"
        # UGV uses motor_controller for low-level control
        assert result.prefilled_answers.get("control_output") == "motor_controller"

    def test_nav2_to_path_planning(self):
        scan = _make_scan(deps=["nav2_msgs", "rclpy"])
        result = codebase_to_qualification(scan)
        # Generic tag
        assert "path_planning" in result.report.control_output
        # Projected to UGV's path_planner option
        assert result.prefilled_answers.get("control_output") == "path_planner"

    def test_moveit_with_robot_arm_to_control_architecture(self):
        scan = _make_scan(deps=["moveit", "moveit_msgs"])
        result = codebase_to_qualification(scan)
        assert result.domain == "robot_arm"
        # robot_arm uses control_architecture, not control_output
        assert "control_architecture" in result.prefilled_answers
        assert result.prefilled_answers["control_architecture"] == "trajectory_to_joints"

    def test_no_control_for_generic_project(self):
        scan = _make_scan(deps=["numpy"])
        result = codebase_to_qualification(scan)
        assert result.report.control_output == []


# ---------------------------------------------------------------------------
# ML framework detection
# ---------------------------------------------------------------------------


class TestMLFrameworkDetection:
    def test_detects_pytorch(self):
        scan = _make_scan(deps=["torch", "torchvision"])
        result = codebase_to_qualification(scan)
        assert "torch" in result.report.ml_frameworks
        assert "torchvision" in result.report.ml_frameworks

    def test_detects_tensorflow(self):
        scan = _make_scan(deps=["tensorflow"])
        result = codebase_to_qualification(scan)
        assert "tensorflow" in result.report.ml_frameworks

    def test_detects_onnx(self):
        scan = _make_scan(deps=["onnxruntime"])
        result = codebase_to_qualification(scan)
        assert "onnxruntime" in result.report.ml_frameworks


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_high_confidence_full_signal(self):
        scan = _make_scan(
            deps=["mavros", "pymavlink", "ultralytics", "simple_pid", "torch"],
            ml_models=[{"path": "yolov8.pt", "format": "pt", "size_bytes": 6_000_000}],
        )
        result = codebase_to_qualification(scan)
        assert result.report.confidence == "high"

    def test_low_confidence_empty_project(self):
        scan = _make_scan(deps=["numpy"])
        result = codebase_to_qualification(scan)
        assert result.report.confidence == "low"


# ---------------------------------------------------------------------------
# Acceptance criteria from issue #41
# ---------------------------------------------------------------------------


class TestIssue41AcceptanceCriteria:
    def test_drone_perception_project_auto_detected(self):
        """Drone perception project → platform=drone, perception=object_detection auto-filled."""
        scan = _make_scan(
            name="drone_yolo_app",
            deps=["mavros", "pymavlink", "ultralytics", "torch", "rclpy"],
        )
        result = codebase_to_qualification(scan)

        # Domain is drone
        assert result.domain == "drone"

        # Perception is object_detection (pre-filled)
        assert "perception_tasks" in result.prefilled_answers
        assert "object_detection" in result.prefilled_answers["perception_tasks"]


# ---------------------------------------------------------------------------
# Bridge integration with GoalQualifier
# ---------------------------------------------------------------------------


class TestBridgeQualifierIntegration:
    def test_prefill_answers_via_qualifier(self):
        """Prefilled answers should populate the qualifier without errors."""
        scan = _make_scan(
            deps=["mavros", "ultralytics", "simple_pid"],
        )
        bridge = codebase_to_qualification(scan)

        qualifier = GoalQualifier()
        qualifier.assess("Drone perception SoC", domain=bridge.domain)
        qualifier.prefill_answers(bridge.prefilled_answers)

        # Answers should now contain the prefilled values
        assert "perception_tasks" in qualifier.answers
        assert "object_detection" in qualifier.answers["perception_tasks"]

    def test_prefill_unknown_question_id_silently_skipped(self):
        """Pre-filling a question_id that doesn't exist should not crash."""
        qualifier = GoalQualifier()
        qualifier.assess("Drone goal", domain="drone")
        # "fake_question_id" doesn't exist; should be silently skipped
        qualifier.prefill_answers({"fake_question_id": "value"})
        assert "fake_question_id" not in qualifier.answers

    def test_control_output_roundtrip_to_qualifier(self):
        """control_output prefills must be accepted by the drone qualifier."""
        scan = _make_scan(deps=["mavros", "simple_pid"])
        bridge = codebase_to_qualification(scan)
        assert bridge.domain == "drone"
        assert bridge.prefilled_answers.get("control_output") == "flight_controller"

        qualifier = GoalQualifier()
        qualifier.assess("drone perception", domain="drone")
        qualifier.prefill_answers(bridge.prefilled_answers)

        # The qualifier should have accepted the control_output answer
        assert qualifier.answers.get("control_output") == "flight_controller"

    def test_control_architecture_roundtrip_to_robot_arm(self):
        """robot_arm uses control_architecture; bridge must round-trip correctly."""
        scan = _make_scan(deps=["moveit", "moveit_msgs"])
        bridge = codebase_to_qualification(scan)
        assert bridge.domain == "robot_arm"
        assert bridge.prefilled_answers.get("control_architecture") == "trajectory_to_joints"

        qualifier = GoalQualifier()
        qualifier.assess("robot arm grasping", domain="robot_arm")
        qualifier.prefill_answers(bridge.prefilled_answers)

        assert qualifier.answers.get("control_architecture") == "trajectory_to_joints"


# ---------------------------------------------------------------------------
# Analysis-aware detection (with kernels)
# ---------------------------------------------------------------------------


class TestAnalysisAwareDetection:
    def test_control_loop_kernel_implies_flight_controller(self):
        scan = _make_scan(deps=["mavros"])
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="pid_loop",
                    source_file="control.py",
                    kernel_type="control_loop",
                ),
            ],
        )
        result = codebase_to_qualification(scan, analysis)
        assert "flight_controller" in result.report.control_output

    def test_ml_inference_kernel_implies_object_detection(self):
        scan = _make_scan(deps=["mavros"])
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="yolo_inference",
                    source_file="detect.py",
                    kernel_type="ml_inference",
                ),
            ],
        )
        result = codebase_to_qualification(scan, analysis)
        assert "object_detection" in result.report.perception_tasks
