"""Tests for goal qualification system."""

from embodied_ai_architect.qualification import (
    GoalQualifier,
    get_domain_template,
    list_domains,
)
from embodied_ai_architect.qualification.models import GoalQualification
from embodied_ai_architect.qualification.qualifier import render_qualification_result
from embodied_ai_architect.qualification.templates import detect_domain, register_template
from embodied_ai_architect.qualification.models import DomainTemplate, Question

# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------


class TestDomainDetection:
    def test_detect_drone(self):
        assert detect_domain("drone perception SoC") == "drone"
        assert detect_domain("UAV obstacle avoidance") == "drone"
        assert detect_domain("quadcopter vision system") == "drone"

    def test_detect_ugv(self):
        assert detect_domain("warehouse AMR navigation") == "ugv"
        assert detect_domain("mobile robot for delivery") == "ugv"
        assert detect_domain("sidewalk delivery robot") == "ugv"

    def test_detect_robot_arm(self):
        assert detect_domain("cobot for assembly tasks") == "robot_arm"
        assert detect_domain("robot arm pick and place") == "robot_arm"
        assert detect_domain("collaborative robot welding") == "robot_arm"

    def test_no_match(self):
        assert detect_domain("something completely unrelated") is None

    def test_list_domains(self):
        domains = list_domains()
        assert "drone" in domains
        assert "ugv" in domains
        assert "robot_arm" in domains


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_drone_template_exists(self):
        t = get_domain_template("drone")
        assert t is not None
        assert t.domain == "drone"
        assert len(t.questions) >= 4

    def test_ugv_template_exists(self):
        t = get_domain_template("ugv")
        assert t is not None
        assert len(t.questions) >= 4

    def test_robot_arm_template_exists(self):
        t = get_domain_template("robot_arm")
        assert t is not None
        assert len(t.questions) >= 5

    def test_robot_arm_has_dual_nervous_system(self):
        t = get_domain_template("robot_arm")
        q = t.get_question("arm_type")
        assert q is not None
        cobot = q.implications.get("cobot_tabletop", {})
        assert cobot.get("custom.architecture") == "dual_nervous_system"
        assert cobot.get("custom.joint_controllers") == "independent_mcu_per_joint"

    def test_dimensions_covered(self):
        for domain in list_domains():
            t = get_domain_template(domain)
            dims = t.dimensions_covered
            assert "platform" in dims
            assert "perception" in dims
            assert "control" in dims

    def test_register_custom_template(self):
        custom = DomainTemplate(
            domain="submarine",
            display_name="Underwater Vehicle",
            description="Underwater ROV/AUV",
            keywords=["submarine", "underwater", "rov", "auv"],
            questions=[
                Question(
                    id="sub_type",
                    dimension="platform",
                    text="What type of underwater vehicle?",
                    options=["rov", "auv"],
                    required=True,
                ),
            ],
        )
        register_template(custom)
        assert "submarine" in list_domains()
        assert detect_domain("underwater ROV inspection") == "submarine"


# ---------------------------------------------------------------------------
# GoalQualification model
# ---------------------------------------------------------------------------


class TestGoalQualification:
    def test_empty_is_not_tangible(self):
        q = GoalQualification()
        assert not q.is_tangible
        assert q.score == 0.0
        assert len(q.missing_dimensions) == 5

    def test_fully_specified(self):
        q = GoalQualification(
            platform_identified=True,
            perception_tasks_enumerated=True,
            control_output_identified=True,
            power_envelope_bounded=True,
            environment_characterized=True,
        )
        assert q.is_tangible
        assert q.score == 1.0
        assert q.missing_dimensions == []

    def test_environment_not_required(self):
        q = GoalQualification(
            platform_identified=True,
            perception_tasks_enumerated=True,
            control_output_identified=True,
            power_envelope_bounded=True,
            environment_characterized=False,
        )
        assert q.is_tangible  # environment is soft


# ---------------------------------------------------------------------------
# Qualifier engine
# ---------------------------------------------------------------------------


class TestQualifier:
    def test_assess_unknown_goal_asks_domain(self):
        q = GoalQualifier()
        result = q.assess("something vague")
        assert not result.is_tangible
        assert result.next_question is not None
        assert result.next_question.id == "_domain_selection"

    def test_assess_drone_goal_detects_domain(self):
        q = GoalQualifier()
        result = q.assess("drone perception SoC")
        assert result.domain == "drone"
        assert not result.is_tangible

    def test_answer_updates_state(self):
        q = GoalQualifier()
        q.assess("drone perception", domain="drone")
        result = q.answer("drone_type", "multirotor_delivery")
        assert result.answers["drone_type"] == "multirotor_delivery"
        assert result.qualification.platform_identified
        assert result.qualification.power_envelope_bounded

    def test_full_drone_qualification(self):
        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        q.answer("perception_tasks", ["obstacle_avoidance", "tracking"])
        q.answer("control_output", ["flight_controller"])
        result = q.answer("operating_environment", "outdoor_urban")

        assert result.qualification.platform_identified
        assert result.qualification.perception_tasks_enumerated
        assert result.qualification.control_output_identified
        assert result.qualification.power_envelope_bounded
        assert result.qualification.environment_characterized
        assert result.is_tangible

    def test_full_robot_arm_qualification(self):
        q = GoalQualifier()
        q.assess("cobot for assembly", domain="robot_arm")
        q.answer("arm_type", "cobot_tabletop")
        q.answer("perception_tasks", ["object_detection", "human_presence"])
        q.answer("control_architecture", "trajectory_to_joints")
        q.answer("safety_requirements", ["power_force_limiting", "collision_detection"])
        result = q.answer("manipulation_task", "assembly")

        assert result.is_tangible
        spec = result.accumulated_spec
        assert spec.get("custom", {}).get("architecture") == "dual_nervous_system"
        assert spec.get("custom", {}).get("joint_controllers") == "independent_mcu_per_joint"

    def test_skip_uses_default(self):
        q = GoalQualifier()
        q.assess("drone", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        q.answer("perception_tasks", ["obstacle_avoidance"])
        q.answer("control_output", ["flight_controller"])
        result = q.skip("operating_environment")

        assert result.qualification.environment_characterized
        assert len(result.assumptions) > 0

    def test_to_design_inputs(self):
        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        q.answer("perception_tasks", ["obstacle_avoidance"])
        q.answer("control_output", ["flight_controller"])

        inputs = q.to_design_inputs()
        assert inputs["platform"] == "drone"
        assert "drone" in inputs["use_case"]
        assert inputs["constraints"] is not None
        assert inputs["constraints"].max_power_watts == 5.0

    def test_implications_merge_lists(self):
        q = GoalQualifier()
        q.assess("drone", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        # base_implications has ["imu", "barometer", "gps"]
        # multirotor_delivery doesn't override modalities
        spec = q.accumulated_spec
        modalities = spec.get("sensors", {}).get("modalities", [])
        assert "imu" in modalities

    def test_domain_selection_flow(self):
        q = GoalQualifier()
        result = q.assess("some system")
        # Should ask for domain
        assert result.next_question.id == "_domain_selection"
        # Answer with drone
        result = q.answer("_domain_selection", "drone")
        assert result.domain == "drone"
        # Next question should be drone-specific
        assert result.next_question is not None
        assert result.next_question.id == "drone_type"


# ---------------------------------------------------------------------------
# Custom answers
# ---------------------------------------------------------------------------


class TestCustomAnswers:
    def test_custom_answer_basic(self):
        from embodied_ai_architect.qualification.models import CustomAnswer

        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")

        custom = CustomAnswer(
            value="imu_odometry",
            description="Dead-reckoning from IMU integration with EKF",
            implications={
                "sensors.modalities": ["imu"],
                "sensors.imu_rate_hz": 400.0,
                "perception.max_latency_ms": 5.0,
                "custom.compute_type": "kalman_filter",
            },
        )
        result = q.answer_custom("perception_tasks", custom)

        assert "imu_odometry" in result.answers["perception_tasks"]
        spec = result.accumulated_spec
        assert spec["sensors"]["imu_rate_hz"] == 400.0
        assert spec["custom"]["compute_type"] == "kalman_filter"
        assert result.qualification.perception_tasks_enumerated

    def test_custom_with_template_options(self):
        from embodied_ai_architect.qualification.models import CustomAnswer

        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")

        custom = CustomAnswer(
            value="imu_odometry",
            description="IMU-based dead reckoning",
            implications={"sensors.imu_rate_hz": 400.0},
        )
        result = q.answer_custom(
            "perception_tasks",
            custom,
            also_selected=["obstacle_avoidance"],
        )

        # Both template and custom answers present
        assert "obstacle_avoidance" in result.answers["perception_tasks"]
        assert "imu_odometry" in result.answers["perception_tasks"]
        # Template implications applied
        spec = result.accumulated_spec
        assert spec["perception"]["max_latency_ms"] == 20.0  # from obstacle_avoidance
        # Custom implications applied
        assert spec["sensors"]["imu_rate_hz"] == 400.0

    def test_custom_answers_recorded(self):
        from embodied_ai_architect.qualification.models import CustomAnswer

        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")

        custom = CustomAnswer(
            value="radar_altimeter",
            description="Radar-based altitude hold",
            implications={"sensors.modalities": ["radar"]},
        )
        q.answer_custom("perception_tasks", custom)

        spec = q.accumulated_spec
        records = spec.get("_custom_answers", [])
        assert len(records) == 1
        assert records[0]["value"] == "radar_altimeter"
        assert records[0]["question_id"] == "perception_tasks"

    def test_multiple_custom_answers(self):
        from embodied_ai_architect.qualification.models import CustomAnswer

        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")

        customs = [
            CustomAnswer(
                value="imu_odometry",
                description="IMU dead reckoning",
                implications={"sensors.imu_rate_hz": 400.0},
            ),
            CustomAnswer(
                value="optical_flow",
                description="Downward-facing optical flow",
                implications={
                    "sensors.modalities": ["optical_flow_sensor"],
                    "perception.min_fps": 60.0,
                },
            ),
        ]
        result = q.answer_custom("perception_tasks", customs)

        assert len(result.answers["perception_tasks"]) == 2
        spec = result.accumulated_spec
        assert spec["sensors"]["imu_rate_hz"] == 400.0
        assert "optical_flow_sensor" in spec["sensors"]["modalities"]

    def test_allow_custom_flag_on_template(self):
        t = get_domain_template("drone")
        perception_q = t.get_question("perception_tasks")
        assert perception_q.allow_custom is True
        assert perception_q.custom_prompt != ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_initial(self):
        q = GoalQualifier()
        result = q.assess("drone SoC", domain="drone")
        rendered = render_qualification_result(result)
        assert "GOAL QUALIFICATION" in rendered
        assert "TANGIBILITY SCORECARD" in rendered
        assert "Still needed" in rendered
        # Question details not in snapshot — presented interactively by caller
        assert "NEXT QUESTION" not in rendered

    def test_render_mid_survey(self):
        """Mid-survey: some dimensions OK, but survey not complete."""
        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        result = q.answer("perception_tasks", ["obstacle_avoidance"])
        rendered = render_qualification_result(result)
        assert "ready for planning" not in rendered

    def test_render_tangible_after_full_survey(self):
        """Tangible only after all questions answered."""
        q = GoalQualifier()
        q.assess("drone SoC", domain="drone")
        q.answer("drone_type", "multirotor_delivery")
        q.answer("perception_tasks", ["obstacle_avoidance"])
        q.answer("control_output", ["flight_controller"])
        q.answer("operating_environment", "outdoor_urban")
        result = q.answer("regulatory", ["no_specific_regulation"])
        rendered = render_qualification_result(result)
        assert "TANGIBLE" in rendered
        assert "ready for planning" in rendered
