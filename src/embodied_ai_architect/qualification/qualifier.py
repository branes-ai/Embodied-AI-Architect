"""Goal qualification engine.

Drives the structured Q&A loop that refines an underspecified goal into
a tangible, actionable specification. Stateful — tracks answers and
accumulated implications across multiple rounds.

Usage:
    qualifier = GoalQualifier()
    result = qualifier.assess("drone perception SoC")
    # result.is_tangible == False, result.next_question is set

    result = qualifier.answer("drone_type", "multirotor_delivery")
    result = qualifier.answer("perception_tasks", ["obstacle_avoidance", "tracking"])
    # ...until result.is_tangible == True

    # Extract design inputs
    constraints, use_case, platform = qualifier.to_design_inputs()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from embodied_ai_architect.qualification.models import (
    CustomAnswer,
    GoalQualification,
    Question,
    QualificationResult,
)
from embodied_ai_architect.qualification.templates import (
    detect_domain,
    get_all_templates,
    get_domain_template,
    list_domains,
)

logger = logging.getLogger(__name__)


class GoalQualifier:
    """Stateful goal qualification engine.

    Holds the accumulated answers and spec fields across a multi-round
    Q&A session. The qualifier is the primary policy component for
    determining whether a design goal is actionable.
    """

    def __init__(self) -> None:
        self._domain: Optional[str] = None
        self._answers: dict[str, Any] = {}
        self._spec_fields: dict[str, Any] = {}
        self._original_goal: str = ""
        self._warnings: list[str] = []
        self._assumptions: list[str] = []

    def reset(self) -> None:
        """Clear all state for a new qualification session."""
        self._domain = None
        self._answers = {}
        self._spec_fields = {}
        self._original_goal = ""
        self._warnings = []
        self._assumptions = []

    @property
    def domain(self) -> Optional[str]:
        return self._domain

    @property
    def answers(self) -> dict[str, Any]:
        return dict(self._answers)

    @property
    def accumulated_spec(self) -> dict[str, Any]:
        return dict(self._spec_fields)

    def assess(self, goal: str, domain: Optional[str] = None) -> QualificationResult:
        """Assess a goal and return qualification result.

        If domain is not specified, attempts to detect it from keywords.
        If no domain can be detected, returns a result asking the user
        to pick one.

        Args:
            goal: Natural language design goal.
            domain: Optional domain override (e.g., "drone", "ugv", "robot_arm").

        Returns:
            QualificationResult with current state and next question.
        """
        self._original_goal = goal

        # Detect or set domain
        if domain:
            self._domain = domain
        else:
            self._domain = detect_domain(goal)

        if self._domain is None:
            # Can't even determine the platform — ask first
            return self._build_domain_selection_result()

        # Apply base implications from domain
        template = get_domain_template(self._domain)
        if template and template.base_implications:
            self._merge_implications(template.base_implications)

        return self._build_result()

    def answer(self, question_id: str, value: Any) -> QualificationResult:
        """Record an answer and return updated qualification result.

        Args:
            question_id: The question being answered.
            value: The answer value (string, list of strings, number, etc.)

        Returns:
            Updated QualificationResult.
        """
        if question_id == "_domain_selection":
            self._domain = value
            template = get_domain_template(self._domain)
            if template and template.base_implications:
                self._merge_implications(template.base_implications)
            return self._build_result()

        self._answers[question_id] = value

        # Apply implications from this answer
        template = get_domain_template(self._domain)
        if template:
            question = template.get_question(question_id)
            if question and question.implications:
                if isinstance(value, list):
                    # Multi-choice: merge implications for all selected values
                    for v in value:
                        if v in question.implications:
                            self._merge_implications(question.implications[v])
                elif isinstance(value, str) and value in question.implications:
                    self._merge_implications(question.implications[value])

        return self._build_result()

    def answer_custom(
        self,
        question_id: str,
        custom: CustomAnswer | list[CustomAnswer],
        also_selected: list[str] | None = None,
    ) -> QualificationResult:
        """Record custom answer(s) not in the template options.

        Custom answers carry their own implications, which are merged into
        the spec alongside any template-option answers from also_selected.

        This is the escape hatch when the template doesn't cover the user's
        design space. The template is a starting point, not a straitjacket.

        Args:
            question_id: The question being answered.
            custom: One or more CustomAnswer objects with value + implications.
            also_selected: Template options also selected (for multi-choice).

        Returns:
            Updated QualificationResult.

        Example:
            custom = CustomAnswer(
                value="imu_odometry",
                description="Dead-reckoning from IMU with EKF",
                implications={
                    "sensors.modalities": ["imu"],
                    "sensors.imu_rate_hz": 400.0,
                    "perception.max_latency_ms": 5.0,
                    "custom.compute_type": "kalman_filter",
                },
            )
            result = qualifier.answer_custom("perception_tasks", custom,
                                              also_selected=["obstacle_avoidance"])
        """
        customs = custom if isinstance(custom, list) else [custom]

        # Collect all values (template + custom)
        all_values = list(also_selected or [])
        for c in customs:
            all_values.append(c.value)

        self._answers[question_id] = all_values

        # Apply template implications for selected options
        template = get_domain_template(self._domain)
        if template and also_selected:
            question = template.get_question(question_id)
            if question and question.implications:
                for opt in also_selected:
                    if opt in question.implications:
                        self._merge_implications(question.implications[opt])

        # Apply custom implications
        for c in customs:
            if c.implications:
                self._merge_implications(c.implications)
            logger.info(
                "Custom answer for '%s': %s — %s",
                question_id,
                c.value,
                c.description,
            )

        # Record that custom options were used (for template evolution tracking)
        custom_record = self._spec_fields.setdefault("_custom_answers", [])
        for c in customs:
            custom_record.append(
                {
                    "question_id": question_id,
                    "value": c.value,
                    "description": c.description,
                    "implications": c.implications,
                }
            )

        return self._build_result()

    def skip(self, question_id: str) -> QualificationResult:
        """Skip a question, using its default if available.

        Args:
            question_id: The question to skip.

        Returns:
            Updated QualificationResult with assumption recorded.
        """
        template = get_domain_template(self._domain)
        if template:
            question = template.get_question(question_id)
            if question:
                if question.default:
                    self._assumptions.append(f"Assumed '{question.default}' for: {question.text}")
                    return self.answer(question_id, question.default)
                elif not question.required:
                    self._answers[question_id] = None
                    self._assumptions.append(f"Skipped optional: {question.text}")
                else:
                    self._warnings.append(
                        f"Required question skipped without default: {question.text}"
                    )
                    self._answers[question_id] = None

        return self._build_result()

    def to_design_inputs(self) -> dict[str, Any]:
        """Convert accumulated spec into design inputs for the SoC pipeline.

        Returns a dict with keys: goal, constraints, use_case, platform,
        ready for create_initial_soc_state().
        """
        from embodied_ai_architect.graphs.soc_state import DesignConstraints

        spec = self._spec_fields

        # Build constraints from accumulated spec fields
        constraint_kwargs: dict[str, Any] = {}
        power = _get_nested(spec, "power.compute_power_watts")
        if power is not None:
            constraint_kwargs["max_power_watts"] = power
        latency = _get_nested(spec, "perception.max_latency_ms")
        if latency is not None:
            constraint_kwargs["max_latency_ms"] = latency
        process = _get_nested(spec, "compute.target_process_nm")
        if process is not None:
            constraint_kwargs["target_process_nm"] = process

        # Build refined goal string
        refined_goal = self._build_refined_goal()

        # Determine use_case and platform
        platform = _get_nested(spec, "platform_type") or self._domain or ""
        use_case = self._infer_use_case()

        return {
            "goal": refined_goal,
            "constraints": DesignConstraints(**constraint_kwargs) if constraint_kwargs else None,
            "use_case": use_case,
            "platform": platform,
            "system_spec": spec,
        }

    # -------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------

    def _build_result(self) -> QualificationResult:
        """Build a QualificationResult from current state."""
        qualification = self._assess_tangibility()
        # Always offer the next unanswered question. The survey runs to
        # completion — every question gathers a requirement that matters.
        # Tangibility is assessed after all questions are answered, not
        # used to short-circuit the survey.
        next_q = self._find_next_question()

        template = get_domain_template(self._domain)
        total_questions = len(template.questions) if template else 0
        answered = len(self._answers)

        return QualificationResult(
            qualification=qualification,
            original_goal=self._original_goal,
            refined_goal=self._build_refined_goal(),
            domain=self._domain,
            answers=dict(self._answers),
            accumulated_spec=dict(self._spec_fields),
            next_question=next_q,
            questions_asked=answered,
            questions_remaining=total_questions - answered,
            warnings=list(self._warnings),
            assumptions=list(self._assumptions),
        )

    def _build_domain_selection_result(self) -> QualificationResult:
        """Build a result that asks the user to select a domain."""
        domains = list_domains()
        templates = get_all_templates()

        options = []
        descriptions = {}
        for d in domains:
            t = templates[d]
            options.append(d)
            descriptions[d] = t.description

        domain_question = Question(
            id="_domain_selection",
            dimension="platform",
            text="What type of platform is this design for?",
            explanation=(
                "I need to know the platform type to ask the right questions. "
                "Different platforms have fundamentally different architectures, "
                "power budgets, and safety requirements."
            ),
            options=options,
            option_descriptions=descriptions,
            required=True,
        )

        return QualificationResult(
            qualification=GoalQualification(),
            original_goal=self._original_goal,
            refined_goal=self._original_goal,
            domain=None,
            answers={},
            accumulated_spec={},
            next_question=domain_question,
            questions_asked=0,
            questions_remaining=0,
            warnings=[
                "Could not determine platform type from goal text. "
                "Please select a platform domain."
            ],
            assumptions=[],
        )

    def _assess_tangibility(self) -> GoalQualification:
        """Evaluate tangibility from accumulated spec fields."""
        spec = self._spec_fields

        platform_ok = _get_nested(spec, "platform_type") is not None
        # Perception is enumerated when the user has answered the perception
        # question (explicitly naming tasks). Additionally check for spec
        # evidence: timing targets or task indicators. The user's explicit
        # answer is the strongest signal — if they said "visual_odometry",
        # perception is enumerated even if the implication didn't set every
        # field perfectly.
        answered_perception = bool(self._answers.get("perception_tasks"))
        has_timing = (
            _get_nested(spec, "perception.max_latency_ms") is not None
            or _get_nested(spec, "perception.min_fps") is not None
        )
        has_task_indicator = (
            bool(_get_nested(spec, "perception.detection_classes"))
            or _get_nested(spec, "perception.tracking") is not None
            or _get_nested(spec, "perception.resolution") is not None
            or _get_nested(spec, "perception.camera_types") is not None
            or _get_nested(spec, "custom.compute_type") is not None
        )
        perception_ok = answered_perception and (has_timing or has_task_indicator)
        control_ok = (
            _get_nested(spec, "actuators.control_rate_hz") is not None
            or _get_nested(spec, "autonomy.decision_rate_hz") is not None
        )
        power_ok = _get_nested(spec, "power.compute_power_watts") is not None
        environment_ok = _get_nested(spec, "sensors.environmental_rating") is not None

        return GoalQualification(
            platform_identified=platform_ok,
            perception_tasks_enumerated=perception_ok,
            control_output_identified=control_ok,
            power_envelope_bounded=power_ok,
            environment_characterized=environment_ok,
        )

    def _find_next_question(self) -> Optional[Question]:
        """Find the next unanswered question, respecting dependencies."""
        template = get_domain_template(self._domain)
        if not template:
            return None

        for q in template.questions:
            if q.id in self._answers:
                continue

            # Check dependency
            if q.depends_on:
                dep_answer = self._answers.get(q.depends_on)
                if dep_answer is None:
                    continue
                if q.depends_on_value and dep_answer != q.depends_on_value:
                    continue

            return q

        return None

    def _merge_implications(self, implications: dict[str, Any]) -> None:
        """Merge implication fields into the accumulated spec.

        Handles dotted paths like "power.compute_power_watts" by building
        nested dicts. Lists are merged (extended), not replaced.
        """
        for path, value in implications.items():
            parts = path.split(".")
            target = self._spec_fields

            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            key = parts[-1]
            existing = target.get(key)

            if isinstance(existing, list) and isinstance(value, list):
                # Merge lists, deduplicating
                merged = list(existing)
                for item in value:
                    if item not in merged:
                        merged.append(item)
                target[key] = merged
            elif isinstance(existing, list) and isinstance(value, str):
                if value not in existing:
                    existing.append(value)
            else:
                target[key] = value

    def _build_refined_goal(self) -> str:
        """Build a refined goal string from original + accumulated context."""
        parts = [self._original_goal]

        template = get_domain_template(self._domain)
        if template:
            platform_q = template.get_question(
                next(
                    (q.id for q in template.questions if q.dimension == "platform"),
                    "",
                )
            )
            if platform_q:
                platform_answer = self._answers.get(platform_q.id)
                if platform_answer:
                    desc = platform_q.option_descriptions.get(platform_answer, platform_answer)
                    parts.append(f"Platform: {desc}")

        perception = self._answers.get("perception_tasks")
        if perception:
            if isinstance(perception, list):
                parts.append(f"Perception: {', '.join(perception)}")
            else:
                parts.append(f"Perception: {perception}")

        power = _get_nested(self._spec_fields, "power.compute_power_watts")
        if power:
            parts.append(f"Compute power: {power}W")

        latency = _get_nested(self._spec_fields, "perception.max_latency_ms")
        if latency:
            parts.append(f"Max latency: {latency}ms")

        return " | ".join(parts)

    def _infer_use_case(self) -> str:
        """Infer use_case string from domain + answers."""
        if self._domain == "drone":
            drone_type = self._answers.get("drone_type", "")
            return f"drone_{drone_type}" if drone_type else "drone"
        elif self._domain == "ugv":
            ugv_type = self._answers.get("ugv_type", "")
            return ugv_type if ugv_type else "ugv"
        elif self._domain == "robot_arm":
            arm_type = self._answers.get("arm_type", "")
            return f"robot_arm_{arm_type}" if arm_type else "robot_arm"
        return self._domain or ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_qualification_result(result: QualificationResult) -> str:
    """Render a qualification result as formatted text for display."""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  GOAL QUALIFICATION")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Goal: {result.original_goal}")
    if result.domain:
        lines.append(f"  Domain: {result.domain}")
    lines.append(f"  Tangibility: {result.qualification.score:.0%}")
    lines.append("")

    # Dimension scorecard
    lines.append("─" * 72)
    lines.append("  TANGIBILITY SCORECARD")
    lines.append("─" * 72)
    lines.append("")

    dims = [
        ("Platform", result.qualification.platform_identified),
        ("Perception tasks", result.qualification.perception_tasks_enumerated),
        ("Control output", result.qualification.control_output_identified),
        ("Power envelope", result.qualification.power_envelope_bounded),
        ("Environment", result.qualification.environment_characterized),
    ]
    for name, ok in dims:
        mark = "OK" if ok else "MISSING"
        lines.append(f"  {mark:>7}  {name}")
    lines.append("")

    survey_complete = result.next_question is None
    if survey_complete and result.is_tangible:
        lines.append("  *** GOAL IS TANGIBLE — ready for planning ***")
    elif survey_complete and not result.is_tangible:
        missing = result.qualification.missing_dimensions
        lines.append(f"  Survey complete but missing: {', '.join(missing)}")
        lines.append("  Some requirements could not be determined from the answers given.")
    else:
        missing = result.qualification.missing_dimensions
        if missing:
            lines.append(f"  Still needed: {', '.join(missing)}")
        else:
            lines.append("  Core dimensions covered — completing remaining questions...")
    lines.append("")

    # Note: next question details are NOT rendered here — the caller
    # (CLI or chat) presents the question interactively with numbered
    # options. Rendering it here would duplicate the display.

    # Warnings
    if result.warnings:
        lines.append("─" * 72)
        for w in result.warnings:
            lines.append(f"  WARNING: {w}")
        lines.append("")

    # Assumptions
    if result.assumptions:
        lines.append("─" * 72)
        lines.append("  ASSUMPTIONS MADE:")
        for a in result.assumptions:
            lines.append(f"    {a}")
        lines.append("")

    # Progress
    lines.append("─" * 72)
    lines.append(
        f"  Questions answered: {result.questions_asked} | "
        f"Remaining: {result.questions_remaining}"
    )
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nested(data: dict, path: str) -> Any:
    """Get a value from a nested dict using dotted path."""
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current
