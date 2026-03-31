"""Data models for goal qualification.

Defines the question templates, qualification state, and result structures.
These are pure data — no business logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    """How a question is presented and answered."""

    SINGLE_CHOICE = "single_choice"  # Pick one from options
    MULTI_CHOICE = "multi_choice"  # Pick one or more
    NUMERIC = "numeric"  # Enter a number
    TEXT = "text"  # Free text
    YES_NO = "yes_no"  # Boolean


class Question(BaseModel):
    """A single question in a domain template.

    Questions carry domain knowledge via `implications`: when a user picks
    an answer, the implications dict pre-fills downstream fields, potentially
    skipping later questions.
    """

    id: str = Field(description="Unique question identifier within the template")
    dimension: str = Field(
        description="Which tangibility dimension this addresses: "
        "platform, perception, control, power, environment, safety, architecture"
    )
    text: str = Field(description="Human-readable question text")
    question_type: QuestionType = Field(default=QuestionType.SINGLE_CHOICE)
    options: list[str] = Field(
        default_factory=list,
        description="Available choices (for single/multi choice)",
    )
    option_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Optional descriptions for each option",
    )
    required: bool = Field(default=True, description="Must be answered to proceed")
    default: Optional[str] = Field(default=None, description="Default answer if skipped")
    depends_on: Optional[str] = Field(
        default=None,
        description="Question ID that must be answered first (conditional question)",
    )
    depends_on_value: Optional[str] = Field(
        default=None,
        description="Only show if depends_on question was answered with this value",
    )
    implications: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="answer_value -> {field: value} to pre-fill in the spec",
    )
    min_selections: int = Field(
        default=1,
        description="Minimum selections for multi_choice",
    )
    numeric_min: Optional[float] = Field(default=None, description="Min value for numeric")
    numeric_max: Optional[float] = Field(default=None, description="Max value for numeric")
    numeric_unit: str = Field(default="", description="Unit label for numeric questions")
    explanation: str = Field(
        default="",
        description="Why this question matters — shown to the user as context",
    )


# ---------------------------------------------------------------------------
# Domain template
# ---------------------------------------------------------------------------


class DomainTemplate(BaseModel):
    """A complete question template for a platform domain.

    Templates are the primary mechanism for encoding domain-specific
    engineering knowledge. Each template defines the questions needed to
    qualify a goal for that domain, ordered by dependency.
    """

    domain: str = Field(description="Domain identifier (e.g., 'drone', 'ugv', 'robot_arm')")
    display_name: str = Field(description="Human-readable domain name")
    description: str = Field(description="What this domain covers")
    keywords: list[str] = Field(description="Keywords that trigger this domain in goal text")
    questions: list[Question] = Field(description="Ordered list of questions")
    base_implications: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields pre-filled just by selecting this domain",
    )

    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    @property
    def dimensions_covered(self) -> set[str]:
        """Which tangibility dimensions this template covers."""
        return {q.dimension for q in self.questions}


# ---------------------------------------------------------------------------
# Qualification result
# ---------------------------------------------------------------------------


class GoalQualification(BaseModel):
    """Per-dimension tangibility assessment."""

    platform_identified: bool = Field(
        default=False,
        description="Platform type resolvable to a specific form factor and subtype",
    )
    perception_tasks_enumerated: bool = Field(
        default=False,
        description="At least one perception task with a quality target",
    )
    control_output_identified: bool = Field(
        default=False,
        description="What downstream system consumes perception output",
    )
    power_envelope_bounded: bool = Field(
        default=False,
        description="Power budget exists or is derivable from platform + mission",
    )
    environment_characterized: bool = Field(
        default=False,
        description="Enough info to determine sensor modalities and protection",
    )

    @property
    def is_tangible(self) -> bool:
        """A goal is tangible when we can enumerate the hw/sw components."""
        return all(
            [
                self.platform_identified,
                self.perception_tasks_enumerated,
                self.control_output_identified,
                self.power_envelope_bounded,
            ]
        )
        # environment_characterized is soft — can be inferred from platform

    @property
    def score(self) -> float:
        """Tangibility score 0.0-1.0."""
        fields = [
            self.platform_identified,
            self.perception_tasks_enumerated,
            self.control_output_identified,
            self.power_envelope_bounded,
            self.environment_characterized,
        ]
        return sum(fields) / len(fields)

    @property
    def missing_dimensions(self) -> list[str]:
        """Which dimensions are still unspecified."""
        missing = []
        if not self.platform_identified:
            missing.append("platform")
        if not self.perception_tasks_enumerated:
            missing.append("perception")
        if not self.control_output_identified:
            missing.append("control")
        if not self.power_envelope_bounded:
            missing.append("power")
        if not self.environment_characterized:
            missing.append("environment")
        return missing


class QualificationResult(BaseModel):
    """Complete result of a qualification assessment."""

    qualification: GoalQualification
    original_goal: str = ""
    refined_goal: str = ""
    domain: Optional[str] = None
    answers: dict[str, Any] = Field(
        default_factory=dict,
        description="question_id -> answer_value for all answered questions",
    )
    accumulated_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated SystemSpec fields from implications",
    )
    next_question: Optional[Question] = None
    questions_asked: int = 0
    questions_remaining: int = 0
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking concerns about the goal",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made when user skipped questions",
    )

    @property
    def is_tangible(self) -> bool:
        return self.qualification.is_tangible
