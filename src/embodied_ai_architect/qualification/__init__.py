"""Goal qualification system for embodied AI design.

Validates that a design goal is specific enough to produce meaningful
results before committing to planning and execution. Uses domain-specific
question templates to guide the user through structured refinement.

Usage:
    from embodied_ai_architect.qualification import (
        GoalQualifier,
        QualificationResult,
        get_domain_template,
        list_domains,
    )

    qualifier = GoalQualifier()
    result = qualifier.assess("drone perception SoC")
    if not result.is_tangible:
        next_q = result.next_question
        # ask user, then:
        result = qualifier.answer(next_q.id, "multirotor_delivery")
"""

from .models import (
    CustomAnswer,
    GoalQualification,
    QualificationResult,
    Question,
    QuestionType,
)
from .qualifier import GoalQualifier
from .templates import get_domain_template, list_domains

__all__ = [
    "CustomAnswer",
    "GoalQualification",
    "GoalQualifier",
    "QualificationResult",
    "Question",
    "QuestionType",
    "get_domain_template",
    "list_domains",
]
