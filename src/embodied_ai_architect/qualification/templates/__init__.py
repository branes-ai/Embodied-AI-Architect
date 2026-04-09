"""Domain-specific question templates for goal qualification.

Each template encodes the engineering knowledge needed to refine a vague
goal into a tangible, actionable specification for a specific platform domain.

Templates are loaded from `_questions.yaml` files in the platform data
directories (issue #49). The YAML files contain the same question trees
that were previously hard-coded in drone.py, ugv.py, and robot_arm.py —
with the same conditional logic (depends_on), option descriptions, and
rich implication mappings.

New domains can be added by:
  1. Creating a `_questions.yaml` in the appropriate platform category dir
  2. OR calling `register_template()` at runtime
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from embodied_ai_architect.qualification.models import (
    DomainTemplate,
    Question,
    QuestionType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML → DomainTemplate loader
# ---------------------------------------------------------------------------

# Platform data directory — resolved relative to the package root.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "platforms"

# Mapping from platform-category directory name to the domain identifier
# that the qualifier knows (e.g., "aerial" → "drone"). Categories not in
# this map use their directory name verbatim.
_CATEGORY_TO_DOMAIN: dict[str, str] = {
    "aerial": "drone",
    "ground_wheeled": "ugv",
    "ground_legged": "ugv",
    "ground_tracked": "ugv",
    "manipulation": "robot_arm",
    "surgical": "robot_arm",
    "lab": "robot_arm",
    "veterinary": "robot_arm",
}


def _question_from_dict(d: dict[str, Any]) -> Question:
    """Build a Question from a YAML dict.

    Only passes non-None values so Pydantic defaults are respected for
    optional fields (custom_prompt defaults to "" not None).
    """
    kwargs: dict[str, Any] = {
        "id": d["id"],
        "dimension": d.get("dimension", "platform"),
        "text": d.get("text", ""),
        "question_type": QuestionType(d.get("question_type", "single_choice")),
        "options": d.get("options", []),
        "option_descriptions": d.get("option_descriptions", {}),
        "required": d.get("required", False),
        "implications": d.get("implications", {}),
        "allow_custom": d.get("allow_custom", False),
    }
    # Only set optional fields when they're actually present and non-None
    # so the Pydantic model's defaults are used otherwise.
    for key in (
        "explanation",
        "default",
        "depends_on",
        "depends_on_value",
        "min_selections",
        "custom_prompt",
    ):
        val = d.get(key)
        if val is not None:
            kwargs[key] = val

    return Question(**kwargs)


def _load_template_from_yaml(path: Path) -> Optional[DomainTemplate]:
    """Load a DomainTemplate from a _questions.yaml file."""
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        questions = [_question_from_dict(qd) for qd in data.get("questions", [])]

        return DomainTemplate(
            domain=data.get("domain", path.parent.name),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            base_implications=data.get("base_implications", {}),
            questions=questions,
        )
    except Exception as e:
        logger.warning("Failed to load template from %s: %s", path, e)
        return None


def _discover_templates() -> dict[str, DomainTemplate]:
    """Scan the data/platforms directory for _questions.yaml files and load them."""
    templates: dict[str, DomainTemplate] = {}

    if not _DATA_DIR.is_dir():
        logger.debug("Platform data directory not found: %s", _DATA_DIR)
        return templates

    for questions_file in _DATA_DIR.rglob("_questions.yaml"):
        tmpl = _load_template_from_yaml(questions_file)
        if tmpl is not None:
            templates[tmpl.domain] = tmpl
            logger.debug("Loaded domain template '%s' from %s", tmpl.domain, questions_file)

    return templates


# ---------------------------------------------------------------------------
# Module-level registry — populated from YAML at import time
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, DomainTemplate] = _discover_templates()


def get_domain_template(domain: str) -> Optional[DomainTemplate]:
    """Get a template by domain identifier."""
    return _TEMPLATES.get(domain)


def list_domains() -> list[str]:
    """List all available domain identifiers."""
    return list(_TEMPLATES.keys())


def get_all_templates() -> dict[str, DomainTemplate]:
    """Get all registered templates."""
    return dict(_TEMPLATES)


def detect_domain(goal_text: str) -> Optional[str]:
    """Detect the most likely domain from goal text keywords.

    First checks domain templates (loaded from _questions.yaml files).
    If no match, falls back to the platform registry for broader coverage.

    Returns the domain identifier or None if no match.
    """
    goal_lower = goal_text.lower()
    best_match = None
    best_score = 0

    for domain, template in _TEMPLATES.items():
        score = sum(1 for kw in template.keywords if kw in goal_lower)
        if score > best_score:
            best_score = score
            best_match = domain

    if best_match:
        return best_match

    # Fall back to platform registry search
    return _detect_domain_from_registry(goal_text)


def detect_domain_with_context(goal_text: str) -> tuple[Optional[str], dict]:
    """Detect domain and return platform context for the best match.

    Returns (domain_name, context_dict). The context_dict contains
    domain knowledge from the matched platform definition that can be
    loaded into the qualification/design planning flow.
    """
    goal_lower = goal_text.lower()
    best_match = None
    best_score = 0

    for domain, template in _TEMPLATES.items():
        score = sum(1 for kw in template.keywords if kw in goal_lower)
        if score > best_score:
            best_score = score
            best_match = domain

    if best_match:
        return best_match, {}

    # Fall back to platform registry — returns richer context
    try:
        from embodied_ai_architect.platforms import PlatformRegistry

        registry = PlatformRegistry()
        matches = registry.search(goal_text, top_k=3, min_score=0.3)
        if matches and len(matches[0].matched_keywords) >= 2:
            top = matches[0]
            # Map platform category to closest domain template, or use category
            domain = _CATEGORY_TO_DOMAIN.get(top.platform.category, top.platform.category)
            context = {
                "platform_id": top.platform_id,
                "platform_name": top.platform.name,
                "score": top.score,
                "matched_keywords": top.matched_keywords,
                "implications": top.platform.implications,
                "context": top.platform.context,
                "attributes": top.platform.attributes,
                "classification": top.platform.classification,
                "alternatives": [
                    {"id": m.platform_id, "name": m.platform.name, "score": m.score}
                    for m in matches[1:]
                ],
            }
            return domain, context
    except Exception:
        # Registry unavailable or failed — degrade gracefully to no match
        logger.debug("Platform registry lookup failed", exc_info=True)

    return None, {}


def _detect_domain_from_registry(goal_text: str) -> Optional[str]:
    """Try to detect domain using the platform registry."""
    try:
        from embodied_ai_architect.platforms import PlatformRegistry

        registry = PlatformRegistry()
        matches = registry.search(goal_text, top_k=1, min_score=0.3)
        if matches and len(matches[0].matched_keywords) >= 2:
            return _CATEGORY_TO_DOMAIN.get(
                matches[0].platform.category, matches[0].platform.category
            )
    except Exception:
        # Registry unavailable or failed — degrade gracefully to no match
        logger.debug("Platform registry fallback failed", exc_info=True)
    return None


def register_template(template: DomainTemplate) -> None:
    """Register a new domain template at runtime.

    This is the extension mechanism for adding new platform domains
    without modifying the core code.
    """
    _TEMPLATES[template.domain] = template
