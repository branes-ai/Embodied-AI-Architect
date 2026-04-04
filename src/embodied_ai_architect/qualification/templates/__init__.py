"""Domain-specific question templates for goal qualification.

Each template encodes the engineering knowledge needed to refine a vague
goal into a tangible, actionable specification for a specific platform domain.

Templates are designed to show a progression of system complexity:
- Drone: single compute node, battery-constrained, mostly perception + control
- UGV: multi-sensor, indoor/outdoor, navigation + manipulation optional
- Robot arm: dual-nervous-system architecture (conscious brain + peripheral
  joint controllers), safety-critical cobot requirements

New domains can be added by creating a module that exports TEMPLATE.
"""

from __future__ import annotations

import logging
from typing import Optional

from embodied_ai_architect.qualification.models import DomainTemplate

from .drone import TEMPLATE as DRONE_TEMPLATE
from .ugv import TEMPLATE as UGV_TEMPLATE
from .robot_arm import TEMPLATE as ROBOT_ARM_TEMPLATE

logger = logging.getLogger(__name__)

# Registry of all domain templates
_TEMPLATES: dict[str, DomainTemplate] = {
    DRONE_TEMPLATE.domain: DRONE_TEMPLATE,
    UGV_TEMPLATE.domain: UGV_TEMPLATE,
    ROBOT_ARM_TEMPLATE.domain: ROBOT_ARM_TEMPLATE,
}


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

    First checks hard-coded domain templates (drone, ugv, robot_arm).
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
            domain = _map_category_to_domain(top.platform.category)
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
            return _map_category_to_domain(matches[0].platform.category)
    except Exception:
        # Registry unavailable or failed — degrade gracefully to no match
        logger.debug("Platform registry fallback failed", exc_info=True)
    return None


def _map_category_to_domain(category: str) -> str:
    """Map a platform registry category to the closest domain template.

    If no domain template exists for the category, return the category
    name itself — the qualifier will handle it as an unrecognized domain
    and fall back to manual selection.
    """
    _CATEGORY_TO_DOMAIN = {
        "aerial": "drone",
        "ground_wheeled": "ugv",
        "ground_legged": "ugv",
        "ground_tracked": "ugv",
        "manipulation": "robot_arm",
        "surgical": "robot_arm",
        "lab": "robot_arm",
        "veterinary": "robot_arm",
    }
    return _CATEGORY_TO_DOMAIN.get(category, category)


def register_template(template: DomainTemplate) -> None:
    """Register a new domain template at runtime.

    This is the extension mechanism for adding new platform domains
    without modifying the core code.
    """
    _TEMPLATES[template.domain] = template
