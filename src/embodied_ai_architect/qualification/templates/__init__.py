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

from typing import Optional

from embodied_ai_architect.qualification.models import DomainTemplate

from .drone import TEMPLATE as DRONE_TEMPLATE
from .ugv import TEMPLATE as UGV_TEMPLATE
from .robot_arm import TEMPLATE as ROBOT_ARM_TEMPLATE

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

    return best_match if best_score > 0 else None


def register_template(template: DomainTemplate) -> None:
    """Register a new domain template at runtime.

    This is the extension mechanism for adding new platform domains
    without modifying the core code.
    """
    _TEMPLATES[template.domain] = template
