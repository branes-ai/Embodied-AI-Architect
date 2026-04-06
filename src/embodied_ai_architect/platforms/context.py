"""Platform context injection for design flows.

Provides functions to search the platform registry for a goal text and
build context strings that can be injected into LLM system prompts or
appended to design state.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_platform_context_for_goal(goal: str) -> dict[str, Any]:
    """Search the platform registry and return context for the best match.

    Returns an empty dict if no match or if the registry is unavailable.
    """
    try:
        from embodied_ai_architect.platforms import PlatformRegistry

        registry = PlatformRegistry()
        matches = registry.search(goal, top_k=3, min_score=0.3)
        if not matches or len(matches[0].matched_keywords) < 2:
            return {}

        top = matches[0]
        return {
            "platform_id": top.platform_id,
            "platform_name": top.platform.name,
            "platform_description": top.platform.description,
            "score": top.score,
            "context": top.platform.context,
            "attributes": top.platform.attributes,
            "classification": top.platform.classification,
            "implications": top.platform.implications,
            "alternatives": [
                {"id": m.platform_id, "name": m.platform.name, "score": m.score}
                for m in matches[1:]
            ],
        }
    except Exception:
        logger.debug("Platform context lookup failed", exc_info=True)
        return {}


def build_context_prompt(platform_ctx: dict[str, Any]) -> str:
    """Build a text block from platform context for LLM injection.

    Returns an empty string if no context available.
    """
    if not platform_ctx:
        return ""

    ctx = platform_ctx.get("context", {})
    if not ctx:
        return ""

    pid = platform_ctx.get("platform_id", "")
    pname = platform_ctx.get("platform_name", "")
    desc = platform_ctx.get("platform_description", "")
    attrs = platform_ctx.get("attributes", {})
    classification = platform_ctx.get("classification", {})

    lines = [
        f"## Platform Context: {pname} ({pid})",
        "",
        desc,
        "",
    ]

    # Classification
    if classification:
        cls_parts = []
        for key in ["locomotion", "manipulation", "load_class", "human_proximity"]:
            val = classification.get(key)
            if val:
                cls_parts.append(f"{key}={val}")
        if cls_parts:
            lines.append(f"Classification: {', '.join(cls_parts)}")

    # Attribute ranges
    if attrs:
        range_parts = []
        for key in ["power_watts", "latency_ms", "cost_usd", "weight_kg"]:
            val = attrs.get(key)
            if isinstance(val, dict) and "typical" in val:
                range_parts.append(
                    f"{key}: {val.get('min', '?')}-{val.get('max', '?')} "
                    f"(typical {val['typical']})"
                )
        if range_parts:
            lines.append("")
            lines.append("Typical attribute ranges:")
            for rp in range_parts:
                lines.append(f"  - {rp}")

    # Architecture
    if ctx.get("typical_architecture"):
        lines.append("")
        lines.append(f"Typical architecture: {ctx['typical_architecture']}")

    # Design considerations
    if ctx.get("design_considerations"):
        lines.append("")
        lines.append(f"Design considerations: {ctx['design_considerations']}")

    # Common pitfalls
    pitfalls = ctx.get("common_pitfalls", [])
    if pitfalls:
        lines.append("")
        lines.append("Common pitfalls:")
        for p in pitfalls[:5]:
            lines.append(f"  - {p}")

    # Regulatory
    regulatory = ctx.get("regulatory", [])
    if regulatory:
        lines.append("")
        lines.append("Regulatory considerations:")
        for r in regulatory[:4]:
            lines.append(f"  - {r}")

    # Reference designs
    refs = ctx.get("reference_designs", [])
    if refs:
        lines.append("")
        lines.append(f"Reference designs: {', '.join(refs[:6])}")

    # Alternatives
    alts = platform_ctx.get("alternatives", [])
    if alts:
        lines.append("")
        alt_strs = [f"{a['name']} ({a['id']}, {a['score']:.2f})" for a in alts]
        lines.append(f"Also consider: {'; '.join(alt_strs)}")

    return "\n".join(lines)
