"""Platform context injection for design flows.

Provides functions to search the platform registry for a goal text and
build context strings that can be injected into LLM system prompts or
appended to design state. Supports composing context from multiple
matched platforms for hybrid systems (e.g., "mobile robot with arm").
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_platform_context_for_goal(
    goal: str, min_score: float = 0.3, max_platforms: int = 3
) -> dict[str, Any]:
    """Search the platform registry and return composed context.

    When multiple platforms match above the threshold and belong to
    different categories, their contexts are merged to support hybrid
    system descriptions (e.g., "mobile robot with manipulator arm").

    Returns an empty dict if no match or if the registry is unavailable.
    """
    try:
        from embodied_ai_architect.platforms import PlatformRegistry

        registry = PlatformRegistry()
        matches = registry.search(goal, top_k=max_platforms + 2, min_score=min_score)
        if not matches or len(matches[0].matched_keywords) < 2:
            return {}

        # Find matches from distinct categories above threshold
        seen_categories: set[str] = set()
        primary_matches = []
        for m in matches:
            if m.platform.category not in seen_categories and m.score >= min_score:
                primary_matches.append(m)
                seen_categories.add(m.platform.category)
                if len(primary_matches) >= max_platforms:
                    break

        if len(primary_matches) == 1:
            # Single match — return as before
            top = primary_matches[0]
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
                    for m in matches
                    if m.platform_id != top.platform_id
                ],
            }

        # Multiple distinct-category matches — compose
        return _compose_multi_platform_context(primary_matches, matches)

    except Exception:
        logger.debug("Platform context lookup failed", exc_info=True)
        return {}


def _compose_multi_platform_context(primary: list, all_matches: list) -> dict[str, Any]:
    """Compose context from multiple platform matches into a unified dict."""
    platform_ids = [m.platform_id for m in primary]
    platform_names = [m.platform.name for m in primary]

    # Merge attributes — use tighter constraints (lower max, higher min)
    merged_attrs = _merge_attributes([m.platform.attributes for m in primary])

    # Merge classification — union of environments, keep all axes
    merged_class = _merge_classifications([m.platform.classification for m in primary])

    # Merge implications — union of perception tasks, sensors, etc.
    merged_impl = _merge_implications([m.platform.implications for m in primary])

    # Merge context blocks
    merged_context = _merge_context_blocks([m.platform.context for m in primary])

    return {
        "platform_id": " + ".join(platform_ids),
        "platform_name": " + ".join(platform_names),
        "platform_description": "; ".join(
            m.platform.description for m in primary if m.platform.description
        ),
        "score": primary[0].score,
        "context": merged_context,
        "attributes": merged_attrs,
        "classification": merged_class,
        "implications": merged_impl,
        "multi_platform": True,
        "platforms": [
            {"id": m.platform_id, "name": m.platform.name, "score": m.score} for m in primary
        ],
        "alternatives": [
            {"id": m.platform_id, "name": m.platform.name, "score": m.score}
            for m in all_matches
            if m.platform_id not in set(platform_ids)
        ],
    }


def _merge_attributes(attr_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge attribute ranges — use tighter constraints."""
    merged: dict[str, Any] = {}
    all_keys: set[str] = set()
    for attrs in attr_list:
        all_keys.update(attrs.keys())

    for key in all_keys:
        values = [a[key] for a in attr_list if key in a]
        if not values:
            continue

        dicts = [v for v in values if isinstance(v, dict)]
        if dicts:
            # Merge range dicts — tighter constraint: higher min, lower max
            mins = [d["min"] for d in dicts if "min" in d]
            maxs = [d["max"] for d in dicts if "max" in d]
            typs = [d["typical"] for d in dicts if "typical" in d]
            merged[key] = {}
            if mins:
                merged[key]["min"] = max(mins)  # tighter: higher minimum
            if maxs:
                merged[key]["max"] = min(maxs)  # tighter: lower maximum
            if typs:
                merged[key]["typical"] = sum(typs) / len(typs)
        else:
            # Non-range value — take first
            merged[key] = values[0]

    return merged


def _merge_classifications(class_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge classifications — union lists, keep all distinct values."""
    merged: dict[str, Any] = {}
    for cls in class_list:
        for key, val in cls.items():
            if key not in merged:
                merged[key] = val
            elif isinstance(val, list) and isinstance(merged[key], list):
                merged[key] = list(set(merged[key] + val))
            elif val != merged[key]:
                # Different scalar values — make a list
                existing = merged[key] if isinstance(merged[key], list) else [merged[key]]
                if val not in existing:
                    existing.append(val)
                merged[key] = existing
    return merged


def _merge_implications(impl_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge implications — union perception tasks, sensor types, etc."""
    merged: dict[str, Any] = {}
    for impl in impl_list:
        for key, val in impl.items():
            if key not in merged:
                merged[key] = val
            elif isinstance(val, dict) and isinstance(merged[key], dict):
                # Recursively merge sub-dicts
                for subkey, subval in val.items():
                    if subkey not in merged[key]:
                        merged[key][subkey] = subval
                    elif isinstance(subval, list) and isinstance(merged[key][subkey], list):
                        merged[key][subkey] = list(set(merged[key][subkey] + subval))
                    elif isinstance(subval, (int, float)) and isinstance(
                        merged[key][subkey], (int, float)
                    ):
                        # Numeric — take tighter (lower for latency, higher for fps)
                        if "latency" in subkey or "max" in subkey:
                            merged[key][subkey] = min(merged[key][subkey], subval)
                        else:
                            merged[key][subkey] = max(merged[key][subkey], subval)
    return merged


def _merge_context_blocks(ctx_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge context blocks — concatenate text, union lists."""
    merged: dict[str, Any] = {}
    for ctx in ctx_list:
        if not ctx:
            continue
        for key, val in ctx.items():
            if key not in merged:
                merged[key] = val
            elif isinstance(val, list) and isinstance(merged[key], list):
                merged[key] = list(set(merged[key] + val))
            elif isinstance(val, str) and isinstance(merged[key], str):
                if val not in merged[key]:
                    merged[key] = merged[key] + "; " + val
    return merged


def build_context_prompt(platform_ctx: dict[str, Any]) -> str:
    """Build a text block from platform context for LLM injection.

    Handles both single-platform and multi-platform (composed) contexts.
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
    is_multi = platform_ctx.get("multi_platform", False)

    if is_multi:
        lines = [
            f"## Composed Platform Context: {pname}",
            "(Hybrid system matching multiple platform categories)",
            "",
            desc,
            "",
        ]
    else:
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
                if isinstance(val, list):
                    cls_parts.append(f"{key}={', '.join(str(v) for v in val)}")
                else:
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
                    f"(typical {val['typical']:.0f})"
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
        alt_strs = [f"{a['name']} ({a['id']}, {a['score']:.2f})" for a in alts[:5]]
        lines.append(f"Also consider: {'; '.join(alt_strs)}")

    return "\n".join(lines)
