"""Goal string parser — extracts numeric constraints from natural language.

Shared helper used by both the GoalQualifier (BUG-006 / #164) and the
``mission new`` CLI command (BUG-007 / #165) so that a goal such as
"Drone perception SoC for YOLO at 30fps under 5W" automatically
populates constraints and spec fields.

Usage:
    from embodied_ai_architect.mission.goal_parser import parse_goal_constraints

    parsed = parse_goal_constraints("Drone perception SoC for YOLO at 30fps under 5W")
    # parsed == {
    #     "power_watts": 5.0,
    #     "latency_ms": 33.333...,
    #     "platform": "drone",
    # }
"""

from __future__ import annotations

import re
from typing import Any

# ── Regex patterns ──────────────────────────────────────────────────────

_POWER_PATTERNS = [
    # "under 5W", "under 5 W", "under 5.0W"
    re.compile(r"under\s+(\d+\.?\d*)\s*[Ww]", re.IGNORECASE),
    # "<5W", "< 5W", "< 5.0 W"
    re.compile(r"<\s*(\d+\.?\d*)\s*[Ww]", re.IGNORECASE),
    # "5W budget", "5 W power", plain "5W" (must not be preceded by digit)
    re.compile(r"(?<!\d)(\d+\.?\d*)\s*[Ww](?:att)?s?\b", re.IGNORECASE),
]

_FPS_PATTERN = re.compile(r"(\d+)\s*fps", re.IGNORECASE)

_COST_PATTERNS = [
    # "<$100", "< $100"
    re.compile(r"<\s*\$\s*(\d+\.?\d*)", re.IGNORECASE),
    # "under $100"
    re.compile(r"under\s+\$\s*(\d+\.?\d*)", re.IGNORECASE),
]

_LATENCY_PATTERN = re.compile(r"(\d+\.?\d*)\s*ms", re.IGNORECASE)

# Platform keyword → canonical platform name
_PLATFORM_KEYWORDS: dict[str, str] = {
    "drone": "drone",
    "uav": "drone",
    "quadrotor": "drone",
    "multirotor": "drone",
    "robot": "robot_arm",
    "cobot": "robot_arm",
    "robotic arm": "robot_arm",
    "amr": "ugv",
    "warehouse": "ugv",
    "ugv": "ugv",
    "ground vehicle": "ugv",
}


def parse_goal_constraints(goal: str) -> dict[str, Any]:
    """Extract numeric constraints and platform from a goal string.

    Returns a dict that may contain any subset of:
        - ``power_watts`` (float)
        - ``latency_ms`` (float)
        - ``cost_usd`` (float)
        - ``platform`` (str)
    """
    result: dict[str, Any] = {}
    if not goal:
        return result

    # ── Power ───────────────────────────────────────────────────────
    for pat in _POWER_PATTERNS:
        m = pat.search(goal)
        if m:
            result["power_watts"] = float(m.group(1))
            break

    # ── FPS → latency ───────────────────────────────────────────────
    m = _FPS_PATTERN.search(goal)
    if m:
        fps = int(m.group(1))
        if fps > 0:
            result["latency_ms"] = 1000.0 / fps

    # ── Explicit latency in ms (only if no fps was found) ───────────
    if "latency_ms" not in result:
        m = _LATENCY_PATTERN.search(goal)
        if m:
            result["latency_ms"] = float(m.group(1))

    # ── Cost ────────────────────────────────────────────────────────
    for pat in _COST_PATTERNS:
        m = pat.search(goal)
        if m:
            result["cost_usd"] = float(m.group(1))
            break

    # ── Platform ────────────────────────────────────────────────────
    goal_lower = goal.lower()
    for keyword, platform in _PLATFORM_KEYWORDS.items():
        if keyword in goal_lower:
            result["platform"] = platform
            break

    return result
