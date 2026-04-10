"""Shared utilities for CLI commands."""


def get_attr_typical(attributes: dict, key: str) -> float | None:
    """Extract the typical value from a min/max/typical attribute dict."""
    val = attributes.get(key)
    if isinstance(val, dict):
        typical = val.get("typical")
        if isinstance(typical, (int, float)) and not isinstance(typical, bool):
            return float(typical)
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None
