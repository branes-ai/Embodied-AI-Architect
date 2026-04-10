"""Shared utilities for CLI commands."""


def get_attr_typical(attributes: dict, key: str) -> float | None:
    """Extract the typical value from a min/max/typical attribute dict."""
    val = attributes.get(key)
    if isinstance(val, dict):
        return val.get("typical")
    if isinstance(val, (int, float)):
        return float(val)
    return None
