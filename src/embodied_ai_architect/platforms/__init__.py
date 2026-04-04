"""Platform Registry — file-based catalog of embodied AI platform definitions.

Loads platform YAML files from data/platforms/, indexes keywords, and provides
a scoring/matching engine that returns ranked platform candidates for a given
user prompt.

Usage:
    from embodied_ai_architect.platforms import PlatformRegistry

    registry = PlatformRegistry()
    matches = registry.search("autonomous sprayer for vineyards")
    # [MatchResult(platform_id='agriculture.pesticide_ground_sprayer', score=0.85, ...), ...]
"""

from embodied_ai_architect.platforms.registry import (
    PlatformDefinition,
    MatchResult,
    PlatformRegistry,
)

__all__ = ["PlatformDefinition", "MatchResult", "PlatformRegistry"]
