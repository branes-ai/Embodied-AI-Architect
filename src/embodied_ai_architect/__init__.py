"""Embodied AI Architect - Design environment for Embodied AI systems.

Hardware/Software Codesign Space Exploration and Optimization Environment.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("embodied-ai-architect")
except PackageNotFoundError:
    # Package not installed (running from source without pip install -e)
    __version__ = "0.0.0.dev"

from .orchestrator import Orchestrator

__all__ = ["Orchestrator", "__version__"]
