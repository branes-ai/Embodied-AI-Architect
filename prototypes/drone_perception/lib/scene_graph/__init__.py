"""Scene graph for 3D object state management."""

from .manager import SceneGraphManager
# Alias TrackedObject as SceneGraphNode for reasoning modules
from ..common import TrackedObject as SceneGraphNode

__all__ = ['SceneGraphManager', 'SceneGraphNode']
