"""
Spatial relationship reasoning for 3D scene understanding.

Analyzes spatial relationships between objects:
- Relative positions (in front, behind, left, right, above, below)
- Distances and proximity
- Object groupings and clusters
- Scene structure understanding
"""

from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..scene_graph import SceneGraphNode


class RelativePosition(Enum):
    """Relative spatial positions."""
    IN_FRONT = "in_front"
    BEHIND = "behind"
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"
    SAME_LEVEL = "same_level"


@dataclass
class SpatialRelation:
    """Spatial relationship between two objects."""

    object_a_id: int
    object_b_id: int
    object_a_class: str
    object_b_class: str

    # Relationships
    relative_position: RelativePosition
    distance: float  # meters

    # Additional context
    relative_angle: float  # degrees (0-360)
    height_difference: float  # meters (positive if A is above B)

    # Confidence
    confidence: float = 1.0

    def describe(self) -> str:
        """Generate natural language description."""
        a_class = self.object_a_class
        b_class = self.object_b_class

        if self.relative_position == RelativePosition.IN_FRONT:
            return f"{a_class} is in front of {b_class} ({self.distance:.1f}m)"
        elif self.relative_position == RelativePosition.BEHIND:
            return f"{a_class} is behind {b_class} ({self.distance:.1f}m)"
        elif self.relative_position == RelativePosition.LEFT:
            return f"{a_class} is to the left of {b_class} ({self.distance:.1f}m)"
        elif self.relative_position == RelativePosition.RIGHT:
            return f"{a_class} is to the right of {b_class} ({self.distance:.1f}m)"
        elif self.relative_position == RelativePosition.ABOVE:
            return f"{a_class} is above {b_class} ({self.height_difference:.1f}m higher)"
        elif self.relative_position == RelativePosition.BELOW:
            return f"{a_class} is below {b_class} ({self.height_difference:.1f}m lower)"
        elif self.relative_position == RelativePosition.NEAR:
            return f"{a_class} is near {b_class} ({self.distance:.1f}m)"
        elif self.relative_position == RelativePosition.FAR:
            return f"{a_class} is far from {b_class} ({self.distance:.1f}m)"
        else:
            return f"{a_class} and {b_class} are {self.distance:.1f}m apart"


class SpatialAnalyzer:
    """
    Analyzes spatial relationships in 3D scenes.

    Provides:
    - Relative position detection
    - Proximity analysis
    - Object grouping/clustering
    - Scene structure understanding
    """

    def __init__(
        self,
        near_threshold: float = 3.0,  # meters
        far_threshold: float = 10.0,  # meters
        height_threshold: float = 0.5,  # meters for above/below
        reference_frame: str = "world"  # "world" or "camera"
    ):
        """
        Initialize spatial analyzer.

        Args:
            near_threshold: Distance threshold for "near" classification
            far_threshold: Distance threshold for "far" classification
            height_threshold: Height difference for above/below detection
            reference_frame: Coordinate frame for spatial relations
        """
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.height_threshold = height_threshold
        self.reference_frame = reference_frame

    def analyze_pair(
        self,
        node_a: SceneGraphNode,
        node_b: SceneGraphNode
    ) -> Optional[SpatialRelation]:
        """
        Analyze spatial relationship between two objects.

        Args:
            node_a: First object
            node_b: Second object

        Returns:
            SpatialRelation describing their relationship
        """
        if node_a.position_3d is None or node_b.position_3d is None:
            return None

        pos_a = node_a.position_3d
        pos_b = node_b.position_3d

        # Calculate distance
        diff = pos_a - pos_b
        distance = np.linalg.norm(diff)

        if distance < 0.01:  # Same position
            return None

        # Calculate relative angle (in XY plane)
        angle_rad = np.arctan2(diff[1], diff[0])
        angle_deg = np.degrees(angle_rad) % 360

        # Height difference
        height_diff = diff[2]

        # Determine primary relationship
        relative_pos = self._classify_position(diff, distance, height_diff)

        return SpatialRelation(
            object_a_id=node_a.track_id,
            object_b_id=node_b.track_id,
            object_a_class=node_a.class_name,
            object_b_class=node_b.class_name,
            relative_position=relative_pos,
            distance=distance,
            relative_angle=angle_deg,
            height_difference=height_diff
        )

    def _classify_position(
        self,
        diff: np.ndarray,
        distance: float,
        height_diff: float
    ) -> RelativePosition:
        """
        Classify relative position based on 3D offset.

        Args:
            diff: Position difference vector (A - B)
            distance: Euclidean distance
            height_diff: Height difference (z component)

        Returns:
            RelativePosition classification
        """
        # Check vertical relationship first
        if abs(height_diff) > self.height_threshold:
            if height_diff > 0:
                return RelativePosition.ABOVE
            else:
                return RelativePosition.BELOW

        # Check horizontal distance
        horizontal_dist = np.linalg.norm(diff[:2])

        if horizontal_dist < self.near_threshold:
            return RelativePosition.NEAR
        elif horizontal_dist > self.far_threshold:
            return RelativePosition.FAR

        # Classify based on angle (in camera frame)
        # Assuming camera looks along +X axis
        angle_rad = np.arctan2(diff[1], diff[0])
        angle_deg = np.degrees(angle_rad) % 360

        # In front: -45° to +45°
        if angle_deg < 45 or angle_deg > 315:
            return RelativePosition.IN_FRONT
        # Right: 45° to 135°
        elif 45 <= angle_deg < 135:
            return RelativePosition.RIGHT
        # Behind: 135° to 225°
        elif 135 <= angle_deg < 225:
            return RelativePosition.BEHIND
        # Left: 225° to 315°
        else:
            return RelativePosition.LEFT

    def analyze_all_pairs(
        self,
        nodes: List[SceneGraphNode]
    ) -> List[SpatialRelation]:
        """
        Analyze all pairwise spatial relationships.

        Args:
            nodes: List of scene graph nodes

        Returns:
            List of spatial relations
        """
        relations = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                relation = self.analyze_pair(nodes[i], nodes[j])
                if relation is not None:
                    relations.append(relation)

        return relations

    def find_nearest_objects(
        self,
        reference_node: SceneGraphNode,
        all_nodes: List[SceneGraphNode],
        k: int = 3
    ) -> List[Tuple[SceneGraphNode, float]]:
        """
        Find K nearest objects to a reference object.

        Args:
            reference_node: Reference object
            all_nodes: All objects in scene
            k: Number of nearest neighbors

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        if reference_node.position_3d is None:
            return []

        distances = []

        for node in all_nodes:
            if node.track_id == reference_node.track_id:
                continue

            if node.position_3d is None:
                continue

            dist = np.linalg.norm(node.position_3d - reference_node.position_3d)
            distances.append((node, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Return top K
        return distances[:k]

    def find_clusters(
        self,
        nodes: List[SceneGraphNode],
        max_cluster_distance: float = 5.0
    ) -> List[Set[int]]:
        """
        Find spatial clusters of objects.

        Uses simple distance-based clustering.

        Args:
            nodes: List of scene graph nodes
            max_cluster_distance: Maximum distance for same cluster

        Returns:
            List of sets of track IDs (one set per cluster)
        """
        # Build distance matrix
        n = len(nodes)
        positions = []
        track_ids = []

        for node in nodes:
            if node.position_3d is not None:
                positions.append(node.position_3d)
                track_ids.append(node.track_id)

        if len(positions) == 0:
            return []

        positions = np.array(positions)

        # Simple greedy clustering
        clusters = []
        assigned = set()

        for i in range(len(positions)):
            if i in assigned:
                continue

            # Start new cluster
            cluster = {track_ids[i]}
            assigned.add(i)

            # Add nearby points
            for j in range(len(positions)):
                if j in assigned:
                    continue

                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= max_cluster_distance:
                    cluster.add(track_ids[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def describe_scene(
        self,
        nodes: List[SceneGraphNode],
        drone_position: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate natural language description of scene.

        Args:
            nodes: Scene graph nodes
            drone_position: Optional drone position for reference

        Returns:
            Text description of scene
        """
        if len(nodes) == 0:
            return "No objects detected in scene."

        # Count objects by class
        class_counts = {}
        for node in nodes:
            class_counts[node.class_name] = class_counts.get(node.class_name, 0) + 1

        # Build description
        desc = f"Scene contains {len(nodes)} objects:\n"

        for class_name, count in sorted(class_counts.items()):
            desc += f"  - {count} {class_name}(s)\n"

        # Find clusters
        clusters = self.find_clusters(nodes)
        if len(clusters) > 1:
            desc += f"\nObjects form {len(clusters)} spatial groups.\n"

        # Describe nearest objects to drone
        if drone_position is not None and len(nodes) > 0:
            # Find closest object
            min_dist = float('inf')
            closest_node = None

            for node in nodes:
                if node.position_3d is not None:
                    dist = np.linalg.norm(node.position_3d - drone_position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = node

            if closest_node is not None:
                desc += f"\nClosest object: {closest_node.class_name} at {min_dist:.1f}m\n"

        return desc
