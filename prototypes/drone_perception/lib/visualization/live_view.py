"""Real-time 3D visualization of tracked objects."""

from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ..common import TrackedObject


class LiveViewer3D:
    """
    Real-time 3D matplotlib viewer for scene graph.

    Displays:
    - Object positions (scatter points)
    - Velocity vectors (arrows)
    - Trajectories (line trails)
    - Object labels
    """

    def __init__(
        self,
        xlim: tuple = (-10, 10),
        ylim: tuple = (-10, 10),
        zlim: tuple = (0, 20),
        show_velocity: bool = True,
        show_trajectories: bool = True,
        figsize: tuple = (12, 8)
    ):
        """
        Initialize 3D viewer.

        Args:
            xlim: X-axis limits (meters)
            ylim: Y-axis limits (meters)
            zlim: Z-axis limits (meters)
            show_velocity: Show velocity vectors
            show_trajectories: Show position trails
            figsize: Figure size
        """
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.show_velocity = show_velocity
        self.show_trajectories = show_trajectories

        # Create figure
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Styling
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.set_zlabel('Z (meters)', fontsize=10)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        # Camera perspective (looking forward)
        self.ax.view_init(elev=20, azim=-90)

        # Color map for different classes
        self.class_colors = {}
        self.color_palette = plt.cm.tab10

    def render(self, objects: List[TrackedObject], frame_info: str = ""):
        """
        Render current scene.

        Args:
            objects: List of tracked objects
            frame_info: Optional info string to display
        """
        self.ax.clear()

        # Reset limits and labels
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')

        # Draw objects
        for obj in objects:
            color = self._get_class_color(obj.class_name)

            # Plot position
            self.ax.scatter(
                obj.position[0],
                obj.position[1],
                obj.position[2],
                c=[color],
                s=100,
                marker='o',
                alpha=0.8
            )

            # Plot trajectory
            if self.show_trajectories and len(obj.position_history) > 1:
                history = np.array(obj.position_history)
                self.ax.plot(
                    history[:, 0],
                    history[:, 1],
                    history[:, 2],
                    c=color,
                    alpha=0.4,
                    linewidth=1
                )

            # Plot velocity vector
            if self.show_velocity:
                vel_mag = np.linalg.norm(obj.velocity)
                if vel_mag > 0.1:  # Only show if moving
                    self.ax.quiver(
                        obj.position[0],
                        obj.position[1],
                        obj.position[2],
                        obj.velocity[0],
                        obj.velocity[1],
                        obj.velocity[2],
                        color=color,
                        alpha=0.6,
                        arrow_length_ratio=0.3,
                        linewidth=2
                    )

            # Label
            label = f"{obj.class_name} #{obj.track_id}"
            vel_str = f"{np.linalg.norm(obj.velocity):.1f}m/s"
            self.ax.text(
                obj.position[0],
                obj.position[1],
                obj.position[2] + 0.5,
                f"{label}\n{vel_str}",
                fontsize=8,
                color=color
            )

        # Title
        title = f"3D Scene Graph"
        if frame_info:
            title += f" - {frame_info}"
        title += f"\nObjects: {len(objects)}"
        self.ax.set_title(title, fontsize=12)

        # Grid
        self.ax.grid(True, alpha=0.3)

        # Update display
        plt.pause(0.001)

    def _get_class_color(self, class_name: str) -> tuple:
        """Get consistent color for a class."""
        if class_name not in self.class_colors:
            # Assign new color
            idx = len(self.class_colors) % 10
            self.class_colors[class_name] = self.color_palette(idx)

        return self.class_colors[class_name]

    def close(self):
        """Close viewer."""
        plt.close(self.fig)

    def save_frame(self, filename: str):
        """Save current view to file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
