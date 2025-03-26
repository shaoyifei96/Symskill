import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import time
from typing import Optional

class MeshcatVisualizer:
    def __init__(self, demo_trajs: Optional[list[np.ndarray]] = None, demo_traj_probs: Optional[np.ndarray] = None):
        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Background"].set_property("visible", False)

        self.demo_trajs = demo_trajs
        self.demo_traj_probs = demo_traj_probs
        self.generated_traj = []

        # Define colors for interpolation
        self.purple = np.array([128, 0, 128])  # Purple RGB
        self.orange = np.array([255, 165, 0])  # Orange RGB
        self.red = np.array([255, 0, 0])  # Red RGB for robot

        self.vis["robot"].set_object(g.Sphere(0.01), 
                                    g.MeshBasicMaterial(color=0xff0000))  # Red color
        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Interpolate between purple (low prob) and orange (high prob)
                rgb = self._interpolate_color(self.demo_traj_probs[i])
                color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=float(color), linewidth=10) # Color based on probability
                ))

    def _interpolate_color(self, probability: float) -> np.ndarray:
        """Interpolate between purple (low probability) and orange (high probability)"""
        rgb = self.purple + probability * (self.orange - self.purple)
        return rgb.astype(int)

    def set_demo_trajs(self, demo_trajs: list[np.ndarray], demo_traj_probs: Optional[np.ndarray] = None):
        self.demo_trajs = demo_trajs
        self.demo_traj_probs = demo_traj_probs if demo_traj_probs is not None else np.ones(len(demo_trajs))
        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Interpolate between purple (low prob) and orange (high prob)
                rgb = self._interpolate_color(self.demo_traj_probs[i])
                color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=float(color), linewidth=10) # Color based on probability
                ))

    def update_robot_position(self, position: np.ndarray):
        self.vis["robot"].set_transform(tf.translation_matrix(position))

    def update_demo_traj_probs(self, demo_traj_probs: list[float]):
        for i in range(len(self.demo_trajs)):
            # Store the new probability
            self.demo_traj_probs[i] = float(demo_traj_probs[i])
            
            # Interpolate between purple (low prob) and orange (high prob)
            rgb = self._interpolate_color(self.demo_traj_probs[i])
            color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            
            # Recreate the line with updated material
            self.vis[f"traj_{i}"].set_object(g.Line(
                g.PointsGeometry(self.demo_trajs[i].T),
                g.MeshBasicMaterial(color=float(color), linewidth=10) # Color based on probability
            ))

    def update_ref_traj(self, ref_traj_idx: int):
        self.vis[f"traj_{ref_traj_idx}"].set_property("color", 0x0000FF)

    def shutdown(self):
        self.vis.close()


if __name__ == "__main__":
    trajectories = []
    for i in range(10):
        trajectories.append(np.random.rand(10, 3))

    visualizer = MeshcatVisualizer(trajectories, list(np.ones(len(trajectories))))
    # Trajectory parameters
    t = 0
    dt = 0.05  # Time step
    steps = 100  # Total duration

    robot_x_pos = np.linspace(0, 1, steps)

    # Live update loop
    for i in range(steps):
        visualizer.update_robot_position(np.array([robot_x_pos[i], 0, 0]))
        visualizer.update_demo_traj_probs(np.random.rand(len(trajectories)))
        # Sleep to simulate real-time updates
        time.sleep(dt)
