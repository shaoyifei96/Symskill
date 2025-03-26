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

        self.vis["robot"].set_object(g.Sphere(0.01))
        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Calculate color based on probability: higher prob is black (0x000000), lower is light gray (0xCCCCCC)
                color_value = int(0xCC * (1.0 - self.demo_traj_probs[i]))
                color = (color_value << 16) | (color_value << 8) | color_value
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color, linewidth=100.0) # Color based on probability
                ))

    def set_demo_trajs(self, demo_trajs: list[np.ndarray], demo_traj_probs: Optional[np.ndarray] = None):
        self.demo_trajs = demo_trajs
        self.demo_traj_probs = demo_traj_probs if demo_traj_probs is not None else np.ones(len(demo_trajs))
        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Calculate color based on probability: higher prob is black (0x000000), lower is light gray (0xCCCCCC)
                color_value = int(0xCC * (1.0 - self.demo_traj_probs[i]))
                color = (color_value << 16) | (color_value << 8) | color_value
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color, linewidth=100.0) # Color based on probability
                ))

    def update_robot_position(self, position: np.ndarray):
        self.vis["robot"].set_transform(tf.translation_matrix(position))

    def update_demo_traj_probs(self, demo_traj_probs: list[float]):
        for i in range(len(self.demo_trajs)):
            # Store the new probability
            self.demo_traj_probs[i] = float(demo_traj_probs[i])
            
            # Calculate color based on probability: higher prob is black (0x000000), lower is light gray (0xCCCCCC)
            color_value = int(0xCC * (1.0 - self.demo_traj_probs[i]))
            color = (color_value << 16) | (color_value << 8) | color_value
            
            # Recreate the line with updated material
            self.vis[f"traj_{i}"].set_object(g.Line(
                g.PointsGeometry(self.demo_trajs[i].T),
                g.MeshBasicMaterial(color=color, linewidth=100.0) # Color based on probability
            ))

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
