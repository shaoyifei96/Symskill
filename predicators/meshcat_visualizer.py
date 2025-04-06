import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R

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
        self.low_prob_color = np.array([128, 0, 128])  # Purple RGB
        self.high_prob_color = np.array([255, 165, 0])  # Orange RGB
        self.robot_color = np.array([255, 0, 0])  # Red RGB for robot
        self.ref_traj_color = np.array([0, 0, 255])  # Blue RGB for ref traj

        # Use Cylinder geometry to show orientation
        cylinder_height = 0.1
        cylinder_radius = 0.01
        self.vis["robot"].set_object(g.Cylinder(cylinder_height, cylinder_radius), 
                                    g.MeshBasicMaterial(color=color_array_to_hex(self.robot_color)))
        self.vis["ref_point"].set_object(g.Cylinder(cylinder_height, cylinder_radius), 
                                     g.MeshBasicMaterial(color=color_array_to_hex(self.ref_traj_color)))

        # Add markers to indicate direction
        box_size = 0.02
        robot_marker_color = np.array([0, 255, 0]) # Green
        ref_marker_color = np.array([255, 0, 0]) # Red
        # Position marker at the top of the cylinder (positive Z direction relative to cylinder)
        marker_transform = tf.translation_matrix([0, cylinder_height / 2.0, 0])
        
        self.vis["robot"]["marker"].set_object(g.Box([3*box_size, box_size, box_size]),
                                             g.MeshBasicMaterial(color=color_array_to_hex(robot_marker_color)))
        self.vis["robot"]["marker"].set_transform(marker_transform)

        self.vis["ref_point"]["marker"].set_object(g.Box([3*box_size, box_size, box_size]),
                                                g.MeshBasicMaterial(color=color_array_to_hex(ref_marker_color)))
        self.vis["ref_point"]["marker"].set_transform(marker_transform)

        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Interpolate between purple (low prob) and orange (high prob)
                rgb = self._interpolate_color(self.demo_traj_probs[i])
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
                ))

    def _interpolate_color(self, probability: float) -> np.ndarray:
        """Interpolate between purple (low probability) and orange (high probability)"""
        rgb = self.low_prob_color + probability * (self.high_prob_color - self.low_prob_color)
        return rgb.astype(int)

    def set_demo_trajs(self, demo_trajs: list[np.ndarray], demo_traj_probs: Optional[np.ndarray] = None):
        if self.demo_trajs is not None: 
            for i in range(len(self.demo_trajs)):
                self.vis[f"traj_{i}"].delete()
        self.demo_trajs = demo_trajs
        self.demo_traj_probs = demo_traj_probs if demo_traj_probs is not None else np.ones(len(demo_trajs))
        if self.demo_trajs is not None and self.demo_traj_probs is not None:
            for i in range(len(self.demo_trajs)):
                # Interpolate between purple (low prob) and orange (high prob)
                rgb = self._interpolate_color(self.demo_traj_probs[i])
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
                ))

    def update_robot_position(self, position: np.ndarray, quaternion: Optional[np.ndarray] = None):
        translation = tf.translation_matrix(position)
        if quaternion is not None:
            rotation = tf.quaternion_matrix(quaternion)
            transform = tf.concatenate_matrices(translation, rotation)
        else:
            transform = translation
        self.vis["robot"].set_transform(transform)

    def update_demo_traj_probs(self, demo_traj_probs: list[float]):
        for i in range(len(self.demo_trajs)):
            # Store the new probability
            self.demo_traj_probs[i] = float(demo_traj_probs[i])
            
            # Interpolate between purple (low prob) and orange (high prob)
            rgb = self._interpolate_color(self.demo_traj_probs[i])
            
            # Recreate the line with updated material
            self.vis[f"traj_{i}"].set_object(g.Line(
                g.PointsGeometry(self.demo_trajs[i].T),
                g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
            ))

    def update_ref_traj(self, ref_traj_idx: int):
        self.vis[f"traj_{ref_traj_idx}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[ref_traj_idx].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(self.ref_traj_color), linewidth=10)
                ))
        
    def update_ref_point(self, ref_point_position: np.ndarray, quaternion: Optional[np.ndarray] = None):
        translation = tf.translation_matrix(ref_point_position)
        if quaternion is not None:
            rotation = tf.quaternion_matrix(quaternion)
            transform = tf.concatenate_matrices(translation, rotation)
        else:
            transform = translation
        self.vis["ref_point"].set_transform(transform)

    def shutdown(self):
        self.vis.close()

def color_array_to_hex(color: np.ndarray) -> int:
    return float((color[0] << 16) | (color[1] << 8) | color[2])

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
    angle_step = 2 * np.pi / steps # Rotate 360 degrees over the duration

    # Live update loop
    for i in range(steps):
        position = np.array([robot_x_pos[i], 0, 0])
        # Rotate around Z-axis
        angle = i * angle_step
        # tf.quaternion_from_euler expects axes convention like 'sxyz', 'rzxz' etc.
        # For simple Z-axis rotation, we can use 'szyx' (static frame, rot z, then y, then x)
        # or directly use math: w = cos(angle/2), x=0, y=0, z=sin(angle/2)
        # using the direct formula for [w, x, y, z] format
        quat_wxyz = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
        
        visualizer.update_robot_position(position, quat_wxyz)
        
        # visualizer.update_demo_traj_probs(np.random.rand(len(trajectories)))
        if i == 0:
            ref_traj_idx = np.random.randint(len(trajectories))
            visualizer.update_ref_traj(ref_traj_idx)
            # Set a fixed position and orientation for the ref_point for testing
            ref_point_pos = trajectories[ref_traj_idx][0]
            ref_point_orient_xyzw = R.from_euler('xyz', [0, 45, 0], degrees=True).as_quat()
            ref_point_orient_wxyz = np.array([ref_point_orient_xyzw[3], ref_point_orient_xyzw[0], ref_point_orient_xyzw[1], ref_point_orient_xyzw[2]])
            visualizer.update_ref_point(ref_point_pos, ref_point_orient_wxyz)

        # Sleep to simulate real-time updates
        time.sleep(dt)