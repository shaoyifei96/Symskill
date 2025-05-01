import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
import logging

class MeshcatVisualizer:
    """
    A visualizer for robot trajectories and motion using Meshcat.
    
    This class provides a 3D visualization of robot motion, trajectories, and 
    velocity vectors. It supports two modes: "se3_lpvds" and "traj_follower".
    
    In both modes, the visualizer shows:
    - The robot as a red cylinder with a direction marker
    - A green velocity arrow showing direction and magnitude of robot velocity
    - Demonstration trajectories with colors indicating probability/scores
    
    In "traj_follower" mode, it additionally shows:
    - A blue reference point 
    - Reference trajectory highlighting
    
    Usage:
        # Initialize the visualizer
        visualizer = MeshcatVisualizer(
            mode="traj_follower",  # or "se3_lpvds"
            demo_trajs=trajectories,  # list of trajectory arrays
            demo_traj_scores=scores   # optional trajectory scores, only for "traj_follower" mode
        )
        
        # Update robot state
        visualizer.update_robot_position(position, quaternion)
        visualizer.update_robot_velocity(velocity)

        # update demo trajectories
        visualizer.set_demo_trajs(trajectories, 
                                  demo_traj_scores # optional, only for "traj_follower" mode
        )
        
        # only for "traj_follower" mode
        visualizer.update_ref_traj(trajectory_index)
        visualizer.update_ref_point(position, quaternion)
        visualizer.update_demo_traj_colors(new_scores)
    """
    def __init__(self, mode: str, demo_trajs: Optional[list[np.ndarray]] = None, demo_traj_scores: Optional[np.ndarray] = None):
        """
        Initialize the Meshcat visualizer.
        
        Args:
            mode: The visualizer mode, either "se3_lpvds" or "traj_follower"
            demo_trajs: A list of demonstration trajectories, each as np.ndarray
            demo_traj_scores: Optional scores/probabilities for each trajectory
        """
        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Background"].set_property("visible", False)

        if mode == "se3_lpvds" or mode == "traj_follower":
            self.mode = mode
            self.se3_lpvds = mode == "se3_lpvds"
            self.traj_follower = mode == "traj_follower"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.robot_transform = tf.identity_matrix() # Store robot transform
        self.demo_trajs = demo_trajs
        self.demo_traj_scores = demo_traj_scores

        self.generated_traj = []

        # Define colors for interpolation
        self.high_prob_color = np.array([128, 0, 128])  # Purple RGB
        self.low_prob_color = np.array([255, 165, 0])  # Orange RGB
        self.robot_color = np.array([255, 0, 0])  # Red RGB for robot
        self.ref_traj_color = np.array([0, 0, 255])  # Blue RGB for ref traj
        self.velocity_color = np.array([0, 255, 0])  # Green RGB for velocity arrow

        cylinder_height = 0.1
        cylinder_radius = 0.01
        self.vis["robot"].set_object(g.Cylinder(cylinder_height, cylinder_radius), 
                                    g.MeshBasicMaterial(color=color_array_to_hex(self.robot_color)))
        if self.traj_follower:
            self.vis["ref_point"].set_object(g.Cylinder(cylinder_height, cylinder_radius), 
                                     g.MeshBasicMaterial(color=color_array_to_hex(self.ref_traj_color)))
        

        # Add markers to indicate direction
        box_size = 0.02
        # Position marker at the top of the cylinder (positive Z direction relative to cylinder)
        marker_transform = tf.translation_matrix([0, cylinder_height / 2.0, 0])
        
        self.vis["robot"]["marker"].set_object(g.Box([3*box_size, box_size, box_size]),
                                             g.MeshBasicMaterial(color=color_array_to_hex(self.robot_color)))
        self.vis["robot"]["marker"].set_transform(marker_transform)
        
        # Add velocity arrow at the top level
        # Initialize with default length and make it invisible initially
        initial_arrow_length = 0.1
        self.vis["velocity_arrow"].set_object(
            g.Cylinder(initial_arrow_length, 0.005), 
            g.MeshBasicMaterial(color=color_array_to_hex(self.velocity_color))
        )
        self.vis["velocity_arrow"].set_property("visible", False)
        
        if self.traj_follower:
            self.vis["ref_point"]["marker"].set_object(g.Box([3*box_size, box_size, box_size]),
                                                g.MeshBasicMaterial(color=color_array_to_hex(self.ref_traj_color)))
            self.vis["ref_point"]["marker"].set_transform(marker_transform)

        if self.demo_trajs is not None:
            self.demo_traj_scores = rescale(self.demo_traj_scores) if self.traj_follower else None
            for i in range(len(self.demo_trajs)):
                if self.traj_follower and self.demo_traj_scores is not None:
                    # Interpolate between purple (low prob) and orange (high prob)
                    rgb = self._interpolate_color(self.demo_traj_scores[i])
                else:
                    rgb = self.high_prob_color
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
                ))

    def _interpolate_color(self, probability: float) -> np.ndarray:
        """
        Interpolate between purple (low probability) and orange (high probability)
        
        Args:
            probability: Value between 0 and 1 indicating probability/score
            
        Returns:
            RGB color as numpy array
        """
        rgb = self.low_prob_color + probability * (self.high_prob_color - self.low_prob_color)
        return rgb.astype(int)

    def set_demo_trajs(self, demo_trajs: list[np.ndarray], demo_traj_scores: Optional[np.ndarray] = None):
        """
        Set or update the demonstration trajectories.
        
        Args:
            demo_trajs: List of trajectory arrays to visualize
            demo_traj_scores: Optional scores/probabilities for each trajectory
        """
        if self.demo_trajs is not None: 
            for i in range(len(self.demo_trajs)):
                self.vis[f"traj_{i}"].delete()
        self.demo_trajs = demo_trajs
        if self.traj_follower:
            self.demo_traj_scores = demo_traj_scores if demo_traj_scores is not None else np.ones(len(demo_trajs))
        if self.demo_trajs is not None:
            self.demo_traj_scores = rescale(self.demo_traj_scores) if self.traj_follower else None
            for i in range(len(self.demo_trajs)):
                if self.traj_follower and self.demo_traj_scores is not None:
                    # Interpolate between purple (low prob) and orange (high prob)
                    rgb = self._interpolate_color(self.demo_traj_scores[i])
                else:
                    rgb = self.high_prob_color
                
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
                ))

    def update_robot_position(self, position: np.ndarray, quaternion: Optional[np.ndarray] = None):
        """
        Update the robot's position and orientation.
        
        Args:
            position: 3D position array [x, y, z]
            quaternion: Optional quaternion for orientation in [w, x, y, z] format
        """
        translation = tf.translation_matrix(position)
        if quaternion is not None:
            rotation = tf.quaternion_matrix(quaternion)
            transform = tf.concatenate_matrices(translation, rotation)
        else:
            transform = translation
        self.vis["robot"].set_transform(transform)
        self.robot_transform = transform # Store the transform

    def update_robot_velocity(self, pos_vel: np.ndarray, ang_vel: Optional[np.ndarray] = None):
        """
        Update the robot's velocity arrow to visualize linear velocity.
        
        Args:
            pos_vel: 3D velocity vector [vx, vy, vz]
            ang_vel: Optional angular velocity (not used currently)
        """
        vel_magnitude = np.linalg.norm(pos_vel)
        if vel_magnitude > 1e-6:
            # Make arrow visible
            self.vis["velocity_arrow"].set_property("visible", True)
            
            # Scale the arrow length based on velocity magnitude
            scale_factor = 0.5
            arrow_length = vel_magnitude * scale_factor
            
            # Update the arrow geometry (length)
            self.vis["velocity_arrow"].set_object(
                g.Cylinder(arrow_length, 0.005),
                g.MeshBasicMaterial(color=color_array_to_hex(self.velocity_color))
            )
            
            # --- Calculate Global Transform ---
            # Get robot's current global position from stored transform
            robot_transform = self.robot_transform
            robot_position = robot_transform[:3, 3]

            # Calculate global rotation to align default Y-axis cylinder with global velocity direction
            vel_normalized = pos_vel / vel_magnitude
            y_axis = np.array([0, 1, 0])
            rotation_axis = np.cross(y_axis, vel_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                angle = np.arccos(np.dot(y_axis, vel_normalized))
                qw = np.cos(angle/2)
                qx, qy, qz = rotation_axis * np.sin(angle/2)
                # This rotation aligns the cylinder's axis (Y) with the global velocity vector
                global_rotation = tf.quaternion_matrix([qw, qx, qy, qz]) 
            else:
                if vel_normalized[1] < 0: # Aligned with -Y
                    global_rotation = tf.quaternion_matrix([0, 1, 0, 0]) # Rotate 180 deg around X
                else: # Aligned with +Y
                    global_rotation = np.eye(4)
            
            # Calculate offset to position the cylinder's base at the origin (0,0,0) of its frame
            # The default cylinder is centered at origin; we need to shift it by half its length 
            # along its *new* orientation (which matches the global velocity direction).
            
            # Get the direction of the cylinder's axis (Y) after global_rotation is applied
            # This is the second column of the rotation matrix, which is also vel_normalized
            arrow_direction_global = global_rotation[0:3, 1] 
            
            half_length = arrow_length / 2.0
            # Calculate the desired CENTER position of the arrow in the global frame
            arrow_center_position = robot_position + half_length * arrow_direction_global # arrow_direction_global is vel_normalized

            # Combine transformations:
            # 1. Apply global rotation (rotates the cylinder around its own origin)
            # 2. Translate the rotated cylinder to the desired center position
            # concatenate_matrices(A, B) applies B then A. So we want T(center) @ R
            final_transform = tf.concatenate_matrices(
                tf.translation_matrix(arrow_center_position), 
                global_rotation
            )
            
            self.vis["velocity_arrow"].set_transform(final_transform)
            # --- End Global Transform Calculation ---

        else:
            # Hide arrow if velocity is zero
            self.vis["velocity_arrow"].set_property("visible", False)

    def update_demo_traj_colors(self, demo_traj_scores: list[float]):
        """
        Update the colors of demonstration trajectories based on new scores.
        
        Args:
            demo_traj_scores: New scores/probabilities for each trajectory
        """
        if self.traj_follower:
            demo_traj_scores = rescale(demo_traj_scores)
            for i in range(len(self.demo_trajs)):
                # Store the new probability
                self.demo_traj_scores[i] = float(demo_traj_scores[i])
                
                # Interpolate between purple (low prob) and orange (high prob)
                rgb = self._interpolate_color(self.demo_traj_scores[i])
                
                # Recreate the line with updated material
                self.vis[f"traj_{i}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[i].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(rgb), linewidth=10) # Color based on probability
                ))
        else:
            logging.warning("Demo traj colors not updated except for traj follower mode")

    def update_ref_traj(self, ref_traj_idx: int):
        """
        Highlight a specific trajectory as the reference trajectory.
        
        Args:
            ref_traj_idx: Index of the trajectory to highlight as reference
        """
        if self.traj_follower:
            self.vis[f"traj_{ref_traj_idx}"].set_object(g.Line(
                    g.PointsGeometry(self.demo_trajs[ref_traj_idx].T),
                    g.MeshBasicMaterial(color=color_array_to_hex(self.ref_traj_color), linewidth=10)
                ))
        else:
            logging.warning("Ref traj not updated except for traj follower mode")
        
    def update_ref_point(self, ref_point_position: np.ndarray, quaternion: Optional[np.ndarray] = None):
        """
        Update the reference point position and orientation.
        
        Args:
            ref_point_position: 3D position array [x, y, z]
            quaternion: Optional quaternion for orientation in [w, x, y, z] format
        """
        if self.traj_follower:
            translation = tf.translation_matrix(ref_point_position)
            if quaternion is not None:
                rotation = tf.quaternion_matrix(quaternion)
                transform = tf.concatenate_matrices(translation, rotation)
            else:
                transform = translation
            self.vis["ref_point"].set_transform(transform)
        else:
            logging.warning("Ref point not updated except for traj follower mode")


def rescale(scores: np.ndarray) -> np.ndarray:
    """
    Rescale scores to range [0,1] based on min and max values.
    
    Args:
        scores: Array of score values
        
    Returns:
        Rescaled scores between 0 and 1
    """
    max_score = np.max(scores)
    min_score = np.min(scores)
    return (scores - min_score) / (max_score - min_score)


def color_array_to_hex(color: np.ndarray) -> int:
    """
    Convert RGB color array to hexadecimal color format for meshcat.
    
    Args:
        color: RGB color array with values 0-255
        
    Returns:
        Hexadecimal color value as float
    """
    return float((color[0] << 16) | (color[1] << 8) | color[2])

if __name__ == "__main__":
    trajectories = []
    for i in range(10):
        trajectories.append(np.random.rand(10, 3))

    # Create visualizer with proper mode parameter
    visualizer = MeshcatVisualizer(mode="traj_follower", demo_trajs=trajectories, 
                                  demo_traj_scores=np.ones(len(trajectories)))
    
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
        
        # Update demo trajectory colors randomly (using the correct method name)
        if i % 10 == 0:  # Update colors every 10 steps
            visualizer.update_demo_traj_colors(np.random.rand(len(trajectories)))
        
        if i == 0:
            ref_traj_idx = np.random.randint(len(trajectories))
            visualizer.update_ref_traj(ref_traj_idx)
            # Set a fixed position and orientation for the ref_point for testing
            ref_point_pos = trajectories[ref_traj_idx][0]
            ref_point_orient_xyzw = R.from_euler('xyz', [0, 45, 0], degrees=True).as_quat()
            ref_point_orient_wxyz = np.array([ref_point_orient_xyzw[3], ref_point_orient_xyzw[0], ref_point_orient_xyzw[1], ref_point_orient_xyzw[2]])
            visualizer.update_ref_point(ref_point_pos, ref_point_orient_wxyz)

        # Example velocity update - creates a circular velocity pattern
        # Velocity magnitude increases with position for better visualization
        velocity_scale = 0.1 + robot_x_pos[i] * 0.3  # Scale from 0.1 to 0.4
        velocity = np.array([np.cos(angle), np.sin(angle), 0]) * velocity_scale
        visualizer.update_robot_velocity(np.array([1, 0, 0]))

        # Sleep to simulate real-time updates
        time.sleep(dt)