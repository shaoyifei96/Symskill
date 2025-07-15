"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set, Optional, Type, Tuple, Any

import numpy as np
import os
import torch
from gym.spaces import Box
import mujoco

import matplotlib.pyplot as plt

from predicators.settings import CFG

from ds_policy import DSPolicy, load_data

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, ParameterizedOption, ParameterizedTerminal, Predicate, State, Type
import torch
from predicators.DS_models.gen_demo_model import DynamicalSystem

from scipy.spatial.transform import Rotation as R
from predicators.utils import calculate_relative_pose

import warnings

class RoboKitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the RoboKitchen environment."""

    moveto_tol: ClassVar[float] = 0.03  # for terminating moving
    max_delta_mag: ClassVar[float] = 1.0  # don't move more than this per step
    max_push_mag: ClassVar[float] = 0.05  # for pushing forward
    # A reasonable home position for the end effector.
    home_pos: ClassVar[Pose3D] = (0.0, 0.37, 2.1)
    # Keep pushing a bit even if the On classifier holds.
    push_lr_thresh_pad: ClassVar[float] = 0.02
    push_microhandle_thresh_pad: ClassVar[float] = 0.02
    turn_knob_tol: ClassVar[float] = 0.02  # for twisting the knob

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"robo_kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type], predicates: Dict[str, Predicate], action_space: Box) -> Set[ParameterizedOption]:
        # Define quaternions using MuJoCo's utilities
        down_quat = np.zeros(4)
        mujoco.mju_euler2Quat(down_quat, np.array([-np.pi, 0.0, -np.pi / 2]), "xyz")

        # End effector facing forward (e.g., toward the knobs.)
        fwd_quat = np.zeros(4)
        mujoco.mju_euler2Quat(fwd_quat, np.array([-np.pi / 2, 0.0, -np.pi / 2]), "xyz")

        # Angled quaternion
        angled_quat = np.zeros(4)
        mujoco.mju_euler2Quat(angled_quat, np.array([-3 * np.pi / 4, 0.0, -np.pi / 2]), "xyz")

        # Types
        gripper = types["gripper_type"]
        handle = types["handle_type"]
        base = types["base_type"]
        left_finger = types["left_finger_type"]
        right_finger = types["right_finger_type"]
        grab = types["grab_type"]
        surface = types["surface_type"]
        object_type = types["object_type"]

        options: Set[ParameterizedOption] = set()

        """---------------------------------- Helper function starts ----------------------------------"""

        def _init_object_of_interest_transform(state: State, objects: Sequence[Object], offset_handle_frame: Optional[np.ndarray] = None):
            """Helper to initialize handle transform data in memory."""
            gripper, object_of_interest, base = objects
            object_of_interest_quat = state.get(object_of_interest, "quaternion")
            object_of_interest_pos = state.get(object_of_interest, "translation")
            object_of_interest_rot = R.from_quat(object_of_interest_quat).as_matrix()
            if offset_handle_frame is not None:
                # Transform offset from handle frame to world frame before adding
                offset_world = object_of_interest_rot @ offset_handle_frame
                object_of_interest_pos = object_of_interest_pos + offset_world
            return object_of_interest_pos, object_of_interest_rot

        def _create_ds_policy(option: str):
            x, x_dot, q, omega, gripper_traj = load_data(CFG.robo_kitchen_task, option, finger=False, transform_to_object_of_interest_frame=True, debug_on=False)
            model_config = {
                'pos_model': {
                    'special_mode': 'none',
                    # 'load_path': f"ds_policy/models/mlp_width128_depth3_{option}.pt",
                },
                'quat_model': {
                    'special_mode': 'simple',
                    # 'save_path': f"ds_policy/models/quat_model_{option}.json",
                    # 'k_init': 10
                }
            }
            demo_traj_probs = np.ones(len(x))
            ds_policy = DSPolicy(x, x_dot, q, omega, gripper_traj, model_config=model_config, dt=1/60, switch=False, demo_traj_probs=demo_traj_probs)
            return ds_policy

        def _create_simple_ds_model():
            """Helper to create and initialize the DS model in memory."""
            # Define model architecture
            class SimpleDS(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.D = torch.nn.Parameter(torch.zeros(3, 3))

                def forward(self, x):
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).float()
                    if len(x.shape) == 1:
                        x = x.unsqueeze(0)
                    return torch.matmul(x, self.D.T).squeeze(0)

            # Create model and load state dict
            model = SimpleDS()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "..", "DS_models", "models", "model.pt")
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            return model

        def frame_transform(source_pos: np.ndarray, source_quat: np.ndarray, reference_pos: np.ndarray, reference_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:                
            source_rotation = R.from_quat(source_quat).as_matrix()

            relative_position = source_pos - reference_pos

            position_in_reference = reference_rot.T @ relative_position
            rotation_in_reference = reference_rot.T @ source_rotation

            return position_in_reference, rotation_in_reference

        def _is_robot_stuck(memory: Dict, gripper_pos: float, velocity_threshold: float = 0.001, stuck_time_threshold: int = 10) -> bool:
            if "prev_gripper_pos" not in memory:
                memory["prev_gripper_pos"] = gripper_pos
                return False

            # Update previous gripper position
            velocity = np.linalg.norm(gripper_pos - memory["prev_gripper_pos"])
            memory["prev_gripper_pos"] = gripper_pos
            # Initialize velocity history if not already in memory
            if "velocity_history" not in memory:
                memory["velocity_history"] = []

            # Add current velocity to history
            memory["velocity_history"].append(velocity)

            # Keep only the most recent velocities for memory efficiency
            if len(memory["velocity_history"]) > stuck_time_threshold + 10:
                memory["velocity_history"] = memory["velocity_history"][-stuck_time_threshold - 10:]

            # Check if we have enough history to make a determination
            if len(memory["velocity_history"]) < stuck_time_threshold:
                return False

            # Check if the robot has been moving slowly for the threshold duration
            recent_velocities = memory["velocity_history"][-stuck_time_threshold:]
            return all(v < velocity_threshold for v in recent_velocities)

        def _process_fail_memory(memory: Dict, option_name: str, gripper: Object, object_of_interest: Object, reference_pos: Optional[np.ndarray] = None, reference_rot: Optional[np.ndarray] = None) -> None:
            """Process fail memory entries for a specific option and update demo trajectory probabilities.
            
            Args:
                memory: Memory dictionary containing fail_memory and ds_policy
                option_name: Name of the option to process fail memory for
                gripper: Gripper object
                object_of_interest: Object of interest object
                reference_pos: Reference position for transformation (use handle_state if None)
                reference_rot: Reference rotation matrix for transformation (use handle_state if None)
            """
            if "fail_memory" not in memory or not memory["fail_memory"]:
                return

            for idx in range(len(memory["fail_memory"])-1, -1, -1):
                if memory["fail_memory"][idx].option_name == option_name:
                    gripper_pos = memory["fail_memory"][idx].state.get(gripper, "translation")
                    gripper_quat = memory["fail_memory"][idx].state.get(gripper, "quaternion")

                    # If reference position/rotation not provided, use object_of_interest state
                    if reference_pos is None or reference_rot is None:
                        object_of_interest_pos = memory["fail_memory"][idx].state.get(object_of_interest, "translation")
                        object_of_interest_quat = memory["fail_memory"][idx].state.get(object_of_interest, "quaternion")
                        ref_pos = object_of_interest_pos
                        ref_rot = R.from_quat(object_of_interest_quat).as_matrix()
                    else:
                        ref_pos = reference_pos
                        ref_rot = reference_rot

                    # Transform gripper state to reference frame
                    gripper_pos_in_ref, gripper_rot_in_ref = frame_transform(
                        gripper_pos, gripper_quat, ref_pos, ref_rot
                    )
                    gripper_quat_in_ref = R.from_matrix(gripper_rot_in_ref).as_quat()

                    # Update demo trajectory probabilities
                    memory["ds_policy"].update_demo_traj_probs(
                        np.concatenate([gripper_pos_in_ref, gripper_quat_in_ref]),
                        "ref_point", penalty=0.8, traj_threshold=0.2, radius=0.02,
                        angle_threshold=np.pi/2, lookahead=10
                    )
                    if CFG.visualizer:
                        CFG.visualizer.update_demo_traj_colors(memory["ds_policy"].demo_traj_probs)

                    # Remove processed entry
                    memory["fail_memory"].pop(idx)

        """---------------------------------- Helper function ends ----------------------------------"""

        """---------------------------------- general move option starts ----------------------------------"""

        def _DS_general_move_static_option_initiable(option: str, state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            gripper, object_of_interest, base = objects
            memory["object_of_interest_pos"] = state.get(object_of_interest, "translation")
            memory["object_of_interest_rot"] = R.from_quat(state.get(object_of_interest, "quaternion")).as_matrix()

            if option not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option=option)
                CFG.option_to_policy[option] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy[option]

            _process_fail_memory(memory, option, gripper, object_of_interest)

            if CFG.visualizer:
                # memory["ds_policy"].init_demo_traj_scores(memory["handle_pos"])
                CFG.visualizer.set_demo_trajs(memory["ds_policy"].x, memory["ds_policy"].demo_traj_probs)

            return True

        def _DS_general_move_dynamic_option_initiable(option:str, state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if option not in CFG.option_to_init_pose:
                memory["object_of_interest_pos"], memory["object_of_interest_rot"] = _init_object_of_interest_transform(state, objects, offset_handle_frame=np.array([0.0, 0.0, 0.0]))
                CFG.option_to_init_pose[option] = [memory["object_of_interest_pos"], memory["object_of_interest_rot"]]
            else:
                memory["object_of_interest_pos"] = CFG.option_to_init_pose[option][0]
                memory["object_of_interest_rot"] = CFG.option_to_init_pose[option][1]

            # if "ds_policy" not in memory:
            #     _create_ds_policy(memory, state, objects, option="move_away", offset_handle_frame=np.array([0.0, 0.0, 0.0]))
            if option not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option=option)
                CFG.option_to_policy[option] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy[option]

            _process_fail_memory(memory, option, objects[0], objects[1],
                                reference_pos=CFG.option_to_init_pose[option][0],
                                reference_rot=CFG.option_to_init_pose[option][1])

            if CFG.visualizer:
                # memory["ds_policy"].init_demo_traj_scores(memory["handle_pos"])
                CFG.visualizer.set_demo_trajs(memory["ds_policy"].x, memory["ds_policy"].demo_traj_probs)

            return True

        def _DS_general_move_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            # Get objects
            gripper, _, base = objects
            object_of_interest_pos = memory["object_of_interest_pos"]
            object_of_interest_rot = memory["object_of_interest_rot"]
            # Get positions
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            pos_in_object_of_interest, rot_in_object_of_interest = frame_transform(gripper_pos, gripper_quat, object_of_interest_pos, object_of_interest_rot)

            expected_relative_rot_object_of_interest = R.from_quat(np.array([0.5, 0.5, 0.5, -0.5]))

            # Compute the difference between the expected relative rotation and the actual relative rotation
            relative_rotation = expected_relative_rot_object_of_interest * R.from_matrix(rot_in_object_of_interest).inv()
            angular_w_object_of_interest = relative_rotation.as_rotvec()
            # angular_w_handle = vee_operator(rel_rot_diff.as_matrix())
            # move that difference to the base frame
            world_w = object_of_interest_rot @ angular_w_object_of_interest

            robot_base_pos = state.get(base, "translation")
            robot_base_quat = state.get(base, "quaternion")
            robot_base_rot = R.from_quat(robot_base_quat).as_matrix()
            robot_base_w = robot_base_rot.T @ world_w
            mag = np.linalg.norm(robot_base_w)
            if mag > 1:
                robot_base_w = robot_base_w / mag

            # Choose between using neural network model or DS policy based on what's in memory
            if "model" in memory:
                # Use neural network model
                net = memory["model"]
                with torch.no_grad():
                    velocity_in_object_of_interest = net(torch.from_numpy(pos_in_object_of_interest).float())

                warnings.warn("Velocity getting scaled, plz remove")
                velocity_in_object_of_interest[0] = velocity_in_object_of_interest[0] * 0.5 #object_of_interest frame x is the direction towards object_of_interest
                velocity_in_object_of_interest[1] = velocity_in_object_of_interest[1] * 0.05
                velocity_in_object_of_interest[2] = velocity_in_object_of_interest[2] * 0.5
                # Transform velocity back to world frame
                velocity_world = object_of_interest_rot @ velocity_in_object_of_interest.numpy()
                velocity_robot_base = robot_base_rot.T @ velocity_world

                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = velocity_robot_base
                arr[3:6] = 0.8 * robot_base_w

            elif "ds_policy" in memory:
                # Use DS policy
                ds_policy = memory["ds_policy"]

                action = ds_policy.get_action(np.concatenate([pos_in_object_of_interest, R.from_matrix(rot_in_object_of_interest).as_quat()]), clf=True, alpha_V=10.0, lookahead=20)
                vel = action[:6] # position + angular velocity

                if CFG.visualizer:
                    rel_gripper_visualizer_rot = np.array([[0, 0, 1], # NOTE: this is a "correction" term: to rotate gripper's frame to visualize in the way we want
                                                          [1, 0, 0],
                                                          [0, 1, 0]])
                    gripper_quat_in_visualizer_xyzw = R.from_matrix(rot_in_object_of_interest @ rel_gripper_visualizer_rot).as_quat()
                    gripper_quat_in_visualizer_wxyz = np.array([gripper_quat_in_visualizer_xyzw[3], gripper_quat_in_visualizer_xyzw[0], gripper_quat_in_visualizer_xyzw[1], gripper_quat_in_visualizer_xyzw[2]])
                    CFG.visualizer.update_robot_position(pos_in_object_of_interest, gripper_quat_in_visualizer_wxyz)
                    CFG.visualizer.update_ref_traj(ds_policy.ref_traj_idx)
                    ref_rot = R.from_quat(ds_policy.quat[ds_policy.ref_traj_idx][ds_policy.ref_point_idx_lookahead]).as_matrix()
                    ref_quat_in_visualizer_xyzw = R.from_matrix(ref_rot @ rel_gripper_visualizer_rot).as_quat()
                    ref_quat_in_visualizer_wxyz = np.array([ref_quat_in_visualizer_xyzw[3], ref_quat_in_visualizer_xyzw[0], ref_quat_in_visualizer_xyzw[1], ref_quat_in_visualizer_xyzw[2]])
                    CFG.visualizer.update_ref_point(ds_policy.x[ds_policy.ref_traj_idx][ds_policy.ref_point_idx_lookahead], ref_quat_in_visualizer_wxyz)

                x_dot_object_of_interest = vel[:3]
                r_dot_object_of_interest = vel[3:]
                x_dot_world = object_of_interest_rot @ x_dot_object_of_interest
                x_dot_robot_base = robot_base_rot.T @ x_dot_world
                r_dot_world = object_of_interest_rot @ r_dot_object_of_interest
                r_dot_robot_base = robot_base_rot.T @ r_dot_world

                mag = np.linalg.norm(r_dot_robot_base)
                if mag > 1:
                    r_dot_robot_base = r_dot_robot_base / mag

                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = x_dot_robot_base
                arr[3:6] = 0.8 * r_dot_robot_base
                # arr[6] = action[6] # gripper
            else:
                # Fallback if neither model is available
                raise ValueError("No DS option policy found")

            # Clip the action to the action space limits
            action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_low, action_high)

            return Action(arr)

        def _DS_general_move_gripper_closed_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            """
            NOTE: this is a cheat. hardcoded gripper closed for move away option
            TODO: should add gripper as another dimension in node to learn
            """
            action = _DS_general_move_option_policy(state, memory, objects, params)

            # Set gripper to closed
            action._arr[6] = 1.0

            return action

        def _DS_general_move_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # handle_pos = memory["handle_pos"]
            gripper, _, base = objects
            gripper_pos = state.get(gripper, "translation")
            # Check if the robot is stuck (velocity too small for too long)
            if _is_robot_stuck(memory, gripper_pos):
                return True
            return False

        """---------------------------------- general move option ends ----------------------------------"""

        """---------------------------------- MoveToInitPoseOption Starts ----------------------------------"""
        """This is a global option to move the gripper to the initial pose"""

        def _move_to_init_pose_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            target_pos = CFG.init_pose[:3]
            target_quat = CFG.init_pose[3:7]
            waypoints = [
                np.array([-0.4, 0.4, 0.5, 0.5, -0.3, 0.6, 0.4]), # 0.21, -1.35, 1.78
                np.concatenate([np.array([0.2, 0.4, 0.5]), R.from_euler("xyz", [0.2, -1.30, 2.2]).as_quat()]),
                np.concatenate([np.array([0.2, -0.2, 0.5]), R.from_euler("xyz", [0.2, -1.30, 2.2]).as_quat()]),
                CFG.init_pose
                # np.concatenate([np.array([-0.6, 0, 0.5]), R.from_euler("xyz", [0.21, -1.35, 2.2]).as_quat()]),
                # np.concatenate([np.array([-0.4, 0.4, 0.5]), R.from_euler("xyz", [0.0, np.pi/4, -np.pi/2]).as_quat()]),
                
                # np.concatenate([np.array([-0.35, 0.35, 0.7]), R.from_euler("xyz", [0.0, 0, 0]).as_quat()]),
                # np.concatenate([CFG.init_pose[:3], np.array([0.5, -0.3, 0.6, 0.4])]),
            ]
            memory["num_waypoints"] = len(waypoints)
            memory["waypoints"] = waypoints
            memory["current_waypoint"] = 0

            gripper, base = objects
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            base_pos = state.get(base, "translation")
            base_quat = state.get(base, "quaternion")
            gripper_pos_in_base, gripper_rot_in_base = frame_transform(gripper_pos, gripper_quat, base_pos, R.from_quat(base_quat).as_matrix())
            gripper_quat_in_base = R.from_matrix(gripper_rot_in_base).as_quat()
            if not np.linalg.norm(gripper_pos_in_base - np.array([-0.03077441,  0.25941396,  0.86963758])) < 0.2:
                #remove all waypoints except the last one
                memory["waypoints"] = [memory["waypoints"][-1]]
                memory["num_waypoints"] = 1

            return True

        def _move_to_init_pose_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            """
            Note: objects contains gripper and base (in this order)
            """
            gripper, base = objects
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            base_pos = state.get(base, "translation")
            base_quat = state.get(base, "quaternion")
            gripper_pos_in_base, gripper_rot_in_base = frame_transform(gripper_pos, gripper_quat, base_pos, R.from_quat(base_quat).as_matrix())
            gripper_quat_in_base = R.from_matrix(gripper_rot_in_base).as_quat()

            return np.linalg.norm(np.concatenate([gripper_pos_in_base, gripper_quat_in_base], axis=0) - memory["waypoints"][-1]) < 0.1
            # in_origin = RoboKitchenEnv._InOrigin_holds(state, [gripper, base])

        def move_to_init_pose_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            """
            Note: objects contains gripper and base (in this order)
            """
            if memory["current_waypoint"] == memory["num_waypoints"]:
                return Action(np.zeros(7, dtype=np.float32))
            gripper, base = objects
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            base_pos = state.get(base, "translation")
            base_quat = state.get(base, "quaternion")
            gripper_pos_in_base, gripper_rot_in_base = frame_transform(gripper_pos, gripper_quat, base_pos, R.from_quat(base_quat).as_matrix())
            gripper_quat_in_base = R.from_matrix(gripper_rot_in_base).as_quat()


            left_finger = None
            right_finger = None 

            for obj in state.data:
                if obj.type.name == "left_finger_type":
                    left_finger = obj
                if obj.type.name == "right_finger_type":
                    right_finger = obj
                if left_finger and right_finger:
                    break

            assert base and left_finger and right_finger

            left_right_finger_dist = calculate_relative_pose(state, left_finger, right_finger, "translation", "quaternion")
            left_right_finger_dist = np.linalg.norm(left_right_finger_dist[:3])

            if np.linalg.norm(np.concatenate([gripper_pos_in_base, gripper_quat_in_base], axis=0) - memory["waypoints"][memory["current_waypoint"]]) < 0.2:
                if memory["current_waypoint"] < memory["num_waypoints"] - 1:
                    memory["current_waypoint"] += 1 

            K_pos = 2.0
            K_rot = 0.5

            target_pos = memory["waypoints"][memory["current_waypoint"]][:3]
            target_quat = memory["waypoints"][memory["current_waypoint"]][3:7]  # Assuming wxyz format

            # --- Calculate world frame velocities ---

            # Linear velocity
            pos_error_world = target_pos - gripper_pos_in_base
            linear_vel_base = K_pos * pos_error_world

            # Angular velocity
            target_rot = R.from_quat(target_quat)
            current_rot = R.from_quat(gripper_quat_in_base)
            error_rot = target_rot * current_rot.inv()
            angular_vel_base = K_rot * error_rot.as_rotvec()

            # --- Visualization ---
            if CFG.visualizer:
                CFG.visualizer.update_robot_position(gripper_pos_in_base, gripper_quat_in_base)
                CFG.visualizer.update_robot_velocity(linear_vel_base)
            # --- Construct action ---
            action = np.zeros(7, dtype=np.float32)
            action[:3] = linear_vel_base
            action[3:6] = angular_vel_base
            # action[3:6] = 0.0
            action[6] = -1.0  # Keep gripper open

            if left_right_finger_dist < 0.10:
                action_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                action_high = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
                action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

            action = np.clip(action, action_low, action_high)

            return Action(action)

        MoveToInitPoseOption = ParameterizedOption(
            "MoveToInitPoseOption",
            types=[gripper, base],
            params_space=Box(-5, 5, (1,)),
            policy=move_to_init_pose_policy,
            initiable=_move_to_init_pose_option_initiable,
            terminal=_move_to_init_pose_option_terminal,
        )
        options.add(MoveToInitPoseOption)

        """---------------------------------- MoveToInitPoseOption Ends ----------------------------------"""

        """---------------------------------- DS_OpenSingleDoor_MoveTowards_option starts ----------------------------------"""

        def _DS_OpenSingleDoor_MoveTowards_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                memory["model"] = _create_simple_ds_model()
            memory["handle_pos"], memory["handle_rot"] = _init_object_of_interest_transform(state, objects, offset_handle_frame=np.array([-0.0, RoboKitchenEnv.offset_inwards_from_handle, 0.0]))
            return True

        # DS_move_option - always initiable, empty policy, never terminates
        def _DS_OpenSingleDoor_MoveTowards_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return _DS_general_move_static_option_initiable(option="OpenSingleDoor_MoveTowards_option", state=state, memory=memory, objects=objects, params=params)

        def _DS_OpenSingleDoor_MoveTowards_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            general_terminal = _DS_general_move_option_terminal(state, memory, objects, params)

            if general_terminal:
                return True

            gripper = objects[0]
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            gripper_pos_in_handle, _ = frame_transform(gripper_pos, gripper_quat, memory["object_of_interest_pos"], memory["object_of_interest_rot"])
            if 'velocity_history' in memory and len(memory['velocity_history']) > 0:
                velocity = memory['velocity_history'][-1]
            else:
                velocity = 0.0

            if np.linalg.norm(gripper_pos_in_handle[0]) <= 0.1 and \
                gripper_pos_in_handle[1] > 0 and \
                velocity < 0.01:
                return True

            return False

        """---------------------------------- DS_OpenSingleDoor_MoveTowards_option ends ----------------------------------"""

        """---------------------------------- DS_OpenSingleDoor_MoveAway_option starts ----------------------------------"""

        # DS_move_away_option - always initiable, empty policy, never terminates
        def _DS_OpenSingleDoor_MoveAway_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                memory["model"] = _create_simple_ds_model()
            memory["handle_pos"], memory["handle_rot"] = _init_object_of_interest_transform(state, objects, offset_handle_frame=np.array([-0.6, -0.6, 0.0]))
            return True

        def _DS_OpenSingleDoor_MoveAway_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return _DS_general_move_dynamic_option_initiable(option="OpenSingleDoor_MoveAway_option", state=state, memory=memory, objects=objects, params=params)

        def _DS_OpenSingleDoor_MoveAway_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return _DS_general_move_option_terminal(state, memory, objects, params)

        """---------------------------------- DS_OpenSingleDoor_MoveAway_option ends ----------------------------------"""

        """---------------------------------- ReachBehindandPull_option starts ----------------------------------"""

        # ReachBehindandPull_option
        def _ReachBehindandPull_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # in memory, add a few waypoints, move first downwards, -z, then move forward, x, then move upwards, z, then -x
            # raise ValueError("ReachBehindandPull_option_initiable is not working, frame of waypoints is not correct")
            # print in red
            waypoints = [
                np.array([0.0, -0.1, 0.0]),
                np.array([0.2, -0.1, 0.0]), 
                np.array([0.2, 0.4, 0.0]),
                np.array([-0.4, 0.4, 0.0])
            ]
            memory["num_waypoints"] = len(waypoints)
            memory["waypoints"] = []
            memory["current_waypoint"] = 0

            for i, waypoint in enumerate(waypoints):
                model = _create_simple_ds_model()
                handle_pos, handle_rot = _init_object_of_interest_transform(
                    state, objects, 
                    offset_handle_frame=waypoint
                )
                # memory[f"model{i}"] = model
                if i == 0:
                    memory["model"] = model
                    memory["handle_pos"] = handle_pos 
                    memory["handle_rot"] = handle_rot

                else:
                    memory[f"model{i}"] = model
                    memory[f"handle_pos{i}"] = handle_pos 
                    memory[f"handle_rot{i}"] = handle_rot
                memory["waypoints"].append(handle_pos)

            return True

        def _ReachBehindandPull_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            memory["handle_pos"], memory["handle_rot"] = _init_object_of_interest_transform(state, objects, offset_handle_frame=np.array([0.0, 0.0, 0.0]))

            if "DS_reach_behind_and_pull_option" not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option="reach_behind_and_pull")
                CFG.option_to_policy["DS_reach_behind_and_pull_option"] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy["DS_reach_behind_and_pull_option"]

            _process_fail_memory(memory, "DS_reach_behind_and_pull_option", objects[0], objects[1])

            if CFG.visualizer:
                # memory["ds_policy"].init_demo_traj_scores(memory["handle_pos"])
                CFG.visualizer.set_demo_trajs(memory["ds_policy"].x, memory["ds_policy"].demo_traj_probs)

            return True

        def _ReachBehindandPull_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            if memory["current_waypoint"] == len(memory["waypoints"]):
                return Action(np.zeros(7, dtype=np.float32))

            # Get current gripper position and transform to handle frame
            gripper_state = state.vec([objects[0]])
            # Store previous state if not already stored
            if "prev_gripper_state" not in memory:
                memory["prev_gripper_state"] = gripper_state

            # Check if gripper is not moving much
            # gripper_movement = np.linalg.norm(gripper_state[:3] - memory["prev_gripper_state"][:3])
            # if gripper_movement < 0.01:
            #     memory["stationary_count"] += 1
            # else:
            #     memory["stationary_count"] = 0

            # # Update previous state
            # memory["prev_gripper_state"] = gripper_state

            # If within threshold of current waypoint or gripper is stuck, move to next one
            print (np.linalg.norm(gripper_state[:3] - memory["waypoints"][memory["current_waypoint"]]))
            if (np.linalg.norm(gripper_state[:3] - memory["waypoints"][memory["current_waypoint"]]) < 0.06):
                memory["current_waypoint"] += 1
                print(f"Reached waypoint {memory['current_waypoint']}")
                if memory["current_waypoint"] == len(memory["waypoints"]):
                    print("Reached end of waypoints")
                    return Action(np.zeros(7, dtype=np.float32))
                # Otherwise use this waypoint's model and transform
                memory["model"] = memory[f"model{memory['current_waypoint']}"]
                memory["handle_pos"] = memory[f"handle_pos{memory['current_waypoint']}"]
                memory["handle_rot"] = memory[f"handle_rot{memory['current_waypoint']}"]

            return _DS_general_move_option_policy(state, memory, objects, params)

        def _ReachBehindandPull_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            gripper, _, base = objects
            gripper_pos = state.get(gripper, "translation")

            # Store previous gripper position if not already in memory
            if "prev_gripper_pos" not in memory:
                memory["prev_gripper_pos"] = gripper_pos
                return False

            # Calculate velocity
            velocity = np.linalg.norm(gripper_pos - memory["prev_gripper_pos"])
            memory["prev_gripper_pos"] = gripper_pos

            # Check if the robot is stuck (velocity too small for too long)
            if _is_robot_stuck(memory, velocity):
                return True

            return False

        """---------------------------------- ReachBehindandPull_option ends ----------------------------------"""

        """---------------------------------- DS_PnPCounterToCab_Pick_option starts ----------------------------------"""

        def _DS_PnPCounterToCab_Pick_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return _DS_general_move_static_option_initiable(option="PnPCounterToCab_Pick_option", state=state, memory=memory, objects=objects, params=params)

        def _DS_PnPCounterToCab_Pick_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            general_terminal = _DS_general_move_option_terminal(state, memory, objects, params)
            if general_terminal:
                return True

            gripper = objects[0]
            gripper_pos = state.get(gripper, "translation")
            gripper_quat = state.get(gripper, "quaternion")
            gripper_pos_in_object_of_interest, _ = frame_transform(gripper_pos, gripper_quat, memory["object_of_interest_pos"], memory["object_of_interest_rot"])
            if 'velocity_history' in memory and len(memory['velocity_history']) > 0:
                velocity = memory['velocity_history'][-1]
            else:
                velocity = 0.0

            if np.linalg.norm(gripper_pos_in_object_of_interest) <= 0.01 and \
                velocity < 0.01:
                return True

            return False

        """---------------------------------- DS_PnPCounterToCab_Pick_option ends ----------------------------------"""

        """---------------------------------- DS_PnPCounterToCab_Place_option starts ----------------------------------"""

        def _DS_PnPCounterToCab_Place_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return _DS_general_move_static_option_initiable(option="PnPCounterToCab_Place_option", state=state, memory=memory, objects=objects, params=params)

        def _DS_PnPCounterToCab_Place_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            general_terminal = _DS_general_move_option_terminal(state, memory, objects, params)
            if general_terminal:
                return True

        """---------------------------------- DS_PnPCounterToCab_Place_option ends ----------------------------------"""

        """---------------------------------- GripperOpen_option starts ----------------------------------"""

        # GripperOpen_option
        def _GripperOpen_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperOpen_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))  # Open gripper

        def _GripperOpen_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Get current gripper quaternion
            left_finger, right_finger = objects
            curr_quat = state.get(left_finger, "quaternion")

            # Store previous quaternion in memory if not already there
            if "prev_quat" not in memory:
                memory["prev_quat"] = curr_quat
                return False

            # Check if quaternion hasn't changed and gripper is open
            quat_unchanged = np.allclose(curr_quat, memory["prev_quat"], atol=1e-3)
            # Use the finger objects passed in
            is_open = RoboKitchenEnv._GripperOpen_holds(state, [left_finger, right_finger])

            # Update memory
            memory["prev_quat"] = curr_quat

            return quat_unchanged and is_open

        """---------------------------------- GripperOpen_option ends ----------------------------------"""

        """---------------------------------- GripperClose_option starts ----------------------------------"""

        # GripperClose_option
        def _GripperClose_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperClose_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))  # Close gripper

        def _GripperClose_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Get current gripper quaternion
            left_finger, right_finger = objects
            left_finger_pos = state.get(left_finger, "translation")
            right_finger_pos = state.get(right_finger, "translation")
            curr_distance = np.linalg.norm(left_finger_pos - right_finger_pos)

            # Store previous distance in memory if not already there
            if "prev_distance" not in memory:
                memory["prev_distance"] = curr_distance
                return False

            # Check if distance hasn't changed and gripper is closed
            distance_unchanged = np.allclose(curr_distance, memory["prev_distance"], atol=1e-3)
            # Use the finger objects passed in
            is_closed = RoboKitchenEnv._GripperClosed_holds(state, [left_finger, right_finger])

            # Update memory
            memory["prev_distance"] = curr_distance

            return distance_unchanged and is_closed

        """---------------------------------- GripperClose_option ends ----------------------------------"""

        """---------------------------------- DummyOption starts ----------------------------------"""

        # Dummy initiable: always returns True.
        def _Dummy_initiable(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        def _Dummy_policy(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Action:
            # Here we assume an action vector of size 7.
            return Action(np.zeros(7, dtype=np.float32))

        def _Dummy_terminal(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        DummyOption = ParameterizedOption(
            "DummyOption",
            types=[],  # Adjust type requirements as needed.
            params_space=Box(-1.0, 1.0, (3,)),  # Example parameter space.
            policy=_Dummy_policy,
            initiable=_Dummy_initiable,
            terminal=_Dummy_terminal,
        )
        options.add(DummyOption)

        """---------------------------------- DummyOption ends ----------------------------------"""

        """---------------------------------- RepositionBaseOption starts ----------------------------------"""

        # RepositionBase initiable: always returns True.
        def _RepositionBase_option_initiable(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            # Extract target position and orientation from the 7D params
            # params is [x, y, z, qx, qy, qz, qw]
            base_target_pos = params[:3]  # 3D position for the base [x, y, z]
            base_target_quat = params[3:7]  # quaternion [qx, qy, qz, qw]
            
            # Store target position and orientation in memory
            memory["base_target_pos"] = base_target_pos
            memory["base_target_quat"] = base_target_quat
            
            # Store initial state to track progress
            # ref_obj, base = objects
            # memory["initial_base_pos"] = state.get(base, "translation")
            # memory["initial_base_quat"] = state.get(base, "quaternion")
            
            return True

        def _RepositionBase_option_policy(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Action:
            ref_obj, base = objects
            
            # Calculate current relative pose using the same method as predicate construction
            # This gives the pose of the base in the reference object's frame
            current_rel_pose = calculate_relative_pose(
                state, ref_obj, base,
                CFG.trans_feat_name, CFG.quat_feat_name
            )
            
            # Target position and orientation from memory (provided by sampler)
            target_pos = memory["base_target_pos"]
            target_quat = memory["base_target_quat"]
            
            # Position error in XY plane - this is in reference object's frame
            pos_error_ref_frame = np.array([target_pos[0] - current_rel_pose[0], 
                                           target_pos[1] - current_rel_pose[1]])
            
            # Convert quaternions to rotation objects for easier manipulation
            current_rot_ref_frame = R.from_quat(current_rel_pose[3:])  # Rotation of base in ref object frame
            target_rot = R.from_quat(target_quat)
            
            # Get current and target euler angles (xyz)
            current_euler = current_rot_ref_frame.as_euler("xyz")
            target_euler = target_rot.as_euler("xyz")
            
            # We only care about yaw (rotation around z-axis)
            yaw_error = target_euler[2] - current_euler[2]
            
            # Normalize yaw error to [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            
            # Calculate control velocities with proportional control
            K_pos = 3.0  # position gain
            K_rot = 2.0  # rotation gain
            
            # Linear velocities in reference object frame
            vel_x_ref_frame = K_pos * pos_error_ref_frame[0]
            vel_y_ref_frame = K_pos * pos_error_ref_frame[1]
            
            # Angular velocity (about z-axis)
            base_rot_vel = K_rot * yaw_error
            
            # Get base orientation in reference frame to transform velocities
            base_quat_in_ref_frame = current_rel_pose[3:]
            base_rot_in_ref_frame = R.from_quat(base_quat_in_ref_frame)
            
            # Transform linear velocities from reference frame to base frame
            # We need the inverse rotation from ref frame to base frame
            vel_ref_frame = np.array([vel_x_ref_frame, vel_y_ref_frame, 0.0])
            vel_base_frame = -base_rot_in_ref_frame.apply(vel_ref_frame)
            
            # Extract x,y velocities in base frame
            base_vel_x = vel_base_frame[0]
            base_vel_y = vel_base_frame[1]
            
            # Limit velocities
            max_vel = 0.5
            base_vel_x = np.clip(base_vel_x, -max_vel, max_vel)
            base_vel_y = np.clip(base_vel_y, -max_vel, max_vel)
            base_rot_vel = np.clip(base_rot_vel, -max_vel, max_vel)
            
            # Create action array (3D: [dx, dy, dyaw]) - in robot base frame
            action = np.array([base_vel_x, base_vel_y, base_rot_vel], dtype=np.float32)
            
            # Hack to expand the action to 7D for compatibility with the environment
            # The first 3 dimensions are the base velocities [dx, dy, dyaw]
            # The remaining 4 dimensions are zeros (not used for base movement)
            action_7d = np.zeros(7, dtype=np.float32)
            action_7d[:3] = action  # Copy the base velocities to the first 3 dimensions
            
            return Action(action_7d)

        def _RepositionBase_option_terminal(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            ref_obj, base = objects
            
            # Get reference object (should be the first object in objects)
            ref_obj = objects[0]
            
            # Calculate current relative pose using the same method as predicate construction
            current_rel_pose = calculate_relative_pose(
                state, ref_obj, base,
                CFG.trans_feat_name, CFG.quat_feat_name
            )
            
            # Target position and orientation from memory
            target_pos = memory["base_target_pos"]
            target_quat = memory["base_target_quat"]
            
            # Position error in XY plane
            pos_error = np.array([target_pos[0] - current_rel_pose[0], 
                                 target_pos[1] - current_rel_pose[1]])
            
            # Convert quaternions to rotation objects
            current_rot = R.from_quat(current_rel_pose[3:])
            target_rot = R.from_quat(target_quat)
            
            # Get current and target euler angles (xyz)
            current_euler = current_rot.as_euler("xyz")
            target_euler = target_rot.as_euler("xyz")
            
            # Calculate yaw error
            yaw_error = target_euler[2] - current_euler[2]
            
            # Normalize yaw error to [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            
            # Define thresholds for position and orientation errors
            pos_threshold = 0.05  # meters
            rot_threshold = 0.1  # radians 6 deg
            
            # Check if position and orientation errors are below thresholds
            pos_close = np.linalg.norm(pos_error) < pos_threshold
            rot_close = abs(yaw_error) < rot_threshold
            
            return pos_close and rot_close

        """---------------------------------- RepositionBaseOption ends ----------------------------------"""

        DS_OpenSingleDoor_MoveTowards_option = ParameterizedOption(
            "DS_OpenSingleDoor_MoveTowards_option",
            types=[gripper, grab, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            initiable=_DS_OpenSingleDoor_MoveTowards_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_OpenSingleDoor_MoveTowards_option_initiable_node,
            terminal=_DS_OpenSingleDoor_MoveTowards_option_terminal,
        )
        options.add(DS_OpenSingleDoor_MoveTowards_option)

        DS_OpenSingleDoor_MoveAway_option = ParameterizedOption(
            "DS_OpenSingleDoor_MoveAway_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_gripper_closed_option_policy,
            initiable=_DS_OpenSingleDoor_MoveAway_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_OpenSingleDoor_MoveAway_option_initiable_node,
            terminal=_DS_OpenSingleDoor_MoveAway_option_terminal,
        )
        options.add(DS_OpenSingleDoor_MoveAway_option)

        ReachBehindandPull_option = ParameterizedOption(
            "ReachBehindandPull_option",
            types=[gripper, handle, base],
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            # policy=_ReachBehindandPull_option_policy,
            initiable=_ReachBehindandPull_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _ReachBehindandPull_option_initiable_node,
            # initiable=_ReachBehindandPull_option_initiable_linear,
            terminal=_ReachBehindandPull_option_terminal,
        )
        options.add(ReachBehindandPull_option)

        PnPCounterToCab_Pick_option = ParameterizedOption(
            "PnPCounterToCab_Pick_option",
            types=[gripper, grab, base],
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            initiable=_DS_PnPCounterToCab_Pick_option_initiable,
            terminal=_DS_PnPCounterToCab_Pick_option_terminal,
        )
        options.add(PnPCounterToCab_Pick_option)

        PnPCounterToCab_Place_option = ParameterizedOption(
            "PnPCounterToCab_Place_option",
            types=[gripper, surface, base],
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_gripper_closed_option_policy,
            initiable=_DS_PnPCounterToCab_Place_option_initiable,
            terminal=_DS_PnPCounterToCab_Place_option_terminal,
        )
        options.add(PnPCounterToCab_Place_option)

        GripperOpen_option = ParameterizedOption(
            "GripperOpen_option",
            types=[left_finger, right_finger],  # Include both finger types
            params_space=Box(-5, 5, (1,)),
            policy=_GripperOpen_option_policy,
            initiable=_GripperOpen_option_initiable,
            terminal=_GripperOpen_option_terminal,
        )
        options.add(GripperOpen_option)

        GripperClose_option = ParameterizedOption(
            "GripperClose_option",
            types=[left_finger, right_finger],  # Include both finger types
            params_space=Box(-5, 5, (1,)),
            policy=_GripperClose_option_policy,
            initiable=_GripperClose_option_initiable,
            terminal=_GripperClose_option_terminal,
        )
        options.add(GripperClose_option)

        RepositionBase_option = ParameterizedOption(
            "RepositionBase_option",
            types=[object_type, base],
            params_space=Box(-5, 5, (7,)),  # [x, y, z, qx, qy, qz, qw] for target base position and orientation
            policy=_RepositionBase_option_policy,
            initiable=_RepositionBase_option_initiable,
            terminal=_RepositionBase_option_terminal,
        )
        options.add(RepositionBase_option)

        return options
