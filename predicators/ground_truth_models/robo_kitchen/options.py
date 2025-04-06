"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set, Optional, Type, Tuple, Any

import numpy as np
import os
import sys
import threading
import random
import torch
from gym.spaces import Box
import mujoco
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import request

import matplotlib.pyplot as plt

from predicators.settings import CFG

from ds_policy import DSPolicy, load_data

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, ParameterizedOption, ParameterizedTerminal, Predicate, State, Type

from predicators.utils import get_pos_quat_from_mujoco_state, xyzw_to_wxyz, wxyz_to_xyzw

import torch
from predicators.DS_models.gen_demo_model import DynamicalSystem

from scipy.spatial.transform import Rotation as R

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
        hinge = types["hinge_type"]
        gripper = types["gripper_type"]
        handle = types["handle_type"]
        base = types["base_type"]

        options: Set[ParameterizedOption] = set()

        """---------------------------------- Helper function starts ----------------------------------"""

        def _init_handle_transform(state: State, objects: Sequence[Object], offset_handle_frame: Optional[np.ndarray] = None):
            """Helper to initialize handle transform data in memory."""
            if len(objects) == 4:
                gripper, handle, base, hinge = objects
            else:
                gripper, handle, base = objects
            handle_pos, handle_quat = get_pos_quat_from_mujoco_state(state, handle)
            handle_rot = R.from_quat(handle_quat).as_matrix()
            if offset_handle_frame is not None:
                # Transform offset from handle frame to world frame before adding
                offset_world = handle_rot @ offset_handle_frame
                handle_pos = handle_pos + offset_world
            return handle_pos, handle_rot

        def _create_ds_policy(option: str):
            x, x_dot, q, omega, gripper_traj = load_data("smoothing_window_21_quat", option, finger=False, transform_to_handle_frame=True, debug_on=False)
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
        
        def _is_robot_stuck(memory: Dict, velocity: float, velocity_threshold: float = 0.005, stuck_time_threshold: int = 10) -> bool:
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
        
        def _process_fail_memory(memory: Dict, option_name: str, reference_pos: Optional[np.ndarray] = None, reference_rot: Optional[np.ndarray] = None) -> None:
            """Process fail memory entries for a specific option and update demo trajectory probabilities.
            
            Args:
                memory: Memory dictionary containing fail_memory and ds_policy
                option_name: Name of the option to process fail memory for
                reference_pos: Reference position for transformation (use handle_state if None)
                reference_rot: Reference rotation matrix for transformation (use handle_state if None)
            """
            if "fail_memory" not in memory or not memory["fail_memory"]:
                return
                
            for idx in range(len(memory["fail_memory"])-1, -1, -1):
                if memory["fail_memory"][idx].option_name == option_name:
                    gripper = RoboKitchenEnv.object_name_to_object("gripper")
                    gripper_pos, gripper_quat = get_pos_quat_from_mujoco_state(memory["fail_memory"][idx].state, gripper)
                    
                    # If reference position/rotation not provided, use handle state
                    if reference_pos is None or reference_rot is None:
                        handle = RoboKitchenEnv.object_name_to_object("handle")
                        handle_pos, handle_quat = get_pos_quat_from_mujoco_state(memory["fail_memory"][idx].state, handle)
                        ref_pos = handle_pos
                        ref_rot = R.from_quat(handle_quat).as_matrix()
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
                        "point", penalty=0.8, traj_threshold=0.2, radius=0.02,
                        angle_threshold=np.pi/2, lookahead=10
                    )
                    
                    # Remove processed entry
                    memory["fail_memory"].pop(idx)

        """---------------------------------- Helper function ends ----------------------------------"""

        """---------------------------------- general move option starts ----------------------------------"""
        
        def _DS_general_move_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            # Get objects
            gripper, _, base = objects
            handle_pos = memory["handle_pos"]
            handle_rot = memory["handle_rot"]
            # Get positions
            gripper_pos, gripper_quat = get_pos_quat_from_mujoco_state(state, gripper)

            pos_in_handle, rot_in_handle = frame_transform(gripper_pos, gripper_quat, handle_pos, handle_rot)
            expected_relative_rot_handle = R.from_quat(np.array([0.5, 0.5, 0.5, -0.5]))

            # Compute the difference between the expected relative rotation and the actual relative rotation
            relative_rotation = expected_relative_rot_handle * R.from_matrix(rot_in_handle).inv()
            angular_w_handle = relative_rotation.as_rotvec()
            world_w = handle_rot @ angular_w_handle
            
            robot_base_pos, robot_base_quat = get_pos_quat_from_mujoco_state(state, base)
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
                    velocity_in_handle = net(torch.from_numpy(pos_in_handle).float())
                
                warnings.warn("Velocity getting scaled, plz remove")
                velocity_in_handle[0] = velocity_in_handle[0] * 0.5 #handle frame x is the direction towards handle
                velocity_in_handle[1] = velocity_in_handle[1] * 0.05
                velocity_in_handle[2] = velocity_in_handle[2] * 0.5
                # Transform velocity back to world frame
                velocity_world = handle_rot @ velocity_in_handle.numpy()
                velocity_robot_base = robot_base_rot.T @ velocity_world
                
                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = velocity_robot_base
                arr[3:6] = 0.3 * robot_base_w
            
            elif "ds_policy" in memory:
                # Use DS policy
                ds_policy = memory["ds_policy"]

                action = ds_policy.get_action(np.concatenate([pos_in_handle, R.from_matrix(rot_in_handle).as_quat()]), clf=True, alpha_V=10.0, lookahead=20)
                vel = action[:6] # position + angular velocity

                if CFG.visualizer:
                    rel_handle_visualizer_rot = np.array([[0, 0, 1],
                                                          [1, 0, 0],
                                                          [0, 1, 0]])
                    CFG.visualizer.update_robot_position(pos_in_handle, xyzw_to_wxyz(R.from_matrix(rel_handle_visualizer_rot @ rot_in_handle).as_quat()))
                    CFG.visualizer.update_ref_traj(ds_policy.ref_traj_idx)
                    CFG.visualizer.update_ref_point(ds_policy.x[ds_policy.ref_traj_idx][ds_policy.ref_point_idx], xyzw_to_wxyz(R.from_matrix(rel_handle_visualizer_rot @ rot_in_handle).as_quat()))
                    
                x_dot_handle = vel[:3]
                r_dot_handle = vel[3:]
                x_dot_world = handle_rot @ x_dot_handle
                x_dot_robot_base = robot_base_rot.T @ x_dot_world
                r_dot_world = handle_rot @ r_dot_handle
                r_dot_robot_base = robot_base_rot.T @ r_dot_world

                mag = np.linalg.norm(r_dot_robot_base)
                if mag > 1:
                    r_dot_robot_base = r_dot_robot_base / mag

                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = x_dot_robot_base
                arr[3:6] = 0.8 * r_dot_robot_base
                # arr[3:6] = 0.8 * robot_base_w # NOTE: this is hardcoded, should be learned
                # arr[6] = action[6] # gripper
            else:
                # Fallback if neither model is available
                raise ValueError("No DS option policy found")

            # Clip the action to the action space limits
            action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_low, action_high)

            return Action(arr)
        
        def _DS_general_move_option_policy_move_away_gripper_closed(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            """
            NOTE: this is a cheat. hardcoded gripper closed for move away option
            TODO: should add gripper as another dimension in node to learn
            """
            action = _DS_general_move_option_policy(state, memory, objects, params)

            # Set gripper to closed
            action._arr[6] = 1.0

            return action

        """---------------------------------- general move option ends ----------------------------------"""

        """---------------------------------- DS_move_towards_option starts ----------------------------------"""

        def _DS_move_towards_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                memory["model"] = _create_simple_ds_model()
            memory["handle_pos"], memory["handle_rot"] = _init_handle_transform(state, objects, offset_handle_frame=np.array([-0.0, RoboKitchenEnv.offset_inwards_from_handle, 0.0]))
            return True

        # DS_move_option - always initiable, empty policy, never terminates
        def _DS_move_towards_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            memory["handle_pos"], memory["handle_rot"] = _init_handle_transform(state, objects, offset_handle_frame=np.array([0.0, 0.0, 0.0]))

            if "DS_move_towards_option" not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option="move_towards")
                CFG.option_to_policy["DS_move_towards_option"] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy["DS_move_towards_option"]
            
            _process_fail_memory(memory, "DS_move_towards_option")
            
            if CFG.visualizer:
                CFG.visualizer.set_demo_trajs(memory["ds_policy"].x, memory["ds_policy"].demo_traj_probs)

            return True

        def _DS_move_towards_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # handle_pos = memory["handle_pos"]
            gripper, _, base = objects
            gripper_pos, gripper_quat = get_pos_quat_from_mujoco_state(state, gripper)
            gripper_pos_in_handle, _ = frame_transform(gripper_pos, gripper_quat, memory["handle_pos"], memory["handle_rot"])
            
            # Store previous gripper position if not already in memory
            if "prev_gripper_pos" not in memory:
                memory["prev_gripper_pos"] = gripper_pos
                return False
            
            # Update previous gripper position
            velocity = np.linalg.norm(gripper_pos - memory["prev_gripper_pos"])
            memory["prev_gripper_pos"] = gripper_pos
            
            # Check if the robot is stuck (velocity too small for too long)
            if _is_robot_stuck(memory, velocity):
                return True
            
            # Original success condition
            if np.linalg.norm(gripper_pos_in_handle[0]) <= 0.1 and \
                gripper_pos_in_handle[1] > 0 and \
                velocity < 0.01:
                return True
            
            return False

        DS_move_towards_option = ParameterizedOption(
            "DS_move_towards_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            initiable=_DS_move_towards_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_move_towards_option_initiable_node,
            terminal=_DS_move_towards_option_terminal,
        )
        options.add(DS_move_towards_option)

        """---------------------------------- DS_move_towards_option ends ----------------------------------"""

        """---------------------------------- DS_move_away_option starts ----------------------------------"""

        # DS_move_away_option - always initiable, empty policy, never terminates
        def _DS_move_away_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                memory["model"] = _create_simple_ds_model()
            memory["handle_pos"], memory["handle_rot"] = _init_handle_transform(state, objects, offset_handle_frame=np.array([-0.6, -0.6, 0.0]))
            return True
        
        def _DS_move_away_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "DS_move_away_option" not in CFG.option_to_init_pose:
                memory["handle_pos"], memory["handle_rot"] = _init_handle_transform(state, objects, offset_handle_frame=np.array([0.0, 0.0, 0.0]))
                CFG.option_to_init_pose["DS_move_away_option"] = [memory["handle_pos"], memory["handle_rot"]]
            else:
                memory["handle_pos"] = CFG.option_to_init_pose["DS_move_away_option"][0]
                memory["handle_rot"] = CFG.option_to_init_pose["DS_move_away_option"][1]

            # if "ds_policy" not in memory:
            #     _create_ds_policy(memory, state, objects, option="move_away", offset_handle_frame=np.array([0.0, 0.0, 0.0]))
            if "DS_move_away_option" not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option="move_away")
                CFG.option_to_policy["DS_move_away_option"] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy["DS_move_away_option"]

            _process_fail_memory(memory, "DS_move_away_option", 
                                reference_pos=CFG.option_to_init_pose["DS_move_away_option"][0],
                                reference_rot=CFG.option_to_init_pose["DS_move_away_option"][1])
            
            if CFG.visualizer:
                CFG.visualizer.set_demo_trajs(memory["ds_policy"].x, memory["ds_policy"].demo_traj_probs)
                
            return True
        
        def _DS_move_away_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            gripper, _, base = objects
            gripper_pos, gripper_quat = get_pos_quat_from_mujoco_state(state, gripper)
            
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

        DS_move_away_option = ParameterizedOption(
            "DS_move_away_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy_move_away_gripper_closed,
            initiable=_DS_move_away_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_move_away_option_initiable_node,
            terminal=_DS_move_away_terminal,
        )
        options.add(DS_move_away_option)

        """---------------------------------- DS_move_away_option ends ----------------------------------"""

        """---------------------------------- ReachBehindandPull_option starts ----------------------------------"""

        # ReachBehindandPull_option
        def _ReachBehindandPull_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # in memory, add a few waypoints, move first downwards, -z, then move forward, x, then move upwards, z, then -x 
            # raise ValueError("ReachBehindandPull_option_initiable is not working, frame of waypoints is not correct")
            #print in red 
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
                handle_pos, handle_rot = _init_handle_transform(
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
            memory["handle_pos"], memory["handle_rot"] = _init_handle_transform(state, objects, offset_handle_frame=np.array([0.0, 0.0, 0.0]))

            if "DS_reach_behind_and_pull_option" not in CFG.option_to_policy:
                memory["ds_policy"] = _create_ds_policy(option="reach_behind_and_pull")
                CFG.option_to_policy["DS_reach_behind_and_pull_option"] = memory["ds_policy"]
            else:
                memory["ds_policy"] = CFG.option_to_policy["DS_reach_behind_and_pull_option"]

            _process_fail_memory(memory, "DS_reach_behind_and_pull_option")

            if CFG.visualizer:
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
            gripper_pos = np.array([state.get(gripper, "x"), state.get(gripper, "y"), state.get(gripper, "z")])
            
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

        """---------------------------------- ReachBehindandPull_option ends ----------------------------------"""

        """---------------------------------- GripperOpen_option starts ----------------------------------"""

        # GripperOpen_option
        def _GripperOpen_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperOpen_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))  # Open gripper

        def _GripperOpen_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Get current gripper angle
            curr_angle = state.get(objects[0], "angle")
            
            # Store previous angle in memory if not already there
            if "prev_angle" not in memory:
                memory["prev_angle"] = curr_angle
                return False
                
            # Check if angle hasn't changed and gripper is open
            angle_unchanged = abs(curr_angle - memory["prev_angle"]) < 1e-3
            is_open = RoboKitchenEnv._GripperOpen_holds(state, objects)
            
            # Update memory
            memory["prev_angle"] = curr_angle
            
            return angle_unchanged and is_open

        GripperOpen_option = ParameterizedOption(
            "GripperOpen_option",
            types=[gripper],  # Adjust type requirements as needed.
            params_space=Box(-5, 5, (1,)),
            policy=_GripperOpen_option_policy,
            initiable=_GripperOpen_option_initiable,
            terminal=_GripperOpen_option_terminal,
        )
        options.add(GripperOpen_option)

        """---------------------------------- GripperOpen_option ends ----------------------------------"""

        """---------------------------------- GripperClose_option starts ----------------------------------"""

        # GripperClose_option
        def _GripperClose_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperClose_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))  # Close gripper

        def _GripperClose_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Get current gripper angle
            curr_angle = state.get(objects[0], "angle")
            
            # Store previous angle in memory if not already there
            if "prev_angle" not in memory:
                memory["prev_angle"] = curr_angle
                return False
                
            # Check if angle hasn't changed and gripper is closed
            angle_unchanged = abs(curr_angle - memory["prev_angle"]) < 1e-3
            is_closed = RoboKitchenEnv._GripperClosed_holds(state, objects)
            
            # Update memory
            memory["prev_angle"] = curr_angle
            
            return angle_unchanged and is_closed

        GripperClose_option = ParameterizedOption(
            "GripperClose_option",
            types=[gripper],  # Adjust type requirements as needed.
            params_space=Box(-5, 5, (1,)),
            policy=_GripperClose_option_policy,
            initiable=_GripperClose_option_initiable,
            terminal=_GripperClose_option_terminal,
        )
        options.add(GripperClose_option)

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

        return options