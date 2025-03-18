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

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ds_policy_path = os.path.join(workspace_root, "DS-Policy/src")
if ds_policy_path not in sys.path:
    sys.path.append(ds_policy_path)

from ds_policy import DSPolicy
from load_tools import load_data

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, ParameterizedOption, ParameterizedTerminal, Predicate, State, Type

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

        # assert _MJKITCHEN_IMPORTED, "See kitchen.py"

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

        def _init_handle_transform(memory: Dict, state: State, objects: Sequence[Object], offset_handle_frame: Optional[np.ndarray] = None) -> None:
            """Helper to initialize handle transform data in memory."""
            gripper, handle, base = objects
            handle_quat = np.array([state.get(handle, "qx"), state.get(handle, "qy"), 
                                  state.get(handle, "qz"), state.get(handle, "qw")])
            handle_pos = np.array([state.get(handle, "x"), state.get(handle, "y"), 
                                 state.get(handle, "z")])
            handle_rot = R.from_quat(handle_quat).as_matrix()
            memory["handle_init_rot"] = handle_rot
            if offset_handle_frame is not None:
                # Transform offset from handle frame to world frame before adding
                offset_world = handle_rot @ offset_handle_frame
                handle_pos = handle_pos + offset_world
            memory["handle_init_pos"] = handle_pos

        def _create_ds_policy(memory: Dict, state: State, objects: Sequence[Object], option: str, offset_handle_frame: Optional[np.ndarray] = None) -> None:
            if option == "move_towards":
                x, x_dot, q, omega = load_data("custom", option="move_towards")
                ds_policy = DSPolicy(x, x_dot, q, omega, dt=1/60, switch=False, use_avg=True)
            elif option == "move_away":
                x, x_dot, q, omega = load_data("custom", option="move_away")
                ds_policy = DSPolicy(x, x_dot, q, omega, dt=1/60, switch=False, use_avg=True)
            
            pos_model_path = f"DS-Policy/models/mlp_width128_depth3_{option}.pt"
            quat_model_path = f"DS-Policy/models/quat_model_{option}.json"
            
            if not os.path.exists(pos_model_path):
                ds_policy.train_pos_model(save_path=pos_model_path, batch_size=10, 
                                        lr_strategy=(1e-3, 1e-4, 1e-5), 
                                        epoch_strategy=(100, 100, 100), 
                                        plot=False, print_every=10)
            else:
                ds_policy.load_pos_model(pos_model_path)
                
            ds_policy.train_quat_model(save_path=quat_model_path, k_init=10)
            memory["ds_policy"] = ds_policy
            _init_handle_transform(memory, state, objects, offset_handle_frame)
            
            # visualizer = RuntimeVisualizer_plotly(x)
            # memory["visualizer"] = visualizer
            # memory["visualizer"]._run()

        def _create_simple_ds_model(memory: Dict, state: State, objects: Sequence[Object], offset_handle_frame: Optional[np.ndarray] = None) -> None:
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
            memory["model"] = model
            _init_handle_transform(memory, state, objects, offset_handle_frame)

        def _DS_move_towards_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "fail_memory" in memory:
                print("fail_memory of DS_move_towards_option")
                print(memory["fail_memory"])
            if "model" not in memory:
                _create_simple_ds_model(memory, state, objects, offset_handle_frame=np.array([0.0, RoboKitchenEnv.offset_inwards_from_handle, 0.0]))
            return True

        # DS_move_option - always initiable, empty policy, never terminates
        def _DS_move_towards_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "fail_memory" in memory:
                print("fail_memory of DS_move_towards_option")
                print(memory["fail_memory"])
            if "ds_policy" not in memory:
                _create_ds_policy(memory, state, objects, option="move_towards", offset_handle_frame=np.array([0.0, RoboKitchenEnv.offset_inwards_from_handle, 0.0]))
                # _create_ds_policy(memory, state, objects, option="move_towards", offset_handle_frame=np.array([0.0, 0.0, 0.0]))

            return True
        
        # DS_move_away_option - always initiable, empty policy, never terminates
        def _DS_move_away_option_initiable_linear(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                _create_simple_ds_model(memory, state, objects, offset_handle_frame=np.array([-0.6, -0.6, 0.0]))
                # NOTE: this means open the door to the left, some doors open to the right and won't work
            return True
        
        def _DS_move_away_option_initiable_node(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "ds_policy" not in memory:
                _create_ds_policy(memory, state, objects, option="move_away", offset_handle_frame=np.array([-0.6, -0.6, 0.0]))
            return True

        
        def _DS_general_move_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            # Get objects
            gripper, _, base = objects
            handle_init_pos = memory["handle_init_pos"]
            handle_init_rot = memory["handle_init_rot"]
            # Get positions
            gripper_pos = np.array([state.get(gripper, "x"), state.get(gripper, "y"), state.get(gripper, "z")])
            gripper_quat = np.array([state.get(gripper, "qx"), state.get(gripper, "qy"), state.get(gripper, "qz"), state.get(gripper, "qw")])
            gripper_rot = R.from_quat(gripper_quat).as_matrix()
            # Compute relative position in world frame
            rel_pos_world = gripper_pos - handle_init_pos
            # Transform relative position to handle frame
            pos_in_handle = handle_init_rot.T @ rel_pos_world
            # Transform gripper rotation to handle frame
            rot_in_handle = handle_init_rot.T @ gripper_rot

            expected_relative_rot_handle = R.from_quat(np.array([0.5, 0.5, 0.5, -0.5]))

            # Compute the difference between the expected relative rotation and the actual relative rotation
            relative_rotation = expected_relative_rot_handle * R.from_matrix(rot_in_handle).inv()
            angular_w_handle = relative_rotation.as_rotvec()
            # angular_w_handle = vee_operator(rel_rot_diff.as_matrix())
            # move that difference to the base frame
            world_w = handle_init_rot @ angular_w_handle

            robot_base_pos = np.array([state.get(base, "x"), state.get(base, "y"), state.get(base, "z")])
            robot_base_quat = np.array([state.get(base, "qx"), state.get(base, "qy"), state.get(base, "qz"), state.get(base, "qw")])
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
                velocity_world = handle_init_rot @ velocity_in_handle.numpy()
                velocity_robot_base = robot_base_rot.T @ velocity_world
                
                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = velocity_robot_base
                arr[3:6] = 0.3 * robot_base_w
            
            elif "ds_policy" in memory:
                # Use DS policy
                ds_policy = memory["ds_policy"]

                if "visualizer" in memory:
                    memory["visualizer"].update_position(pos_in_handle, ds_policy.ref_traj_idx)
                
                vel = ds_policy.get_action(np.concatenate([pos_in_handle, R.from_matrix(rot_in_handle).as_quat()]), clf=True, alpha_V=100.0, lookahead=10)
                x_dot_handle = vel[:3]
                r_dot_handle = vel[3:]
                
                # Transform velocity back to world frame
                x_dot_world = handle_init_rot @ x_dot_handle
                x_dot_robot_base = robot_base_rot.T @ x_dot_world
                r_dot_world = handle_init_rot @ r_dot_handle
                r_dot_robot_base = robot_base_rot.T @ r_dot_world

                mag = np.linalg.norm(r_dot_robot_base)
                if mag > 1:
                    r_dot_robot_base = r_dot_robot_base / mag

                # Create action array
                arr = np.zeros(7, dtype=np.float32)
                arr[:3] = x_dot_robot_base
                arr[3:6] = r_dot_robot_base
            
            else:
                # Fallback if neither model is available
                raise ValueError("No DS option policy found")

            # Clip the action to the action space limits
            action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_low, action_high)

            return Action(arr)

        def _DS_move_towards_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            handle_init_pos = memory["handle_init_pos"]
            gripper, _, base = objects
            gripper_pos = np.array([state.get(gripper, "x"), state.get(gripper, "y"), state.get(gripper, "z")])
            
            # Store previous gripper position if not already in memory
            if "prev_gripper_pos" not in memory:
                memory["prev_gripper_pos"] = gripper_pos
                return False
            
            # Update previous gripper position
            velocity = np.linalg.norm(gripper_pos - memory["prev_gripper_pos"])
            memory["prev_gripper_pos"] = gripper_pos
            
            if np.linalg.norm(gripper_pos - handle_init_pos) <= RoboKitchenEnv.offset_inwards_from_handle and velocity < 0.01:
                return True
            return False
        
        def _DS_move_away_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            return False

        DS_move_towards_option = ParameterizedOption(
            "DS_move_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            initiable=_DS_move_towards_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_move_towards_option_initiable_node,
            terminal=_DS_move_towards_option_terminal,
        )


        DS_move_away_option = ParameterizedOption(
            "DS_move_away_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_general_move_option_policy,
            initiable=_DS_move_away_option_initiable_linear if CFG.robo_kitchen_policy_model == "simple_ds" else _DS_move_away_option_initiable_node,
            terminal=_DS_move_away_terminal,
        )

        options.add(DS_move_towards_option)
        options.add(DS_move_away_option)

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

        return options

class RuntimeVisualizer_plotly:
    """
    Visualize demo trajectories and generated trajectory at runtime.
    """
    
    def __init__(self, demo_trajs=None):
        self.app = dash.Dash(__name__)
        self.demo_trajs = demo_trajs or []
        
        # Lists to hold our streaming runtime data
        self.runtime_xs = []
        self.runtime_ys = []
        self.runtime_zs = []
        
        # Track which trajectory is being followed
        self.last_ref_traj_idx = None
        self.current_ref_traj_idx = None
        
        # Flag to control whether to run the server
        self.running = False
        
        # Create the initial figure with demo trajectories
        self.init_fig = self._create_initial_figure()
        
        self.app.layout = html.Div([
            html.H3("RoboKitchen Trajectory Visualization"),
            dcc.Graph(id='live-graph', figure=self.init_fig),
            dcc.Interval(
                id='interval-component',
                interval=200,  # 200 ms = 5 updates per second
                n_intervals=0
            )
        ])
        
        # Set up callback for live updates
        @self.app.callback(
            Output('live-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph_live(n):
            return self._update_figure()
    
    def _create_initial_figure(self):
        """Create the initial figure with demo trajectories"""
        fig = go.Figure()
        
        # Add each demo trajectory as a separate trace
        for i, traj in enumerate(self.demo_trajs):
            # Extract positions (assuming first 3 columns are x,y,z)
            xs = traj[:, 0]
            ys = traj[:, 1]
            zs = traj[:, 2]
            
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='lines',
                    line=dict(width=2, color=f'rgba(0, 0, 255, 0.5)'),
                    name=f'Demo {i+1}'
                )
            )
        
        # Add empty trace for runtime data
        fig.add_trace(
            go.Scatter3d(
                x=self.runtime_xs,
                y=self.runtime_ys,
                z=self.runtime_zs,
                mode='lines+markers',
                line=dict(width=1, color='red'),
                marker=dict(size=2, color='red'),
                name='Current Execution'
            )
        )
        
        
        all_demo_points = np.vstack(self.demo_trajs)
        x_min, y_min, z_min = np.min(all_demo_points[:, :3], axis=0)
        x_max, y_max, z_max = np.max(all_demo_points[:, :3], axis=0)
        
        # Add some padding
        padding = 0.1
        x_range = [x_min - padding, x_max + padding]
        y_range = [y_min - padding, y_max + padding]
        z_range = [z_min - padding, z_max + padding]
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=x_range, title='X'),
                yaxis=dict(range=y_range, title='Y'),
                zaxis=dict(range=z_range, title='Z'),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def _update_figure(self):
        """Update the figure with new runtime data"""
        fig = go.Figure(self.init_fig)
        
        # Update the runtime trace (last trace)
        fig.data[-1].x = self.runtime_xs
        fig.data[-1].y = self.runtime_ys
        fig.data[-1].z = self.runtime_zs

        # Update the reference trajectory index
        fig.data[self.last_ref_traj_idx].line.color = 'rgba(0, 0, 255, 0.5)'
        fig.data[self.current_ref_traj_idx].line.color = 'rgba(0, 255, 0, 0.5)'
        return fig
    
    def update_position(self, pos, ref_traj_idx):
        """Add a new position to the runtime data"""
        self.runtime_xs.append(pos[0])
        self.runtime_ys.append(pos[1])
        self.runtime_zs.append(pos[2])
        
        # Update the reference trajectory index
        self.last_ref_traj_idx = self.current_ref_traj_idx
        self.current_ref_traj_idx = ref_traj_idx
    
    def _run(self):
        """Start the Dash server in a separate thread"""
        server_thread = threading.Thread(
            target=self.app.run_server,
            kwargs={"debug": False},
            daemon=True
        )
        server_thread.start()