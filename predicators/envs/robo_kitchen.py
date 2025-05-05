"""A Kitchen environment wrapping robosuite kitchen."""

import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import robocasa.macros as macros
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, Observation, Predicate, State, Type, Video
import matplotlib
from collections import OrderedDict
from termcolor import colored
import warnings
import os
import mujoco
import time
import logging
from scipy.spatial.transform import Rotation as R

from robosuite.devices import Keyboard


# Disable JAX debug messages
logging.getLogger("jax._src.cache_key").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)

# Constants from demo files
MAX_CARTESIAN_DISPLACEMENT = 1.0
MAX_ROTATION_DISPLACEMENT = 1.0


class RoboKitchenEnv(BaseEnv):
    """Kitchen environment using robosuite."""

    door_open_thresh = np.deg2rad(80)  # rad
    door_half_open_thresh = 0.4  # rad
    grab_close_distance_thresh = 0.02  # m
    gripper_fingers_distance_thresh = 0.08  # m
    place_close_distance_thresh = 0.1  # m

    online_door_open_thresh = np.deg2rad(60)  # rad
    online_place_close_distance_thresh = 0.1  # m

    # Types
    object_type = Type("object_type", ["translation", "quaternion"])
    grab_type = Type("grab_type", ["translation", "quaternion"], parent=object_type)
    base_type = Type("base_type", ["translation", "quaternion"], parent=object_type)
    gripper_type = Type("gripper_type", ["translation", "quaternion"], parent=object_type)
    left_finger_type = Type("left_finger_type", ["translation", "quaternion"], parent=object_type)
    right_finger_type = Type("right_finger_type", ["translation", "quaternion"], parent=object_type)
    cabinet_type = Type("cabinet_type", ["translation", "quaternion"], parent=object_type)
    handle_type = Type("handle_type", ["translation", "quaternion"], parent=grab_type)
    surface_type = Type("surface_type", ["translation", "quaternion"], parent=object_type)
    thing_type = Type("thing_type", ["translation", "quaternion"], parent=grab_type)

    obj_name_to_type = {
        "handle": handle_type,
        "left_door_handle": handle_type,
        "right_door_handle": handle_type,
        "gripper": gripper_type,
        "left_finger": left_finger_type,
        "right_finger": right_finger_type,
        "cabinet": cabinet_type,
        "robot0_base": base_type,
        "obj": thing_type,
        "bottom": surface_type,
    }

    tasks_extended = [
        "Lift",
        "Stack",
        "NutAssembly",
        "NutAssemblySingle",
        "NutAssemblySquare",
        "NutAssemblyRound",
        "PickPlace",
        "PickPlaceSingle",
        "PickPlaceMilk",
        "PickPlaceBread",
        "PickPlaceCereal",
        "PickPlaceCan",
        "Door",
        "Wipe",
        "ToolHang",
        "TwoArmLift",
        "TwoArmPegInHole",
        "TwoArmHandover",
        "TwoArmTransport",
        "Kitchen",
        "KitchenDemo",
        "CupcakeCleanup",
        "OrganizeBakingIngredients",
        "PastryDisplay",
        "FillKettle",
        "HeatMultipleWater",
        "VeggieBoil",
        "ArrangeTea",
        "KettleBoiling",
        "PrepareCoffee",
        "ArrangeVegetables",
        "BreadSetupSlicing",
        "ClearingTheCuttingBoard",
        "MeatTransfer",
        "OrganizeVegetables",
        "BowlAndCup",
        "CandleCleanup",
        "ClearingCleaningReceptacles",
        "CondimentCollection",
        "DessertAssembly",
        "DrinkwareConsolidation",
        "FoodCleanup",
        "DefrostByCategory",
        "MicrowaveThawing",
        "QuickThaw",
        "ThawInSink",
        "AssembleCookingArray",
        "FryingPanAdjustment",
        "MealPrepStaging",
        "SearingMeat",
        "SetupFrying",
        "BreadSelection",
        "CheesyBread",
        "PrepareToast",
        "SweetSavoryToastSetup",
        "PrepForTenderizing",
        "PrepMarinatingMeat",
        "ColorfulSalsa",
        "SetupJuicing",
        "SpicyMarinade",
        "HeatMug",
        "MakeLoadedPotato",
        "SimmeringSauce",
        "WaffleReheat",
        "WarmCroissant",
        "BeverageSorting",
        "RestockBowls",
        "RestockPantry",
        "StockingBreakfastFoods",
        "CleanMicrowave",
        "CountertopCleanup",
        "PrepForSanitizing",
        "PushUtensilsToSink",
        "DessertUpgrade",
        "PanTransfer",
        "PlaceFoodInBowls",
        "PrepareSoupServing",
        "ServeSteak",
        "WineServingPrep",
        "ArrangeBreadBasket",
        "BeverageOrganization",
        "DateNight",
        "SeasoningSpiceSetup",
        "SetBowlsForSoup",
        "SizeSorting",
        "BreadAndCheese",
        "CerealAndBowl",
        "MakeFruitBowl",
        "VeggieDipPrep",
        "YogurtDelightPrep",
        "MultistepSteaming",
        "SteamInMicrowave",
        "SteamVegetables",
        "ManipulateDrawer",
        "OpenDrawer",
        "CloseDrawer",
        "DrawerUtensilSort",
        "OrganizeCleaningSupplies",
        "PantryMishap",
        "ShakerShuffle",
        "SnackSorting",
        "DryDishes",
        "DryDrinkware",
        "PreSoakPan",
        "SortingCleanup",
        "StackBowlsInSink",
        "AfterwashSorting",
        "ClearClutter",
        "DrainVeggies",
        "PrewashFoodAssembly",
        "PnPCoffee",
        "CoffeeSetupMug",
        "CoffeeServeMug",
        "CoffeePressButton",
        "ManipulateDoor",
        "OpenDoor",
        "OpenSingleDoor",
        "OpenDoubleDoor",
        "CloseDoor",
        "CloseSingleDoor",
        "CloseDoubleDoor",
        "MicrowavePressButton",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
        "NavigateKitchen",
        "PnP",
        "PnPCounterToCab",
        "PnPCabToCounter",
        "PnPCounterToSink",
        "PnPSinkToCounter",
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "PnPCounterToStove",
        "PnPStoveToCounter",
        "ManipulateSinkFaucet",
        "TurnOnSinkFaucet",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
        "ManipulateStoveKnob",
        "TurnOnStove",
        "TurnOffStove",
        "StoreFruit",
    ]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        print(f"ALL_KITCHEN_ENVIRONMENTS: {ALL_KITCHEN_ENVIRONMENTS}")

        if self._using_gui:
            pass
            # assert not CFG.make_test_videos or CFG.make_failure_videos, \
            #     "Turn off --use_gui to make videos in robo kitchen env"

        self._pred_name_to_pred = self.create_predicates()
        self._env = None  # Will be created in reset
        self._env_raw = None
        self.task_selected = CFG.robo_kitchen_task
        if self.task_selected not in self.tasks_extended:
            raise ValueError(f"Task {self.task_selected} not supported")
        print(colored(f"Selected task: {self.task_selected}", "green"))

        self.device = None  # control device

    def get_objects_of_interest(self, task_name: str) -> List[Object]:
        """Get the object of interest for the task."""
        # by default, there are robot and gripper objects
        if task_name == "OpenSingleDoor":
            return [self.object_name_to_object("handle")]
        elif task_name == "PnPCounterToCab":
            return [self.object_name_to_object("obj")]
        elif task_name == "StoreFruit":
            return [self.object_name_to_object("handle"), self.object_name_to_object("obj")]
        else:
            raise ValueError(f"Task {task_name} not supported")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create tasks for training."""
        using_recorded_data = True
        if using_recorded_data:
            tasks = []

            # Get the demo dataset path
            from robocasa.utils.dataset_registry import get_ds_path

            dataset_path = get_ds_path(self.task_selected, ds_type="human_raw")

            if dataset_path is None or not os.path.exists(dataset_path):
                print(colored(f"Unable to find dataset for {self.task_selected}. Downloading...", "yellow"))
                from robocasa.scripts.download_datasets import download_datasets

                download_datasets(tasks=[self.task_selected], ds_types=["human_raw"])
                dataset_path = get_ds_path(self.task_selected, ds_type="human_raw")

            # Load the demos
            import h5py

            f = h5py.File(dataset_path, "r")
            demos = list(f["data"].keys())

            # Sort demos by index
            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]

            # Create tasks from each demo
            for task_idx in range(CFG.num_train_tasks):
                if task_idx >= len(demos):
                    break

                # Get demo data
                demo = f[f"data/{demos[task_idx]}"]
                # Get initial state info from first timestep of datagen_info
                initial_state = {}
                for key in demo["datagen_info"].keys():
                    initial_state[key] = demo["datagen_info"][key][0]
                initial_state["model"] = demo.attrs["model_file"]
                initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

                # Create observation
                obs = {"state_info": initial_state, "obs_images": []}

                # Get goal description from task name
                goal_description = self.task_selected

                # Create task
                task = EnvironmentTask(obs, goal_description)
                tasks.append(task)

            f.close()
            return tasks
        else:
            return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Create tasks for testing."""
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    def _get_tasks(self, num: int, train_or_test: str) -> List[EnvironmentTask]:
        """Create a list of tasks"""
        tasks = []

        for task_idx in range(num):
            # For now just use OpenSingleDoor as the default task
            task_name = self.task_selected
            # check if task_name is in available_tasks
            if task_name not in ALL_KITCHEN_ENVIRONMENTS:
                raise ValueError(f"Task {task_name} not supported")
            goal_description = task_name
            seed = task_idx

            # Get initial observation
            init_obs = self._reset_initial_state(seed, train_or_test, task_name)
            # let's not do that since we are not using reset from initial state
            # init_obs = {}
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)

        return tasks

    def goal_reached(self) -> bool:
        # check success
        # check door state using quaternions
        state = self.state_info_to_state(self._current_observation["state_info"])
        goal_desc = self.task_selected

        if goal_desc == "OpenSingleDoor":
            handle = self.object_name_to_object("handle")
            cabinet = self.object_name_to_object("cabinet")
            if self._DoorOpen_holds(state, [handle, cabinet]):
                return True
        elif goal_desc == "PnPCounterToCab" or goal_desc == "StoreFruit":
            obj = self.object_name_to_object("obj")
            bottom = self.object_name_to_object("bottom")
            if self._OnSurface_holds(state, [obj, bottom]):
                return True
        # elif goal_desc == "StoreFruit":
        #     handle = self.object_name_to_object("handle")
        #     bottom = self.object_name_to_object("bottom")
        #     cabinet = self.object_name_to_object("cabinet")
        #     obj = self.object_name_to_object("obj")
        #     if self._DoorOpen_holds(state, [handle, cabinet]) and self._OnSurface_holds(state, [obj, bottom]):
        #         return True
        else:
            return False

    def _reset_initial_state(self, seed: int, train_or_test: str, task_name: str, complex_config: bool = False) -> Observation:
        """Reset the environment to an initial state based on the seed."""
        # Create or recreate environment if needed
        warnings.warn("Resetting environment to initial state from seed not implemented for robosuite kitchen")
        if self._env is None:
            complex_config = True  # NOTE: this should be removed. only for mac
            if complex_config:
                robot_type = "PandaOmron"
                controller_config = load_composite_controller_config(robot=robot_type)

                config = {
                    "env_name": task_name,
                    "robots": robot_type,
                    "controller_configs": controller_config,
                    "layout_ids": [0],
                    "style_ids": None,
                    "translucent_robot": True,
                }

                print(colored(f"Initializing environment for task: {task_name}", "yellow"))

                self._env_raw = robosuite.make(
                    **config,
                    has_renderer=self._using_gui,
                    has_offscreen_renderer=False,
                    render_camera="robot0_frontview",
                    ignore_done=True,
                    use_camera_obs=False,
                    control_freq=20,
                    renderer="mjviewer",
                    # seed=4,
                )

                self._env = VisualizationWrapper(self._env_raw)
                self.ep_meta = self._env.get_ep_meta()
            else:
                print(f"Creating env for task: {task_name}, seed: {seed}, gui: {self._using_gui}")
                self._env = create_env(
                    env_name=task_name,
                    render_onscreen=self._using_gui,
                    seed=seed + 4,  # this seed the third demo opens to the right, will have replan
                )

        # Reset environment with seed
        obs = self._env.reset()

        if CFG.use_teleop:
            self.device = Keyboard(
                env=self._env,
                pos_sensitivity=4.0,
                rot_sensitivity=4.0,
            )
            self.device.start_control()

        # Update objects of interest based on task
        self.objects_of_interest = self.get_objects_of_interest(task_name)

        # Get contact information
        contact_set = self.get_object_level_contacts()

        # Get initial gripper in base pose
        initial_eef_pos_in_base = self._env.robots[0]._hand_pos["right"]
        initial_eef_orn_mat_in_base = self._env.robots[0]._hand_orn["right"]
        initial_eef_quat_in_base = T.mat2quat(initial_eef_orn_mat_in_base)
        # CFG.init_pose = np.concatenate([initial_eef_pos_in_base, initial_eef_quat_in_base])
        self.initial_eef_pos_quat = np.concatenate([initial_eef_pos_in_base, initial_eef_quat_in_base])

        # Return observation
        return {"state_info": obs, "obs_images": [], "contact_set": contact_set}

    def get_object_level_contacts(self) -> set[Tuple[Object, Object]]:
        """Get all contacts between objects in the environment, default to have robot and gripper, in addition to the objects of interest
        this has to be a method not a class method since we need env access"""

        # only support panda robot for now
        contacts = set()
        # robot_contacts = self._env.get_contacts(self._env.robots[0].robot_model.models[0]) # robot
        gripper_contact = self._env.get_contacts(self._env.robots[0].robot_model.models[1])  # gripper
        # filter down to only include objects of interest

        object_names = [obj.name for obj in self.objects_of_interest]
        # robot_obj = self.object_name_to_object("robot")
        gripper_obj = self.object_name_to_object("gripper")

        # for contact in robot_contacts: # each contact is a string
        #     for obj_name in object_names:
        #         if obj_name in contact:
        #             obj = self.object_name_to_object(obj_name)
        #             contacts.add((obj, robot_obj))
        for contact in gripper_contact:
            for obj_name in object_names:
                if obj_name in contact:
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((gripper_obj, obj))

        return contacts

    @classmethod
    def get_name(cls) -> str:
        return "robo_kitchen"

    @classmethod
    def create_predicates(cls) -> Dict[str, Predicate]:
        """Exposed for perceiver."""
        preds = {
            Predicate("ReadyGrabObj", [cls.gripper_type, cls.object_type], cls._ReadyGrabObj_holds),
            Predicate("GripperOpen", [cls.left_finger_type, cls.right_finger_type], cls._GripperOpen_holds),
            Predicate("GripperClosed", [cls.left_finger_type, cls.right_finger_type], cls._GripperClosed_holds),
            Predicate("DoorOpen", [cls.handle_type, cls.cabinet_type], cls._DoorOpen_holds),
            Predicate("DoorClosed", [cls.handle_type, cls.cabinet_type], cls._DoorClosed_holds),
            Predicate("InContact", [cls.object_type, cls.object_type], cls._InContact_holds),
            Predicate("OnSurface", [cls.thing_type, cls.surface_type], cls._OnSurface_holds),
            Predicate("DoorHalfOpen", [cls.handle_type, cls.cabinet_type], cls._DoorHalfOpen_holds),
            Predicate("InOrigin", [cls.gripper_type, cls.base_type], cls._InOrigin_holds),
        }

        return {p.name: p for p in preds}

    def simulate(self, state: State, action: Action) -> State:
        """Get next state from current state and action."""
        # Implement simulation logic
        raise NotImplementedError("Simulate not implemented for robosuite kitchen")

    def viz_type_frames(self, target_type: Type):
        """Show frames of objects of the given type."""
        objects = self._current_state.get_objects(target_type)
        for obj in objects:
            pos = self._current_state.get(obj, "translation")
            quat = self._current_state.get(obj, "quaternion")
            self.mjshowframe(pos, quat, name=obj.name)

    def step(self, action: Action) -> Observation:
        """Execute action and return observation.

        Convert 7D predicators action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        to 12D robocasa action [right_pose(6), right_gripper(1), base(3), torso(1), extra(1)]
        """

        # Debugging: show frames of gripper, target and surface
        self.viz_type_frames(self.gripper_type)
        self.viz_type_frames(self.grab_type)
        self.viz_type_frames(self.surface_type)
        self.viz_type_frames(self.cabinet_type)
        self.viz_type_frames(self.base_type)

        if CFG.use_teleop:
            input_ac_dict = self.device.input2action(mirror_actions=True)
            # print(f"input_ac_dict: {input_ac_dict}")
            # action_keyboard = self._env.robots[0].create_action_vector(input_ac_dict)

        # Scale the action
        pos_delta = action.arr[:3] * MAX_CARTESIAN_DISPLACEMENT
        rot_delta = action.arr[3:6] * MAX_ROTATION_DISPLACEMENT
        gripper_cmd = action.arr[6]

        # Create 12D robocasa action:
        # - First 6D: right arm pose (position + rotation)
        # - Next 1D: right gripper
        # - Next 3D: base (no movement)
        # - Next 1D: torso (no movement)
        # - Last 1D: extra dimension (not used)
        env_action = np.zeros(12, dtype=np.float32)

        env_action[0:3] = pos_delta  # position control
        env_action[3:6] = rot_delta  # rotation control
        env_action[6] = gripper_cmd  # gripper control
        if CFG.use_teleop:
            env_action[7:10] = input_ac_dict["base"]
        # env_action[7:10] are zeros (no base movement)
        # env_action[10] is zero (no torso movement)
        # env_action[11] is zero (extra dimension)

        # Execute action in environment (Robosuite:Mujoco Env)
        obs, _, _, _ = self._env.step(env_action)

        contact_set = self.get_object_level_contacts()

        observation = {"state_info": obs, "obs_images": [], "contact_set": contact_set}

        self._current_observation = observation
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Reset environment to initial state for the given task."""
        self._current_task = self.get_task(train_or_test, task_idx)
        task_name = self._current_task.goal_description
        warnings.warn("Resetting environment to initial state from not implemented, just reset the env")
        self._current_observation = self._reset_initial_state(seed=task_idx, train_or_test=train_or_test, task_name=task_name)
        return self._copy_observation(self._current_observation)

    def render(self, action: Optional[Action] = None, caption: Optional[str] = None) -> Video:  # this renders the robot observation, not the viewer??
        """Render current state."""
        return self._env.render()

    def mjprint(self, text, auto_clean=False):
        """Print text in the viewer."""
        if self._env_raw is not None:
            self._env_raw.viewer.mjprint(text, auto_clean=auto_clean)

    def mjshowframe(self, xyz, quat=(1, 0, 0, 0), size=0.1, name=None, keep=False):
        """Show frame in the viewer."""
        if self._env_raw is not None:
            self._env_raw.viewer.mjshowframe(xyz, quat=quat, size=size, name=name, keep=keep)

    @property
    def action_space(self) -> Box:
        """7D action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]"""
        return Box(-1.0, 1.0, (7,), np.float32)

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        return {
            self._pred_name_to_pred["DoorOpen"],
            self._pred_name_to_pred["OnSurface"],
        }
        goal_desc = self.task_selected
        goal_preds = set()
        if goal_desc == "OpenSingleDoor":
            goal_preds = {self._pred_name_to_pred["DoorOpen"]}
        elif goal_desc == "PnPCounterToCab":
            goal_preds = {self._pred_name_to_pred["OnSurface"]}
        elif goal_desc == "StoreFruit":
            goal_preds = {self._pred_name_to_pred["DoorOpen"], self._pred_name_to_pred["OnSurface"]}
        return goal_preds

    @property
    def inContact_predicate(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        return set([self._pred_name_to_pred["InContact"]])

    @property
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        # Initialize predicates similar to kitchen.py

        return set(self._pred_name_to_pred.values())

    @property
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        return {
            self.object_type,
            self.base_type,
            self.gripper_type,
            self.left_finger_type,
            self.right_finger_type,
            self.cabinet_type,
            self.handle_type,
            self.surface_type,
            self.thing_type,
            self.grab_type,
        }

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def _copy_observation(self, obs: Observation) -> Observation:
        """Create copy of observation."""
        return copy.deepcopy(obs)

    def render_state_plt(self, state: State, task: EnvironmentTask, action: Optional[Action] = None, caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    @classmethod
    def object_name_to_object(cls, obj_name: str) -> Object:
        """Made public for perceiver."""
        if obj_name in cls.obj_name_to_type:
            return Object(obj_name, cls.obj_name_to_type[obj_name])
        else:
            return None
            raise ValueError(f"Object {obj_name} not found in obj_name_to_type")

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any], contact_set: set[Tuple[Object, Object]] = None) -> State:

        if hasattr(CFG, "load_approach") and CFG.load_approach:
            cls.door_open_thresh = cls.online_door_open_thresh  # rad
            cls.place_close_distance_thresh = cls.online_place_close_distance_thresh  # m

        state_dict = {}

        # Process any other objects with standard format
        for key, val in state_info.items():
            if key.endswith("_pos_quat"):
                obj_name = key[:-9]
                obj = cls.object_name_to_object(obj_name)
                translation = np.array([val[0], val[1], val[2]])
                quaternion = np.array([val[3], val[4], val[5], val[6]])
                if obj is not None:
                    state_dict[obj] = {"translation": translation, "quaternion": quaternion}
            elif key.endswith("_quat"):
                obj_name = key[:-5]  # Remove _pos
                translation = np.array(state_info[key[:-5] + "_pos"])
                quaternion = np.array(val)
                obj = cls.object_name_to_object(obj_name)
                if obj is not None:
                    state_dict[obj] = {"translation": translation, "quaternion": quaternion}

        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = {}
        state.items_in_contact = contact_set  # when defaults, it means Not populated, when empty means no contact
        cls._current_state = state
        return state

    @classmethod
    def _InOrigin_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is at origin."""
        gripper, base = objects

        # Get the predefined initial relative pose (stored as class variables)
        initial_pos_rel = CFG.init_pose[:3]
        initial_quat_rel = CFG.init_pose[3:]
        initial_rot_rel = R.from_quat(initial_quat_rel)

        gripper_pos_world = state.get(gripper, "translation")
        gripper_quat_world = state.get(gripper, "quaternion")  # Assuming xyzw format
        base_pos_world = state.get(base, "translation")
        base_quat_world = state.get(base, "quaternion")  # Assuming xyzw format

        # Convert current world quaternions (xyzw) to Scipy Rotations
        gripper_rot_world = R.from_quat(gripper_quat_world)  # Scipy expects xyzw
        base_rot_world = R.from_quat(base_quat_world)  # Scipy expects xyzw

        # Calculate current gripper pose relative to the current base pose
        # Position: rot_base_inv * (pos_gripper - pos_base)
        current_pos_rel = base_rot_world.inv().apply(gripper_pos_world - base_pos_world)
        # Orientation: rot_base_inv * rot_gripper
        current_rot_rel = base_rot_world.inv() * gripper_rot_world

        # Define tolerances
        pos_tolerance = 0.2  # meters
        angle_tolerance = np.deg2rad(90)  # radians

        # Check position distance
        pos_diff = np.linalg.norm(current_pos_rel - initial_pos_rel)
        pos_close = pos_diff < pos_tolerance

        # Check orientation difference (angle of relative rotation between current and initial)
        delta_rot = initial_rot_rel.inv() * current_rot_rel
        # Use magnitude of rotation vector as the angle difference
        angle_diff = np.linalg.norm(delta_rot.as_rotvec())
        ori_close = angle_diff < angle_tolerance

        return pos_close and ori_close

    @classmethod
    def _ReadyGrabObj_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is ready to grip handle."""

        gripper, obj = objects
        # Check if position of gripper is close to handle
        gripper_pos = state.get(gripper, "translation")
        gripper_quat = state.get(gripper, "quaternion")
        obj_pos = state.get(obj, "translation")
        obj_quat = state.get(obj, "quaternion")
        obj_pos_in_gripper, _ = frame_transform(obj_pos, obj_quat, gripper_pos, R.from_quat(gripper_quat).as_matrix())
        if np.linalg.norm(obj_pos_in_gripper) <= cls.grab_close_distance_thresh:
            return True
        return False

    @classmethod
    def _GripperOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is open by measuring distance between fingers."""
        left_finger, right_finger = objects

        # Get positions of both fingers
        left_pos = state.get(left_finger, "translation")
        right_pos = state.get(right_finger, "translation")

        # Calculate distance between fingers
        distance = np.linalg.norm(left_pos - right_pos)

        # If distance is greater than threshold, gripper is open
        return distance > cls.gripper_fingers_distance_thresh

    @classmethod
    def _GripperClosed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is closed by measuring distance between fingers."""
        left_finger, right_finger = objects

        # Get positions of both fingers
        left_pos = state.get(left_finger, "translation")
        right_pos = state.get(right_finger, "translation")

        # Calculate distance between fingers
        distance = np.linalg.norm(left_pos - right_pos)

        # If distance is less than threshold, gripper is closed
        return distance <= cls.gripper_fingers_distance_thresh

    @classmethod
    def _DoorOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if door is open by comparing rotation between door and cabinet."""
        door, cabinet = objects

        # Get quaternions from the objects passed in
        door_quat = state.get(door, "quaternion")
        cabinet_quat = state.get(cabinet, "quaternion")

        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation

        door_rot = Rotation.from_quat(door_quat)
        cabinet_rot = Rotation.from_quat(cabinet_quat)

        # Calculate relative rotation
        rel_rot = cabinet_rot.inv() * door_rot

        # Extract rotation value (approximation for hinge rotation)
        rot_vec = rel_rot.as_rotvec()
        rotation_value = np.linalg.norm(rot_vec)  # Total rotation angle in radians
        # print(f"rotation_value: {rotation_value}")
        return rotation_value > cls.door_open_thresh

    @classmethod
    def _DoorClosed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if door is closed by comparing rotation between door and cabinet."""
        door, cabinet = objects

        # Get quaternions from the objects passed in
        door_quat = state.get(door, "quaternion")
        cabinet_quat = state.get(cabinet, "quaternion")

        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation

        door_rot = Rotation.from_quat(door_quat)
        cabinet_rot = Rotation.from_quat(cabinet_quat)

        # Calculate relative rotation
        rel_rot = cabinet_rot.inv() * door_rot

        # Extract rotation value (approximation for hinge rotation)
        rot_vec = rel_rot.as_rotvec()
        rotation_value = np.linalg.norm(rot_vec)  # Total rotation angle in radians

        return rotation_value <= cls.door_open_thresh

    @classmethod
    def _DoorHalfOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if door is open by comparing rotation between door and cabinet."""
        door, cabinet = objects

        # Get quaternions from the objects passed in
        door_quat = state.get(door, "quaternion")
        cabinet_quat = state.get(cabinet, "quaternion")

        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation

        door_rot = Rotation.from_quat(door_quat)
        cabinet_rot = Rotation.from_quat(cabinet_quat)

        # Calculate relative rotation
        rel_rot = cabinet_rot.inv() * door_rot

        # Extract rotation value (approximation for hinge rotation)
        rot_vec = rel_rot.as_rotvec()
        rotation_value = np.linalg.norm(rot_vec)  # Total rotation angle in radians

        return rotation_value > cls.door_half_open_thresh

    @classmethod
    def _InContact_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if two objects are in contact using robosuite's contact checking."""
        obj1, obj2 = objects
        return (obj1, obj2) in state.items_in_contact or (obj2, obj1) in state.items_in_contact

    @classmethod
    def _OnSurface_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is at location."""
        obj, surface = objects
        obj_pos = state.get(obj, "translation")
        obj_quat = state.get(obj, "quaternion")
        surface_pos = state.get(surface, "translation")
        surface_quat = state.get(surface, "quaternion")
        obj_pos_in_surface, _ = frame_transform(obj_pos, obj_quat, surface_pos, R.from_quat(surface_quat).as_matrix())
        near_surface = obj_pos_in_surface[2] <= cls.place_close_distance_thresh
        in_surface = abs(obj_pos_in_surface[0]) <= 0.13 and abs(obj_pos_in_surface[1]) <= 0.13
        # print(obj_pos_in_surface[0], obj_pos_in_surface[1])
        # print(near_surface, in_surface)
        return near_surface and in_surface


def frame_transform(pos_in_init: np.ndarray, quat_in_init: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot_in_init = R.from_quat(quat_in_init).as_matrix()

    rel_pos_init = pos_in_init - target_pos

    pos_in_target = target_rot.T @ rel_pos_init
    rot_in_target = target_rot.T @ rot_in_init

    return pos_in_target, rot_in_target
