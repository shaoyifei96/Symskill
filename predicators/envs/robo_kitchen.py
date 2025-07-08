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

if CFG.use_teleop:
    from robosuite.devices import Keyboard


# Disable JAX debug messages
logging.getLogger("jax._src.cache_key").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)

# Constants from demo files
MAX_CARTESIAN_DISPLACEMENT = 1.0
MAX_ROTATION_DISPLACEMENT = 1.0

# gripper - base offset in base frame

init_delta_gripper_base = np.array([ 0.24262412, -0.00722384,  0.58795444])
init_delta_gripper_base_rot = np.array([ 0.99227682,  0.03468661, -0.11850566,  0.01183061])

# q1 = np.array([x1, y1, z1, w1])  # First quaternion
# q2 = np.array([x2, y2, z2, w2])  # Second quaternion

# # Convert to Rotation objects
# r1 = R.from_quat(q1)
# r2 = R.from_quat(q2)

# # Get relative rotation (q2 * q1^-1)
# relative_rot = r2 * r1.inv()

# # Get the resulting quaternion
# result_quat = relative_rot.as_quat()

class RoboKitchenEnv(BaseEnv):
    """Kitchen environment using robosuite."""

    door_open_thresh = np.deg2rad(80)  # rad
    door_close_thresh = np.deg2rad(10)  # rad
    knob_on_thresh = 0.35  # rad
    door_half_open_thresh = 0.4  # rad
    grab_close_distance_thresh = 0.02  # m
    gripper_fingers_distance_thresh = 0.08  # m
    place_close_y_thresh = 0.1  # m
    place_close_xy_thresh = 0.15  # m

    online_door_open_thresh = np.deg2rad(70)  # rad
    online_door_close_thresh = np.deg2rad(10)  # rad

    # Types
    object_type = Type("object_type", ["translation", "quaternion"])
    grab_type = Type("grab_type", ["translation", "quaternion"], parent=object_type)
    knob_type = Type("knob_type", ["translation", "quaternion"], parent=grab_type)
    door_type = Type("door_type", ["translation", "quaternion"], parent=object_type)
    base_type = Type("base_type", ["translation", "quaternion"], parent=object_type)
    gripper_type = Type("gripper_type", ["translation", "quaternion"], parent=object_type)
    left_finger_type = Type("left_finger_type", ["translation", "quaternion"], parent=object_type)
    right_finger_type = Type("right_finger_type", ["translation", "quaternion"], parent=object_type)
    cabinet_type = Type("cabinet_type", ["translation", "quaternion"], parent=object_type)
    handle_type = Type("handle_type", ["translation", "quaternion"], parent=grab_type)
    surface_type = Type("surface_type", ["translation", "quaternion"], parent=object_type)
    thing_type = Type("thing_type", ["translation", "quaternion"], parent=grab_type)
    stove_type = Type("stove_type", ["translation", "quaternion", "on"], parent=object_type)
    microwave_type = Type("microwave_type", ["translation", "quaternion", "on"], parent=object_type)
    microwave_button_type = Type("microwave_button_type", ["translation", "quaternion"], parent=grab_type)
    drawer_type = Type("drawer_type", ["translation", "quaternion"], parent=object_type)
    sink_faucet_handle_type = Type("sink_faucet_handle_type", ["translation", "quaternion", "on"], parent=object_type)
    sink_type = Type("sink_type", ["translation", "quaternion"], parent=object_type)

    obj_name_to_type = {
        # "handle": handle_type,
        # "left_door_handle": handle_type,
        # "right_door_handle": handle_type,
        "door": door_type,
        "leftdoor": door_type,
        "rightdoor": door_type,
        "gripper": gripper_type,
        "left_finger": left_finger_type,
        "right_finger": right_finger_type,
        "cabinet": cabinet_type,
        "robot0_base": base_type,
        "obj": thing_type,
        "bottom": surface_type,
        "knob": knob_type,
        "stovetop": stove_type,
        "microwave": microwave_type,
        "microwave_start_button": microwave_button_type,
        "drawer": cabinet_type,  # The drawer fixture (stationary cabinet structure)
        "drawer_inner_box": drawer_type,  # The movable sliding part
        "sink_faucet_handle": sink_faucet_handle_type,  # The sink faucet object
        "sink": sink_type,  # The sink object
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
        "StoreFruitFull",
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
            # return [self.object_name_to_object("handle")]
            return [self.object_name_to_object("door")]
        elif task_name == "OpenDoubleDoor":
            # return [self.object_name_to_object("left_door_handle"), self.object_name_to_object("right_door_handle")]
            return [self.object_name_to_object("leftdoor"), self.object_name_to_object("rightdoor")]
        elif task_name == "CloseSingleDoor":
            return [self.object_name_to_object("door")]
        elif task_name == "CloseDoubleDoor":
            return [self.object_name_to_object("leftdoor"), self.object_name_to_object("rightdoor")]
        elif task_name == "PnPCounterToCab":
            return [self.object_name_to_object("obj")]
        elif task_name == "StoreFruit":
            # return [self.object_name_to_object("handle"), self.object_name_to_object("obj")]
            return [self.object_name_to_object("door"), self.object_name_to_object("obj")]
        elif task_name == "StoreFruitFull":
            return [self.object_name_to_object("door"), self.object_name_to_object("obj")]
        elif task_name == "TurnOnStove":
            return [self.object_name_to_object("knob"), self.object_name_to_object("stovetop")]
        elif task_name == "TurnOffStove":
            return [self.object_name_to_object("knob"), self.object_name_to_object("stovetop")]
        elif task_name == "TurnOnMicrowave":
            return [self.object_name_to_object("microwave_start_button")]
        elif task_name == "CloseDrawer":
            return [self.object_name_to_object("drawer_inner_box"), self.object_name_to_object("drawer")]
        elif task_name == "PnPCounterToStove":
            return [self.object_name_to_object("obj"), self.object_name_to_object("bottom")]
        elif task_name == "OpenDrawer":
            return [self.object_name_to_object("drawer_inner_box"), self.object_name_to_object("drawer")]
        elif task_name == "TurnOnSinkFaucet":
            return [self.object_name_to_object("sink_faucet_handle"), self.object_name_to_object("sink")]
        elif task_name == "TurnOffSinkFaucet":
            return [self.object_name_to_object("sink_faucet_handle"), self.object_name_to_object("sink")]
        else:
            raise ValueError(f"Task {task_name} not supported")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create tasks for training."""
        if CFG.load_approach:
            return []
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
            # handle = self.object_name_to_object("handle")
            # cabinet = self.object_name_to_object("cabinet")
            # if self._DoorOpen_holds(state, [handle, cabinet]):
            #     return True
            door = self.object_name_to_object("door")
            cabinet = self.object_name_to_object("cabinet")
            if self._DoorOpen_holds(state, [door, cabinet]):
                return True
        elif goal_desc == "OpenDoubleDoor":
            # left_handle = self.object_name_to_object("left_door_handle")
            # right_handle = self.object_name_to_object("right_door_handle")
            # cabinet = self.object_name_to_object("cabinet")
            # if self._DoorOpen_holds(state, [left_handle, cabinet]) and self._DoorOpen_holds(state, [right_handle, cabinet]):
            #     return True
            left_door = self.object_name_to_object("leftdoor")
            right_door = self.object_name_to_object("rightdoor")
            cabinet = self.object_name_to_object("cabinet")
            if self._DoorOpen_holds(state, [left_door, cabinet]) and self._DoorOpen_holds(state, [right_door, cabinet]):
                return True
        elif goal_desc == "CloseSingleDoor":
            door = self.object_name_to_object("door")
            cabinet = self.object_name_to_object("cabinet")
            if self._DoorClosed_holds(state, [door, cabinet]):
                return True
        elif goal_desc == "CloseDoubleDoor":
            left_door = self.object_name_to_object("leftdoor")
            right_door = self.object_name_to_object("rightdoor")
            cabinet = self.object_name_to_object("cabinet")
            if self._DoorClosed_holds(state, [left_door, cabinet]) and self._DoorClosed_holds(state, [right_door, cabinet]):
                return True
        elif goal_desc == "PnPCounterToCab":
            obj = self.object_name_to_object("obj")
            bottom = self.object_name_to_object("bottom")
            if self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "TurnOnStove":
            stove = self.object_name_to_object("stovetop")
            if stove is not None and self._StoveOn_holds(state, [stove]):
                return True
        elif goal_desc == "StoreFruit":
            door = self.object_name_to_object("door")
            bottom = self.object_name_to_object("bottom")
            cabinet = self.object_name_to_object("cabinet")
            obj = self.object_name_to_object("obj")
            if self._DoorOpen_holds(state, [door, cabinet]) and self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "StoreFruitFull":
            door = self.object_name_to_object("door")
            bottom = self.object_name_to_object("bottom")
            cabinet = self.object_name_to_object("cabinet")
            obj = self.object_name_to_object("obj")
            if self._DoorClosed_holds(state, [door, cabinet]) and self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "TurnOnMicrowave":
            microwave = self.object_name_to_object("microwave")
            if microwave is not None and self._MicrowaveOn_holds(state, [microwave]):
                return True
        elif goal_desc == "CloseDrawer":
            drawer_inner_box = self.object_name_to_object("drawer_inner_box")
            drawer_cabinet = self.object_name_to_object("drawer")
            if self._DrawerClosed_holds(state, [drawer_inner_box, drawer_cabinet]):
                return True
        elif goal_desc == "OpenDrawer":
            drawer_inner_box = self.object_name_to_object("drawer_inner_box")
            drawer_cabinet = self.object_name_to_object("drawer")
            if self._DrawerOpen_holds(state, [drawer_inner_box, drawer_cabinet]):
                return True
        elif goal_desc == "TurnOffStove":
            stove = self.object_name_to_object("stovetop")
            if stove is not None and self._StoveOff_holds(state, [stove]):
                return True
        elif goal_desc == "TurnOnSinkFaucet":
            sink_faucet_handle = self.object_name_to_object("sink_faucet_handle")
            if sink_faucet_handle is not None and self._SinkFaucetOn_holds(state, [sink_faucet_handle]):
                return True
        elif goal_desc == "TurnOffSinkFaucet":
            sink_faucet_handle = self.object_name_to_object("sink_faucet_handle")
            if sink_faucet_handle is not None and self._SinkFaucetOff_holds(state, [sink_faucet_handle]):
                return True
        else:
            raise ValueError(f"Goal description {goal_desc} not supported")

    def _reset_initial_state(self, seed: int, train_or_test: str, task_name: str, complex_config: bool = False) -> Observation:
        """Reset the environment to an initial state based on the seed."""
        # Create or recreate environment if needed
        warnings.warn("Resetting environment to initial state from seed not implemented for robosuite kitchen")
        if self._env is None:
            complex_config = True  # NOTE: this should be removed. only for mac
            if complex_config:
                robot_type = "PandaOmron"
                controller_config = load_composite_controller_config(robot=robot_type)
                if CFG.robo_kitchen_task == "OpenDrawer":
                    layout_ids = [0]
                else:
                    layout_ids = [3]
                
                # top handle sink requries style traditional 1 (5), traditional 2 (6), transitional 2 (11), mediterranean (9)
                # for now just keep 6 for all tasks

                config = {
                    "env_name": task_name,
                    "robots": robot_type,
                    "controller_configs": controller_config,
                    "layout_ids": layout_ids,
                    "style_ids": [6], # this combination of layout and style makes sure the stove is stovetop, so similar to demos for turn on stove
                    "translucent_robot": True,
                }

                print(colored(f"Initializing environment for task: {task_name}", "yellow"))

                self._env_raw = robosuite.make(
                    **config,
                    has_renderer=self._using_gui,
                    has_offscreen_renderer=not self._using_gui,
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

        self.num = 0
        self.default_contact_num = len(self._env_raw.sim.data.contact)
        self.default_contact_pairs = [(self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2)) for contact in self._env_raw.sim.data.contact]

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

        # try:
        #     if self.num > 5:
        #         self.num += 1
        #         if len(self._env_raw.sim.data.contact) > self.default_contact_num:
        #             print(self.num, len(self._env_raw.sim.data.contact), end=": ")
        #             for contact in self._env_raw.sim.data.contact:
        #                 pair = (self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2))
        #                 if pair not in self.default_contact_pairs:
        #                     print(pair[0], pair[1], contact.dist)
        #     else:
        #         self.num += 1
        #         self.default_contact_num = len(self._env_raw.sim.data.contact)
        #         self.default_contact_pairs = [(self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2)) for contact in self._env_raw.sim.data.contact]
        # except AttributeError:
        #     self.num = 0
        #     self.default_contact_num = len(self._env_raw.sim.data.contact)
        #     self.default_contact_pairs = [(self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2)) for contact in self._env_raw.sim.data.contact]

        contact_name_to_object = {"g18": "door", "g16": "door", "g27": "door"}
        object_names = [obj.name for obj in self.objects_of_interest]

        # only support panda robot for now
        contacts = set()

        robot_body_contact = self._env.get_contacts(self._env.robots[0].robot_model)  # robot
        robot_body_obj = self.object_name_to_object("gripper") # use gripper as the robot object
        global_contact_pairs = [(self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2)) for contact in self._env_raw.sim.data.contact]
        link7_contact_pairs = [pair for pair in global_contact_pairs if "link7" in pair[0] or "link7" in pair[1]]
        for contact in robot_body_contact:
            for obj_name in object_names:
                for contact_name in contact_name_to_object:
                    if contact_name in contact:
                        contact = contact_name_to_object[contact_name]
                
                # Check if object name is in contact string, or if contact ends with object name suffix
                if obj_name in contact and link7_contact_pairs:
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((robot_body_obj, obj))
                elif "_" in obj_name and contact.endswith(obj_name.split("_", 1)[1]) and link7_contact_pairs:
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((robot_body_obj, obj))

        # Get all contacts and filter for gripper-related ones
        gripper_contact = []
        for i in range(self._env_raw.sim.data.ncon):
            contact = self._env_raw.sim.data.contact[i]
            g1 = self._env_raw.sim.model.geom_id2name(contact.geom1)
            g2 = self._env_raw.sim.model.geom_id2name(contact.geom2)
            # Check if either geom belongs to gripper/finger
            if any(name in g1 for name in ["gripper", "finger", "finger1", "finger2", "fingertip", "fingerpad"]):
                # Add the non-gripper geom (g2)
                gripper_contact.append(g2)
            elif any(name in g2 for name in ["gripper", "finger", "finger1", "finger2", "fingertip", "fingerpad"]):
                # Add the non-gripper geom (g1)
                gripper_contact.append(g1)

        # Process gripper contacts to create contact pairs
        gripper_obj = self.object_name_to_object("gripper")
        # Remove duplicates by converting to set
        unique_gripper_contacts = set(gripper_contact)
        
        for contact in unique_gripper_contacts:
            for obj_name in object_names:
                for contact_name in contact_name_to_object:
                    if contact_name in contact:
                        contact = contact_name_to_object[contact_name]
                
                # Check if object name is in contact string, or if contact ends with object name suffix  
                if obj_name in contact:
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((gripper_obj, obj))
                elif "_" in obj_name and contact.endswith(obj_name.split("_", 1)[1]):
                    obj = self.object_name_to_object(obj_name)
                    contacts.add((gripper_obj, obj))
                
                # Special handling for drawer contacts - drawers appear as "door" in contact names
                # For CloseDrawer task, map door contacts to drawer_inner_box (the movable part)
                if obj_name == "drawer_inner_box" and "door" in contact:
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
            Predicate("Dummy", [], cls._Dummy_holds),
            Predicate("ReadyGrabObj", [cls.gripper_type, cls.object_type], cls._ReadyGrabObj_holds),
            Predicate("GripperOpen", [cls.left_finger_type, cls.right_finger_type], cls._GripperOpen_holds),
            Predicate("GripperClosed", [cls.left_finger_type, cls.right_finger_type], cls._GripperClosed_holds),
            # Predicate("DoorOpen", [cls.handle_type, cls.cabinet_type], cls._DoorOpen_holds),
            Predicate("DoorOpen", [cls.door_type, cls.cabinet_type], cls._DoorOpen_holds),
            Predicate("DoorClosed", [cls.door_type, cls.cabinet_type], cls._DoorClosed_holds),
            Predicate("DrawerClosed", [cls.drawer_type, cls.cabinet_type], cls._DrawerClosed_holds),
            Predicate("DrawerOpen", [cls.drawer_type, cls.cabinet_type], cls._DrawerOpen_holds),
            Predicate("InContact", [cls.object_type, cls.object_type], cls._InContact_holds),
            Predicate("OnSurface", [cls.thing_type, cls.surface_type], cls._OnSurface_holds),
            Predicate("DoorHalfOpen", [cls.handle_type, cls.cabinet_type], cls._DoorHalfOpen_holds),
            Predicate("KnobTurnedOn", [cls.knob_type, cls.stove_type], cls._KnobTurnedOn_holds),
            Predicate("InOrigin", [cls.gripper_type, cls.base_type], cls._InOrigin_holds),
            Predicate("MicrowaveOn", [cls.microwave_type], cls._MicrowaveOn_holds),
            Predicate("StoveOn", [cls.stove_type], cls._StoveOn_holds),
            Predicate("StoveOff", [cls.stove_type], cls._StoveOff_holds),
            Predicate("SinkFaucetOn", [cls.sink_faucet_handle_type], cls._SinkFaucetOn_holds),
            Predicate("SinkFaucetOff", [cls.sink_faucet_handle_type], cls._SinkFaucetOff_holds),
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
        self.viz_type_frames(self.stove_type)
        self.viz_type_frames(self.knob_type)
        self.viz_type_frames(self.door_type)
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
        if CFG.use_teleop is None: # none for WBC
            arm_ratio = 0.8
            arm_pos = np.array([arm_ratio * pos_delta[0], arm_ratio * pos_delta[1], pos_delta[2]])
            base_pos = (1.0 - arm_ratio) * pos_delta[0:2]
            env_action[0:3] = arm_pos  # position control
            env_action[3:6] = rot_delta  # rotation control
            env_action[6] = gripper_cmd  # gripper control
            env_action[7:9] = base_pos
            # gripper_obj = self._current_state.get_objects(self.gripper_type)[0]
            # base_obj = self._current_state.get_objects(self.base_type)[0]
            # gripper_pos, gripper_quat = get_gripper_in_base_frame(self._current_state, gripper_obj, base_obj)
            # # # convert gripper_quat to euler angles
            # # # euler_angles = R.from_quat(gripper_quat).as_euler("xyz", degrees=False)
            # # # print(f"euler_angles: {euler_angles}")
            # # find delta pos and quat to control the base
            # delta_pos = gripper_pos[0:2] - init_delta_gripper_base[0:2]
            # distance = np.linalg.norm(gripper_pos)
            # # get angle between robot and gripper with atan2
            # angle = np.arctan2(gripper_pos[1], gripper_pos[0])
            # # print(f"angle: {angle}, {distance}")
            # # print(f"gripper_pos: {gripper_pos}")
            # # print(f"delta_pos: {delta_pos}")
            # # set deadzone to 0.01
            
            # if delta_pos[0] > -0.01 and delta_pos[0] < 0.25:
            #     delta_pos[0] = 0.0
            # if np.linalg.norm(delta_pos[1]) < 0.2:
            #     delta_pos[1] = 0.0
            # env_action[8] = env_action[8] + delta_pos[1] * 0.3 # keep base and arm close in y
        else: # either teleop or no teleop
            env_action[0:3] = pos_delta  # position control
            env_action[3:6] = rot_delta  # rotation control
            env_action[6] = gripper_cmd  # gripper control
        
        if CFG.use_teleop: # keyboard teleop populate other fields 
            env_action[7:10] = input_ac_dict["base"]

        # # if np.linalg.norm(angle) < 0.1:
        # #     angle = 0.0
        # env_action[9] = angle *0.3
        # delta_quat = gripper_quat - init_delta_gripper_base_rot
        # if np.linalg.norm(delta_quat) < 0.05:
        #     delta_quat = np.zeros(4)
        # env_action[7:10] = delta_pos
        # env_action[10] = gripper_quat

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
        if self._env_raw is not None and hasattr(self._env_raw, "viewer") and self._env_raw.viewer is not None:
            self._env_raw.viewer.mjprint(text, auto_clean=auto_clean)

    def mjshowframe(self, xyz, quat=(1, 0, 0, 0), size=0.1, name=None, keep=False):
        """Show frame in the viewer."""
        if self._env_raw is not None and hasattr(self._env_raw, "viewer") and self._env_raw.viewer is not None:
            self._env_raw.viewer.mjshowframe(xyz, quat=quat, size=size, name=name, keep=keep)

    def mjshowellipse(self, xyz, quat=(1,0,0,0), size=(0.1, 0.1, 0.1), color=(1, 0, 0), alpha=0.5, name=None, base_pos=None, base_quat=None):
        """Show ellipse in the viewer."""
        if self._env_raw is not None and hasattr(self._env_raw, "viewer") and self._env_raw.viewer is not None:
            if base_pos is not None and base_quat is not None:
                # base_quat and quat are xyzw

                # Convert base and relative quaternions to Rotation objects
                base_rot = R.from_quat(base_quat)
                rel_rot = R.from_quat(quat)

                # Transform position: world_pos = base_pos + base_rot * rel_pos
                xyz_world = base_pos + base_rot.apply(xyz)

                # Transform orientation: world_rot = base_rot * rel_rot
                world_rot = base_rot * rel_rot
                quat_world = world_rot.as_quat() # Convert back to xyzw

                # Update xyz and quat to be in world frame
                xyz = xyz_world
                quat = quat_world
            self._env_raw.viewer.mjshowellipse(xyz, quat=quat, size=size, color=color, alpha=alpha, name=name)

    def show_option_cluster_predicates(self, curr_option):
        """Show predicates in the viewer."""

        def show_cluster_predicates(predicates, color=(1, 0, 0), alpha=0.1, prefix=""):
            for pred in predicates:
                cluster_predicates = []
                predicate = pred.predicate
                if "RelCovCluster" in predicate.name:
                    cluster_predicates = [predicate]
                    import re
                    pattern = r'\w+-in-\w+-frame'
                    match = re.search(pattern, pred._str)
                    if match:
                        name = f"{prefix}_{match.group(0)}"
                    else:
                        name = f"{prefix}_{pred._str.split('-')[0]}"
                else:
                    pred_key = (predicate.name, pred.entities[0].type.name, pred.entities[1].type.name)
                    name = f"{prefix}_{pred._str}"
                    if pred_key in CFG.dict_contact_predicate_to_rel_pose_predicates:
                        cluster_predicates = CFG.dict_contact_predicate_to_rel_pose_predicates[pred_key]

                for i, cluster_predicate in enumerate(cluster_predicates):
                    ref_type = cluster_predicate.types[0]
                    ref_obj = None
                    ref_frame = None
                    for obj in curr_option.objects:
                        if obj.type == ref_type:
                            ref_obj = obj
                            break
                    if ref_obj is None:
                        continue
                    for s in self._current_state:
                        if s.name == ref_obj.name:
                            ref_frame = np.concatenate(self._current_state[s])
                            break
                    if ref_frame is None:
                        continue
                    cluster_cov = np.linalg.inv(cluster_predicate._classifier.inv_covariance_matrix_trans)
                    mahalanobis_threshold = cluster_predicate._classifier.mahalanobis_threshold_trans
                    pos, quat = cluster_predicate._classifier.trans_center[:3], cluster_predicate._classifier.rot_center.as_quat()
                    eigvals, eigvecs = np.linalg.eigh(cluster_cov)
                    eigvals = np.abs(eigvals)
                    a, b, c = np.sqrt(mahalanobis_threshold * eigvals)

                    if i == 0:
                        self.mjshowellipse(pos, quat, size=(a, b, c), name=name, base_pos=ref_frame[:3], base_quat=ref_frame[3:7], alpha=alpha, color=color)
                    else:
                        self.mjshowellipse(pos, quat, size=(a, b, c), base_pos=ref_frame[:3], base_quat=ref_frame[3:7], alpha=alpha, color=color)

        if hasattr(curr_option, "parent") and hasattr(curr_option.parent, "operator"):
            preconditions = curr_option.parent.operator.preconditions
            add_effects = curr_option.parent.operator.add_effects
            delete_effects = curr_option.parent.operator.delete_effects
            ignore_effects = curr_option.parent.operator.ignore_effects

            show_cluster_predicates(preconditions, color=(0, 1, 0), prefix="pre")
            show_cluster_predicates(add_effects, color=(0, 0, 1), prefix="add")
            show_cluster_predicates(delete_effects, color=(1, 0, 0), prefix="del")
            show_cluster_predicates(ignore_effects, color=(1, 0.5, 0), prefix="ignore")

    @property
    def action_space(self) -> Box:
        """7D action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]"""
        return Box(-1.0, 1.0, (7,), np.float32)

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        # return set()
        # return {
        #     self._pred_name_to_pred["DoorOpen"],
        #     self._pred_name_to_pred["OnSurface"],
        #     self._pred_name_to_pred["DoorClosed"],
        #     self._pred_name_to_pred["KnobTurnedOn"],
        # }
        goal_desc = self.task_selected
        goal_preds = set()
        if goal_desc == "OpenSingleDoor":
            goal_preds = {self._pred_name_to_pred["DoorOpen"]}
        elif goal_desc == "PnPCounterToCab":
            goal_preds = {self._pred_name_to_pred["OnSurface"]}
        elif goal_desc == "CloseSingleDoor":
            goal_preds = {self._pred_name_to_pred["DoorClosed"]}
        elif goal_desc == "StoreFruit":
            goal_preds = {
                self._pred_name_to_pred["OnSurface"],
                # self._pred_name_to_pred["DoorClosed"]
            }
        elif goal_desc == "TurnOnMicrowave":
            goal_preds = {self._pred_name_to_pred["MicrowaveOn"]}
        elif goal_desc == "TurnOnStove":
            goal_preds = {self._pred_name_to_pred["StoveOn"]}
        elif goal_desc == "TurnOffStove":
            goal_preds = {self._pred_name_to_pred["StoveOff"]}
        elif goal_desc == "CloseDrawer":
            goal_preds = {self._pred_name_to_pred["DrawerClosed"]}
        elif goal_desc == "OpenDrawer":
            goal_preds = {self._pred_name_to_pred["DrawerOpen"]}
        elif goal_desc == "TurnOnSinkFaucet":
            goal_preds = {self._pred_name_to_pred["SinkFaucetOn"]}
        elif goal_desc == "TurnOffSinkFaucet":
            goal_preds = {self._pred_name_to_pred["SinkFaucetOff"]}
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
            self.door_type,
            self.knob_type,
            self.stove_type,
            self.microwave_type,
            self.microwave_button_type,
            self.sink_faucet_handle_type,
            self.sink_type,
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
            cls.door_close_thresh = cls.online_door_close_thresh  # rad

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

        # Add the 'on' feature to the microwave object
        if "microwave_on" in state_info:
            mic_obj = cls.object_name_to_object("microwave")
            if mic_obj is not None and mic_obj in state_dict:
                state_dict[mic_obj]["on"] = np.array([state_info["microwave_on"]])

        # Add the 'on' feature to the stove object
        if "stove_on" in state_info:
            stove_obj = cls.object_name_to_object("stovetop")
            if stove_obj is not None and stove_obj in state_dict:
                state_dict[stove_obj]["on"] = np.array([state_info["stove_on"]])

        # Add the 'on' feature to the sink faucet object
        if "sink_faucet_on" in state_info:
            sink_faucet_obj = cls.object_name_to_object("sink_faucet_handle")
            if sink_faucet_obj is not None and sink_faucet_obj in state_dict:
                state_dict[sink_faucet_obj]["on"] = np.array([state_info["sink_faucet_on"]])

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
    def _Dummy_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Dummy predicate for testing."""
        return True

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

        return rotation_value <= cls.door_close_thresh

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
        on_surface_top = 0.0 <= obj_pos_in_surface[2] <= cls.place_close_y_thresh
        in_surface_region = abs(obj_pos_in_surface[0]) <= 0.13 and abs(obj_pos_in_surface[1]) <= 0.13
        # print(obj_pos_in_surface[0], obj_pos_in_surface[1])
        # print(near_surface, in_surface)
        return on_surface_top and in_surface_region

    @classmethod
    def _KnobTurnedOn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if knob is on stove."""
        knob, stove = objects

        # Get quaternions from the objects passed in
        knob_quat = state.get(knob, "quaternion")
        stove_quat = state.get(stove, "quaternion")

        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation
        from scipy.spatial.transform import Rotation as R

        knob_rot = Rotation.from_quat(knob_quat)
        stove_rot = Rotation.from_quat(stove_quat)

        # Calculate relative rotation
        rel_rot = stove_rot.inv() * knob_rot

        # Extract rotation value (approximation for hinge rotation)
        rot_vec = rel_rot.as_rotvec()
        rotation_value = np.linalg.norm(rot_vec)

        is_on = cls.knob_on_thresh <= np.abs(rotation_value) <= 2 * np.pi - cls.knob_on_thresh

        return is_on

    @classmethod
    def _MicrowaveOn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if the microwave is on."""
        microwave, = objects
        return state.get(microwave, "on")[0] > 0.5

    @classmethod
    def _StoveOn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if the stove is on."""
        stove, = objects
        return state.get(stove, "on")[0] > 0.5
    
    @classmethod
    def _StoveOff_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if the stove is off."""
        stove, = objects
        return state.get(stove, "on")[0] < 0.5
    
    @classmethod
    def _SinkFaucetOn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if the sink faucet is on."""
        sink_faucet_handle, = objects
        return state.get(sink_faucet_handle, "on")[0] > 0.5
    
    @classmethod
    def _SinkFaucetOff_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if the sink faucet is off."""
        sink_faucet_handle, = objects
        return state.get(sink_faucet_handle, "on")[0] < 0.5

    @classmethod
    def _DrawerClosed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if drawer is closed by checking the slide joint position.
        
        For drawers, closed means the slide joint position is close to 0.
        Unlike doors which rotate, drawers slide linearly.
        """
        drawer, cabinet = objects
        
        # For now, we'll use the same position-based approach as doors
        # but interpret it differently for drawers
        drawer_pos = state.get(drawer, "translation")
        cabinet_pos = state.get(cabinet, "translation")
        
        # Calculate relative position - for a closed drawer, it should be
        # very close to the cabinet's position in the Y dimension (slide axis)
        rel_pos = drawer_pos - cabinet_pos
        
        # For a closed drawer, the Y displacement should be minimal
        # (drawers slide along Y-axis according to the XML)
        drawer_close_thresh = 0.05  # meters - threshold for considering drawer closed
        
        return abs(rel_pos[1]) < drawer_close_thresh
    
    @classmethod
    def _DrawerOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if drawer is open by checking the slide joint position.
        
        For drawers, open means the slide joint position is close to 1.
        Unlike doors which rotate, drawers slide linearly.
        """
        drawer, cabinet = objects
        
        # For now, we'll use the same position-based approach as doors
        # but interpret it differently for drawers
        drawer_pos = state.get(drawer, "translation")
        cabinet_pos = state.get(cabinet, "translation")

        # Calculate relative position - for an open drawer, it should be
        # very close to the cabinet's position in the Y dimension (slide axis)
        rel_pos = drawer_pos - cabinet_pos
        
        # For an open drawer, the Y displacement should be significant
        # (drawers slide along Y-axis according to the XML)
        drawer_open_thresh = 0.2  # meters - threshold for considering drawer open
        
        return abs(rel_pos[1]) > drawer_open_thresh

    def close(self) -> None:
        """Close the Robosuite environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_raw = None # Assuming _env_raw is closed by _env.close() or managed by it
            if self.device is not None and hasattr(self.device, 'stop_control'): # Check for Keyboard device
                self.device.stop_control()
                self.device = None
        logging.info("RoboKitchenEnv closed.")


def frame_transform(pos_in_init: np.ndarray, quat_in_init: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot_in_init = R.from_quat(quat_in_init).as_matrix()

    rel_pos_init = pos_in_init - target_pos

    pos_in_target = target_rot.T @ rel_pos_init
    rot_in_target = target_rot.T @ rot_in_init

    return pos_in_target, rot_in_target
def get_gripper_in_base_frame(state: State, gripper: Object, base: Object) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get gripper's position and quaternion in base frame.
    
    Args:
        state: Current state containing object poses
        gripper: Gripper object
        base: Base object
    
    Returns:
        Tuple of (position, quaternion) in base frame
    """
    # Get base and gripper poses in world frame
    base_pos = state.get(base, "translation")  # [x, y, z]
    base_quat = state.get(base, "quaternion")  # [x, y, z, w]
    gripper_pos = state.get(gripper, "translation")  # [x, y, z]
    gripper_quat = state.get(gripper, "quaternion")  # [x, y, z, w]
    
    # Convert quaternions to rotation matrices
    base_rot = R.from_quat(base_quat).as_matrix()
    gripper_rot = R.from_quat(gripper_quat).as_matrix()
    
    # Calculate position in base frame
    # First subtract base position to get relative position in world frame
    rel_pos_world = gripper_pos - base_pos
    # Then rotate to base frame
    rel_pos_base = base_rot.T @ rel_pos_world
    
    # Calculate quaternion in base frame
    # First get relative rotation in world frame
    rel_rot_world = gripper_rot @ base_rot.T
    # Convert back to quaternion
    rel_quat_base = R.from_matrix(rel_rot_world).as_quat()
    
    return rel_pos_base, rel_quat_base
