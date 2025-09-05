
"""A Kitchen environment wrapping robosuite kitchen."""

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast
import time
# import rospy

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

# from predicators.envs.ros_hardware_interface import ROSHardwareInterface
# import tf
# from geometry_msgs.msg import PoseStamped

# Disable JAX debug messages
logging.getLogger("jax._src.cache_key").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)

# Constants from demo files
MAX_CARTESIAN_DISPLACEMENT = 0.8
MAX_ROTATION_DISPLACEMENT = 1.0

# gripper - base offset in base frame

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
    place_close_z_thresh = 0.10  # m
    place_close_xy_thresh = 0.15  # m
    gripper_obj_far_thresh = 0.25  # m

    online_door_open_thresh = np.deg2rad(70)  # rad
    online_door_close_thresh = np.deg2rad(10)  # rad

    # Types
    object_type = Type("object_type", ["translation", "quaternion"])
    grab_type = Type("grab_type", ["translation", "quaternion"], parent=object_type)
    knob_type = Type("knob_type", ["translation", "quaternion"], parent=grab_type)
    door_type = Type("door_type", ["translation", "quaternion"], parent=object_type)
    base_type = Type("base_type", ["translation", "quaternion"], parent=object_type)
    gripper_type = Type("gripper_type", ["translation", "quaternion"], parent=object_type)
    wrist_type = Type("wrist_type", ["translation", "quaternion"], parent=object_type)
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
    container_type = Type("container_type", ["translation", "quaternion"], parent=object_type)
    counter_type = Type("counter_type", ["translation", "quaternion"], parent=object_type)
    cookware_type = Type("cookware_type", ["translation", "quaternion"], parent=object_type)
    lid_type = Type("lid_type", ["translation", "quaternion"], parent=object_type)

    obj_name_to_type = {
        # "handle": handle_type,
        # "left_door_handle": handle_type,
        # "right_door_handle": handle_type,
        # "door": door_type,
        # "leftdoor": door_type,
        # "rightdoor": door_type,
        "gripper": gripper_type,
        "wrist": wrist_type,
        "left_finger": left_finger_type,
        "right_finger": right_finger_type,
        # "cabinet": cabinet_type,
        "robot0_base": base_type,
        # "obj": thing_type,
        # "bottom": surface_type,
        # "counter": counter_type,
        # "knob": knob_type,
        # "stovetop": stove_type,
        # "microwave": microwave_type,
        # "microwave_start_button": microwave_button_type,
        # "drawer": cabinet_type,  # The drawer fixture (stationary cabinet structure)
        # "drawer_inner_box": drawer_type,  # The movable sliding part
        # "sink_faucet_handle": sink_faucet_handle_type,  # The sink faucet object
        # "sink": sink_type,  # The sink object
        # CookCheeseAndTomatoes
        "plate": container_type,
        # "tomato": thing_type,
        # "cheese": thing_type,
        # "pan": container_type,
        # PnPStoveToCounter
        # "container": container_type,
        # "obj_container": container_type,
        # "door_obj": thing_type, # opensingledoor data have door obj in the cabinet
        # "dummy_object": object_type,
        # "vegetable1": thing_type,
        # "vegetable2": thing_type,
        # "cutting_board": container_type,
        # "obj1": cookware_type,
        # "obj2": thing_type,
        # "mug": thing_type,
        # "cab_door": door_type,
        "lid": lid_type,
        "dishrack": cabinet_type,
        # "bowl": container_type,
        "pan": cookware_type,
        "banana": thing_type,
        "block": thing_type,
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
        "CookCheeseAndTomatoes",
        "PnPCabToCounterTomato",
    ]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # print(f"ALL_KITCHEN_ENVIRONMENTS: {ALL_KITCHEN_ENVIRONMENTS}")

        if self._using_gui:
            pass
            # assert not CFG.make_test_videos or CFG.make_failure_videos, \
            #     "Turn off --use_gui to make videos in robo kitchen env"

        self._pred_name_to_pred = self.create_predicates()
        self._env = None  # Will be created in reset
        self._env_raw = None
        self.task_selected = CFG.robo_kitchen_task
        if self.task_selected not in self.tasks_extended and self.task_selected not in CFG.mocap_tasks:
            raise ValueError(f"Task {self.task_selected} not supported")
        print(colored(f"Selected task: {self.task_selected}", "green"))

        self.device = None  # control device
        print(colored("Initializing ROS Hardware Interface...", "yellow"))
        # self._hw_interface = ROSHardwareInterface(init_gripper_open=True)
        print(colored("ROSHardwareInterface initialized.", "green"))
        # self._robot_base_frame = self._hw_interface.robot_base_frame
        self._video_frames = []  # For saving video frames when GUI is not enabled
        self._frame_counter = 0  # To track steps for frame saving

    def get_objects_of_interest(self, task_name: str) -> List[Object]:
        """Get the object of interest for the task. These are objects involved in contact."""
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
        elif task_name == "PnPCabToCounter":
            return [self.object_name_to_object("obj")]
        elif task_name == "PnPStoveToCounter":
            return [self.object_name_to_object("obj")]
        elif task_name == "PnPCounterToStove":
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
        elif task_name == "OpenDrawer":
            return [self.object_name_to_object("drawer_inner_box"), self.object_name_to_object("drawer")]
        elif task_name == "TurnOnSinkFaucet":
            return [self.object_name_to_object("sink_faucet_handle"), self.object_name_to_object("sink")]
        elif task_name == "TurnOffSinkFaucet":
            return [self.object_name_to_object("sink_faucet_handle"), self.object_name_to_object("sink")]
        elif task_name == "CookCheeseAndTomatoes":
            return [self.object_name_to_object("tomato"), self.object_name_to_object("cheese"), self.object_name_to_object("plate")]
        elif task_name == "PnPCabToCounterTomato":
            return [self.object_name_to_object("tomato"), self.object_name_to_object("plate")]
        elif task_name == "ArrangeVegetables":
            return [self.object_name_to_object("tomato"), self.object_name_to_object("plate")]
        elif task_name == "PreSoakPan":
            return [self.object_name_to_object("tomato"), self.object_name_to_object("plate")]
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
            if task_name not in ALL_KITCHEN_ENVIRONMENTS and task_name not in CFG.mocap_tasks:
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
            doors = self.object_name_to_objects("door", test_time=True)
            assert len(doors) == 1, "Expected exactly one door object"
            door = doors[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            if self._DoorOpen_holds(state, [door, cabinet]):
                return True
        elif goal_desc == "OpenDoubleDoor":
            # left_handle = self.object_name_to_object("left_door_handle")
            # right_handle = self.object_name_to_object("right_door_handle")
            # cabinet = self.object_name_to_object("cabinet")
            # if self._DoorOpen_holds(state, [left_handle, cabinet]) and self._DoorOpen_holds(state, [right_handle, cabinet]):
            #     return True
            left_doors = self.object_name_to_objects("leftdoor", test_time=True)
            assert len(left_doors) == 1, "Expected exactly one left door object"
            left_door = left_doors[0]
            right_doors = self.object_name_to_objects("rightdoor", test_time=True)
            assert len(right_doors) == 1, "Expected exactly one right door object"
            right_door = right_doors[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            if self._DoorOpen_holds(state, [left_door, cabinet]) and self._DoorOpen_holds(state, [right_door, cabinet]):
                return True
        elif goal_desc == "CloseSingleDoor":
            doors = self.object_name_to_objects("door", test_time=True)
            assert len(doors) == 1, "Expected exactly one door object"
            door = doors[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            if self._DoorClosed_holds(state, [door, cabinet]):
                return True
        elif goal_desc == "CloseDoubleDoor":
            left_doors = self.object_name_to_objects("leftdoor", test_time=True)
            assert len(left_doors) == 1, "Expected exactly one left door object"
            left_door = left_doors[0]
            right_doors = self.object_name_to_objects("rightdoor", test_time=True)
            assert len(right_doors) == 1, "Expected exactly one right door object"
            right_door = right_doors[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            if self._DoorClosed_holds(state, [left_door, cabinet]) and self._DoorClosed_holds(state, [right_door, cabinet]):
                return True
        elif goal_desc == "PnPCounterToCab":
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            bottoms = self.object_name_to_objects("bottom", test_time=True)
            assert len(bottoms) == 1, "Expected exactly one bottom object"
            bottom = bottoms[0]
            if self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "PnPCabToCounter":
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            counters = self.object_name_to_objects("counter", test_time=True)
            assert len(counters) == 1, "Expected exactly one counter object"
            counter = counters[0]
            if self._OnCounter_holds(state, [obj, counter]):
                return True
        elif goal_desc == "PnPStoveToCounter":
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            containers = self.object_name_to_objects("container", test_time=True)
            assert len(containers) == 1, "Expected exactly one container object"
            container = containers[0]
            if self._InContainer_holds(state, [obj, container]):
                return True
        elif goal_desc == "PnPCounterToStove":
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            containers = self.object_name_to_objects("container", test_time=True)
            assert len(containers) == 1, "Expected exactly one container object"
            container = containers[0]
            if self._InContainer_holds(state, [obj, container]):
                return True
        elif goal_desc == "TurnOnStove":
            stoves = self.object_name_to_objects("stovetop", test_time=True)
            assert len(stoves) == 1, "Expected exactly one stove object"
            stove = stoves[0]
            if stove is not None and self._StoveOn_holds(state, [stove]):
                return True
        elif goal_desc == "StoreFruit":
            doors = self.object_name_to_objects("door", test_time=True)
            assert len(doors) == 1, "Expected exactly one door object"
            door = doors[0]
            bottoms = self.object_name_to_objects("bottom", test_time=True)
            assert len(bottoms) == 1, "Expected exactly one bottom object"
            bottom = bottoms[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            if self._DoorOpen_holds(state, [door, cabinet]) and self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "StoreFruitFull":
            doors = self.object_name_to_objects("door", test_time=True)
            assert len(doors) == 1, "Expected exactly one door object"
            door = doors[0]
            bottoms = self.object_name_to_objects("bottom", test_time=True)
            assert len(bottoms) == 1, "Expected exactly one bottom object"
            bottom = bottoms[0]
            cabinets = self.object_name_to_objects("cabinet", test_time=True)
            assert len(cabinets) == 1, "Expected exactly one cabinet object"
            cabinet = cabinets[0]
            objs = self.object_name_to_objects("obj", test_time=True)
            assert len(objs) == 1, "Expected exactly one object"
            obj = objs[0]
            if self._DoorClosed_holds(state, [door, cabinet]) and self._OnSurface_holds(state, [obj, bottom]):
                return True
        elif goal_desc == "TurnOnMicrowave":
            microwaves = self.object_name_to_objects("microwave", test_time=True)
            assert len(microwaves) == 1, "Expected exactly one microwave object"
            microwave = microwaves[0]
            if microwave is not None and self._MicrowaveOn_holds(state, [microwave]):
                return True
        elif goal_desc == "CloseDrawer":
            drawer_inner_boxes = self.object_name_to_objects("bottom", test_time=True)
            assert len(drawer_inner_boxes) == 1, "Expected exactly one drawer inner box object"
            drawer_inner_box = drawer_inner_boxes[0]
            drawer_cabinets = self.object_name_to_objects("drawer", test_time=True)
            assert len(drawer_cabinets) == 1, "Expected exactly one drawer cabinet object"
            drawer_cabinet = drawer_cabinets[0]
            if self._DrawerClosed_holds(state, [drawer_inner_box, drawer_cabinet]):
                return True
        elif goal_desc == "OpenDrawer":
            drawer_inner_boxes = self.object_name_to_objects("bottom", test_time=True)
            assert len(drawer_inner_boxes) == 1, "Expected exactly one drawer inner box object"
            drawer_inner_box = drawer_inner_boxes[0]
            drawer_cabinets = self.object_name_to_objects("drawer", test_time=True)
            assert len(drawer_cabinets) == 1, "Expected exactly one drawer cabinet object"
            drawer_cabinet = drawer_cabinets[0]
            if self._DrawerOpen_holds(state, [drawer_inner_box, drawer_cabinet]):
                return True
        elif goal_desc == "TurnOffStove":
            stoves = self.object_name_to_objects("stovetop", test_time=True)
            assert len(stoves) == 1, "Expected exactly one stovetop object"
            stove = stoves[0]
            if stove is not None and self._StoveOff_holds(state, [stove]):
                return True
        elif goal_desc == "TurnOnSinkFaucet":
            sink_faucet_handles = self.object_name_to_objects("sink_faucet_handle", test_time=True)
            assert len(sink_faucet_handles) == 1, "Expected exactly one sink faucet handle object"
            sink_faucet_handle = sink_faucet_handles[0]
            if sink_faucet_handle is not None and self._SinkFaucetOn_holds(state, [sink_faucet_handle]):
                return True
        elif goal_desc == "TurnOffSinkFaucet":
            sink_faucet_handles = self.object_name_to_objects("sink_faucet_handle", test_time=True)
            assert len(sink_faucet_handles) == 1, "Expected exactly one sink faucet handle object"
            sink_faucet_handle = sink_faucet_handles[0]
            if sink_faucet_handle is not None and self._SinkFaucetOff_holds(state, [sink_faucet_handle]):
                return True
        elif goal_desc == "CookCheeseAndTomatoes":
            tomato = self.object_name_to_object("tomato_1", test_time=True)
            assert tomato is not None, "Expected exactly one tomato object"
            cheese = self.object_name_to_object("cheese_2", test_time=True)
            assert cheese is not None, "Expected exactly one cheese object"
            plate1 = self.object_name_to_object("plate_1", test_time=True)
            assert plate1 is not None, "Expected exactly one plate object"
            plate2 = self.object_name_to_object("plate_2", test_time=True)
            assert plate2 is not None, "Expected exactly one plate object"
            if self._InContainer_holds(state, [tomato, plate1]) and self._InContainer_holds(state, [cheese, plate2]):
                return True
        elif goal_desc == "PnPCabToCounterTomato":
            obj = self.object_name_to_object("tomato_1", test_time=True)
            assert obj is not None, "Expected exactly one object"
            plate = self.object_name_to_object("plate")
            assert plate is not None, "Expected exactly one plate object"
            if self._OnSurface_holds(state, [obj, plate]):
                return True
            
        
        else:
            return False

    def _reset_initial_state(self,
                               seed: int,
                               train_or_test: str,
                               task_name: str,
                               initial_state_info: Optional[Dict] = None
                               ) -> Observation:
        """Reset the environment to an initial state based on the seed."""
        # Create or recreate environment if needed
        # warnings.warn("Resetting environment to initial state from seed not implemented for robosuite kitchen")
        self._env = "DummyEnv"
        self._env_raw = None
        # Check if demo-based reset is enabled
        if CFG.demo_reset_enabled:
            demo_observation = self._reset_from_demo()
            current_state = demo_observation
        else:
            current_state = self._get_current_observation(task_name)
        return {"state_info": current_state, "obs_images": [], "contact_set": set()}
        #     complex_config = True  # NOTE: this should be removed. only for mac
        #     if complex_config:
        #         robot_type = "PandaOmron"
        #         controller_config = load_composite_controller_config(robot=robot_type)

        #         config = {
        #             "env_name": task_name,
        #             "robots": robot_type,
        #             "controller_configs": controller_config,
        #             "layout_ids": 3,
        #             "style_ids": 0,
        #             "layout_ids": [3],
        #             "style_ids": None,
        #             "translucent_robot": True,
        #         }

        #         print(colored(f"Initializing environment for task: {task_name}", "yellow"))

        #         self._env_raw = robosuite.make(
        #             **config,
        #             has_renderer=self._using_gui,
        #             has_offscreen_renderer=not self._using_gui,
        #             render_camera="robot0_frontview",
        #             ignore_done=True,
        #             use_camera_obs=False,
        #             control_freq=20,
        #             renderer="mjviewer",
        #             # seed=4,
        #         )

        #         self._env = VisualizationWrapper(self._env_raw)
        #         self.ep_meta = self._env.get_ep_meta()
        #     else:
        #         print(f"Creating env for task: {task_name}, seed: {seed}, gui: {self._using_gui}")
        #         self._env = create_env(
        #             env_name=task_name,
        #             render_onscreen=self._using_gui,
        #             seed=seed + 4,  # this seed the third demo opens to the right, will have replan
        #         )

        # # Reset environment with seed
        # obs = self._env.reset()

        # if CFG.use_teleop:
        #     self.device = Keyboard(
        #         env=self._env,
        #         pos_sensitivity=4.0,
        #         rot_sensitivity=4.0,
        #     )
        #     self.device.start_control()

        # # Update objects of interest based on task
        # self.objects_of_interest = self.get_objects_of_interest(task_name)

        # # Get contact information
        # contact_set = self.get_object_level_contacts()

        # self.num = 0
        # self.default_contact_num = len(self._env_raw.sim.data.contact)
        # self.default_contact_pairs = [(self._env_raw.sim.model.geom_id2name(contact.geom1), self._env_raw.sim.model.geom_id2name(contact.geom2)) for contact in self._env_raw.sim.data.contact]

        # # Get initial gripper in base pose
        # initial_eef_pos_in_base = self._env.robots[0]._hand_pos["right"]
        # initial_eef_orn_mat_in_base = self._env.robots[0]._hand_orn["right"]
        # initial_eef_quat_in_base = T.mat2quat(initial_eef_orn_mat_in_base)
        # # CFG.init_pose = np.concatenate([initial_eef_pos_in_base, initial_eef_quat_in_base])
        # self.initial_eef_pos_quat = np.concatenate([initial_eef_pos_in_base, initial_eef_quat_in_base])

        # # Return observation
        # return {"state_info": obs, "obs_images": [], "contact_set": contact_set}

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
    def _get_current_observation(self, task_name: str) -> Observation:
        """Get the current observation from the hardware."""
        state_info = {}

        # state_info["gripper_pos_quat"] = np.concatenate([ee_pos_world, ee_quat_world])
        


        # gripper_positions = self._hw_interface.get_gripper_positions()
        # gripper_width = 0.0
        # if gripper_positions is not None:
        #     gripper_width = abs(gripper_positions[0]) + abs(gripper_positions[1])
        # state_info["gripper_width"] = gripper_width

        # Get poses from motion capture
        # Handle Pose
        # door_pose_msg = self._hw_interface.get_door_pose(wait_for_message=False, timeout=2.0)
        # # transform the handle pose to the base frame
        # door_pose_msg = self._hw_interface.transform_pose(door_pose_msg, self._robot_base_frame)
        # pos = door_pose_msg.pose.position
        # quat = door_pose_msg.pose.orientation
        # state_info["door_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # Cabinet Pose
        # cabinet_pose_msg = self._hw_interface.get_cabinet_pose(wait_for_message=False, timeout=2.0)
        # # transform the cabinet pose to the base frame
        # cabinet_pose_msg = self._hw_interface.transform_pose(cabinet_pose_msg, self._robot_base_frame)
        # pos = cabinet_pose_msg.pose.position
        # quat = cabinet_pose_msg.pose.orientation
        # state_info["cabinet_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))


        # End-Effector Pose (Gripper) this has slight offset wrt gripper alt pose, mocap is more consistent
        # ee_pose_msg = self._hw_interface.get_ee_pose(wait_for_message=False, timeout=2.0)
        # ee_pose_in_base = self._hw_interface.transform_pose(ee_pose_msg, self._robot_base_frame)
        # assert ee_pose_msg is not None, "No end-effector pose message received" # mocap is in world frame now


        # pos = ee_pose_in_base.pose.position
        # quat = ee_pose_in_base.pose.orientation
        # ee_pos_base = np.array([pos.x, pos.y, pos.z])
        # ee_quat_base = np.array([quat.x, quat.y, quat.z, quat.w])
        # state_info["gripper_pos_quat"] = np.concatenate([ee_pos_base, ee_quat_base])
        # Gripper Alternate Pose (Mocap)
        gripper_alt_pose_msg = self._hw_interface.get_gripper_alt_pose(wait_for_message=False, timeout=2.0)
        # transform the gripper alternate pose to the base frame
        gripper_alt_pose_msg = self._hw_interface.transform_pose(gripper_alt_pose_msg, self._robot_base_frame)
        pos = gripper_alt_pose_msg.pose.position
        quat = gripper_alt_pose_msg.pose.orientation
        state_info["gripper_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # try:
        #     bottom_surface_pose_lookup = self._hw_interface.tf_listener.lookupTransform(
        #         "mocap_world",
        #         "bottom",
        #         rospy.Time(0)
        #     )
        #     bottom_surface_pose_msg = PoseStamped()
        #     bottom_surface_pose_msg.header.frame_id = self._robot_base_frame
        #     bottom_surface_pose_msg.pose.position.x = bottom_surface_pose_lookup[0][0]
        #     bottom_surface_pose_msg.pose.position.y = bottom_surface_pose_lookup[0][1]
        #     bottom_surface_pose_msg.pose.position.z = bottom_surface_pose_lookup[0][2]
        #     bottom_surface_pose_msg.pose.orientation.x = bottom_surface_pose_lookup[1][0]
        #     bottom_surface_pose_msg.pose.orientation.y = bottom_surface_pose_lookup[1][1]
        #     bottom_surface_pose_msg.pose.orientation.z = bottom_surface_pose_lookup[1][2]
        #     bottom_surface_pose_msg.pose.orientation.w = bottom_surface_pose_lookup[1][3]
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        #     rospy.logwarn(f"Failed to get bottom surface pose: {e}")
        #     bottom_surface_pose_msg = None
        # transform the bottom surface pose to the base frame
        # bottom_surface_pose_msg = self._hw_interface.transform_pose(bottom_surface_pose_msg, self._robot_base_frame)
        # pos = bottom_surface_pose_msg.pose.position
        # quat = bottom_surface_pose_msg.pose.orientation
        # state_info["bottom_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))


        state_info["robot0_base_pos_quat"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        gripper_positions = self._hw_interface.get_gripper_positions()
        if gripper_positions is not None:
            state_info["left_finger_pos_quat"] = np.array([-gripper_positions[0], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            state_info["right_finger_pos_quat"] = np.array([gripper_positions[1], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Add wrist object - get actual link7 pose from TF instead of using gripper pose
        link7_msg = self._hw_interface.get_link7_pose(timeout=0.5)
        if link7_msg is not None:
            # Use actual link7 pose from TF lookup
            link7_msg = self._hw_interface.transform_pose(link7_msg, self._robot_base_frame)
            pos = link7_msg.pose.position
            quat = link7_msg.pose.orientation
            state_info["wrist_pos_quat"] = np.array([pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w])
        else:
            # Fallback to gripper pose if TF lookup fails
            print("Warning: Could not get link7 pose from TF, falling back to gripper pose for wrist")
            gripper_pos = state_info["gripper_pos_quat"][:3]
            gripper_quat = state_info["gripper_pos_quat"][3:]
            state_info["wrist_pos_quat"] = np.concatenate([gripper_pos, gripper_quat])
        
        # Get new mocap object poses - only include if we have valid data
        # Bowl pose
        bowl_msg = self._hw_interface.get_bowl_pose(wait_for_message=False, timeout=2.0)
        if bowl_msg is not None:
            bowl_msg = self._hw_interface.transform_pose(bowl_msg, self._robot_base_frame)
            if bowl_msg is not None:
                pos = bowl_msg.pose.position
                quat = bowl_msg.pose.orientation
                state_info["bowl_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # Lid pose
        lid_msg = self._hw_interface.get_lid_pose(wait_for_message=False, timeout=2.0)
        if lid_msg is not None:
            lid_msg = self._hw_interface.transform_pose(lid_msg, self._robot_base_frame)
            if lid_msg is not None:
                pos = lid_msg.pose.position
                quat = lid_msg.pose.orientation
                state_info["lid_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # Pan pose
        pan_msg = self._hw_interface.get_pan_pose(wait_for_message=False, timeout=2.0)
        if pan_msg is not None:
            pan_msg = self._hw_interface.transform_pose(pan_msg, self._robot_base_frame)
            if pan_msg is not None:
                pos = pan_msg.pose.position
                quat = pan_msg.pose.orientation
                state_info["pan_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # Dishrack pose
        dishrack_msg = self._hw_interface.get_dishrack_pose(wait_for_message=False, timeout=2.0)
        if dishrack_msg is not None:
            dishrack_msg = self._hw_interface.transform_pose(dishrack_msg, self._robot_base_frame)
            if dishrack_msg is not None:
                pos = dishrack_msg.pose.position
                quat = dishrack_msg.pose.orientation
                state_info["dishrack_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        # Mug pose
        mug_msg = self._hw_interface.get_mug_pose(wait_for_message=False, timeout=2.0)
        if mug_msg is not None:
            mug_msg = self._hw_interface.transform_pose(mug_msg, self._robot_base_frame)
            if mug_msg is not None:
                pos = mug_msg.pose.position
                quat = mug_msg.pose.orientation
                state_info["mug_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))


        # banana pose
        banana_msg = self._hw_interface.get_banana_pose(wait_for_message=False, timeout=2.0)
        if banana_msg is not None:
            banana_msg = self._hw_interface.transform_pose(banana_msg, self._robot_base_frame)
            if banana_msg is not None:
                pos = banana_msg.pose.position
                quat = banana_msg.pose.orientation
                state_info["banana_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))
        # Get object pose (if relevant for the task) - only include if we have valid data
        # obj_msg = self._hw_interface.get_object_pose()
        # if obj_msg is not None:
        #     obj_msg = self._hw_interface.transform_pose(obj_msg, self._robot_base_frame)
        #     if obj_msg is not None:
        #         pos = obj_msg.pose.position
        #         quat = obj_msg.pose.orientation
        #         state_info["obj_pos_quat"] = np.concatenate(([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w]))

        contact_set = set()

        # self.objects_of_interest = self.get_objects_of_interest(task_name)

        return state_info
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
            Predicate("OnCounter", [cls.thing_type, cls.counter_type], cls._OnCounter_holds),
            Predicate("DoorHalfOpen", [cls.handle_type, cls.cabinet_type], cls._DoorHalfOpen_holds),
            Predicate("KnobTurnedOn", [cls.knob_type, cls.stove_type], cls._KnobTurnedOn_holds),
            Predicate("InOrigin", [cls.gripper_type, cls.base_type], cls._InOrigin_holds),
            Predicate("MicrowaveOn", [cls.microwave_type], cls._MicrowaveOn_holds),
            Predicate("StoveOn", [cls.stove_type], cls._StoveOn_holds),
            Predicate("StoveOff", [cls.stove_type], cls._StoveOff_holds),
            Predicate("SinkFaucetOn", [cls.sink_faucet_handle_type], cls._SinkFaucetOn_holds),
            Predicate("SinkFaucetOff", [cls.sink_faucet_handle_type], cls._SinkFaucetOff_holds),
            Predicate("InContainer", [cls.thing_type, cls.container_type], cls._InContainer_holds),
            # below are hardware predicates
            Predicate("LidOnDishrack", [cls.lid_type, cls.cabinet_type], cls._LidOnDishrack_holds),
            Predicate("PourInPan", [cls.thing_type, cls.container_type], cls._PourInPan_holds),
            Predicate("InCookware", [cls.thing_type, cls.cookware_type], cls._InCookware_holds),
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
        gripper_cmd_raw = action.arr[6]

        # Create 12D robocasa action:
        # - First 6D: right arm pose (position + rotation)
        # - Next 1D: right gripper
        # - Next 3D: base (no movement)
        # - Next 1D: torso (no movement)
        # - Last 1D: extra dimension (not used)
        # env_action = np.zeros(12, dtype=np.float32)

        # env_action[0:3] = pos_delta  # position control
        # env_action[3:6] = rot_delta  # rotation control
        # env_action[6] = gripper_cmd  # gripper control
        # if CFG.use_teleop:
        #     env_action[7:10] = input_ac_dict["base"]
        # # env_action[7:10] are zeros (no base movement)
        # # env_action[10] is zero (no torso movement)
        # # env_action[11] is zero (extra dimension)

        # # Execute action in environment (Robosuite:Mujoco Env)
        # obs, _, _, _ = self._env.step(env_action)

        # contact_set = self.get_object_level_contacts()
        action_velocities = np.concatenate([pos_delta, rot_delta])
        self._hw_interface.scale_and_publish_twist_action(action_velocities)


        gripper_cmd_hw = 1.0 if gripper_cmd_raw > 0 else 0.0
        self._hw_interface.set_gripper_state(gripper_cmd_hw)

        ob = self._get_current_observation(self.task_selected)

        observation = {"state_info": ob, "obs_images": [], "contact_set": set(  )}

        self._current_observation = observation
        
        # Visualize bounding boxes if enabled
        if CFG.robo_kitchen_modulation_mode is not None:
            CFG.robo_kitchen_obstacles = {}
            if CFG.robo_kitchen_task in CFG.mocap_tasks:
                # print("Getting mocap object bboxes")
                self._get_mocap_object_bboxes()
            else:
                self._get_object_bboxes()
        # Visualize robot arm spheres each step (GUI only)
        # self._visualize_robot_arm_spheres()
        # Video frame saving logic (only if GUI is not enabled)
        # if not self._using_gui:
        #     self._frame_counter += 1
        #     # Save a frame from the center camera
        #     frame = self._env.sim.render(camera_name="robot0_agentview_center", height=512, width=768)
        #     self._video_frames.append(frame)
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Reset environment to initial state for the given task."""
        self._current_task = self.get_task(train_or_test, task_idx)
        task_name = self._current_task.goal_description
        warnings.warn("Resetting environment to initial state from not implemented, just reset the env")
        self._current_observation = self._reset_initial_state(seed=task_idx, train_or_test=train_or_test, task_name=task_name)
        return self._copy_observation(self._current_observation)

    def _reset_from_demo(self) -> Optional[Observation]:
        """Reset environment from demo data at specified timestep.
        
        Returns:
            Observation from demo data, or None if reset fails
        """
        from predicators.demo_utils import load_demo_dataset, get_demo_state_at_timestep, validate_demo_reset_config
        
        # Validate configuration
        if not validate_demo_reset_config():
            return None
            
        # Load demo dataset
        demo_dataset = load_demo_dataset()
        if demo_dataset is None:
            return None
            
        # Get state from demo at specified timestep
        demo_state = get_demo_state_at_timestep(
            demo_dataset, 
            CFG.demo_reset_task_idx, 
            CFG.demo_reset_timestep
        )
        
        if demo_state is None:
            return None
            
        return demo_state

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

    def _get_object_bboxes(self):
        """Visualize bounding boxes of all objects and fixtures in the environment."""
        if not (self._env_raw and hasattr(self._env_raw, "viewer") and self._env_raw.viewer is not None):
            return
        
        door_ids = []
        for name in CFG.robo_kitchen_obj_names:
            # Match names like 'door_{num}_pos_quat' and extract the num
            match = re.match(r"door_(\d+)$", name)
            if match:
                door_ids.append(int(match.group(1)))

        # These are from each task's _get_obj_cfgs(), e.g. "door", "distr_counter", etc.
        # In terms of mapping to CFG.robo_kitchen_obj_names, 
        # objs are fine because they are not added mujoco id in CFG.robo_kitchen_obj_names, 
        # but doors are added mujoco id (so we know which door belongs to which cabinet).
        if len(door_ids) > 0:
            for id in door_ids:
                cabinet = next(
                    (fx for fx in getattr(self._env_raw, "fixtures", {}).values()
                    if self._env_raw.sim.model.body_name2id(fx.root_body) == id),
                    None
                )
                if cabinet is not None:
                    # For single door
                    door_body_name = cabinet.door_name
                    # Compute bounding box points for the door panel body
                    door_bbox_points = self._get_body_bbox_points(door_body_name, ignore_handle=False)
                    door_obstacle_name = f'door_{id}'
                    
                    # Process door obstacle
                    if door_bbox_points is not None:
                        center, quat_xyzw, radii = self._fit_bbox_ellipsoid(door_bbox_points, door_obstacle_name)
                        if CFG.robo_kitchen_visualize_bboxes:
                            self._visualize_bbox_ellipsoid(door_bbox_points, center, quat_xyzw, radii, "door"+str(id))
                    
                    # Add cabinet bottom as separate obstacle
                    bottom_geom_name = f"{cabinet.name}_bottom"
                    bottom_bbox_points = self._get_geom_bbox_points(bottom_geom_name, ignore_handle=True)
                    
                    if bottom_bbox_points is not None:
                        bottom_obstacle_name = f'cabinet_bottom_{id}'
                        center, quat_xyzw, radii = self._fit_bbox_ellipsoid(bottom_bbox_points, bottom_obstacle_name)
                        if CFG.robo_kitchen_visualize_bboxes:
                            self._visualize_bbox_ellipsoid(bottom_bbox_points, center, quat_xyzw, radii, f"bottom_{id}")
                    else:
                        logging.warning(f"Bottom geom {bottom_geom_name} not found for cabinet {id}")
                        
                else:
                    logging.warning(f"door_{id}'s cabinet not found in fixtures")
        
        for obj_name in self._env_raw.obj_body_id:
            obstacle_name = obj_name # what to store in CFG.robo_kitchen_obstacles
            
            obj_model = self._env_raw.objects.get(obj_name)
            if obj_model is None:
                continue
            
            # Get object position and orientation
            obj_pos = self._env_raw.sim.data.body_xpos[self._env_raw.obj_body_id[obj_name]]
            obj_quat_wxyz = self._env_raw.sim.data.body_xquat[self._env_raw.obj_body_id[obj_name]]
            # Convert from wxyz to xyzw format
            obj_quat_xyzw = np.array([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])
            
            # Get bounding box points
            try:
                bbox_points = obj_model.get_bbox_points(trans=obj_pos, rot=obj_quat_xyzw)
            except Exception:
                # Skip objects that don't have proper bounding box implementation
                continue
            center, quat_xyzw, radii = self._fit_bbox_ellipsoid(bbox_points, obstacle_name) # obstacle_name is what to store in CFG.robo_kitchen_obstacles
            if CFG.robo_kitchen_visualize_bboxes:
                self._visualize_bbox_ellipsoid(bbox_points, center, quat_xyzw, radii, obj_name) # obj_name is only used for generating a color (doesn't matter)
        
        # Visualize all fixtures (cabinets, doors, drawers, counters, etc.)
        # if hasattr(self._env_raw, 'fixtures'):
        #     for fixture_name, fixture_model in self._env_raw.fixtures.items():
        #         if not hasattr(fixture_model, 'get_bbox_points'):
        #             continue
                
        #         try:
        #             # Get fixture position and orientation from its root body
        #             fixture_body_id = self._env_raw.sim.model.body_name2id(fixture_model.root_body)
        #             fixture_pos = self._env_raw.sim.data.body_xpos[fixture_body_id]
        #             fixture_quat_wxyz = self._env_raw.sim.data.body_xquat[fixture_body_id]
        #             # Convert from wxyz to xyzw format
        #             fixture_quat_xyzw = np.array([fixture_quat_wxyz[1], fixture_quat_wxyz[2], fixture_quat_wxyz[3], fixture_quat_wxyz[0]])
                    
        #             # Get bounding box points
        #             bbox_points = fixture_model.get_bbox_points(trans=fixture_pos, rot=fixture_quat_xyzw)
                    
        #             # Visualize bounding box as wireframe
        #             self._visualize_bbox_wireframe(bbox_points, f"fixture_{fixture_name}")
        #         except Exception:
        #             # Skip fixtures that don't have proper bounding box implementation
        #             continue

    def _get_mocap_object_bboxes(self):
        """Get bounding boxes for mocap tasks using predefined sizes from settings and object poses from observations."""
        if not hasattr(self, '_current_observation') or self._current_observation is None:
            return
            
        state_info = self._current_observation.get("state_info", {})
        
        # Object types to exclude from obstacles (robot parts, etc.)
        excluded_types = {
            "base_type", "gripper_type", "wrist_type", 
            "left_finger_type", "right_finger_type"
        }
        state =  self.state_info_to_state(state_info)
        for obj in state:
            obj_name = obj.name
            # print(f"Processing state_info entry: {obj_name}")
            # Skip non-object entries
            obj_type_name = obj.type.name
                            
            # Skip robot parts and other excluded types
            if obj_type_name in excluded_types:
                # print(f"Skipping state_info entry '{obj_name}'")
                continue
                
            # Check if we have predefined sizes for this object type
            if obj_type_name not in CFG.mocap_object_sizes:
                # Log a warning for unknown object types (optional)
                # if obj_type_name not in excluded_types:
                # print(f"Warning: No bounding box size defined for object type '{obj_type_name}' ")
                continue
                
            # Get half-extents from settings
            half_extents = np.array(CFG.mocap_object_sizes[obj_type_name])
            
            # Get object pose from state
            try:
                # Objects use "translation" and "quaternion" fields
                obj_pos       = state.get(obj, "translation")  # if object has translation feature
                obj_quat_xyzw = state.get(obj, "quaternion")   # if object has quaternion feature
                
                # Ensure they are numpy arrays
                if not isinstance(obj_pos, np.ndarray):
                    obj_pos = np.array(obj_pos)
                if not isinstance(obj_quat_xyzw, np.ndarray):
                    obj_quat_xyzw = np.array(obj_quat_xyzw)
                    
            except (AttributeError, KeyError, TypeError):

                print(f"Skipping state_info entry '{obj_name}' without proper pose information")
                # Skip objects without proper pose information
                continue
                
            # Create bounding box points from half-extents
            bbox_points = self._create_bbox_points_from_pose(obj_pos, obj_quat_xyzw, half_extents)
            
            # Fit ellipsoid and store in obstacles
            if bbox_points is not None:
                obstacle_name = obj_name
                center, quat_xyzw, radii = self._fit_bbox_ellipsoid(bbox_points, obstacle_name)
                
                # Visualize if enabled
                if CFG.robo_kitchen_visualize_bboxes:
                    self._visualize_bbox_ellipsoid(bbox_points, center, quat_xyzw, radii, obj_name)

    def _create_bbox_points_from_pose(self, position, quaternion_xyzw, half_extents):
        """Create 8 bounding box corner points from object pose and half-extents.
        
        Args:
            position: 3D position [x, y, z]
            quaternion_xyzw: Quaternion [x, y, z, w] 
            half_extents: Half-sizes along each axis [x_half, y_half, z_half]
            
        Returns:
            List of 8 corner points in world coordinates
        """
        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat(quaternion_xyzw).as_matrix()
        
        # Create 8 corner offsets in local frame
        bbox_offsets = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    local_offset = np.array([dx * half_extents[0], 
                                           dy * half_extents[1], 
                                           dz * half_extents[2]])
                    bbox_offsets.append(local_offset)
        
        # Transform to world coordinates
        bbox_points = []
        for offset in bbox_offsets:
            world_point = position + rot_matrix @ offset
            bbox_points.append(world_point)
            
        return bbox_points

    def _visualize_bbox_wireframe(self, bbox_points, obj_name):
        """Visualize bounding box as wireframe cube using edges."""
        

    def _bbox_to_min_ellipsoid(self, bbox_points):
        """Analytically convert 8 bounding-box vertices to centre, orientation (quat xyzw) and
        radii (a, b, c) of the smallest-volume ellipsoid whose axes are aligned with the
        box axes.  Radii are √3 times the half-lengths of the box."""

        import numpy as np

        if len(bbox_points) != 8:
            return None

        pts = np.asarray(bbox_points)
        centre = pts.mean(axis=0)

        # Principal directions via SVD (works for any oriented rectangular box)
        _, _, vh = np.linalg.svd(pts - centre, full_matrices=False)
        R_box = vh.T  # Columns are principal axes

        # Ensure a right-handed coordinate frame (determinant +1)
        if np.linalg.det(R_box) < 0:
            R_box[:, -1] *= -1

        local = (pts - centre) @ R_box  # Express vertices in box frame
        half_lengths = np.max(np.abs(local), axis=0)

        radii = half_lengths * np.sqrt(3.0)

        quat_xyzw = R.from_matrix(R_box).as_quat()

        return centre, quat_xyzw, radii

    def _fit_bbox_ellipsoid(self, bbox_points, obj_name):
        """Draw the analytical minimal ellipsoid (√3-scaled) enclosing the box."""

        res = self._bbox_to_min_ellipsoid(bbox_points)
        if res is None:
            return

        centre, quat_xyzw, radii = res
        if CFG.robo_kitchen_modulation_mode == "ellipsoid":
            CFG.robo_kitchen_obstacles[obj_name] = (bbox_points, (centre, radii, quat_xyzw))

        return centre, quat_xyzw, radii

    def _visualize_bbox_ellipsoid(self, bbox_points, centre, quat_xyzw, radii, obj_name):
        if len(bbox_points) != 8:
            return
        
        # Generate unique, consistent colour for this object
        color = self._object_hash_color(obj_name)
        
        # Define the 12 edges of a cube (connecting the 8 vertices)
        # Based on the vertex ordering from get_bbox_points:
        # 0: [-1, -1, -1], 1: [+1, -1, -1], 2: [-1, +1, -1], 3: [-1, -1, +1]
        # 4: [+1, +1, +1], 5: [-1, +1, +1], 6: [+1, -1, +1], 7: [+1, +1, -1]
        edges = [
            # Bottom face (z = -1): back_left → back_right → front_right → front_left → back_left
            (0, 1), (1, 7), (7, 2), (2, 0),
            # Top face (z = +1): back_left → back_right → front_right → front_left → back_left  
            (3, 6), (6, 4), (4, 5), (5, 3),
            # Vertical edges: connect corresponding bottom and top vertices
            (0, 3), (1, 6), (2, 5), (7, 4)
        ]
        
        # Draw each edge as a thin cylinder NOTE: this is commented out since it is not working.
        # for i, (start_idx, end_idx) in enumerate(edges):
        #     start_point = np.array(bbox_points[start_idx])
        #     end_point = np.array(bbox_points[end_idx])
            
        #     # Calculate midpoint and direction
        #     midpoint = (start_point + end_point) / 2
        #     direction = end_point - start_point
        #     length = np.linalg.norm(direction)
            
        #     if length > 0:
        #         # Calculate orientation quaternion for the cylinder
        #         # Default cylinder axis is along z, we want it along the edge direction
        #         z_axis = np.array([0, 0, 1])
        #         edge_direction = direction / length
                
        #         # Calculate rotation quaternion to align z-axis with edge direction
        #         if np.allclose(edge_direction, z_axis):
        #             quat = np.array([1, 0, 0, 0])  # no rotation needed
        #         elif np.allclose(edge_direction, -z_axis):
        #             quat = np.array([0, 1, 0, 0])  # 180 degree rotation around x
        #         else:
        #             # General case: rotate z-axis to align with edge direction
        #             cross = np.cross(z_axis, edge_direction)
        #             dot = np.dot(z_axis, edge_direction)
        #             quat_w = 1 + dot
        #             quat = np.array([quat_w, cross[0], cross[1], cross[2]])
        #             quat = quat / np.linalg.norm(quat)
                
        #         # Draw thin cylinder as edge
        #         self.mjshowellipse(
        #             xyz=midpoint,
        #             quat=quat,
        #             size=(0.003, 0.003, length/2),  # thin cylinder
        #             color=color,
        #             alpha=0.8,
        #             name=None  # No text label
        #         )

        # Visualize each vertex as a small sphere for clarity
        for idx, vertex in enumerate(bbox_points):
            self.mjshowellipse(
                xyz=np.array(vertex),
                quat=(1, 0, 0, 0),  # orientation irrelevant for spheres
                size=(0.01, 0.01, 0.01),  # small sphere radius
                color=color,
                alpha=0.9,
                name=None  # no text label
            )

        self.mjshowellipse(
            xyz=centre,
            quat=quat_xyzw,
            size=tuple(radii),
            color=color,
            alpha=0.5,
            name=None,
        )

        # BEGIN ROBOT ARM SPHERE METHODS
    def _init_robot_arm_spheres(self):
        """Compute 2-4 bounding spheres per robot link (Panda Omron) using an
        axis-aligned sweep in the link frame.

        The sphere parameters are stored in
        ``self.robot_arm_spheres`` as a list of tuples
        ``(body_name, local_offset, radius)`` where
          • *body_name*      - Mujoco body name (string)  
          • *local_offset*   - 3-vector, centre expressed in the **link** frame  
          • *radius*         - scalar, metres

        We treat every geom belonging to the link, expand its extent in the
        link frame, then wrap the aggregated AABB with the minimal enclosing
        sphere (centre = box midpoint, radius = max corner distance).  Only
        links whose body name starts with ``robot0`` (the Panda arm prefix in
        RoboCasa) are considered.
        """
        import numpy as np  # local import to avoid circular issues

        # Guard against running before the mujoco simulator exists.
        if not getattr(self, "_env_raw", None):
            self.robot_arm_spheres = []
            return

        model = self._env_raw.sim.model
        data = self._env_raw.sim.data

        robot_prefix = "robot0"                     # Panda Omron prefix
        spheres: list[tuple[str, np.ndarray, float]] = []

        for body_id in range(model.nbody):
            body_name = model.body_id2name(body_id)
            if body_name is None or not body_name.startswith(robot_prefix):
                continue

            # World‑frame pose of the link.
            body_pos = data.body_xpos[body_id].copy()
            body_xmat = data.body_xmat[body_id].reshape(3, 3).copy()

            # Grow an axis‑aligned box in the **link** frame that encloses
            # every geom attached to this body.
            ext_min = np.array([ np.inf,  np.inf,  np.inf])
            ext_max = np.array([-np.inf, -np.inf, -np.inf])

            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) != body_id:
                    continue

                geom_type = int(model.geom_type[g])
                size      = model.geom_size[g].copy()
                geom_pos  = data.geom_xpos[g].copy()
                geom_xmat = data.geom_xmat[g].reshape(3, 3).copy()

                # Centre of the geom expressed in the link frame
                centre_local = body_xmat.T @ (geom_pos - body_pos)

                if geom_type == 2:                                   # sphere
                    r = size[0]
                    ext_min = np.minimum(ext_min, centre_local - r)
                    ext_max = np.maximum(ext_max, centre_local + r)

                elif geom_type in (3, 4):                            # capsule / cyl
                    r, half_len = size[0], size[1]
                    axis_local  = body_xmat.T @ geom_xmat[:, 0]      # local x‑axis
                    for sign in (-1.0, +1.0):
                        end_pt = centre_local + sign * axis_local * half_len
                        ext_min = np.minimum(ext_min, end_pt - r)
                        ext_max = np.maximum(ext_max, end_pt + r)

                else:                                                # box or mesh
                    # size = half‑extents in geom frame; gather all eight corners
                    sx, sy, sz = size if geom_type == 1 else (
                        size[0],
                        size[1] if size[1] > 0 else size[0],
                        size[2] if size[2] > 0 else size[0],
                    )
                    corners = np.array(
                        [[ sx,  sy,  sz], [ sx,  sy, -sz], [ sx, -sy,  sz], [ sx, -sy, -sz],
                         [-sx,  sy,  sz], [-sx,  sy, -sz], [-sx, -sy,  sz], [-sx, -sy, -sz]]
                    )
                    # Express corners in the link frame
                    rot_local = body_xmat.T @ geom_xmat
                    corners_local = (rot_local @ corners.T).T + centre_local
                    ext_min = np.minimum(ext_min, corners_local.min(axis=0))
                    ext_max = np.maximum(ext_max, corners_local.max(axis=0))

            if np.any(np.isinf(ext_min)):
                # No geoms for this link (should not happen)
                continue

            # --- Split the link into 2‑4 spheres along its longest dimension ---
            ext         = ext_max - ext_min
            main_idx    = int(np.argmax(ext))              # index of longest axis
            length_main = float(ext[main_idx])

            # Radius: half of the largest minor extent (approximates cross‑section)
            minor_ext    = np.delete(ext, main_idx)
            cross_radius = 0.5 * float(np.max(minor_ext))
            if cross_radius < 1e-4:
                cross_radius = 0.5 * length_main  # degenerate case (thin link)

            # Choose sphere count so they overlap slightly; clamp to [2,4]
            n_spheres = int(np.ceil(length_main / (cross_radius * 1.5)))
            n_spheres = max(2, min(4, n_spheres))

            step = length_main / n_spheres
            for i in range(n_spheres):
                centre_local = ext_min.copy()
                centre_local[main_idx] += (i + 0.5) * step   # centre of slice
                # For the two minor axes, take the midpoint of their extents
                for ax in range(3):
                    if ax != main_idx:
                        centre_local[ax] = 0.5 * (ext_min[ax] + ext_max[ax])

                spheres.append(
                    (body_name, centre_local.astype(float), float(cross_radius))
                )

        self.robot_arm_spheres = spheres

    def _visualize_robot_arm_spheres(self):
        """Render the per-link bounding spheres computed in
        ``_init_robot_arm_spheres``.  Each sphere is drawn as an MJViewer
        ellipsoid (equal axes → sphere).  Called every control step when the
        GUI viewer is available.
        """
        if not getattr(self, "robot_arm_spheres", None):
            return

        # Skip if there is no active Mujoco viewer (e.g. off‑screen rendering).
        if not (self._env_raw and hasattr(self._env_raw, "viewer")
                and self._env_raw.viewer is not None):
            return

        model = self._env_raw.sim.model
        data  = self._env_raw.sim.data

        for body_name, local_offset, radius in self.robot_arm_spheres:
            try:
                body_id = model.body_name2id(body_name)
            except Exception:
                continue  # body was removed or renamed

            body_pos  = data.body_xpos[body_id]
            body_xmat = data.body_xmat[body_id].reshape(3, 3)

            centre_world = body_pos + body_xmat @ local_offset

            # Light blue, semi‑transparent
            self.mjshowellipse(
                xyz   = centre_world,
                quat  = (1, 0, 0, 0),                   # identity orientation
                size  = (radius, radius, radius),
                color = (0.1, 0.4, 1.0),
                alpha = 0.30,
                name  = None,
            )
        # END ROBOT ARM SPHERE METHODS

    def _object_hash_color(self, obj_name: str):
        """Return a bright, deterministic RGB colour for a given object name."""
        import hashlib

        hash_hex = hashlib.md5(obj_name.encode()).hexdigest()
        r = int(hash_hex[0:2], 16) / 255.0
        g = int(hash_hex[2:4], 16) / 255.0
        b = int(hash_hex[4:6], 16) / 255.0

        # Elevate brightness: map from [0,1] → [0.3,1]
        r = 0.3 + 0.7 * r
        g = 0.3 + 0.7 * g
        b = 0.3 + 0.7 * b
        return (r, g, b)

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
            goal_preds = {self._pred_name_to_pred["OnSurface"],  self._pred_name_to_pred["GripperFarFromObj"]}   
        elif goal_desc == "PnPStoveToCounter":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
        elif goal_desc == "PnPCabToCounter":
            goal_preds = {self._pred_name_to_pred["OnCounter"]}
        elif goal_desc == "PnPCounterToStove":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
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
        elif goal_desc == "PnPCabToCounterTomato":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
        elif goal_desc == "MocapOpenLid": 
            goal_preds = {self._pred_name_to_pred["LidOnDishrack"]}
        elif goal_desc == "MocapPourWater":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
        elif goal_desc == "MocapTest":
            goal_preds = {self._pred_name_to_pred["LidOnDishrack"]}
        elif goal_desc == "MocapOpenLidPourWater":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
        elif goal_desc == "MocapPnPBanana":
            goal_preds = {self._pred_name_to_pred["InCookware"]}
        elif goal_desc == "MocapOpenLidPnPBanana":
            goal_preds = {self._pred_name_to_pred["InCookware"]}
        elif goal_desc == "MocapMulti":
            goal_preds = {self._pred_name_to_pred["InCookware"]}
        elif goal_desc == "MocapMultiVision":
            goal_preds = {self._pred_name_to_pred["InCookware"]}
        elif goal_desc == "ArrangeVegetables":
            goal_preds = {self._pred_name_to_pred["InContainer"]}
        else:
            raise NotImplementedError(f"Goal description {goal_desc} not implemented for {CFG.robo_kitchen_task}")
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
    def object_name_to_objects(cls, obj_name: str, test_time: bool = False) -> List[Object]:
        """
        Made public for perceiver.
        If test_time is True, this function searches for object names in CFG.robo_kitchen_obj_names that match obj_name. Return name in CFG.robo_kitchen_obj_names.
        If test_time is False, this function returns obj_name object, if obj_name is in cls.obj_name_to_type.
        """
        if not test_time:
            if obj_name in cls.obj_name_to_type:
                return [Object(obj_name, cls.obj_name_to_type[obj_name])]
            else:
                return []
        found_names = []
        found_objects = []
        obj_name_no_num = obj_name
        if "_" in obj_name and obj_name.split("_")[-1].isdigit():
            obj_name_no_num = "_".join(obj_name.split("_")[:-1])
        if obj_name_no_num not in cls.obj_name_to_type:
            return []
        for robo_kitchen_obj_name in CFG.robo_kitchen_obj_names:
            robo_kitchen_obj_name_no_num = robo_kitchen_obj_name
            if "_" in robo_kitchen_obj_name_no_num and robo_kitchen_obj_name_no_num.split("_")[-1].isdigit():
                robo_kitchen_obj_name_no_num = "_".join(robo_kitchen_obj_name_no_num.split("_")[:-1])
            if obj_name_no_num == robo_kitchen_obj_name_no_num:
                found_names.append(robo_kitchen_obj_name)
        for found_name in found_names:
            found_objects.append(Object(found_name, cls.obj_name_to_type[obj_name_no_num]))
# deal with case with objects that we need to add offline for user demo data, where no env is avaliable.
        if len(found_objects) > 1:
            warnings.warn(f"Expected exactly 1 object for {obj_name}, got {len(found_objects)}")
            # raise ValueError(f"Expected exactly 1 object for {obj_name}, got {len(found_objects)}")
            # if len(found_objects) != 2:
            #     raise ValueError(f"Expected exactly 2 objects for {obj_name}, got {len(found_objects)}")
            
            # # Check if they have the same base name (without pos_quat)
            # obj1_base = found_objects[0].name[:-9] if found_objects[0].name.endswith("pos_quat") else found_objects[0].name
            # obj2_base = found_objects[1].name[:-9] if found_objects[1].name.endswith("pos_quat") else found_objects[1].name
            
            # if obj1_base != obj2_base:
            #     raise ValueError(f"Objects have different base names: {obj1_base} and {obj2_base}")
            
            # # Keep the one that ends with pos_quat
            # if found_objects[0].name.endswith("pos_quat"):
            #     found_objects.pop(1)
            # elif found_objects[1].name.endswith("pos_quat"):
            #     found_objects.pop(0)
            # else:
            #     raise ValueError(f"Neither object ends with pos_quat: {found_objects[0].name} and {found_objects[1].name}")
        return found_objects
    
    @classmethod
    def object_name_to_object(cls, obj_name: str, test_time: bool = False) -> Object:
        """
        Made public for perceiver.
        Use this function at test time only when you have the exact obj_name, i.e. with mujoco id. Returns name in CFG.robo_kitchen_obj_names.
        """
        if not test_time:
            if obj_name in cls.obj_name_to_type:
                return Object(obj_name, cls.obj_name_to_type[obj_name])
            else:
                return None
            
        obj_name_no_pos_quat = obj_name
        if obj_name.endswith("pos_quat"):
            obj_name_no_pos_quat = obj_name[:-9]

        # obj_name is name_id, we need to find if it is in cls.obj_name_to_type
        if "_" in obj_name_no_pos_quat and obj_name_no_pos_quat.split("_")[-1].isdigit():
            last_num_characters = len(obj_name_no_pos_quat.split("_")[-1]) + 1
            obj_name_no_num = obj_name_no_pos_quat[:-last_num_characters]
        else:
            obj_name_no_num = obj_name_no_pos_quat
            
        if obj_name_no_num in cls.obj_name_to_type:
            return Object(obj_name_no_pos_quat, cls.obj_name_to_type[obj_name_no_num])
        
        return None
        
        # for robo_kitchen_obj_name in CFG.robo_kitchen_obj_names:
        #     robo_kitchen_obj_name_no_pos_quat = robo_kitchen_obj_name
        #     if robo_kitchen_obj_name.endswith("pos_quat"):
        #         robo_kitchen_obj_name_no_pos_quat = robo_kitchen_obj_name[:-9]
        #     if obj_name_no_pos_quat == robo_kitchen_obj_name_no_pos_quat:
        #         obj_name_raw = obj_name_no_pos_quat
        #         if "_" in obj_name and obj_name.split("_")[-1].isdigit():
        #             obj_name_raw = "_".join(obj_name.split("_")[:-1])
        #         if obj_name_raw in cls.obj_name_to_type:
        #             return Object(robo_kitchen_obj_name, cls.obj_name_to_type[obj_name_raw])
        return None

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any], contact_set: set[Tuple[Object, Object]] = None) -> State:
        if isinstance(state_info, State):
            return state_info
        
        if hasattr(CFG, "load_approach") and CFG.load_approach:
            cls.door_open_thresh = cls.online_door_open_thresh  # rad
            cls.door_close_thresh = cls.online_door_close_thresh  # rad

        state_dict = {}

        # Process any other objects with standard format
        for key, val in state_info.items():
            if key.endswith("_pos_quat"):
                obj_name = key[:-9]
                obj = cls.object_name_to_object(obj_name, test_time=True)
                translation = np.array([val[0], val[1], val[2]])
                quaternion = np.array([val[3], val[4], val[5], val[6]])
                if obj is not None:
                    state_dict[obj] = {"translation": translation, "quaternion": quaternion}
            elif key.endswith("_quat"):
                obj_name = key[:-5]  # Remove _pos
                translation = np.array(state_info[key[:-5] + "_pos"])
                quaternion = np.array(val)
                obj = cls.object_name_to_object(obj_name, test_time=True)
                if obj is not None:
                    state_dict[obj] = {"translation": translation, "quaternion": quaternion}

        # Add the 'on' feature to the microwave object
        if "microwave_on" in state_info:
            mic_objs = cls.object_name_to_objects("microwave", test_time=True)
            for mic_obj in mic_objs:
                state_dict[mic_obj]["on"] = np.array([state_info["microwave_on"]])

        # Add the 'on' feature to the stove object
        if "stove_on" in state_info:
            stove_objs = cls.object_name_to_objects("stovetop", test_time=True)
            for stove_obj in stove_objs:
                state_dict[stove_obj]["on"] = np.array([state_info["stove_on"]])

        # Add the 'on' feature to the sink faucet object
        if "sink_faucet_on" in state_info:
            sink_faucet_objs = cls.object_name_to_objects("sink_faucet_handle", test_time=True)
            for sink_faucet_obj in sink_faucet_objs:
                if sink_faucet_obj in state_dict:
                    state_dict[sink_faucet_obj]["on"] = np.array([state_info["sink_faucet_on"]])

        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = {}
        state.items_in_contact = contact_set  # when defaults, it means Not populated, when empty means no contact
        cls._current_state = state
        return state


    @classmethod
    def _GripperFarFromObj_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is far from object."""
        gripper, obj = objects
        gripper_pos = state.get(gripper, "translation")
        obj_pos = state.get(obj, "translation")
        gripper_obj_far = np.linalg.norm(gripper_pos - obj_pos) > cls.gripper_obj_far_thresh
        return gripper_obj_far

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
        logging.info(f"finger distance: {distance}")
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
    def _OnSurface_holds(cls, state: State, objects: Sequence[Object], thresh: float = 0.2, vertical_thresh: float = None) -> bool:
        """Check if object is at location."""
        vertical_thresh = vertical_thresh or cls.place_close_z_thresh
        obj, surface = objects
        obj_pos = state.get(obj, "translation")
        obj_quat = state.get(obj, "quaternion")
        surface_pos = state.get(surface, "translation")
        surface_quat = state.get(surface, "quaternion")
        obj_pos_in_surface, _ = frame_transform(obj_pos, obj_quat, surface_pos, R.from_quat(surface_quat).as_matrix())
        on_surface_top = 0.0 <= obj_pos_in_surface[2] <= vertical_thresh
        in_surface_region = abs(obj_pos_in_surface[0]) <= thresh and abs(obj_pos_in_surface[1]) <= thresh
        # print(obj_pos_in_surface[0], obj_pos_in_surface[1])
        # print(near_surface, in_surface)
        return on_surface_top and in_surface_region
    
    @classmethod
    def _OnCounter_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is on counter."""
        obj, counter = objects
        return cls._OnSurface_holds(state, [obj, counter], thresh = 1.3, vertical_thresh = 0.6)

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
        drawer_close_thresh = 0.23  # meters - threshold for considering drawer closed
        
        return abs(rel_pos[1]) > drawer_close_thresh
    
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
        drawer_open_thresh = 0.05  # meters - threshold for considering drawer open
        return abs(rel_pos[1]) < drawer_open_thresh
    
    @classmethod
    def _InContainer_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is in container."""
        obj, container = objects
        obj_pos = state.get(obj, "translation")
        obj_quat = state.get(obj, "quaternion")
        container_pos = state.get(container, "translation")
        container_quat = state.get(container, "quaternion")
        obj_pos_in_container, _ = frame_transform(obj_pos, obj_quat, container_pos, R.from_quat(container_quat).as_matrix())
        in_container = 0.0 <= obj_pos_in_container[2] <= cls.place_close_z_thresh
        in_container_region = abs(obj_pos_in_container[0]) <= 0.1 and abs(obj_pos_in_container[1]) <= 0.1
        return in_container and in_container_region
    
    @classmethod
    def _LidOnDishrack_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if lid is on dishrack."""
        return False # hardware, so we can run this task without checking this predicate
    @classmethod
    def _InCookware_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is in cookware."""
        return False # hardware, so we can run this task without checking this predicate

    @classmethod
    def _PourInPan_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object is in pan."""
        return False

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

    def save_episode_video(self, filename="episode.mp4"):
        """Save the collected video frames as a video and clear the buffer."""
        if not self._using_gui and self._video_frames:
            from predicators import utils
            utils.save_video(filename, self._video_frames)
            self._video_frames = []
            self._frame_counter = 0

    def _get_geom_bbox_points_from_indices(self, geom_indices: List[int], ignore_handle: bool = True):
        """Return 8 world-coordinate bounding box corner points for the given geom indices.

        The bounding box is computed as an axis-aligned bounding box that encloses
        all specified geoms. This works for visualization purposes
        and does not assume any specific geom type (box / sphere / cylinder).
        Returns None if no valid geoms are provided.
        """
        if len(geom_indices) == 0:
            return None

        model = self._env_raw.sim.model
        data = self._env_raw.sim.data

        min_xyz = np.array([np.inf, np.inf, np.inf])
        max_xyz = np.array([-np.inf, -np.inf, -np.inf])

        for g in geom_indices:
            # Option A: skip handle-related geoms so bounding box thickness is not inflated
            if ignore_handle:
                g_name = model.geom_id2name(g)
                body_name_of_geom = model.body_id2name(model.geom_bodyid[g])
                if (g_name and "handle" in g_name.lower()) or (
                    body_name_of_geom and "handle" in body_name_of_geom.lower()
                ):
                    continue

            size = model.geom_size[g].copy()

            geom_type = int(model.geom_type[g])  # 2: sphere, 3: capsule, 4: cylinder, 5: box, etc.

            if geom_type == 2:  # sphere, size[0] = radius
                r = size[0]
                extents = np.array([r, r, r])
            elif geom_type in (3, 4):  # capsule or cylinder
                r = size[0]
                half_len = size[1]
                # Long axis (x) length = half_len + r (radius adds to each side)
                extents = np.array([half_len + r, r, r])
            else:  # box or other types assume extents directly in size
                # For boxes, MuJoCo stores half-size extents already
                sx = size[0]
                sy = size[1] if size[1] > 0 else sx
                sz = size[2] if size[2] > 0 else sx
                extents = np.array([sx, sy, sz])

            xpos = data.geom_xpos[g]
            xmat = data.geom_xmat[g].reshape(3, 3)

            # Iterate over 8 corner combinations
            for dx in (-1, 1):
                for dy in (-1, 1):
                    for dz in (-1, 1):
                        local_point = np.array([dx * extents[0], dy * extents[1], dz * extents[2]])
                        world_point = xpos + xmat.dot(local_point)
                        min_xyz = np.minimum(min_xyz, world_point)
                        max_xyz = np.maximum(max_xyz, world_point)

        minx, miny, minz = min_xyz
        maxx, maxy, maxz = max_xyz

        bbox_points = [
            np.array([minx, miny, minz]),
            np.array([maxx, miny, minz]),
            np.array([minx, maxy, minz]),
            np.array([minx, miny, maxz]),
            np.array([maxx, maxy, maxz]),
            np.array([minx, maxy, maxz]),
            np.array([maxx, miny, maxz]),
            np.array([maxx, maxy, minz]),
        ]

        return bbox_points

    def _get_geom_bbox_points(self, geom_name: str, ignore_handle: bool = True):
        """Return 8 world-coordinate bounding box corner points for a single geom.

        Args:
            geom_name: Name of the geom to get bounding box for
            ignore_handle: Whether to skip handle-related geoms (not applicable for single geom)
        
        Returns:
            List of 8 corner points or None if geom not found
        """
        try:
            model = self._env_raw.sim.model
            geom_id = model.geom_name2id(geom_name)
            return self._get_geom_bbox_points_from_indices([geom_id], ignore_handle)
        except Exception:
            logging.warning(f"Geom {geom_name} not found in simulation model")
            return None

    def _get_body_bbox_points(self, body_name: str, ignore_handle: bool = True):
        """Return 8 world-coordinate bounding box corner points for the given MuJoCo body.

        The bounding box is computed as an axis-aligned bounding box that encloses
        all geoms that belong to the body. This works for visualization purposes
        and does not assume any specific geom type (box / sphere / cylinder).
        Returns None if the body is not found or contains no geoms.
        """
        try:
            model = self._env_raw.sim.model
            target_body_id = model.body_name2id(body_name)
        except Exception:
            logging.warning(f"Body {body_name} not found in simulation model")
            return None

        # ------------------------------------------------------------------
        # Gather all descendant body ids (including the target body itself).
        # MuJoCo keeps a tree of bodies, where model.body_parentid gives the
        # parent of a body (root's parent is -1). We include a geom if the
        # body it is attached to is the target body or lies in its subtree.
        # ------------------------------------------------------------------
        parent = model.body_parentid

        def _is_descendant(child_id: int, ancestor_id: int) -> bool:
            """Return True iff ancestor_id is on the path from child to root.

            Handles malformed parent arrays where the root body's parent id equals
            itself (common in some MuJoCo models) to avoid infinite loops.
            """
            visited = set()
            while child_id != -1:
                if child_id == ancestor_id:
                    return True
                if child_id in visited:
                    # Cycle detected (e.g.
                    # parent[child_id] == child_id). Break to avoid infinite loop.
                    break
                visited.add(child_id)
                next_id = parent[child_id]
                if next_id == child_id:
                    # Reached a self-parenting root
                    break
                child_id = next_id
            return False

        geom_indices = [
            g for g in range(model.ngeom) if _is_descendant(model.geom_bodyid[g], target_body_id)
        ]

        if len(geom_indices) == 0:
            logging.warning(f"No geoms found for body {body_name} (including descendants)")
            return None

        return self._get_geom_bbox_points_from_indices(geom_indices, ignore_handle)


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

