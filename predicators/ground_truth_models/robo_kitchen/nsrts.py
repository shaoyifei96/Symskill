"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, Type, Variable


class RoboKitchenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the RoboKitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"robo_kitchen"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type], predicates: Dict[str, Predicate], options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        gripper_type = types["gripper_type"]
        left_finger_type = types["left_finger_type"]
        right_finger_type = types["right_finger_type"]
        cabinet_type = types["cabinet_type"]
        handle_type = types["handle_type"]
        base_type = types["base_type"]
        thing_type = types["thing_type"]
        surface_type = types["surface_type"]
        object_type = types["object_type"]
        door_type = types["door_type"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        left_finger = Variable("?left_finger", left_finger_type)
        right_finger = Variable("?right_finger", right_finger_type)
        cabinet = Variable("?cabinet", cabinet_type)
        handle = Variable("?handle", handle_type)
        base = Variable("?base", base_type)
        thing = Variable("?thing", thing_type)
        surface = Variable("?surface", surface_type)
        obj = Variable("?object", object_type)
        door = Variable("?door", door_type)

        # Options
        DS_OpenSingleDoor_MoveTowards_option = options["DS_OpenSingleDoor_MoveTowards_option"]
        DS_OpenSingleDoor_MoveAway_option = options["DS_OpenSingleDoor_MoveAway_option"]
        GripperOpen_option = options["GripperOpen_option"]
        GripperClose_option = options["GripperClose_option"]
        DummyOption = options["DummyOption"]
        ReachBehindandPull_option = options["ReachBehindandPull_option"]
        PnPCounterToCab_Pick_option = options["PnPCounterToCab_Pick_option"]
        PnPCounterToCab_Place_option = options["PnPCounterToCab_Place_option"]
        MoveToInitPoseOption = options["MoveToInitPoseOption"]
        # Predicates
        ReadyGrabObj = predicates["ReadyGrabObj"]
        GripperClosed = predicates["GripperClosed"]
        GripperOpen = predicates["GripperOpen"]
        DoorClosed = predicates["DoorClosed"]
        DoorOpen = predicates["DoorOpen"]
        InContact = predicates["InContact"]
        DoorHalfOpen = predicates["DoorHalfOpen"]
        OnSurface = predicates["OnSurface"]
        InOrigin = predicates["InOrigin"]

        nsrts = set()

        # ToInitialState
        parameters = [gripper, base]
        preconditions = set()
        maintain_effects = set()
        # add_effects = {LiftedAtom(InOrigin, [gripper, base])}
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()
        option = MoveToInitPoseOption
        option_vars = [gripper, base]

        def to_initial_state_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)
        to_initial_state_nsrt = NSRT(
            "ToInitialState",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            to_initial_state_sampler,
            maintain_effects,
        )

        # ReachBehindandPull
        parameters = [gripper, door, handle, base, cabinet, left_finger, right_finger]
        preconditions = {LiftedAtom(DoorHalfOpen, [handle, cabinet]), LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {LiftedAtom(DoorOpen, [door, cabinet])}
        # add_effects = set() # NOTE: this avoids using reach_behind_and_pull_option
        delete_effects = {LiftedAtom(DoorHalfOpen, [handle, cabinet])}
        ignore_effects = set()
        option = ReachBehindandPull_option
        option_vars = [gripper, handle, base]

        def reach_behind_and_pull_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        reach_behind_and_pull_nsrt = NSRT(
            "ReachBehindandPull",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            reach_behind_and_pull_sampler,
            maintain_effects,
        )

        # OpenGripper
        parameters = [left_finger, right_finger]
        preconditions = set()
        maintain_effects = set()
        add_effects = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        delete_effects = {LiftedAtom(GripperClosed, [left_finger, right_finger])}
        ignore_effects = set()
        option = GripperOpen_option
        option_vars = [left_finger, right_finger]

        def open_gripper_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        open_gripper_nsrt = NSRT(
            "OpenGripper",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            open_gripper_sampler,
            maintain_effects,
        )

        # MoveToHandle
        parameters = [gripper, handle, base, left_finger, right_finger]
        preconditions = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        add_effects = {LiftedAtom(ReadyGrabObj, [gripper, handle])}
        delete_effects = set()
        ignore_effects = set()
        option = DS_OpenSingleDoor_MoveTowards_option
        option_vars = [gripper, handle, base]

        def move_to_handle_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        move_to_handle_nsrt = NSRT(
            "MoveToHandle",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            move_to_handle_sampler,
            maintain_effects,
        )

        # MoveToThing
        parameters = [gripper, thing, base, left_finger, right_finger]
        preconditions = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        add_effects = {LiftedAtom(ReadyGrabObj, [gripper, thing])}
        delete_effects = set()
        ignore_effects = set()
        option = PnPCounterToCab_Pick_option
        option_vars = [gripper, thing, base]

        def move_to_thing_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        move_to_thing_nsrt = NSRT(
            "MoveToThing",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            move_to_thing_sampler,
            maintain_effects,
        )

        # GrabObj
        parameters = [gripper, obj, left_finger, right_finger]
        preconditions = {LiftedAtom(ReadyGrabObj, [gripper, obj]), LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {
            LiftedAtom(GripperClosed, [left_finger, right_finger]),
            LiftedAtom(InContact, [gripper, obj]),
        }
        delete_effects = {
            # LiftedAtom(ReadyGrabObj, [gripper, obj]),
            LiftedAtom(GripperOpen, [left_finger, right_finger]),
        }
        ignore_effects = set()
        option = GripperClose_option
        option_vars = [left_finger, right_finger]

        def grab_obj_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        grab_obj_nsrt = NSRT(
            "GrabObj",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            grab_obj_sampler,
            maintain_effects,
        )

        # MoveToAndGrabHandle
        parameters = [gripper, handle, base, left_finger, right_finger]
        preconditions = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {
            LiftedAtom(GripperClosed, [left_finger, right_finger]), 
            # LiftedAtom(InContact, [gripper, handle]),
            LiftedAtom(ReadyGrabObj, [gripper, handle]),
        }
        delete_effects = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        ignore_effects = set()
        option = DummyOption
        option_vars = []

        def move_to_and_grab_handle_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        move_to_and_grab_handle_nsrt = NSRT(
            "MoveToAndGrabHandle",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            move_to_and_grab_handle_sampler,
            maintain_effects,   
        )

        # PullOpenDoor
        parameters = [gripper, door, handle, cabinet, base, left_finger, right_finger]
        preconditions = {
            LiftedAtom(DoorClosed, [door, cabinet]),
            LiftedAtom(GripperClosed, [left_finger, right_finger]),
            # LiftedAtom(InContact, [gripper, handle]),
            LiftedAtom(ReadyGrabObj, [gripper, handle]),
        }
        maintain_effects = {
            # LiftedAtom(InContact, [gripper, handle]),
            LiftedAtom(ReadyGrabObj, [gripper, handle]),
            LiftedAtom(GripperClosed, [left_finger, right_finger]),
        }
        add_effects = {LiftedAtom(DoorOpen, [door, cabinet])}
        delete_effects = {LiftedAtom(DoorClosed, [door, cabinet])}
        ignore_effects = set()
        option = DS_OpenSingleDoor_MoveAway_option
        option_vars = [gripper, handle, base]

        def pull_open_door_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        pull_open_door_nsrt = NSRT(
            "PullOpenDoor",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            pull_open_door_sampler,
            maintain_effects,
        )

        # PlaceThingOnSurface
        parameters = [gripper, thing, surface, base, left_finger, right_finger]
        preconditions = {
            LiftedAtom(ReadyGrabObj, [gripper, thing]),
            LiftedAtom(GripperClosed, [left_finger, right_finger]), 
        }
        maintain_effects = {
            LiftedAtom(ReadyGrabObj, [gripper, thing]),
            LiftedAtom(GripperClosed, [left_finger, right_finger]),
        }
        add_effects = {LiftedAtom(OnSurface, [thing, surface])}
        delete_effects = set()
        ignore_effects = set()
        option = PnPCounterToCab_Place_option
        option_vars = [gripper, surface, base]

        def place_thing_on_surface_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        place_thing_on_surface = NSRT(
            "PlaceThingOnSurface",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            place_thing_on_surface_sampler,
            maintain_effects,   
        )

        # nsrts.add(open_gripper_nsrt)
        # # nsrts.add(move_to_and_grab_handle_nsrt)  # open_gripper + move_to_handle + grab_handle
        # nsrts.add(grab_obj_nsrt)
        # nsrts.add(move_to_handle_nsrt)
        # nsrts.add(move_to_thing_nsrt)
        # nsrts.add(pull_open_door_nsrt)
        # # nsrts.add(reach_behind_and_pull_nsrt)
        # nsrts.add(place_thing_on_surface)
        nsrts.add(to_initial_state_nsrt)

        return nsrts
