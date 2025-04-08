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
        door_type = types["door_type"]
        handle_type = types["handle_type"]
        base_type = types["base_type"]
        object_type = types["object_type"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        left_finger = Variable("?left_finger", left_finger_type)
        right_finger = Variable("?right_finger", right_finger_type)
        cabinet = Variable("?cabinet", cabinet_type)
        door = Variable("?door", door_type)
        handle = Variable("?handle", handle_type)
        base = Variable("?base", base_type)

        # Options
        DS_move_towards_option = options["DS_move_towards_option"]
        DS_move_away_option = options["DS_move_away_option"]
        GripperOpen_option = options["GripperOpen_option"]
        GripperClose_option = options["GripperClose_option"]
        DummyOption = options["DummyOption"]
        ReachBehindandPull_option = options["ReachBehindandPull_option"]
        # Predicates
        ReadyGrabHandle = predicates["ReadyGrabHandle"]
        GripperClosed = predicates["GripperClosed"]
        GripperOpen = predicates["GripperOpen"]
        HingeClosed = predicates["HingeClosed"]
        HingeOpen = predicates["HingeOpen"]
        InContact = predicates["InContact"]
        DoorHalfOpen = predicates["DoorHalfOpen"]

        nsrts = set()

        # ReachBehindandPull
        parameters = [gripper, handle, base, door, cabinet, left_finger, right_finger]
        preconditions = {LiftedAtom(DoorHalfOpen, [door, cabinet]), LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {LiftedAtom(HingeOpen, [door, cabinet])}
        # add_effects = set() # NOTE: this avoids using reach_behind_and_pull_option
        delete_effects = {LiftedAtom(DoorHalfOpen, [door, cabinet])}
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
        nsrts.add(reach_behind_and_pull_nsrt)


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
        nsrts.add(open_gripper_nsrt)

        # MoveToHandle
        parameters = [gripper, handle, base, left_finger, right_finger]
        preconditions = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        add_effects = {LiftedAtom(ReadyGrabHandle, [gripper, handle])}
        delete_effects = set()
        ignore_effects = set()
        option = DS_move_towards_option
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
        nsrts.add(move_to_handle_nsrt)

        # GrabHandle
        parameters = [gripper, handle, left_finger, right_finger]
        preconditions = {LiftedAtom(ReadyGrabHandle, [gripper, handle]), LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {
            LiftedAtom(GripperClosed, [left_finger, right_finger]), 
            LiftedAtom(InContact, [gripper, handle]),
        }
        delete_effects = {LiftedAtom(ReadyGrabHandle, [gripper, handle]), LiftedAtom(GripperOpen, [left_finger, right_finger])}
        ignore_effects = set()
        option = GripperClose_option
        option_vars = [left_finger, right_finger]

        def grab_handle_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        grab_handle_nsrt = NSRT(
            "GrabHandle",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            grab_handle_sampler,
            maintain_effects,
        )
        nsrts.add(grab_handle_nsrt)

        # MoveToAndGrabHandle
        parameters = [gripper, handle, base, left_finger, right_finger]
        preconditions = {LiftedAtom(GripperOpen, [left_finger, right_finger])}
        maintain_effects = set()
        add_effects = {
            LiftedAtom(GripperClosed, [left_finger, right_finger]), 
            LiftedAtom(InContact, [gripper, handle]),
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
        # nsrts.add(move_to_and_grab_handle_nsrt)

        # PullOpenDoor
        parameters = [gripper, handle, door, cabinet, base, left_finger, right_finger]
        preconditions = {
            LiftedAtom(HingeClosed, [door, cabinet]), 
            LiftedAtom(GripperClosed, [left_finger, right_finger]), 
            LiftedAtom(InContact, [gripper, handle]),
        }
        maintain_effects = {
            LiftedAtom(InContact, [gripper, handle]),
        }
        add_effects = {LiftedAtom(HingeOpen, [door, cabinet])}
        delete_effects = {LiftedAtom(HingeClosed, [door, cabinet])}
        ignore_effects = set()
        option = DS_move_away_option
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
        nsrts.add(pull_open_door_nsrt)

        return nsrts
