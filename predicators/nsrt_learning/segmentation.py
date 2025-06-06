"""Methods for segmenting low-level trajectories into segments."""

from typing import Callable, List, Optional, Set, Tuple
import time

import numpy as np
from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, Predicate, Segment, State
from robocasa.scripts.playback_dataset import reset_to
from predicators.envs.robo_kitchen import RoboKitchenEnv


def segment_trajectory(
    ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: Optional[List[Set[GroundAtom]]] = None, low_speed_only: bool = False, low_speed_threshold: float = 0.001
) -> List[Segment]:
    """Segment a ground atom trajectory."""
    # Start with the segmenters that don't need atom_seq. Still pass it in
    # because if it was provided, it can be used to avoid calling abstract.
    start_time = time.time()
    if CFG.segmenter == "option_changes":
        return _segment_with_option_changes(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "every_step":
        return _segment_with_switch_function(ll_traj, predicates, atom_seq, lambda _: True)
    # All segmenters below need atom_seq. Create it if it wasn't passed in.
    if atom_seq is None:
        atom_seq = [utils.abstract(s, predicates) for s in ll_traj.states]
    if CFG.segmenter == "atom_changes":
        return _segment_with_atom_changes(ll_traj, predicates, atom_seq, low_speed_only, low_speed_threshold)
    if CFG.segmenter == "atom_changes_add_effects_only":
        return _segment_with_atom_changes_add_effects_only(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "atom_changes_low_speed_check":
        # The new segmenter inherently uses low speed checks.
        # Pass the threshold.
        return _segment_with_atom_changes_low_speed_check(ll_traj, predicates, atom_seq, low_speed_threshold)
    if CFG.segmenter == "oracle":
        return _segment_with_oracle(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "contacts":
        return _segment_with_contact_changes(ll_traj, predicates, atom_seq)
    print(f"Segmentation took {time.time() - start_time:.2f} seconds.")
    raise NotImplementedError(f"Unrecognized segmenter: {CFG.segmenter}.")

def _segment_with_atom_changes_add_effects_only(ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory based on atom changes, but only considering add effects."""
    def _switch_fn(t: int) -> bool:
        return not atom_seq[t + 1].issubset(atom_seq[t])
    return _segment_with_switch_function(ll_traj, predicates, atom_seq, _switch_fn)

def _segment_with_atom_changes(
    ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: List[Set[GroundAtom]], low_speed_only: bool = False, low_speed_threshold: float = 0.001
) -> List[Segment]:
    """Segment a trajectory whenever the abstract state changes."""

    def _switch_fn(t: int) -> bool:
        return atom_seq[t] != atom_seq[t + 1]

    if low_speed_only:

        def switch_fn_with_goal(t: int) -> Tuple[bool, bool]:
            return atom_seq[t] != atom_seq[t + 1], list(predicates)[-1] in atom_seq[t]

        return _segment_with_switch_function(ll_traj, predicates, atom_seq, switch_fn_with_goal, low_speed_only, low_speed_threshold)
    else:

        return _segment_with_switch_function(ll_traj, predicates, atom_seq, _switch_fn, low_speed_only, low_speed_threshold)


def _segment_with_contact_changes(ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory based on contact changes.

    Since environments do not expose contacts, this is implemented in an
    environment-specific way. We assume that some predicates represent
    contacts and we look for changes in those contact predicates.
    """

    if CFG.env == "robo_kitchen":
        keep_pred_names = {"InContact"}
    elif CFG.env == "stick_button":
        keep_pred_names = {"Grasped", "Pressed"}
    elif CFG.env in ("cover", "cover_multistep_options", "pybullet_cover"):
        keep_pred_names = {"Covers", "HandEmpty", "Holding"}
    elif CFG.env in ("blocks", "pybullet_blocks"):
        keep_pred_names = {"Holding", "On", "OnTable"}
    elif CFG.env == "doors":
        keep_pred_names = {"TouchingDoor", "InRoom"}
    elif CFG.env == "touch_point":
        keep_pred_names = {"Touched"}
    elif CFG.env == "coffee":
        keep_pred_names = {"Holding", "HandEmpty", "MachineOn", "CupFilled"}
    elif CFG.env == "exit_garage":
        keep_pred_names = {"ObstacleCleared", "CarHasExited"}
    else:
        raise NotImplementedError("Contact-based segmentation not implemented " f"for environment {CFG.env}.")
    include_last_segment = False

    env = get_or_create_env(CFG.env)
    keep_preds = {p for p in env.predicates if p.name in keep_pred_names}
    assert len(keep_preds) == len(keep_pred_names)
    all_keep_atoms = []
    for state in ll_traj.states:
        all_keep_atoms.append(utils.abstract(state, keep_preds))
        # print(state.items_in_contact)

    def _switch_fn(t: int) -> bool:
        return all_keep_atoms[t] != all_keep_atoms[t + 1]

    return _segment_with_switch_function(ll_traj, predicates, atom_seq, _switch_fn, include_last_segment)


def _segment_with_option_changes(ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: Optional[List[Set[GroundAtom]]]) -> List[Segment]:
    """Segment a trajectory whenever the (assumed known) option changes."""

    def _switch_fn(t: int) -> bool:
        # Segment by checking whether the option changes on the next step.
        option_t = ll_traj.actions[t].get_option()
        # As a special case, if this is the last timestep, then use the
        # option's terminal function to check if it completed, or see if the
        # termination was due to max_num_steps_option_rollout.
        if t == len(ll_traj.actions) - 1:
            # Calculate the number of steps since the option changed.
            backward_t = t
            while backward_t > 0:
                if ll_traj.actions[backward_t - 1].get_option() is not option_t:
                    break
                backward_t -= 1
            option_duration = t - backward_t + 1
            if option_duration >= CFG.max_num_steps_option_rollout:
                return True
            return option_t.terminal(ll_traj.states[t + 1])
        return option_t is not ll_traj.actions[t + 1].get_option()

    return _segment_with_switch_function(ll_traj, predicates, atom_seq, _switch_fn)


def _segment_with_oracle(ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory using oracle NSRTs.

    If options are known, just uses _segment_with_option_changes().

    Otherwise, starting at the beginning of the trajectory, keeps track of
    which oracle ground NSRTs are applicable. When any of them have their
    effects achieved, that marks the switch point between segments.
    """
    if ll_traj.actions and ll_traj.actions[0].has_option():
        assert CFG.option_learner == "no_learning"
        return _segment_with_option_changes(ll_traj, predicates, atom_seq)
    env = get_or_create_env(CFG.env)
    env_options = get_gt_options(env.get_name())
    gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
    objects = list(ll_traj.states[0])
    ground_nsrts = {ground_nsrt for nsrt in gt_nsrts for ground_nsrt in utils.all_ground_nsrts(nsrt, objects)}
    atoms = atom_seq[0]
    all_expected_next_atoms = [utils.apply_operator(n, atoms) for n in utils.get_applicable_operators(ground_nsrts, atoms)]

    def _switch_fn(t: int) -> bool:
        nonlocal all_expected_next_atoms  # update at each switch point
        next_atoms = atom_seq[t + 1]
        # Check if any of the current NSRT effects hold.
        for expected_next_atoms in all_expected_next_atoms:
            # Check if we have reached the expected next atoms.
            if expected_next_atoms != next_atoms:
                continue
            # Time to segment. Update the expected next atoms.
            applicable_nsrts = utils.get_applicable_operators(ground_nsrts, next_atoms)
            all_expected_next_atoms = [utils.apply_operator(n, next_atoms) for n in applicable_nsrts]
            return True
        # Not yet time to segment.
        return False

    return _segment_with_switch_function(ll_traj, predicates, atom_seq, _switch_fn)


def _segment_with_switch_function(
    ll_traj: LowLevelTrajectory,
    predicates: Set[Predicate],
    atom_seq: Optional[List[Set[GroundAtom]]],
    switch_fn: Callable[[int], bool],
    low_speed_only: bool = False,
    low_speed_threshold: float = 0.001,
    include_last_segment: bool = False,
) -> List[Segment]:
    """Helper for other segmentation methods.

    The switch_fn takes in a timestep and returns True if the trajectory
    should be segmented at the end of that timestep.
    """
    segments = []
    assert len(ll_traj.states) > 0
    current_segment_states: List[State] = []
    current_segment_actions: List[Action] = []
    if atom_seq is not None:
        assert len(ll_traj.states) == len(atom_seq)
        current_segment_init_atoms = atom_seq[0]
    else:
        s0 = ll_traj.states[0]
        current_segment_init_atoms = utils.abstract(s0, predicates)

    t_last_switch = -1
    for t in range(len(ll_traj.actions)):
        current_segment_states.append(ll_traj.states[t])
        current_segment_actions.append(ll_traj.actions[t])
        if low_speed_only:
            switch_fn_result, goal_reached = switch_fn(t)
        else:
            switch_fn_result = switch_fn(t)
        if switch_fn_result:
            if low_speed_only:
                # gripper_obj = list(ll_traj.states[0].get_objects(gripper))
                if t > 0:
                    gripper = RoboKitchenEnv.gripper_type
                    gripper_obj = list(ll_traj.states[t - 1].get_objects(gripper))[0]
                    # gripper_obj_prev = list(ll_traj.states[t].get_objects(gripper))
                    gripper_obj_prev = ll_traj.states[t - 1].get(gripper_obj, "translation")
                    gripper_obj_curr = ll_traj.states[t].get(gripper_obj, "translation")
                    # get distance between gripper_obj and gripper_obj_prev
                    dist = np.linalg.norm(gripper_obj_curr - gripper_obj_prev)
                    if dist > low_speed_threshold and not goal_reached:  # 1cm /sec since 10 Hz
                        continue
            # Include the final state as the end of this segment.
            t_last_switch = t
            current_segment_states.append(ll_traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states, current_segment_actions, _train_task_idx=ll_traj._train_task_idx)
            if atom_seq is not None:
                current_segment_final_atoms = atom_seq[t + 1]
                # Compute maintain_atoms: intersection of all atom sets in the segment
                segment_atom_sets = atom_seq[(t - len(current_segment_states) + 2):(t + 1)] 
                # we know there is a change at t+2, so maitain effects are valid until t+1
                if segment_atom_sets:
                    maintain_atoms = set.intersection(*segment_atom_sets)
                else:
                    maintain_atoms = set()
            else:
                st1 = ll_traj.states[t + 1]
                current_segment_final_atoms = utils.abstract(st1, predicates)
                # If atom_seq is None, we can't compute maintain_atoms
                maintain_atoms = None
            if ll_traj.actions[t].has_option():
                segment = Segment(current_segment_traj, current_segment_init_atoms, current_segment_final_atoms, ll_traj.actions[t].get_option())
            else:
                # If we're in option learning mode, include the default option
                # here; replaced later during option learning.
                segment = Segment(current_segment_traj, current_segment_init_atoms, current_segment_final_atoms, maintain_atoms=maintain_atoms)
            # Set maintain_atoms on the segment
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
            current_segment_init_atoms = current_segment_final_atoms

    # robocasa needs last segment to learn
    if include_last_segment and t_last_switch != len(ll_traj.actions) - 1:
        current_segment_states = []
        current_segment_actions = []
        current_final_atoms = set()  # not changed so mark as empty

        for t in range(t_last_switch + 1, len(ll_traj.actions)):
            current_segment_states.append(ll_traj.states[t])
            current_segment_actions.append(ll_traj.actions[t])
        current_segment_states.append(ll_traj.states[-1])
        current_segment_traj = LowLevelTrajectory(current_segment_states, current_segment_actions, _train_task_idx=ll_traj._train_task_idx)
        segments.append(Segment(current_segment_traj, current_segment_init_atoms, current_final_atoms))
    # Don't include the last segment because it didn't result in a switch.
    # E.g., with option_changes, the option may not have terminated.
    return segments


def _segment_with_atom_changes_low_speed_check(
    ll_traj: LowLevelTrajectory, predicates: Set[Predicate], atom_seq: List[Set[GroundAtom]], low_speed_threshold: float  # Keep predicates for potential future use/consistency?
) -> List[Segment]:
    """Segment a trajectory based on atom changes checked only when gripper
    speed is low compared to the previous timestep.

    Specifically, segment at time t if:
    1. speed(t) <= low_speed_threshold
    2. atom_seq[t+1] != atom_seq[start_of_segment]
    """
    segments = []
    assert len(ll_traj.states) > 0
    assert len(ll_traj.states) == len(atom_seq)

    current_segment_states: List[State] = []
    current_segment_actions: List[Action] = []
    current_segment_init_atoms = atom_seq[0]
    # This currently assumes RoboKitchenEnv for gripper checks.
    try:
        gripper = RoboKitchenEnv.gripper_type
    except AttributeError:
        raise NotImplementedError("Low speed segmentation currently only supports RoboKitchenEnv")

    debug = False
    have_segmented = False
    have_ended = False
    for t in range(len(ll_traj.actions)):
        current_segment_states.append(ll_traj.states[t])
        current_segment_actions.append(ll_traj.actions[t])

        should_segment = False
        # Check speed only if we have previous state (t > 0)
        if t > 0:
            if debug:
                if not have_segmented and t > 140:
                    have_segmented = True
                    should_segment = True
                if not have_ended and t == len(ll_traj.actions) - 1:
                    have_ended = True
                    should_segment = True
            else:
                try:
                    # Get gripper object - assumes one gripper object exists
                    gripper_objs_prev = list(ll_traj.states[t - 1].get_objects(gripper))
                    gripper_objs_curr = list(ll_traj.states[t].get_objects(gripper))

                    if not gripper_objs_prev or not gripper_objs_curr:
                        raise ValueError("Gripper object not found.")

                    gripper_obj_prev = gripper_objs_prev[0]
                    gripper_obj_curr = gripper_objs_curr[0]  # Use current state's gripper object ref

                    gripper_pos_prev = ll_traj.states[t - 1].get(gripper_obj_prev, "translation")
                    gripper_pos_curr = ll_traj.states[t].get(gripper_obj_curr, "translation")
                    dist = np.linalg.norm(gripper_pos_curr - gripper_pos_prev)

                    if dist <= low_speed_threshold:
                        # Speed is low, now check if atoms changed since segment start
                        next_atoms = atom_seq[t + 1]
                        if next_atoms != current_segment_init_atoms:
                            should_segment = True
                except (IndexError, KeyError, ValueError) as e:
                    # Handle cases where gripper object or its translation might not be found
                    print(f"Warning: Gripper speed check failed at timestep {t}: {e}")
                    pass  # Don't segment if speed check fails

        if should_segment:
            # Include the final state as the end of this segment.
            current_segment_states.append(ll_traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states, current_segment_actions)
            # Use the atoms from t+1 as the final atoms for this segment
            current_segment_final_atoms = atom_seq[t + 1]

            if ll_traj.actions[t].has_option():
                # This case might be less relevant if options aren't used with this segmenter,
                # but handle it for completeness.
                segment = Segment(current_segment_traj, current_segment_init_atoms, current_segment_final_atoms, ll_traj.actions[t].get_option())
            else:
                segment = Segment(current_segment_traj, current_segment_init_atoms, current_segment_final_atoms)

            segments.append(segment)
            # Reset for next segment
            current_segment_states = []
            current_segment_actions = []
            current_segment_init_atoms = current_segment_final_atoms

    # Note: Unlike _segment_with_switch_function, trajectories ending without
    # a low-speed atom change don't create a final segment.

    return segments
