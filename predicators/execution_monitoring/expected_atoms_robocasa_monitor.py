"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met."""

import logging
from typing import Dict, Set, Tuple, Optional, List
import numpy as np

from predicators import utils
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import State, VLMPredicate, Object, GroundAtom, OptionFailureInfo
from predicators.envs.robo_kitchen import RoboKitchenEnv


class ExpectedAtomsRobocasaExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    def __init__(self) -> None:
        super().__init__()
        self._failure_memory: List[OptionFailureInfo] = []
        # Track the current executing option and its start time
        self._running_option_name: str = None
        self._last_option_name: str = None
        self._option_start_timestep: int = 0
        self._max_option_exe_timesteps: int = 500  # Maximum timesteps before considering option failed
        self._current_nsrt_step = 0
        self._NSRT_plan_executed = False

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms_robocasa"
    
    def _record_failure(self, option_name: str, state: State, reason_of_failure: str) -> None:
        """Record failure."""
        # gripper = RoboKitchenEnv.object_name_to_object("gripper")
        # gripper_state = state.vec([gripper])
        failure_info = OptionFailureInfo(option_name, reason_of_failure, state)
        self._failure_memory.append(failure_info)

    def step(self, state: State) -> bool:
        """Returns True if the agent should replan."""
        
        # Basic validation checks
        if not self._validate_approach():
            return False

        if self._NSRT_plan_executed:
            self._NSRT_plan_executed = False
            failure_reason = self._format_failure_reason("Exhausted", set())
            self._record_failure(self._running_option_name, state, failure_reason)
            return True
        # Update option tracking
        new_option_bool, last_option_name = self._update_option_tracking()

        # Get next expected atoms and increment timestep
        if self._current_nsrt_step + 1 < len(self._approach_info):
            next_expected_atoms, _ = self._approach_info[self._current_nsrt_step + 1]
        else:
            next_expected_atoms = set()
        _, current_maintain_effects = self._approach_info[self._current_nsrt_step]
        assert isinstance(next_expected_atoms, set)
        self._curr_plan_timestep += 1

        # Check for timeout
        if self._check_option_timeout():
            failure_reason = self._format_failure_reason("Timeout", set())
            self._record_failure(self._running_option_name, state, failure_reason)
            return True

        # Check maintain effects
        unsat_maintain_effects = self._check_maintain_effects(state, current_maintain_effects)
        if unsat_maintain_effects:
            failure_reason = self._format_failure_reason("Maintain", unsat_maintain_effects)
            self._record_failure(self._running_option_name, state, failure_reason)
            return True

        # if new option, check new predicates are satisfied
        if new_option_bool:
            unsat_atoms = self._check_predicates(state, next_expected_atoms)
            # it seems in nsrt_plan_to_greedy_policy, there is check for unsat atoms already
            # TODO: test if this function really checks for unsat atoms    # unsat_atoms = {}
            if unsat_atoms:
                failure_reason = self._format_failure_reason("New", unsat_atoms)
                self._record_failure(last_option_name, state, failure_reason)
                return True
            # if no unsat atoms, increment nsrt step, means we're moving to next NSRT
            self._current_nsrt_step += 1
        return False
    
    def _format_failure_reason(self, prefix: str, atoms: Set[GroundAtom]) -> str:
        """Format a failure reason string with a prefix and a set of atoms."""
        failure_reason = f"<{prefix}>:"
        for atom in atoms:
            failure_reason += f"{str(atom)}, "
        logging.info("Exe Monitor failed because of: " + failure_reason)
        return failure_reason

    def _check_maintain_effects(self, state: State, maintain_effects: Set[GroundAtom]) -> Set[GroundAtom]:
        """Check which maintain effects are unsatisfied in current state."""
        # unsat_maintain_effects = {
        #     atom
        #     for atom in maintain_effects
        #     if not atom.holds(state)
        # }
        # if unsat_maintain_effects:
        #     return unsat_maintain_effects
        # return set()
        return self._check_predicates(state, maintain_effects)

    def _validate_approach(self) -> bool:
        """Validate that we're using a supported planning approach."""
        if self._action is not None and self._action.has_option():
            self._running_option_name = self._action.get_option().name
            
        assert "oracle" in CFG.approach or "active_sampler" in CFG.approach \
            or "maple_q" in CFG.approach or \
            "grammar_search_invention" in CFG.approach\
            or "clustering_invention" in CFG.approach
            
            
        if not self._approach_info:  # pragma: no cover
            return False
        return True

    def _update_option_tracking(self) -> bool:
        """Update tracking of current and previous options. Returns True if option changed."""
        new_option_bool = False
        last_option_name = None
        if self._running_option_name is not None:
            if self._running_option_name != self._last_option_name:
                if self._last_option_name is not None:
                    new_option_bool = True
                    last_option_name = self._last_option_name
                self._last_option_name = self._running_option_name
                self._option_start_timestep = self._curr_plan_timestep
                logging.info(f"Starting new option: {self._running_option_name}")
        return new_option_bool, last_option_name

    def _check_option_timeout(self) -> bool:
        """Check if current option has exceeded max timesteps."""
        if self._last_option_name is not None:
            time_in_option = self._curr_plan_timestep - self._option_start_timestep
            if time_in_option > self._max_option_exe_timesteps:
                logging.info(f"Option {self._running_option_name} exceeded max timesteps "
                           f"({time_in_option} > {self._max_option_exe_timesteps})")
                return True
        return False

    def _check_predicates(self, state: State, next_expected_atoms: Set) -> Set:
        """Check which predicates are unsatisfied in current state."""
        next_expected_vlm_atoms = set(
            atom for atom in next_expected_atoms
            if isinstance(atom.predicate, VLMPredicate))
            
        non_vlm_unsat_atoms = set()
        
        
        # converting contact predicates to rel_pose predicates
        for atom in next_expected_atoms - next_expected_vlm_atoms:
            if (atom.predicate.name, atom.entities[0].type.name, atom.entities[1].type.name) in CFG.dict_contact_predicate_to_rel_pose_predicates:
                if not utils.check_dict_contact_predicate_to_rel_pose_predicates(atom, state):
                    non_vlm_unsat_atoms.add(atom)
        
        # {
        #     atom
        #     for atom in (next_expected_atoms - next_expected_vlm_atoms)
        #     if not atom.holds(state)
        # }
        
        vlm_unsat_atoms = set()
        if len(next_expected_vlm_atoms) > 0:
            vlm_unsat_atoms = utils.query_vlm_for_atom_vals(
                next_expected_vlm_atoms, state)  # pragma: no cover
                
        return non_vlm_unsat_atoms | vlm_unsat_atoms


    def reset(self, task, reset_failure_memory: bool = True) -> None:
        """Reset the monitor for a new task."""
        super().reset(task)
        self._running_option_name = None
        self._last_option_name = None
        self._option_start_timestep = 0
        self._current_nsrt_step = 0
        if reset_failure_memory:
            self._failure_memory = []
        # Note: we don't reset failure memory as we want to keep track across episodes

