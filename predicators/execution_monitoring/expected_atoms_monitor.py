"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met."""

import logging
from typing import Dict, Set, Tuple, Optional
import numpy as np

from predicators import utils
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import State, VLMPredicate, Object


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    def __init__(self) -> None:
        super().__init__()
        # Store failure information: (option_name, objects) -> (failure_pos, failure_count)
        self._failure_memory: Dict[Tuple[str, Tuple[Object, ...]], Tuple[np.ndarray, int]] = {}
        self._last_failed_option: Optional[Tuple[str, Tuple[Object, ...]]] = None
        # Track the current executing option and its start time
        self._running_option_name: str = None
        self._last_option_name: str = None
        self._option_start_timestep: int = 0
        self._max_option_timesteps: int = 200  # Maximum timesteps before considering option failed
        self._current_expected_atom_step = 1

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def step(self, state: State) -> bool:
        """Returns True if the agent should replan."""
        
        # Basic validation checks
        if not self._validate_approach():
            return False

        # Update option tracking
        new_option_bool = self._update_option_tracking()

        # Get next expected atoms and increment timestep
        if len(self._approach_info) <= 1:
            return False
        next_expected_atoms = self._approach_info[self._current_expected_atom_step]
        assert isinstance(next_expected_atoms, set)
        self._curr_plan_timestep += 1

        # Check for timeout
        if self._check_option_timeout():
            return True

        # Check predicates
        unsat_atoms = self._check_predicates(state, next_expected_atoms)

        # Handle option transitions and replanning
        return self._handle_option_transition(new_option_bool, unsat_atoms)

    def _validate_approach(self) -> bool:
        """Validate that we're using a supported planning approach."""
        if self._action is not None and self._action.has_option():
            self._running_option_name = self._action.get_option().name
            
        assert "oracle" in CFG.approach or "active_sampler" in CFG.approach \
            or "maple_q" in CFG.approach or \
            "grammar_search_invention" in CFG.approach
            
        if not self._approach_info:  # pragma: no cover
            return False
        return True

    def _update_option_tracking(self) -> bool:
        """Update tracking of current and previous options. Returns True if option changed."""
        new_option_bool = False
        if self._running_option_name is not None:
            if self._running_option_name != self._last_option_name:
                if self._last_option_name is not None:
                    new_option_bool = True
                self._last_option_name = self._running_option_name
                self._option_start_timestep = self._curr_plan_timestep
                logging.info(f"Starting new option: {self._running_option_name}")
        return new_option_bool

    def _check_option_timeout(self) -> bool:
        """Check if current option has exceeded max timesteps."""
        if self._last_option_name is not None:
            time_in_option = self._curr_plan_timestep - self._option_start_timestep
            if time_in_option > self._max_option_timesteps:
                logging.info(f"Option {self._running_option_name} exceeded max timesteps "
                           f"({time_in_option} > {self._max_option_timesteps})")
                return True
        return False

    def _check_predicates(self, state: State, next_expected_atoms: Set) -> Set:
        """Check which predicates are unsatisfied in current state."""
        next_expected_vlm_atoms = set(
            atom for atom in next_expected_atoms
            if isinstance(atom.predicate, VLMPredicate))
            
        non_vlm_unsat_atoms = {
            a
            for a in (next_expected_atoms - next_expected_vlm_atoms)
            if not a.holds(state)
        }
        
        vlm_unsat_atoms = set()
        if len(next_expected_vlm_atoms) > 0:
            vlm_unsat_atoms = utils.query_vlm_for_atom_vals(
                next_expected_vlm_atoms, state)  # pragma: no cover
                
        return non_vlm_unsat_atoms | vlm_unsat_atoms

    def _handle_option_transition(self, new_option_bool: bool, unsat_atoms: Set) -> bool:
        """Handle option transitions and determine if replanning is needed."""
        if new_option_bool:
            self._current_expected_atom_step += 1
            if unsat_atoms:
                logging.info(
                    "Expected atoms execution monitor triggered replanning "
                    f"because of these atoms: {unsat_atoms}")
                return True
        return False

    def reset(self, task) -> None:
        """Reset the monitor for a new task."""
        super().reset(task)
        self._running_option_name = None
        self._last_option_name = None
        self._option_start_timestep = 0
        self._current_expected_atom_step = 1
        # Note: we don't reset failure memory as we want to keep track across episodes

    def get_failure_memory(self) -> Dict[Tuple[str, Tuple[Object, ...]], Tuple[np.ndarray, int]]:
        """Get the current failure memory."""
        return self._failure_memory

    def get_last_failed_option(self) -> Optional[Tuple[str, Tuple[Object, ...]]]:
        """Get the most recently failed option."""
        return self._last_failed_option
