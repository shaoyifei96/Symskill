"""Utilities for working with demo data for environment reset and planning."""

import logging
import os
from typing import Optional, Tuple

import dill as pkl

from predicators.settings import CFG
from predicators.structs import Dataset, LowLevelTrajectory, State


def load_demo_dataset(dataset_path: Optional[str] = None) -> Optional[Dataset]:
    """Load demo dataset from disk.
    
    Args:
        dataset_path: Path to dataset file. If None, uses CFG.demo_reset_dataset_path
        
    Returns:
        Dataset object or None if loading fails
    """
    if dataset_path is None:
        dataset_path = CFG.demo_reset_dataset_path
        
    if dataset_path is None:
        logging.warning("No dataset path provided for demo reset")
        return None
        
    if not os.path.exists(dataset_path):
        logging.warning(f"Demo dataset not found at {dataset_path}")
        return None
        
    try:
        with open(dataset_path, "rb") as f:
            dataset = pkl.load(f)
        logging.info(f"Loaded demo dataset with {len(dataset.trajectories)} trajectories from {dataset_path}")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load demo dataset from {dataset_path}: {e}")
        return None


def get_demo_state_at_timestep(demo_dataset: Dataset, 
                              task_idx: int, 
                              timestep: int) -> Optional[State]:
    """Get the state from a demo trajectory at a specific timestep.
    
    Args:
        demo_dataset: Dataset containing demo trajectories
        task_idx: Index of the demo trajectory to use
        timestep: Timestep within the trajectory
        
    Returns:
        State object at the specified timestep, or None if invalid indices
    """
    if demo_dataset is None:
        logging.warning("Demo dataset is None")
        return None
        
    if task_idx >= len(demo_dataset.trajectories):
        logging.warning(f"Task index {task_idx} out of range for dataset with {len(demo_dataset.trajectories)} trajectories")
        return None
        
    trajectory = demo_dataset.trajectories[task_idx]
    
    if timestep >= len(trajectory.states):
        logging.warning(f"Timestep {timestep} out of range for trajectory with {len(trajectory.states)} states")
        return None
        
    logging.info(f"Retrieved demo state from task {task_idx}, timestep {timestep}")
    return trajectory.states[timestep]


def get_demo_trajectory_info(demo_dataset: Dataset, task_idx: int) -> Optional[Tuple[int, LowLevelTrajectory]]:
    """Get information about a demo trajectory.
    
    Args:
        demo_dataset: Dataset containing demo trajectories
        task_idx: Index of the demo trajectory
        
    Returns:
        Tuple of (trajectory_length, trajectory) or None if invalid index
    """
    if demo_dataset is None:
        return None
        
    if task_idx >= len(demo_dataset.trajectories):
        return None
        
    trajectory = demo_dataset.trajectories[task_idx]
    return len(trajectory.states), trajectory


def validate_demo_reset_config() -> bool:
    """Validate demo reset configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if not CFG.demo_reset_enabled:
        return True
        
    if CFG.demo_reset_dataset_path is None:
        logging.error("demo_reset_enabled is True but demo_reset_dataset_path is None")
        return False
        
    if CFG.demo_reset_task_idx < 0:
        logging.error(f"demo_reset_task_idx must be non-negative, got {CFG.demo_reset_task_idx}")
        return False
        
    if CFG.demo_reset_timestep < 0:
        logging.error(f"demo_reset_timestep must be non-negative, got {CFG.demo_reset_timestep}")
        return False
        
    return True
