"""Create offline datasets by collecting demonstrations."""

import functools
import glob
import logging
import os
import re
from typing import Callable, List, Set, Tuple

import dill as pkl
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.cogman import CogMan, run_episode_and_get_states
from predicators.envs import BaseEnv

from predicators.envs.robo_kitchen import RoboKitchenEnv
from robocasa.scripts.playback_dataset import reset_to


from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, \
    ParameterizedOption, State, Task
from robocasa.utils.dataset_registry import get_ds_path
from PIL import Image
import csv


def create_demo_data(env: BaseEnv,
                     known_options: Set[ParameterizedOption],
                     annotate_with_gt_ops: bool,
                     robocasa_task: str = None) -> Dataset:
    """Create offline datasets by collecting demos.
    
    Args:
        env: The environment to load demonstrations for
        train_tasks: List of training tasks
        known_options: Set of known parameterized options
        annotate_with_gt_ops: Whether to annotate with ground truth operators
        robocasa_task: If provided, load demonstrations from robocasa dataset
                      instead of collecting new ones
    """
    if robocasa_task is not None:
        # Try to load cached dataset first if enabled
        if CFG.robo_kitchen_load_dataset:
            dataset_fname = f"generated_datasets/robokitchen__{robocasa_task}__{CFG.num_train_tasks}.pkl"
            if os.path.exists(dataset_fname):
                with open(dataset_fname, "rb") as f:
                    dataset = pkl.load(f)
                return dataset
            else:
                raise ValueError(f"Dataset not found at {dataset_fname}")
        else:
            # Create dataset from appropriate source
            if CFG.robo_kitchen_user_demo and robocasa_task in CFG.mocap_tasks:
                dataset = create_demo_data_from_mocap(env, CFG.path_to_user_demo[robocasa_task], robocasa_task)
            elif CFG.robo_kitchen_user_demo: # sim demo collected using spacemouse
                env._reset_initial_state(seed=0, train_or_test="train", task_name=robocasa_task)
                dataset = create_demo_data_from_user_demo(env, CFG.path_to_user_demo[robocasa_task], robocasa_task)
            else:
                dataset = create_demo_data_from_robocasa(env, known_options, robocasa_task)
        
        # Save dataset if enabled
        if CFG.robo_kitchen_save_dataset:
            dataset_fname = f"generated_datasets/robokitchen__{robocasa_task}__{CFG.num_train_tasks}.pkl"
            with open(dataset_fname, "wb") as f:
                pkl.dump(dataset, f)
        
        return dataset


def _create_demo_data_with_loading(env: BaseEnv, train_tasks: List[Task],
                                   known_options: Set[ParameterizedOption],
                                   dataset_fname_template: str,
                                   dataset_fname: str) -> Dataset:
    """Create demonstration data while handling loading from disk.

    This method takes care of three cases: the demonstrations on disk
    are exactly the desired number, too many, or too few. Note that we
    can only load datasets with annotations of exactly the right size;
    attempting to load annotations for smaller or bigger datasets will
    fail.
    """
    if os.path.exists(dataset_fname):
        # Case 1: we already have a file with the exact name that we need
        # (i.e., the correct amount of data).
        with open(dataset_fname, "rb") as f:
            dataset = pkl.load(f)
        logging.info(f"\n\nLOADED DATASET OF {len(dataset.trajectories)} "
                     "DEMONSTRATIONS")
        return dataset
    fnames_with_less_data = {}  # used later, in Case 3
    for fname in os.listdir(CFG.data_dir):
        regex_match = re.match(dataset_fname_template, fname)
        if not regex_match:
            continue
        num_train_tasks = int(regex_match.groups()[0])
        assert num_train_tasks != CFG.num_train_tasks  # would be Case 1
        # Case 2: we already have a file with MORE data than we need. Load
        # and truncate this data.
        if num_train_tasks > CFG.num_train_tasks:
            with open(os.path.join(CFG.data_dir, fname), "rb") as f:
                dataset = pkl.load(f)
            logging.info("\n\nLOADED AND TRUNCATED DATASET OF "
                         f"{len(dataset.trajectories)} DEMONSTRATIONS")
            assert not dataset.has_annotations
            # To truncate, note that we can't simply take the first
            # `CFG.num_train_tasks` elements of `dataset.trajectories`,
            # because some of these might have a `train_task_idx` that is
            # out of range (if there were errors in the course of
            # collecting those demonstrations). The correct thing to do
            # here is to truncate based on the value of `train_task_idx`.
            return Dataset([
                traj for traj in dataset.trajectories
                if traj.train_task_idx < CFG.num_train_tasks
            ])
        # Save the names of all datasets that have less data than
        # we need, to be used in Case 3.
        fnames_with_less_data[num_train_tasks] = fname
    if not fnames_with_less_data:
        # Give up: we did not find any data file we can load from.
        raise ValueError(f"Cannot load data: {dataset_fname}")
    # Case 3: we already have a file with LESS data than we need. Load
    # this data and generate some more. Specifically, we load from the
    # file with the maximum data among all files that have less data
    # than we need, then we generate the remaining demonstrations.
    train_tasks_start_idx = max(fnames_with_less_data)
    fname = fnames_with_less_data[train_tasks_start_idx]
    with open(os.path.join(CFG.data_dir, fname), "rb") as f:
        dataset = pkl.load(f)
    loaded_trajectories = dataset.trajectories
    generated_dataset = _generate_demonstrations(
        env,
        train_tasks,
        known_options,
        train_tasks_start_idx=train_tasks_start_idx,
        annotate_with_gt_ops=False)
    generated_trajectories = generated_dataset.trajectories
    logging.info(f"\n\nLOADED DATASET OF {len(loaded_trajectories)} "
                 "DEMONSTRATIONS")
    logging.info(
        f"CREATED {len(generated_dataset.trajectories)} DEMONSTRATIONS")
    dataset = Dataset(loaded_trajectories + generated_trajectories)
    with open(dataset_fname, "wb") as f:
        pkl.dump(dataset, f)
    return dataset


def _generate_demonstrations(env: BaseEnv, train_tasks: List[Task],
                             known_options: Set[ParameterizedOption],
                             train_tasks_start_idx: int,
                             annotate_with_gt_ops: bool) -> Dataset:
    """Use the demonstrator to generate demonstrations, one per training task
    starting from train_tasks_start_idx."""
    if CFG.demonstrator == "oracle":
        # Instantiate CogMan with the oracle approach (to be used as the
        # demonstrator). This requires creating a perceiver and
        # execution monitor according to settings from CFG.
        options = get_gt_options(env.get_name())
        oracle_approach = OracleApproach(
            env.predicates,
            options,
            env.types,
            env.action_space,
            train_tasks,
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized,
            bilevel_plan_without_sim=CFG.offline_data_bilevel_plan_without_sim)
        perceiver = create_perceiver(CFG.perceiver)
        execution_monitor = create_execution_monitor(CFG.execution_monitor)
        cogman = CogMan(oracle_approach, perceiver, execution_monitor)
    else:  # pragma: no cover
        # Disable all built-in keyboard shortcuts.
        keymaps = {k for k in plt.rcParams if k.startswith("keymap.")}
        for k in keymaps:
            plt.rcParams[k].clear()
        # Create the environment-specific method for turning events into
        # actions. This should also log instructions.
        event_to_action = env.get_event_to_action_fn()
    trajectories = []
    if annotate_with_gt_ops:
        annotations = []
    num_tasks = min(len(train_tasks), CFG.max_initial_demos)
    for idx, task in enumerate(train_tasks):
        if idx < train_tasks_start_idx:  # ignore demos before this index
            continue
        if CFG.make_demo_videos or CFG.make_demo_images:
            video_monitor = utils.VideoMonitor(env.render)
        else:
            video_monitor = None

        # Note: we assume in main.py that demonstrations are only generated
        # for train tasks whose index is less than CFG.max_initial_demos. If
        # you modify code around here, make sure that this invariant holds.
        if idx >= CFG.max_initial_demos:
            break
        try:
            if CFG.demonstrator == "oracle":
                # In this case, we use the instantiated cogman to generate
                # demonstrations. Importantly, we want to access state-action
                # trajectories, not observation-action ones.
                env_task = env.get_train_tasks()[idx]
                cogman.reset(env_task)
                traj, _, _ = run_episode_and_get_states(
                    cogman,
                    env,
                    "train",
                    idx,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure,
                        utils.HumanDemonstrationFailure,
                    },
                    monitor=video_monitor)
            else:  # pragma: no cover
                # Otherwise, we get human input demos.
                caption = (f"Task {idx+1} / {num_tasks}\nPlease demonstrate "
                           f"achieving the goal:\n{task.goal}")
                policy = functools.partial(human_demonstrator_policy, env,
                                           caption, event_to_action)
                termination_function = task.goal_holds
                traj, _ = utils.run_policy(
                    policy,
                    env,
                    "train",
                    idx,
                    termination_function=termination_function,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure,
                        utils.HumanDemonstrationFailure,
                    },
                    monitor=video_monitor)
        except (ApproachTimeout, ApproachFailure,
                utils.EnvironmentFailure) as e:
            logging.warning("WARNING: Approach failed to solve with error: "
                            f"{e}")
            continue
        # Check that the goal holds at the end. Print a warning if not.
        if not task.goal_holds(traj.states[-1]):  # pragma: no cover
            logging.warning("WARNING: Oracle failed on training task.")
            continue
        if CFG.demonstrator == "human":  # pragma: no cover
            logging.info("Successfully collected human demonstration of "
                         f"length {len(traj.states)} for task {idx+1} / "
                         f"{num_tasks}.")
        # Add is_demo flag and task index information into the trajectory.
        traj = LowLevelTrajectory(traj.states,
                                  traj.actions,
                                  _is_demo=True,
                                  _train_task_idx=idx)
        # To prevent cheating by option learning approaches, remove all oracle
        # options from the trajectory actions, unless the options are known
        # (via CFG.included_options or CFG.option_learner = 'no_learning').
        if CFG.demonstrator == "oracle":
            for act in traj.actions:
                if act.get_option().parent not in known_options:
                    assert CFG.option_learner != "no_learning"
                    act.unset_option()
        trajectories.append(traj)
        # If we're also annotating with ground truth operators,
        # then get the last nsrt_plan and add the name of the
        # nsrt used to the list of annotations.
        if annotate_with_gt_ops:
            last_nsrt_plan = oracle_approach.get_last_nsrt_plan()
            annotations.append(list(last_nsrt_plan))
        if CFG.make_demo_videos:
            assert video_monitor is not None
            video = video_monitor.get_video()
            outfile = f"{CFG.env}__{CFG.seed}__demo__task{idx}.mp4"
            utils.save_video(outfile, video)
        if CFG.make_demo_images:
            assert video_monitor is not None
            video = video_monitor.get_video()
            width = len(str(len(train_tasks)))
            task_number = str(idx).zfill(width)
            outfile_prefix = f"{CFG.env}__{CFG.seed}__demo__task{task_number}"
            utils.save_images(outfile_prefix, video)
    if annotate_with_gt_ops:
        dataset = Dataset(trajectories, annotations)
    else:
        dataset = Dataset(trajectories)
    return dataset


def human_demonstrator_policy(env: BaseEnv, caption: str,
                              event_to_action: Callable[
                                  [State, matplotlib.backend_bases.Event],
                                  Action],
                              state: State) -> Action:  # pragma: no cover
    """Collect actions from a human interacting with a GUI."""
    # Temporarily change the backend to one that supports a GUI.
    # We do this here because we don't want the rest of the codebase
    # to use GUI-based Matplotlib.
    cur_backend = matplotlib.get_backend()
    matplotlib.use("Qt5Agg")
    # Render the state.
    fig = env.render_plt(caption=caption)
    container = {}

    def _handler(event: matplotlib.backend_bases.Event) -> None:
        container["action"] = event_to_action(state, event)

    keyboard_cid = fig.canvas.mpl_connect("key_press_event", _handler)
    mouse_cid = fig.canvas.mpl_connect("button_press_event", _handler)
    # Hang until either a mouse press or a keyboard press.
    plt.waitforbuttonpress()
    fig.canvas.mpl_disconnect(keyboard_cid)
    fig.canvas.mpl_disconnect(mouse_cid)
    plt.close()
    if "action" not in container:
        logging.warning("WARNING: Event handler failed. Its error message "
                        "should be printed above. Terminating task.")
        raise utils.HumanDemonstrationFailure("Event handler failed!")
    # Revert to the previous backend.
    matplotlib.use(cur_backend)
    return container["action"]


def create_demo_data_from_user_demo(env: RoboKitchenEnv,
                                    path_to_demo: str,
                                 task_name: str) -> Dataset:
    """Create offline datasets by loading user demonstrations.
    This is real data so no contact information, contact left empty!!!
    
    Args:
        env: The environment to load demonstrations for
        path_to_demo: Path to the user demo directory containing pickle files named demo_0_obs.pkl, demo_1_obs.pkl, etc.
        task_name: Name of the robocasa task to load (e.g. 'PnPCounterToCab')
    
    Returns:
        Dataset containing the loaded demonstrations
    """
    
    # Find all demo pickle files in the directory
    demo_obs_files = glob.glob(os.path.join(path_to_demo, "*_obs.pkl"))
    # action_files = glob.glob(os.path.join(path_to_demo, "*_actions.pkl"))
    demo_obs_files.sort()  # Sort to ensure consistent ordering
    
    if not demo_obs_files:
        raise ValueError(f"No *_obs.pkl files found in directory {path_to_demo}")
    
    trajectories = []
    
    for demo_idx, demo_file_path in enumerate(demo_obs_files):
        # Show progress
        if demo_idx >= CFG.num_train_tasks:
            break
        logging.info(f"Processing demo {demo_idx+1} / {min(CFG.num_train_tasks, len(demo_obs_files))}")
        
        # Load pickle file containing list of observations
        with open(demo_file_path, "rb") as f:
            observations = pkl.load(f)
        with open(demo_file_path.replace("_obs.pkl", "_actions.pkl"), "rb") as f:
            actions = pkl.load(f)
        
        if not isinstance(observations, list) or len(observations) == 0:
            logging.warning(f"Skipping {demo_file_path}: not a valid list of observations")
            continue
            
        # Create list of State objects from observations
        states = []
        action_ref = []
        frames_center = []
        frames_left = []
        frames_right = []
        
        for obs, action in zip(observations, actions):
            # Since this is real data, contact information is not available
            # Create empty contact set as mentioned in the docstring
            contact_set = set()
            
            # Create state object from observation
            state = RoboKitchenEnv.state_info_to_state(obs, contact_set)
            states.append(state)
            
            action_ref.append(Action(action))
        
        # Create LowLevelTrajectory
        traj = LowLevelTrajectory(
            _states=list(states),
            _actions=action_ref[:-1],
            _is_demo=True,
            _train_task_idx=demo_idx,
            _raw_robosuite_states=None,  # Not available for user demos
            _model_file=None,  # Not available for user demos
            _ep_meta=None  # Not available for user demos
        )
        trajectories.append(traj)
        
        # Save video if frames were collected
        if not CFG.use_gui and frames_center:
            center_video_save_name = f"robokitchen__{task_name}__{demo_idx}__center.mp4"
            utils.save_video(center_video_save_name, frames_center)
    
    return Dataset(trajectories)


def create_demo_data_from_robocasa(env: RoboKitchenEnv,
                                 known_options: Set[ParameterizedOption],
                                 task_name: str) -> Dataset:
    """Create offline datasets by loading robocasa demonstrations.
    
    Args:
        env: The environment to load demonstrations for
        train_tasks: List of training tasks
        known_options: Set of known parameterized options
        task_name: Name of the robocasa task to load (e.g. 'PnPCounterToCab')
    
    Returns:
        Dataset containing the loaded demonstrations
    """
    # Get path to robocasa dataset
    dataset_path = get_ds_path(task_name, ds_type="human_raw")

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}")

    # Load HDF5 file
    with h5py.File(dataset_path, "r") as f:        
        trajectories = []

        # Each demonstration is stored in a group like "demo_0", "demo_1", etc.
        demos = list(f["data"].keys())

        for demo_idx, demo_key in enumerate(demos):
            # Show progress
            if demo_idx >= CFG.num_train_tasks:
                break
            logging.info(f"Processing demo {demo_idx+1} / {min(CFG.num_train_tasks, len(demos))}")

            # Get demo data
            demo = f[f"data/{demo_key}"]

            # Get states and actions
            # Create list of State objects from state info at each timestep
            states = []
            frames_center = []
            frames_left = []
            frames_right = []
            # first_key = next(iter(demo["datagen_info"]))
            actions = demo["actions"][()]  # Get actions array
            raw_robosuite_states = demo["states"][()]

            # Reset to initial state
            reset_state = {
                "states": demo["states"][0],
                "model": demo.attrs["model_file"],
                "ep_meta": demo.attrs.get("ep_meta", None)
            }
            env._reset_initial_state(seed=0, train_or_test="train", task_name=task_name)
            # seed here does not matter, since we are reset_to later
            reset_to(env._env, reset_state)

            # Get initial state info
            # state_info = {}
            # for key in demo["datagen_info"].keys():
            #     state_info[key] = demo["datagen_info"][key][0]

            # Get contact information for initial state
            # contact_set = env.get_object_level_contacts()

            # # Create and store initial state
            # state = env.state_info_to_state(state_info, contact_set)
            # states.append(state)

            # since we would like observation at each timestep, let us skip the resetted state, start with t=1

            # Process each timestep by executing actions

            # num actions = num states -1
            # state 0 we reset to initial state, so first state to save is
            # states 0 1 2 3 4 5
            # actions 0 1 2 3 4 5
            # need to remove action 0 and action 5, remove states 0
            # in the dataset, the number of states is longer for some reason, so run actions to the end

            for t in range(len(raw_robosuite_states)-1):  # -1 since we skip last action
                # Execute action in environment
                obs, _, _, _ = env._env.step(actions[t])

                # Get state info for next timestep
                # state_info = {}
                # for key in demo["datagen_info"].keys():
                #     state_info[key] = demo["datagen_info"][key][t+1]

                # Get contact information
                contact_set = env.get_object_level_contacts()

                # Create state object
                state = RoboKitchenEnv.state_info_to_state(obs, contact_set) # state here is the predicator state
                states.append(state)
                
                if not CFG.use_gui:
                    center_frame = env._env.sim.render(camera_name="robot0_agentview_center", height=512, width=768)
                    left_frame = env._env.sim.render(camera_name="robot0_agentview_left", height=512, width=768)
                    right_frame = env._env.sim.render(camera_name="robot0_agentview_right", height=512, width=768)
                    frames_center.append(Image.fromarray(center_frame[::-1]))
                    frames_left.append(Image.fromarray(left_frame[::-1]))
                    frames_right.append(Image.fromarray(right_frame[::-1]))

                # Optional: Check if execution matches recorded trajectory
                state_playback = np.array(env._env.sim.get_state().flatten())
                if not np.all(np.abs(demo["states"][t+1] - state_playback) < 0.3):
                    err = np.linalg.norm(demo["states"][t+1] - state_playback)
                    logging.warning(f"Playback diverged by {err} at step {t}")
            # Smooth contact sets using a moving window
            window_size = CFG.robo_kitchen_contact_smoothing_window  # Number of timesteps to look at

            # Store original contacts to prevent smoothing from affecting later operations
            original_contacts = [state.items_in_contact.copy() for state in states]

            for t in range(len(states)):
                # Get contact sets from window using original contacts
                window_contacts = []

                # For points near start, use shifted window that fits
                if t < window_size//2:
                    window_start = 0
                    window_end = window_size
                # For points near end, use shifted window that fits
                elif t >= len(states) - window_size//2:
                    window_start = len(states) - window_size
                    window_end = len(states)
                # For middle points use centered window
                else:
                    window_start = t - window_size//2
                    window_end = t + window_size//2 + 1

                # Get contacts in window
                for w in range(window_start, window_end):
                    window_contacts.append(original_contacts[w])

                # For each possible contact pair, use mode over window
                all_pairs = set()
                for contact_set in window_contacts:
                    all_pairs.update(contact_set)

                smoothed_contacts = set()
                for pair in all_pairs:
                    # Count occurrences of this pair in window
                    count = sum(1 for contact_set in window_contacts if pair in contact_set)
                    # Add to smoothed set if pair appears in majority of window
                    # if count > window_size//2:
                    if count > 0:
                        smoothed_contacts.add(pair)
                # Update contact set for this timestep
                states[t].items_in_contact = smoothed_contacts

            # Convert actions to predicators Action objects
            action_objs = []
            for action in actions[1:-1]:  # Skip first and last action
                # Create Action object - you may need to adjust this based on your action space
                action_obj = Action(action)
                action_objs.append(action_obj)
            # Create LowLevelTrajectory
            traj = LowLevelTrajectory(
                _states=list(states),
                _actions=action_objs,
                _is_demo=True,
                _train_task_idx=demo_idx,
                _raw_robosuite_states=raw_robosuite_states,
                _model_file=demo.attrs["model_file"],
                _ep_meta=demo.attrs.get("ep_meta", None)
            )
            trajectories.append(traj)

            if not CFG.use_gui:
                center_video_save_name = f"robokitchen__{task_name}__{demo_idx}__center.mp4"
                # left_video_save_name = f"robokitchen__{task_name}__{demo_idx}__left.mp4"
                # right_video_save_name = f"robokitchen__{task_name}__{demo_idx}__right.mp4"
                utils.save_video(center_video_save_name, frames_center)
                # utils.save_video(left_video_save_name, frames_left)
                # utils.save_video(right_video_save_name, frames_right)

    return Dataset(trajectories)


def create_demo_data_from_mocap(env: RoboKitchenEnv,
                               path_to_mocap: str,
                               task_name: str) -> Dataset:
    """Create offline datasets by loading mocap demonstrations.
    
    Args:
        env: The environment to load demonstrations for
        path_to_mocap: Path to the mocap data directory containing CSV files
        task_name: Name of the task (e.g. 'yifei_open_lid', 'yifei_pour_pot')
    
    Returns:
        Dataset containing the loaded mocap demonstrations
    """
    import csv
    import numpy as np
    from collections import defaultdict
    
    # Find all CSV files in the directory
    mocap_files = glob.glob(os.path.join(path_to_mocap, "*.csv"))
    mocap_files.sort()  # Sort to ensure consistent ordering
    
    if not mocap_files:
        raise ValueError(f"No CSV files found in directory {path_to_mocap}")
    
    trajectories = []
    
    for demo_idx, mocap_file_path in enumerate(mocap_files):
        # Show progress
        if demo_idx >= CFG.num_train_tasks:
            break
        logging.info(f"Processing mocap demo {demo_idx+1} / {min(CFG.num_train_tasks, len(mocap_files))}")
        
        # Parse mocap CSV file
        mocap_data = parse_mocap_csv(mocap_file_path)
        
        if not mocap_data['frames']:
            logging.warning(f"Skipping {mocap_file_path}: no valid mocap data")
            continue
        
        # Convert mocap data to states and actions
        states = []
        actions = []
        
        # Set initial environment state (you may need to customize this based on your environment)
        # env._reset_initial_state(seed=0, train_or_test="train", task_name="PnPCabToCounterTomato") 
        # using tomato task since it is smallest with all objects
        
        # Process each timestep of mocap data
        for frame_idx, frame_data in enumerate(mocap_data['frames']):
            # Create state from mocap data
            state = create_state_from_mocap_frame(frame_data)
            states.append(state)
            
            # Create actions (if not the last frame)
            if frame_idx < len(mocap_data['frames']) - 1:
                next_frame_data = mocap_data['frames'][frame_idx + 1]
                action = create_action_from_mocap_transition(frame_data, next_frame_data, mocap_data['objects'])
                actions.append(action)
        
        # Create LowLevelTrajectory
        traj = LowLevelTrajectory(
            _states=states,
            _actions=actions,
            _is_demo=True,
            _train_task_idx=demo_idx,
            _raw_robosuite_states=None,  # Not available for mocap demos
            _model_file=None,  # Not available for mocap demos
            _ep_meta={'mocap_file': mocap_file_path, 'task_name': task_name}
        )
        trajectories.append(traj)
    
    return Dataset(trajectories)


def parse_mocap_csv_simple(csv_file_path: str) -> dict:
    """Parse a simplified mocap CSV file with format: sequence,stamp,topic,px,py,pz,qx,qy,qz,qw
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed mocap data with structure:
        {
            'metadata': {},
            'objects': [...],
            'frames': [...]
        }
    """
    import csv
    import numpy as np
    from collections import defaultdict
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {'metadata': {}, 'objects': [], 'frames': []}
    
    # Group data by sequence number (frame)
    frames_data = defaultdict(lambda: {'frame': None, 'time': None, 'object_data': {}})
    object_names = set()
    
    for row in rows:
        sequence = int(row['sequence'])
        timestamp = float(row['stamp'])
        topic = row['topic']
        
        # Extract object name from topic (e.g., "/natnet_ros/umi_body/pose" -> "umi_body")
        if '/natnet_ros/' in topic and '/pose' in topic:
            object_name = topic.replace('/natnet_ros/', '').replace('/pose', '')
        else:
            # Fallback parsing for other topic formats
            parts = topic.split('/')
            object_name = parts[-2] if len(parts) >= 2 else 'unknown'
        
        object_names.add(object_name)
        
        # Parse position and quaternion
        position = [float(row['px']), float(row['py']), float(row['pz'])]
        rotation = [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
        
        # Store frame data
        frames_data[sequence]['frame'] = sequence
        frames_data[sequence]['time'] = timestamp
        frames_data[sequence]['object_data'][object_name] = {
            'position': position,
            'rotation': rotation
        }
    
    # Create objects list (for compatibility with existing code)
    objects = []
    for obj_name in sorted(object_names):
        objects.append({
            'name': obj_name,
            'id': obj_name,
            'type': 'rigid_body',
            'rotation_cols': [],  # Not used in simple format
            'position_cols': []   # Not used in simple format
        })
    
    # Convert frames data to list format, only including frames with all objects
    frames = []
    for sequence in sorted(frames_data.keys()):
        frame_data = frames_data[sequence]
        # Only include frames that have data for all objects
        if len(frame_data['object_data']) == len(object_names):
            frames.append(frame_data)
        else:
            logging.warning(f"Skipping frame {sequence} because it does not have data for all objects")
    
    return {
        'metadata': {'format': 'simple'},
        'objects': objects,
        'frames': frames
    }


def parse_mocap_csv(csv_file_path: str) -> dict:
    """Parse a mocap CSV file and extract structured data.
    
    This function automatically detects the CSV format and uses the appropriate parser.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed mocap data with structure:
        {
            'metadata': {...},
            'objects': [...],
            'frames': [...]
        }
    """
    # Detect CSV format by reading the first line
    with open(csv_file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if it's the simple format
    if first_line.startswith('sequence,stamp,topic,px,py,pz,qx,qy,qz,qw'):
        return parse_mocap_csv_simple(csv_file_path)
    else:
        # Use the original complex parser
        return parse_mocap_csv_complex(csv_file_path)


def parse_mocap_csv_complex(csv_file_path: str) -> dict:
    """Parse a complex mocap CSV file with metadata and structured headers.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed mocap data with structure:
        {
            'metadata': {...},
            'objects': [...],
            'frames': [...]
        }
    """
    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Parse metadata from first row
    metadata = {}
    header_row = rows[0]
    for i in range(0, len(header_row), 2):
        if i + 1 < len(header_row):
            key = header_row[i]
            value = header_row[i + 1]
            metadata[key] = value
    
    # Parse object information
    object_types = rows[2][2:]  # Skip first two cells (empty and "Time (Seconds)")
    object_names = rows[3][2:]  # Skip first two cells (empty and "Name")  
    object_ids = rows[4][2:]    # Skip first two cells (empty and ID header)
    data_types = rows[5][2:]    # Skip first two cells (empty and data type header)
    column_headers = rows[6][2:]  # Skip first two cells ("Frame" and "Time (Seconds)")
    
    # Group columns by object
    objects = []
    col_idx = 0  # This tracks the index in the data_values array (excluding Frame and Time)
    current_object = None
    
    for i, (obj_type, obj_name, obj_id, data_type, col_header) in enumerate(zip(
        object_types, object_names, object_ids, data_types, column_headers)):
        
        if current_object is None or current_object['name'] != obj_name:
            # New object
            if current_object is not None:
                objects.append(current_object)
            current_object = {
                'name': obj_name,
                'id': obj_id,
                'type': obj_type,
                'rotation_cols': [],
                'position_cols': []
            }
        
        # Add column info to current object
        if data_type == 'Rotation':
            current_object['rotation_cols'].append({'header': col_header, 'index': col_idx})
        elif data_type == 'Position':
            current_object['position_cols'].append({'header': col_header, 'index': col_idx})
        
        col_idx += 1
    
    # Don't forget the last object
    if current_object is not None:
        objects.append(current_object)
    
    # Parse frame data
    frames = []
    for row in rows[7:]:  # Skip header rows
        if not row or not row[0]:  # Skip empty rows
            continue
        
        frame_num = int(row[0])
        timestamp = float(row[1])
        data_values = [float(x) if x else 0.0 for x in row[2:]]
        
        frame_data = {
            'frame': frame_num,
            'time': timestamp,
            'object_data': {}
        }
        
        # Extract data for each object
        for obj in objects:
            obj_data = {
                'rotation': [],
                'position': []
            }
            
            # Get rotation data (quaternion: X, Y, Z, W)
            for col_info in obj['rotation_cols']:
                if col_info['index'] < len(data_values):
                    obj_data['rotation'].append(data_values[col_info['index']])
            
            # Get position data (X, Y, Z)
            for col_info in obj['position_cols']:
                if col_info['index'] < len(data_values):
                    obj_data['position'].append(data_values[col_info['index']])
            
            frame_data['object_data'][obj['name']] = obj_data
        
        frames.append(frame_data)
    
    return {
        'metadata': metadata,
        'objects': objects,
        'frames': frames
    }


def create_state_from_mocap_frame(frame_data: dict) -> State:
    """Create a State object from mocap frame data.
    
    Args:
        env: The environment
        frame_data: Single frame of mocap data
        objects: List of object definitions
        
    Returns:
        State object representing the mocap frame
    """
    # This is a placeholder implementation - you'll need to customize this
    # based on how your RoboKitchenEnv.state_info_to_state works
    
    # Create a mock observation dictionary with mocap data
    obs = {}
    
    # Handle gripper and finger poses specially
    gripper_data = None
    finger_data = None
    
    # Map mocap objects to environment state representation
    for obj_name, obj_data in frame_data['object_data'].items():
        if obj_data['position'] and obj_data['rotation']:
            # Position (x, y, z)
            pos = np.array(obj_data['position'])
            # Rotation quaternion (x, y, z, w)  
            quat = np.array(obj_data['rotation'])
            assert len(pos) == 3 and len(quat) == 4, f"Position and quaternion must have 3 and 4 elements respectively, but got {len(pos)} and {len(quat)} for {obj_name}"
            
            # Handle special cases for gripper and finger
            if obj_name == "umi_body":
                # Rename umi_body to gripper
                gripper_data = {'pos': pos, 'quat': quat}
                obs["gripper_pos_quat"] = np.concatenate([pos, quat])
            elif obj_name == "umi_finger":
                # Store finger data for symmetric processing
                finger_data = {'pos': pos, 'quat': quat}
            elif not obj_name.startswith("umi_"):
                # Store other objects normally
                obs[f"{obj_name}_pos_quat"] = np.concatenate([pos, quat])
    
    # Create symmetric left and right finger poses from single umi_finger marker
    if finger_data is not None and gripper_data is not None:
        # Assume the single finger marker represents the center between left and right fingers
        finger_pos = finger_data['pos']
        finger_quat = finger_data['quat']
        gripper_pos = gripper_data['pos']
        
        # Calculate offset vector from gripper to finger (this represents the finger center)
        finger_offset = finger_pos - gripper_pos
        
        # Create symmetric left and right finger positions
        # Assume fingers are symmetric about the gripper's local y-axis
        # We'll offset them by a small amount (e.g., 0.02m = 2cm) in the gripper's local x-axis
        finger_separation = 0.02  # 2cm separation between fingers
        
        # Convert quaternion to rotation matrix to get local coordinate system
        gripper_rot = R.from_quat([gripper_data['quat'][0], gripper_data['quat'][1], gripper_data['quat'][2], gripper_data['quat'][3]])  # x, y, z, w
        gripper_rot_matrix = gripper_rot.as_matrix()
        
        # Local x-axis of gripper (for finger separation)
        local_x_axis = gripper_rot_matrix[:, 0]
        
        # Calculate left and right finger positions
        left_finger_pos = finger_pos - (finger_separation / 2) * local_x_axis
        right_finger_pos = finger_pos + (finger_separation / 2) * local_x_axis
        
        # Both fingers have the same orientation as the original finger marker
        obs["left_finger_pos_quat"] = np.concatenate([left_finger_pos, finger_quat])
        obs["right_finger_pos_quat"] = np.concatenate([right_finger_pos, finger_quat])
    
    # Create empty contact set (mocap doesn't provide contact information)
    contact_set = set()
    
    # Convert to State object using your environment's method
    # You may need to modify this call based on your actual state representation
    state = RoboKitchenEnv.observation_to_state_mocap(obs)
    
    return state


def create_action_from_mocap_transition(current_frame: dict, next_frame: dict, objects: list) -> Action:
    """Create an Action object from the mocap data, for ds_policy, the first 6 elements are not used
    the last element is the gripper command, which is average,
    here just get the distance between the gripper and the finger, and assign to -1, 1
    
    Args:
        current_frame: Current frame mocap data
        next_frame: Next frame mocap data  
        objects: List of object definitions
        
    Returns:
        Action object representing the transition
    """
    # This is a placeholder implementation - you'll need to customize this
    # based on your action space and how you want to represent mocap transition
    action_array = np.zeros(7)

    # Ensure we have exactly 7D action space (3D gripper + 3D finger + 1D gripper command)
    gripper_pos = current_frame['object_data']['umi_body']['position']
    finger_pos = current_frame['object_data']['umi_finger']['position']
    gripper_command = np.linalg.norm(np.array(gripper_pos) - np.array(finger_pos))
    # print(f"gripper_command: {gripper_command}")
    action_array[-1] = 1 if gripper_command < 0.063 else -1
    assert len(action_array) == 7, f"Action array must have 7 elements, but got {len(action_array)}"
    
    return Action(np.array(action_array))
