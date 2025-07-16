import os
from predicators.settings import CFG
import pytest
from unittest.mock import patch
import pickle

from predicators.main import main as predicators_main
from predicators import utils
import shutil
import glob 

results_dir = "results"
base_results_dir = "sym_skill_base_results"
online_learning_cycle = None
saved_approaches_dir = "saved_approaches"

BASE_SIMULATED_ARGV = [
    'predicators/main.py',  # The first element of sys.argv is the script name
    "--env", "robo_kitchen",
    # "--use_gui", # github action does not support gui
    "--approach", "clustering_invention",
    "--seed", "0",
    "--bilevel_plan_without_sim", "True",
    "--debug",
    "--excluded_predicates", "all_goal",
    "--option_learner", "ds_policy",
    "--execution_monitor", "expected_atoms_robocasa",
    # some flags to override settings.py to ensure consistency
    # "--num_train_tasks", "10",
    # "--num_test_tasks", "1",
    "--results_dir", results_dir,
    "--use_learnt_goal_predicates", "True",
    "--use_teleop", "False", #single stage task does not need motion of the base
]

ROBO_KITCHEN_TASK_NAMES = [
    "OpenSingleDoor",
    "PnPCounterToCab", # learn precondition of door open to pick place
    "CloseSingleDoor", # learn closing door has precondition of door open
    "OpenSingleDoor", # learn opening door has precondition of door closed 
]

COMPOSITE_SIMULATED_ARGV = [
    'predicators/main.py',  # The first element of sys.argv is the script name
    "--env", "robo_kitchen",
    # "--use_gui", # github action does not support gui
    "--approach", "clustering_invention",
    "--seed", "0",
    "--bilevel_plan_without_sim", "True",
    "--debug",
    "--excluded_predicates", "all_goal",
    "--option_learner", "ds_policy",
    "--execution_monitor", "expected_atoms_robocasa",
    # some flags to override settings.py to ensure consistency
    # "--num_train_tasks", "10",
    # "--num_test_tasks", "1",
    "--results_dir", results_dir,
    "--use_learnt_goal_predicates", "True",
    "--use_teleop", "False", #single stage task does not need motion of the base
    "--load_approach"
]
COMPOSITE_TASK_NAMES = [
    "StoreFruit",
]


@pytest.mark.parametrize("robo_kitchen_task_name", ROBO_KITCHEN_TASK_NAMES)
def test_main(robo_kitchen_task_name):
    """
    Tests the main() function for various robo_kitchen_task configurations
    by simulating the command-line arguments.
    """
    CFG.dict_contact_predicate_to_rel_pose_predicates = {}
    CFG.dict_gt_goal_predicate_to_dummy_goal_predicates = {}
    # Create a copy of the base arguments for this specific test run
    current_argv = list(BASE_SIMULATED_ARGV)
    # Add the current robo_kitchen_task to the arguments
    current_argv.extend(["--robo_kitchen_task", robo_kitchen_task_name])

    with patch('sys.argv', current_argv):
        predicators_main()

        # start checking log files
        outfile = (f"{results_dir}/{utils.get_config_path_str()}__{online_learning_cycle}.pkl")
        assert os.path.exists(outfile)
        with open(outfile, 'rb') as f:
            log_data = pickle.load(f)
        results = log_data['results']
        # Remove all files in the saved_approaches folder after running

        # for f in glob.glob(f"{saved_approaches_dir}/*"):
        #     try:
        #         if os.path.isfile(f) or os.path.islink(f):
        #             os.remove(f)
        #         elif os.path.isdir(f):
        #             shutil.rmtree(f)
        #     except Exception as e:
        #         print(f"Failed to delete {f}. Reason: {e}")
        assert results['num_solved'] == results['num_total'] #

@pytest.mark.parametrize("robo_kitchen_task_name", COMPOSITE_TASK_NAMES)
def test_composite(robo_kitchen_task_name):
    """
    Tests the main() function for various robo_kitchen_task configurations
    by simulating the command-line arguments.
    """
    # Create a copy of the base arguments for this specific test run
    current_argv = list(COMPOSITE_SIMULATED_ARGV)
    # Add the current robo_kitchen_task to the arguments
    current_argv.extend(["--robo_kitchen_task", robo_kitchen_task_name])

    with patch('sys.argv', current_argv):
        predicators_main()

        # start checking log files
        outfile = (f"{results_dir}/{utils.get_config_path_str()}__{online_learning_cycle}.pkl")
        assert os.path.exists(outfile)
        with open(outfile, 'rb') as f:
            log_data = pickle.load(f)
        results = log_data['results']
        # Remove all files in the saved_approaches folder after running

        # for f in glob.glob(f"{saved_approaches_dir}/*"):
        #     try:
        #         if os.path.isfile(f) or os.path.islink(f):
        #             os.remove(f)
        #         elif os.path.isdir(f):
        #             shutil.rmtree(f)
        #     except Exception as e:
        #         print(f"Failed to delete {f}. Reason: {e}")
        assert results['num_solved'] == results['num_total'] #



        # start comparing results to base_results   
        # assert results['offline_learning_trajs_states_nums'] == base_results['offline_learning_trajs_states_nums']
        # assert compare_nsrt_rel_cluster_types(results['offline_learning_nsrt_rel_cluster_types'], base_results['offline_learning_nsrt_rel_cluster_types'])
        # end comparing results to base_results

def compare_nsrt_rel_cluster_types(nsrt_rel_cluster_types_1, nsrt_rel_cluster_types_2):
    """
    nsrt_rel_cluster_types_1 and nsrt_rel_cluster_types_2 are lists of dicts
    each dict corresponds to a single NSRT
    each dict has keys: preconditions, maintain_effects, add_effects, delete_effects, ignore_effects
    each value is a list of sets, where each set is a cluster's types
    return True if nsrt_rel_cluster_types_1 and nsrt_rel_cluster_types_2 are the same
    return False otherwise
    """
    if len(nsrt_rel_cluster_types_1) != len(nsrt_rel_cluster_types_2):
        print(f"nsrt_rel_cluster_types_1 and nsrt_rel_cluster_types_2 have different lengths: {len(nsrt_rel_cluster_types_1)} and {len(nsrt_rel_cluster_types_2)}")
        return False
    for i, nsrt_1 in enumerate(nsrt_rel_cluster_types_1):
        found_match = False
        for j, nsrt_2 in enumerate(nsrt_rel_cluster_types_2):
            if compare_two_nsrt_rel_cluster_types(nsrt_1, nsrt_2):
                found_match = True
                break
        if not found_match:
            print(f"No match found for nsrt_1: {nsrt_1}")
            return False
    return True

def compare_two_nsrt_rel_cluster_types(nsrt_1, nsrt_2):
    """
    nsrt_1 and nsrt_2 are dicts with keys: preconditions, maintain_effects, add_effects, delete_effects, ignore_effects
    each value is a list of sets, where each set is a cluster's types
    return True if nsrt_1 and nsrt_2 are the same
    return False otherwise
    """
    for k, v in nsrt_1.items():
        if v != nsrt_2[k]:
            return False
    return True

# NOTE: this is for testing purposes only. This file should be run with pytest
if __name__ == "__main__":
    # start checking log files
    outfile = "results/robo_kitchen__clustering_invention__OpenSingleDoor__0__all_goal______None.pkl"
    assert os.path.exists(outfile)
    with open(outfile, 'rb') as f:
        log_data = pickle.load(f)
    results = log_data['results']
    base_outfile = f"{base_results_dir}/robo_kitchen__clustering_invention__OpenSingleDoor__0__all_goal______None.pkl"
    assert os.path.exists(base_outfile)
    with open(base_outfile, 'rb') as f:
        base_log_data = pickle.load(f)
    base_results = base_log_data['results']

    # start comparing results to base_results
    # offline learning
    assert results['offline_learning_trajs_states_nums'] == base_results['offline_learning_trajs_states_nums']
    assert compare_nsrt_rel_cluster_types(results['offline_learning_nsrt_rel_cluster_types'], base_results['offline_learning_nsrt_rel_cluster_types'])
    # online testing
    succ_rate = results['num_solved'] / results['num_total']
    base_succ_rate = base_results['num_solved'] / base_results['num_total']
    assert abs(succ_rate - base_succ_rate) < 0.3
    assert (results['replan_total_count'] - base_results['replan_total_count']) / base_results['replan_total_count'] < 0.3
    # end comparing results to base_results

    # end checking log files