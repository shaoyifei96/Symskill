import pytest
from unittest.mock import patch

from predicators.main import main as predicators_main

BASE_SIMULATED_ARGV = [
    'predicators/main.py',  # The first element of sys.argv is the script name
    "--env", "robo_kitchen",
    "--use_gui",
    "--approach", "clustering_invention",
    "--seed", "0",
    "--bilevel_plan_without_sim", "True",
    "--debug",
    "--excluded_predicates", "all_goal",
    "--option_learner", "ds_policy",
    "--execution_monitor", "expected_atoms_robocasa",
    # some flags to override settings.py to ensure consistency
    "--num_train_tasks", "10",
    "--num_test_tasks", "1",
    # "--test", "True"  # User-added argument
]

ROBO_KITCHEN_TASKS_TO_TEST = [
    "PnPCounterToCab",
    "OpenSingleDoor",
    "CloseSingleDoor",
]


@pytest.mark.parametrize("robo_kitchen_task_name", ROBO_KITCHEN_TASKS_TO_TEST)
def test_main(robo_kitchen_task_name):
    """
    Tests the main() function for various robo_kitchen_task configurations
    by simulating the command-line arguments.
    """
    # Create a copy of the base arguments for this specific test run
    current_argv = list(BASE_SIMULATED_ARGV)
    # Add the current robo_kitchen_task to the arguments
    current_argv.extend(["--robo_kitchen_task", robo_kitchen_task_name])

    with patch('sys.argv', current_argv):
        predicators_main()

    # TODO: Add more specific assertions here based on the expected behavior 
    # of main() for each specific robo_kitchen_task_name.
    # For example, you might check logs, created files, or the state of certain objects if possible.
    assert True  # Placeholder if main() doesn't call sys.exit() or for basic run check
