"""Definitions of option learning strategies."""

from __future__ import annotations

import abc
import copy
import logging
from typing import ClassVar, Dict, List, Sequence, Set, Tuple, Any

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.envs.blocks import BlocksEnv
from predicators.ground_truth_models import get_gt_options
from predicators.ml_models import ImplicitMLPRegressor, MLPRegressor, Regressor
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, Array, Datastore, Object, OptionSpec, \
    ParameterizedOption, Segment, State, STRIPSOperator, Variable, \
    VarToObjSub
from predicators.utils import OptionExecutionFailure

from ds_policy import DSPolicy, UnifiedModelConfig, transform_frame, compute_vel_traj
from scipy.spatial.transform import Rotation as R

def create_option_learner(action_space: Box) -> _OptionLearnerBase:
    """Create an option learner given its name."""
    if CFG.option_learner == "no_learning":
        return KnownOptionsOptionLearner()
    if CFG.option_learner == "oracle":
        return _OracleOptionLearner()
    if CFG.option_learner == "direct_bc":
        return _DirectBehaviorCloningOptionLearner(action_space)
    if CFG.option_learner == "implicit_bc":
        return _ImplicitBehaviorCloningOptionLearner(action_space)
    if CFG.option_learner == "direct_bc_nonparameterized":
        return _DirectBehaviorCloningOptionLearner(action_space,
                                                   is_parameterized=False)
    if CFG.option_learner == "ds_policy":
        return _DSOptionLearner(action_space)
    raise NotImplementedError(f"Unknown option_learner: {CFG.option_learner}")


def create_rl_option_learner() -> _RLOptionLearnerBase:
    """Create an RL option learner given its name."""
    if CFG.nsrt_rl_option_learner == "dummy_rl":
        return _DummyRLOptionLearner()
    raise NotImplementedError(f"Unknown option_learner: {CFG.option_learner}")


class _OptionLearnerBase(abc.ABC):
    """Struct defining an option learner, which has an abstract method for
    learning option specs and an abstract method for annotating data segments
    with options."""

    @abc.abstractmethod
    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        """Given datastores and STRIPS operators that were fit on them, learn
        option specs, which are tuples of (ParameterizedOption,
        Sequence[Variable]).

        The returned option specs should be one-to-one with the given
        strips_ops / datastores (which are already one-to-one with each
        other).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        """Figure out which option was executed within the given segment.
        Modify the segment in-place to include this option, via
        segment.set_option().

        At this point, we know which ParameterizedOption was used, and
        we know the option_vars. This information is included in the
        given option_spec. But we don't know what parameters were used
        in the option, which this method should figure out.
        """
        raise NotImplementedError("Override me!")


class KnownOptionsOptionLearner(_OptionLearnerBase):
    """The "option learner" that's used when we're in the code path where
    CFG.option_learner is "no_learning".

    This option learner assumes that all of the actions that are
    received already have an option attached to them
    (action.has_option() is True). Since options are already known,
    "learning" is a bit of a misnomer. What this class is doing is just
    extracting the known options from the actions and creating option
    specs for the STRIPSOperator objects.
    """

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        # Since we're not actually doing option learning, the data already
        # contains the options. So, we just extract option specs from the data.
        option_specs = []
        for datastore in datastores:
            param_option = None
            option_vars = []
            for i, (segment, var_to_obj) in enumerate(datastore):
                option = segment.actions[0].get_option()
                if i == 0:
                    obj_to_var = {o: v for v, o in var_to_obj.items()}
                    assert len(var_to_obj) == len(obj_to_var)
                    param_option = option.parent
                    option_vars = [obj_to_var[o] for o in option.objects]
                else:
                    assert param_option == option.parent
                    option_args = [var_to_obj[v] for v in option_vars]
                    assert option_args == option.objects
                # Make sure the option is consistent within a trajectory.
                for a in segment.actions:
                    option_a = a.get_option()
                    assert param_option == option_a.parent
                    option_args = [var_to_obj[v] for v in option_vars]
                    assert option_args == option_a.objects
            assert param_option is not None and option_vars is not None, \
                "No data in this datastore?"
            option_specs.append((param_option, option_vars))
        return option_specs

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        # If we're not doing option learning, the segments will already have
        # the options, so there is nothing to do here.
        pass


class _OracleOptionLearner(_OptionLearnerBase):
    """The option learner that just cheats by looking up ground truth options
    from the environment.

    Useful for testing.
    """

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        env_options = get_gt_options(CFG.env)
        option_specs: List[OptionSpec] = []
        if CFG.env == "cover":
            assert len(strips_ops) == 4
            PickPlace = [
                option for option in env_options if option.name == "PickPlace"
            ][0]
            # All strips operators use the same PickPlace option,
            # which has no parameters.
            for _ in strips_ops:
                option_specs.append((PickPlace, []))
        elif CFG.env == "blocks":
            assert len(strips_ops) == 4
            Pick = [option for option in env_options
                    if option.name == "Pick"][0]
            Stack = [
                option for option in env_options if option.name == "Stack"
            ][0]
            PutOnTable = [
                option for option in env_options if option.name == "PutOnTable"
            ][0]
            for op in strips_ops:
                if {atom.predicate.name for atom in op.preconditions} in \
                   ({"GripperOpen", "Clear", "OnTable"},
                    {"GripperOpen", "Clear", "On"}):
                    # PickFromTable or Unstack operators
                    gripper_open_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"
                    ][0]
                    block = clear_atom.variables[0]
                    option_specs.append((Pick, [robot, block]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Clear", "Holding"}:
                    # Stack operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"
                    ][0]
                    otherblock = clear_atom.variables[0]
                    option_specs.append((Stack, [robot, otherblock]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Holding"}:
                    # PutOnTable operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    option_specs.append((PutOnTable, [robot]))
        return option_specs

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        if CFG.env == "cover":
            param_opt, opt_vars = option_spec
            assert not opt_vars
            assert len(segment.actions) == 1
            # In the cover env, the action is itself the option parameter.
            params = segment.actions[0].arr
            option = param_opt.ground([], params)
            segment.set_option(option)
        if CFG.env == "blocks":
            param_opt, opt_vars = option_spec
            assert len(segment.actions) == 1
            act = segment.actions[0].arr
            robby = [obj for obj in segment.states[1]
                     if obj.name == "robby"][0]
            # Transform action array back into parameters.
            if param_opt.name == "Pick":
                assert len(opt_vars) == 2
                picked_blocks = [
                    obj for obj in segment.states[1]
                    if obj.type.name == "block"
                    and segment.states[0].get(obj, "held") < 0.5 <
                    segment.states[1].get(obj, "held")
                ]
                assert len(picked_blocks) == 1
                block = picked_blocks[0]
                params = np.zeros(0, dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)
            elif param_opt.name == "PutOnTable":
                x, y, _, _ = act
                params = np.array([
                    (x - BlocksEnv.x_lb) / (BlocksEnv.x_ub - BlocksEnv.x_lb),
                    (y - BlocksEnv.y_lb) / (BlocksEnv.y_ub - BlocksEnv.y_lb)
                ])
                option = param_opt.ground([robby], params)
                segment.set_option(option)
            elif param_opt.name == "Stack":
                assert len(opt_vars) == 2
                dropped_blocks = [
                    obj for obj in segment.states[1]
                    if obj.type.name == "block"
                    and segment.states[1].get(obj, "held") < 0.5 <
                    segment.states[0].get(obj, "held")
                ]
                assert len(dropped_blocks) == 1
                block = dropped_blocks[0]
                params = np.zeros(0, dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)


class _ActionConverter(abc.ABC):
    """Maps environment actions to a reduced action space and back."""

    @abc.abstractmethod
    def env_to_reduced(self, env_action_arr: Array) -> Array:
        """Map an environment action to a reduced action."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reduced_to_env(self, reduced_action_arr: Array) -> Array:
        """Map a reduced action to an environment action."""
        raise NotImplementedError("Override me!")


class _IdentityActionConverter(_ActionConverter):
    """A trivial action space converter, useful for testing."""

    def env_to_reduced(self, env_action_arr: Array) -> Array:
        return env_action_arr.copy()

    def reduced_to_env(self, reduced_action_arr: Array) -> Array:
        return reduced_action_arr.copy()


class _KinematicActionConverter(_ActionConverter):
    """Uses CFG.pybullet_robot to convert the 9D action space into 4D.

    Assumes that the gripper does not rotate.

    Creates a new PyBullet connection for the robot.
    """
    _gripper_open: ClassVar[float] = 1.0
    _gripper_closed: ClassVar[float] = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._init()

    def _init(self) -> None:
        # Create a new PyBullet connection and robot.
        self._physics_client_id = p.connect(p.DIRECT)
        # Create the robot.
        self._robot = create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                       self._physics_client_id)
        # The rotation is assumed to be fixed, so record it once.
        qx, qy, qz, qw = self._robot.get_state()[3:7]
        self._ee_orn = (qx, qy, qz, qw)

    def __setstate__(self, state: Dict) -> None:
        # Recreate the object to avoid issues with the PyBullet client.
        del state  # unused
        self._init()

    def env_to_reduced(self, env_action_arr: Array) -> Array:
        # Forward kinematics.
        assert env_action_arr.shape == (9, )
        pose = self._robot.forward_kinematics(env_action_arr.tolist())
        x, y, z = pose.position
        # Average the two fingers.
        left_finger = env_action_arr[self._robot.left_finger_joint_idx]
        right_finger = env_action_arr[self._robot.right_finger_joint_idx]
        fingers = (left_finger + right_finger) / 2.
        # Round the fingers.
        dist_to_open = abs(fingers - self._robot.open_fingers)
        dist_to_closed = abs(fingers - self._robot.closed_fingers)
        if dist_to_closed < dist_to_open:
            gripper = self._gripper_closed
        else:
            gripper = self._gripper_open
        return np.array([x, y, z, gripper], dtype=np.float32)

    def reduced_to_env(self, reduced_action_arr: Array) -> Array:
        # Inverse kinematics.
        x, y, z, gripper = reduced_action_arr
        # Gripper to fingers.
        dist_to_open = abs(gripper - self._gripper_open)
        dist_to_closed = abs(gripper - self._gripper_closed)
        if dist_to_closed < dist_to_open:
            fingers = self._robot.closed_fingers
        else:
            fingers = self._robot.open_fingers
        try:
            pose = Pose((x, y, z), self._ee_orn)
            joints = self._robot.inverse_kinematics(pose, validate=True)
        except InverseKinematicsError:
            raise OptionExecutionFailure("IK failure in action conversion.")
        joints[self._robot.left_finger_joint_idx] = fingers
        joints[self._robot.right_finger_joint_idx] = fingers
        return np.array(joints, dtype=np.float32)


def create_action_converter() -> _ActionConverter:
    """Create an action space converter based on CFG."""
    name = CFG.option_learning_action_converter
    if name == "identity":
        return _IdentityActionConverter()
    if name == "kinematic":
        return _KinematicActionConverter()
    raise NotImplementedError(f"Unknown action space converter: {name}")


class _LearnedNeuralParameterizedOption(ParameterizedOption):
    """A parameterized option that "implements" an operator.

    * The option objects correspond to the operator parameters.
    * The option initiable corresponds to the operator preconditions.
    * The option terminal corresponds to the operator effects.
    * The option policy is implemented as a neural regressor that takes in the
      states of the option objects and the input continuous parameters (see
      below), all concatenated into a 1D vector, and outputs an action.

    The continuous parameters of the option are the main thing to note: they
    correspond to a desired change in state for a subset of the option objects
    and a subset of those object features. The objects that are changing are
    changing_var_order, which are the same as the keys of changing_var_to_feat.
    The feature indices that change are the values in changing_var_to_feat. All
    other objects and features are assumed to have no change in their state.

    Note that the option terminal, which corresponds to the operator effects,
    already describes how we want the objects to change. But these effects only
    characterize a *set* of desired state changes. For example, a Pick operator
    with Holding(?x) in the add effects says that we want to end up in *some*
    state where we are holding ?x, but there are in general an infinite number
    of such states. The role of the continuous parameters is to identify one
    specific state in this set.

    The hope is that by sampling different continuous parameters, the option
    will be able to navigate to different states in the effect set, giving the
    diversity of transition samples that we need to do bilevel planning.

    Also note that the parameters correspond to a state *change*, rather than
    an absolute state. This distinction is useful for learning; relative changes
    lead to better data efficiency / generalization than absolute ones. However,
    it's also important to note that the change here is a delta between the
    initial state that the option is executed from and the final state where the
    option is terminated. In the middle of executing the option, the desired
    delta between the current state and the final state is different from what
    it was in the initial state. To handle this, we save the *absolute* desired
    state in the memory of the option during initialization, and compute the
    updated delta on each call to the policy.

    The is_parameterized kwarg is for a baseline that learns a policy without
    continuous parameters. If it is False, the parameter space is null.
    """

    def __init__(self,
                 name: str,
                 operator: STRIPSOperator,
                 regressor: Regressor,
                 changing_var_to_feat: Dict[Variable, List[int]],
                 changing_var_order: List[Variable],
                 action_space: Box,
                 action_converter: _ActionConverter,
                 is_parameterized: bool = True) -> None:
        assert set(changing_var_to_feat).issubset(set(operator.parameters))
        types = [v.type for v in operator.parameters]
        option_param_dim = sum(
            len(idxs) for idxs in changing_var_to_feat.values())
        if is_parameterized:
            params_space = Box(low=-np.inf,
                               high=np.inf,
                               shape=(option_param_dim, ),
                               dtype=np.float32)
        else:
            params_space = Box(0, 1, (0, ), dtype=np.float32)
        self.operator = operator
        self._regressor = regressor
        self._changing_var_to_feat = changing_var_to_feat
        self._changing_var_order = changing_var_order
        self._action_space = action_space
        self._action_converter = action_converter
        self._is_parameterized = is_parameterized
        super().__init__(name,
                         types,
                         params_space,
                         policy=self._regressor_based_policy,
                         initiable=self._precondition_based_initiable,
                         terminal=self._optimized_effect_based_terminal)

    def _precondition_based_initiable(self, state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
        if self._is_parameterized:
            # The memory here is used to store the absolute params, based on
            # the relative params and the object states.
            memory["params"] = params  # store for sanity checking in policy
            var_to_obj = dict(zip(self.operator.parameters, objects))
            state_params = _create_absolute_option_param(
                state, self._changing_var_to_feat, self._changing_var_order,
                var_to_obj)
            memory["absolute_params"] = state_params + params
        # Check if initiable based on preconditions.
        grounded_op = self.operator.ground(tuple(objects))
        return all(pre.holds(state) for pre in grounded_op.preconditions)

    def _regressor_based_policy(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
        if self._is_parameterized:
            # Compute the updated relative goal.
            assert np.allclose(params, memory["params"])
            var_to_obj = dict(zip(self.operator.parameters, objects))
            state_params = _create_absolute_option_param(
                state, self._changing_var_to_feat, self._changing_var_order,
                var_to_obj)
            relative_goal_vec = memory["absolute_params"] - state_params
        else:
            relative_goal_vec = []
        x = np.hstack(([1.0], state.vec(objects), relative_goal_vec))
        action_arr = self._regressor.predict(x)
        if np.isnan(action_arr).any():
            raise OptionExecutionFailure("Option policy returned nan.")
        # Convert the action back to the original space.
        action_arr = self._action_converter.reduced_to_env(action_arr)
        # Clip the action.
        if CFG.env == "robo_kitchen":
            # Clip the action size to the robo_kitchen action space
            action_arr = action_arr[:len(self._action_space.low)]
            
        action_arr = np.clip(action_arr, self._action_space.low,
                             self._action_space.high)
        return Action(np.array(action_arr, dtype=np.float32))

    def _optimized_effect_based_terminal(self, state: State, memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> bool:
        if self._is_parameterized:
            assert np.allclose(params, memory["params"])
        terminate = self.effect_based_terminal(state, objects)
        # Optimization: remember the most recent state and terminate early if
        # the state is repeated, since this option will never get unstuck.
        if "last_state" in memory and memory["last_state"].allclose(state):
            return True
        if terminate:
            return True
        memory["last_state"] = state
        return False

    def effect_based_terminal(self, state: State,
                              objects: Sequence[Object]) -> bool:
        """Terminate when the option's corresponding operator's effects have
        been reached."""
        grounded_op = self.operator.ground(tuple(objects))
        if all(e.holds(state) for e in grounded_op.add_effects) and \
            not any(e.holds(state) for e in grounded_op.delete_effects):
            return True
        return False

    def get_rel_option_param_from_state(self, state: State, memory: Dict,
                                        objects: Sequence[Object]) -> Array:
        """Get the relative parameter that is passed into the option's
        regressor."""
        var_to_obj = dict(zip(self.operator.parameters, objects))
        curr_state_changing_feat = _create_absolute_option_param(
            state, self._changing_var_to_feat, self._changing_var_order,
            var_to_obj)
        subgoal_state_changing_feat = memory["absolute_params"]
        relative_param = subgoal_state_changing_feat - curr_state_changing_feat
        return relative_param


class _BehaviorCloningOptionLearner(_OptionLearnerBase):
    """Learn _LearnedNeuralParameterizedOption objects by behavior cloning.

    See the docstring for _LearnedNeuralParameterizedOption for a description
    of the option structure.

    In this paradigm, the option initiable and termination are determined from
    the operators, so the main thing that needs to be learned is the option
    policy. We learn this policy by behavior cloning (fitting a regressor
    via supervised learning) in learn_option_specs().

    The is_parameterized kwarg is for a baseline that learns a policy without
    continuous parameters. If it is False, the parameter space is null.
    """

    def __init__(self,
                 action_space: Box,
                 is_parameterized: bool = True) -> None:
        super().__init__()
        # Actions are clipped to stay within the action space.
        self._action_space = action_space
        # Actions can be converted to a reduced space for learning.
        self._action_converter = create_action_converter()
        # See class docstring.
        self._is_parameterized = is_parameterized
        # While learning the policy, we record the map from each segment to
        # the option parameterization, so we don't need to recompute it in
        # update_segment_from_option_spec.
        self._segment_to_grounding: Dict[Segment, Tuple[Sequence[Object],
                                                        Array]] = {}

    @abc.abstractmethod
    def _create_regressor(self) -> Regressor:
        raise NotImplementedError("Override me!")

    def learn_option_specs(self, 
                           strips_ops: List[STRIPSOperator], # List of operators to learn options for, each operator has a datastore
                           datastores: List[Datastore] # List of datastores for each operator, each containing a list of segments
                           ) -> List[OptionSpec]:
        option_specs: List[Tuple[ParameterizedOption, List[Variable]]] = []

        assert len(strips_ops) == len(datastores)

        for op, datastore in zip(strips_ops, datastores):
            logging.info(f"\nLearning option for NSRT {op.name}")

            X_regressor: List[Array] = []
            Y_regressor = []

            # The superset of all objects whose states we may include in the
            # option params is op.parameters. But we only want to include
            # those objects that have state changes (in at least some data).
            # We do this for learning efficiency; including all objects would
            # likely work too, but may require more data for the model to
            # realize that those objects' parameters can be ignored.
            # Furthermore, we only want to include the features of the objects
            # that exhibit some change in the data.
            changing_var_to_feat = self._get_changing_features(datastore)
            # Just to avoid confusion, we will insist that the order of the
            # changing parameters is consistent with the order of the operator
            # parameters.
            changing_var_order = sorted(changing_var_to_feat,
                                        key=op.parameters.index)
            for segment, var_to_obj in datastore:
                all_objects_in_operator = [
                    var_to_obj[v] for v in op.parameters
                ]

                # We're accomplishing two things here: (1) computing
                # option_param so it can later be stored into the segment, and
                # (2) computing final_param which is the absolute goal vector
                # so we can later compute relative goal vectors when we iterate
                # through the segment's states (below).
                if self._is_parameterized:
                    init_state = segment.states[0]
                    final_state = segment.states[-1]
                    init_param = _create_absolute_option_param( # this is pose of init_state, only changing_var_to_feat is included
                        init_state, changing_var_to_feat, changing_var_order,
                        var_to_obj)
                    final_param = _create_absolute_option_param(
                        final_state, changing_var_to_feat, changing_var_order,
                        var_to_obj)
                    option_param = final_param - init_param
                else:
                    option_param = np.zeros((0, ), dtype=np.float32)

                # Store the option parameterization for this segment so we can
                # use it in update_segment_from_option_spec.
                self._segment_to_grounding[segment] = (all_objects_in_operator,
                                                       option_param)
                del option_param  # not used after this

                # Next, create the input vectors from all object states named
                # in the operator parameters (not just the changing ones).
                # Note that each segment contributions multiple data points,
                # one per action.
                assert len(segment.states) == len(segment.actions) + 1
                for state, action in zip(segment.states, segment.actions):
                    state_features = state.vec(all_objects_in_operator)
                    state_features = _flatten_and_convert_to_array(state_features)
                    if self._is_parameterized:
                        # Compute the relative goal vector for this segment.
                        state_param = _create_absolute_option_param(
                            state, changing_var_to_feat, changing_var_order,
                            var_to_obj)
                        rel_goal_vec = (final_param - state_param).tolist()
                    else:
                        rel_goal_vec = []
                    # Add a bias term for regression.
                    x = np.hstack(([1.0], state_features, rel_goal_vec))
                    X_regressor.append(x)
                    # Convert to the reduced action space.
                    arr = self._action_converter.env_to_reduced(action.arr)
                    Y_regressor.append(arr)

            X_arr_regressor = np.array(X_regressor, dtype=np.float32)
            Y_arr_regressor = np.array(Y_regressor, dtype=np.float32)
            regressor = self._create_regressor()
            logging.info("Fitting regressor with X shape: "
                         f"{X_arr_regressor.shape}, Y shape: "
                         f"{Y_arr_regressor.shape}.")
            regressor.fit(X_arr_regressor, Y_arr_regressor)

            # Construct the ParameterizedOption for this operator.
            name = f"{op.name}LearnedOption"
            parameterized_option = _LearnedNeuralParameterizedOption(
                name,
                op,
                regressor,
                changing_var_to_feat,
                changing_var_order,
                self._action_space,
                self._action_converter,
                is_parameterized=self._is_parameterized)
            option_specs.append((parameterized_option, list(op.parameters)))

        return option_specs

    @staticmethod
    def _get_changing_features(
            datastore: Datastore) -> Dict[Variable, List[int]]:
        """Returns a dict from variables to changing feature indices.

        If a variable has no changing feature indices, it is not
        included in the returned dict.
        """

        def compare_nested_arrays(a, b, atol=1e-3):
            """Helper function to compare potentially nested arrays with tolerance."""
            # Special Case: Both are numpy arrays with object data (contains other arrays)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == object and b.dtype == object:
                if a.shape != b.shape:
                    raise ValueError("Arrays have different shapes.")
                return all(compare_nested_arrays(x, y, atol) for x, y in zip(a, b))
            # Original Case
            return np.allclose(a, b, atol=atol)

        # Create sets of features first, because we want to use set update,
        # but then convert the features to a sorted list at the end.
        changing_var_to_feat_set: Dict[Variable, Set[int]] = {}
        for segment, var_to_obj in datastore:
            start = segment.states[0]
            end = segment.states[-1]
            for v, o in var_to_obj.items():
                # if np.allclose(start[o], end[o]):
                if compare_nested_arrays(start[o], end[o]):
                    continue
                if v not in changing_var_to_feat_set:
                    changing_var_to_feat_set[v] = set()
                changed_indices = set()
                for i in range(len(start[o])):
                    if isinstance(start[o][i], np.ndarray):
                        for j in range(len(start[o][i])):
                            if abs(start[o][i][j] - end[o][i][j]) > 1e-7:
                                changed_indices.add((i, j))
                    elif abs(start[o][i] - end[o][i]) > 1e-7:
                        changed_indices.add(i)
                changing_var_to_feat_set[v].update(changed_indices)
        changing_var_to_feat = {
            v: sorted(f)
            for v, f in changing_var_to_feat_set.items()
        }
        return changing_var_to_feat

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        objects, params = self._segment_to_grounding[segment]
        param_opt, opt_vars = option_spec
        assert all(o.type == v.type for o, v in zip(objects, opt_vars))
        option = param_opt.ground(objects, params)
        segment.set_option(option)


class _DirectBehaviorCloningOptionLearner(_BehaviorCloningOptionLearner):
    """Use an MLPRegressor for regression."""

    def _create_regressor(self) -> Regressor:
        return MLPRegressor(seed=CFG.seed,
                            hid_sizes=CFG.mlp_regressor_hid_sizes,
                            max_train_iters=CFG.mlp_regressor_max_itr,
                            clip_gradients=CFG.mlp_regressor_clip_gradients,
                            clip_value=CFG.mlp_regressor_gradient_clip_value,
                            learning_rate=CFG.learning_rate,
                            weight_decay=CFG.weight_decay,
                            use_torch_gpu=CFG.use_torch_gpu,
                            train_print_every=CFG.pytorch_train_print_every)


class _DSOptionLearner(_OptionLearnerBase):

    def __init__(self, action_space: Box, is_parameterized: bool = True) -> None:
        super().__init__()
        # Actions are clipped to stay within the action space.
        self._action_space = action_space
        # Actions can be converted to a reduced space for learning.
        self._action_converter = create_action_converter()
        # See class docstring.
        self._is_parameterized = is_parameterized
        # While learning the policy, we record the map from each segment to
        # the option parameterization, so we don't need to recompute it in
        # update_segment_from_option_spec.
        self._segment_to_grounding: Dict[Segment, Tuple[Sequence[Object],
                                                        Array]] = {}

    def learn_option_specs(self, 
                           strips_ops: List[STRIPSOperator], # List of operators to learn options for, each operator has a datastore
                           datastores: List[Datastore] # List of datastores for each operator, each containing a list of segments
                           ) -> List[OptionSpec]:
        option_specs: List[Tuple[ParameterizedOption, List[Variable]]] = []

        assert len(strips_ops) == len(datastores)

        dt = 1 / 60

        for op, datastore in zip(strips_ops, datastores):
            logging.info(f"\nLearning option for NSRT {op.name}")
            
            # Process segments to extract trajectories for DSPolicy
            x = []  # position trajectories
            x_dot = []  # velocity trajectories
            quat = []  # quaternion trajectories
            omega = []  # angular velocity trajectories
            gripper = []  # gripper state trajectories if available
            
            # Process each segment to extract position, orientation and velocity data
            for segment, var_to_obj in datastore:
                # NOTE: we assume the first parameter is the object of interest and the second parameter is the gripper
                obj_of_interest = var_to_obj[op.parameters[0]]
                gripper = var_to_obj[op.parameters[1]]
                
                OOI_pos_traj = []
                OOI_quat_traj = []
                gripper_pos_traj = []
                gripper_quat_traj = []
                
                # Extract position and orientation from states
                for state in segment.states:
                    obj_states = state.vec([obj_of_interest, gripper]) # array of shape (4,), each element is an array

                    OOI_pos_traj.append(obj_states[0])
                    OOI_quat_traj.append(obj_states[1])
                    gripper_pos_traj.append(obj_states[2])
                    gripper_quat_traj.append(obj_states[3])

                OOI_pos_traj = np.array(OOI_pos_traj)
                OOI_quat_traj = np.array(OOI_quat_traj)
                OOI_rot_traj = np.array([R.from_quat(q).as_matrix() for q in OOI_quat_traj])
                gripper_pos_traj = np.array(gripper_pos_traj)
                gripper_quat_traj = np.array(gripper_quat_traj)
                gripper_rot_traj = np.array([R.from_quat(q).as_matrix() for q in gripper_quat_traj])

                gripper_pos_traj_OOI_frame, gripper_rot_traj_OOI_frame = transform_frame(gripper_pos_traj, gripper_rot_traj, OOI_pos_traj, OOI_rot_traj)
                gripper_quat_traj_OOI_frame = np.array([R.from_matrix(r).as_quat() for r in gripper_rot_traj_OOI_frame])
                gripper_vel_traj_OOI_frame, gripper_ang_vel_traj_OOI_frame = compute_vel_traj(gripper_pos_traj_OOI_frame, gripper_rot_traj_OOI_frame, dt)

                # Add segment data to overall dataset
                x.append(gripper_pos_traj_OOI_frame)
                x_dot.append(gripper_vel_traj_OOI_frame)
                quat.append(gripper_quat_traj_OOI_frame)
                omega.append(gripper_ang_vel_traj_OOI_frame)

            check_DSPolicy_input_data(x, x_dot, quat, omega, gripper)
            
            # Configure DS Policy
            unified_config = UnifiedModelConfig(
                mode="se3_lpvds",
                k_init=1
            )
            
            # Create DSPolicy
            ds_policy = DSPolicy(
                x=x,
                x_dot=x_dot,
                quat=quat,
                omega=omega,
                gripper=gripper,
                unified_config=unified_config,
                dt=dt,
                switch=False
            )
            
            # Create a ParameterizedOption that uses DSPolicy
            name = f"{op.name}DSOption"
            parameterized_option = _LearnedDSParameterizedOption(
                name,
                op,
                ds_policy,
                is_parameterized=self._is_parameterized
            )
            
            option_specs.append((parameterized_option, list(op.parameters)))
            
        return option_specs
    
    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        objects, params = self._segment_to_grounding[segment]
        param_opt, opt_vars = option_spec
        assert all(o.type == v.type for o, v in zip(objects, opt_vars))
        option = param_opt.ground(objects, params)
        segment.set_option(option)


def check_DSPolicy_input_data(x: List[np.ndarray], x_dot: List[np.ndarray], quat: List[np.ndarray], omega: List[np.ndarray], gripper: List[np.ndarray]) -> bool:
    assert len(x) == len(x_dot) == len(quat) == len(omega)
    for i in range(len(x)):
        assert x[i].shape[0] == x_dot[i].shape[0] == quat[i].shape[0] == omega[i].shape[0]
        assert x[i].shape[1] == 3
        assert x_dot[i].shape[1] == 3
        assert quat[i].shape[1] == 4
        assert omega[i].shape[1] == 3
    return True


class _LearnedDSParameterizedOption(ParameterizedOption):
    """
    A parameterized option that uses DSPolicy for action selection.
    """
    
    def __init__(self,
                 name: str,
                 operator: STRIPSOperator,
                 ds_policy: DSPolicy,  # DSPolicy object
                 is_parameterized: bool = True) -> None:
        types = [v.type for v in operator.parameters]
        self.operator = operator
        self._ds_policy = ds_policy
        self._is_parameterized = is_parameterized
        super().__init__(name,
                         types,
                         params_space=Box(0, 1, (0, ), dtype=np.float32),
                         policy=self._DS_based_policy,
                         initiable=self._precondition_based_initiable,
                         terminal=self._optimized_effect_based_terminal)
    
    def _precondition_based_initiable(self, state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
        # Check if initiable based on preconditions.
        grounded_op = self.operator.ground(tuple(objects))
        return all(pre.holds(state) for pre in grounded_op.preconditions)
    
    def _DS_based_policy(self, state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> Action:
        # NOTE: assume we have 3 objects: obj_of_interest, gripper, base
        obj_of_interest = objects[0]
        gripper = objects[1]
        base = objects[2]

        OOI_pos = state.get(obj_of_interest, "translation")
        OOI_quat = state.get(obj_of_interest, "quaternion")
        OOI_rot = R.from_quat(OOI_quat).as_matrix()
        gripper_pos = state.get(gripper, "translation")
        gripper_quat = state.get(gripper, "quaternion")
        base_pos = state.get(base, "translation")
        base_quat = state.get(base, "quaternion")
        base_rot = R.from_quat(base_quat).as_matrix()
        gripper_pos_OOI_frame, gripper_rot_OOI_frame = transform_frame(gripper_pos, gripper_quat, OOI_pos, OOI_rot)
        gripper_quat_OOI_frame = R.from_matrix(gripper_rot_OOI_frame).as_quat()
        
        
        # Get action from DS Policy
        action = self._ds_policy.get_action(
            np.concatenate([gripper_pos_OOI_frame, gripper_quat_OOI_frame]), 
            clf=True,  # Use Control Lyapunov Function
            alpha_V=10.0,  # CLF parameter
            lookahead=5  # Number of steps to look ahead
        )
        pos_vel_OOI_frame = action[:3]
        ang_vel_OOI_frame = action[3:]
        pos_vel_world_frame = OOI_rot @ pos_vel_OOI_frame
        pos_vel_base_frame = base_rot.T @ pos_vel_world_frame
        ang_vel_world_frame = OOI_rot @ ang_vel_OOI_frame
        ang_vel_base_frame = base_rot.T @ ang_vel_world_frame
        
        mag = np.linalg.norm(ang_vel_base_frame)
        if mag > 1:
            ang_vel_base_frame = ang_vel_base_frame / mag
        
        action_arr = np.zeros(7, dtype=np.float32)
        action_arr[:3] = pos_vel_base_frame
        action_arr[3:6] = ang_vel_base_frame
        action_arr[6] = 0.0 # gripper
        
        action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        action_arr = np.clip(action_arr, action_low, action_high)
        
        return Action(action_arr)
    
    def _optimized_effect_based_terminal(self, state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> bool:
        # NOTE: based on optimized_effect_based_terminal in _LearnedNeuralParameterizedOption
        if self._is_parameterized:
            assert np.allclose(params, memory["params"])
        terminate = self.effect_based_terminal(state, objects)
        # Optimization: remember the most recent state and terminate early if
        # the state is repeated, since this option will never get unstuck.
        if "last_state" in memory and memory["last_state"].allclose(state):
            return True
        if terminate:
            return True
        memory["last_state"] = state
        return False
    
    def effect_based_terminal(self, state: State,
                            objects: Sequence[Object]) -> bool:
        """Terminate when the option's corresponding operator's effects have
        been reached."""
        # NOTE: based on effect_based_terminal in _LearnedNeuralParameterizedOption
        grounded_op = self.operator.ground(tuple(objects))
        if all(e.holds(state) for e in grounded_op.add_effects) and \
           not any(e.holds(state) for e in grounded_op.delete_effects):
            return True
        return False


class _ImplicitBehaviorCloningOptionLearner(_BehaviorCloningOptionLearner):
    """Use an ImplicitMLPRegressor for regression."""

    def _create_regressor(self) -> Regressor:
        # Pull out the constants that have long names.
        num_neg = CFG.implicit_mlp_regressor_num_negative_data_per_input
        num_sam = CFG.implicit_mlp_regressor_num_samples_per_inference
        num_itr = CFG.implicit_mlp_regressor_derivative_free_num_iters
        sigma = CFG.implicit_mlp_regressor_derivative_free_sigma_init
        shrink_scale = CFG.implicit_mlp_regressor_derivative_free_shrink_scale
        num_ticks = CFG.implicit_mlp_regressor_grid_num_ticks_per_dim
        return ImplicitMLPRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.implicit_mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every,
            num_negative_data_per_input=num_neg,
            num_samples_per_inference=num_sam,
            temperature=CFG.implicit_mlp_regressor_temperature,
            inference_method=CFG.implicit_mlp_regressor_inference_method,
            derivative_free_num_iters=num_itr,
            derivative_free_sigma_init=sigma,
            derivative_free_shrink_scale=shrink_scale,
            grid_num_ticks_per_dim=num_ticks)


class _RLOptionLearnerBase(abc.ABC):
    """Struct defining an option learner that learns via reinforcement
    learning, which has an abstract method for updating the policy associated
    with an option."""

    @abc.abstractmethod
    def update(
        self, option: _LearnedNeuralParameterizedOption,
        experience: List[List[Tuple[State, Array, Action, int, State]]]
    ) -> _LearnedNeuralParameterizedOption:
        """Updates a _LearnedNeuralParameterizedOption via reinforcement
        learning.

        The inner list of `experience` corresponds to the expeirence
        from one execution of the option.
        """
        raise NotImplementedError("Override me!")


class _DummyRLOptionLearner(_RLOptionLearnerBase):
    """Does not update the policy associated with a
    _LearnedNeuralParameterizedOption."""

    def update(
        self, option: _LearnedNeuralParameterizedOption,
        experience: List[List[Tuple[State, Array, Action, int, State]]]
    ) -> _LearnedNeuralParameterizedOption:
        # Don't actually update the option at all.
        # Update would be made to option._regressor, which requires changing the
        # code in ml_models.py so that you can train without re-initializing the
        # network. Update would also be made to the policy of the parameterized
        # option itself, e.g. to perform both exploitation and exploration.
        return copy.deepcopy(option)


def _create_absolute_option_param(state: State,
                                  changing_var_to_feat: Dict[Variable,
                                                             List[int]],
                                  var_order: Sequence[Variable],
                                  var_to_obj: VarToObjSub) -> Array:
    """From state, which includes objects, extract only changing variables according to changing_var_to_feat"""
    vec = []
    for v in var_order:
        obj = var_to_obj[v]
        obj_vec = state[obj]
        for idx in changing_var_to_feat[v]:
            if isinstance(idx, (list, tuple)):
                i, j = idx
                vec.append(obj_vec[i][j])
            else:
                vec.append(obj_vec[idx])
    return np.array(vec, dtype=np.float32)


def _flatten_and_convert_to_array(vec: List) -> Array:
    """Flatten any nested arrays in the vector and convert to a numpy array.
    
    Args:
        vec: A list that may contain numpy arrays
        
    Returns:
        A flattened numpy array
    """
    if any(isinstance(x, np.ndarray) for x in vec):
        # Flatten any nested arrays
        flattened_vec = []
        for item in vec:
            if isinstance(item, np.ndarray):
                flattened_vec.extend(item.flatten())
            else:
                flattened_vec.append(item)
        return np.array(flattened_vec, dtype=np.float32)
    else:
        return np.array(vec, dtype=np.float32)
