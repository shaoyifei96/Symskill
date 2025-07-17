"""An approach that invents predicates by clustering features and selecting via
beam search."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
import os
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
from itertools import combinations_with_replacement, product

import numpy as np
from gym.spaces import Box
# We will use Agglomerative Clustering as described.
# May need `pip install scikit-learn`
from predicators.envs import get_or_create_env
from predicators.envs.robo_kitchen import RoboKitchenEnv
from sklearn.cluster import AgglomerativeClustering, DBSCAN
# Import HDBSCAN (may need `pip install hdbscan`)
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R
# Need linalg for inverse and norm
from numpy.linalg import inv, norm, det, LinAlgError

import matplotlib
# Import dill for pickling
import dill as pkl

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import _BinaryClassifier, _ProgrammaticClassifier, _UnaryClassifier
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.planning import PlanningFailure, PlanningTimeout, run_task_plan_once
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtomTrajectory, NSRT, LiftedAtom, Object, ParameterizedOption, Predicate, Segment, State, Task, Type, STRIPSOperator, GroundAtom, DummyPredicate
import warnings
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse # For 2D ellipses
import numpy.linalg # For eigh
import matplotlib.cm as cm # Import cm for colormaps
import matplotlib.colors as mcolors # Import colors for normalization
import ruptures as rpt
import os
from ds_policy import DSPolicy, compute_vel_traj, UnifiedModelConfig
################################################################################
#                          Programmatic classifiers                            #
################################################################################


@dataclass(frozen=True, eq=False, repr=False)
class _NegationClassifier(_ProgrammaticClassifier):
    """Negate a given classifier."""

    body: Predicate

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        return not self.body.holds(s, o)

    def __str__(self) -> str:
        return f"NOT-{self.body}"

    def pretty_str(self) -> Tuple[str, str]:
        vars_str, body_str = self.body.pretty_str()
        return vars_str, f"¬{body_str}"

    def __getattr__(self, name: str) -> Any:
        """Expose attributes of the body predicate's classifier."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(self.body._classifier, name)
        except AttributeError as e:
            # Raise a new AttributeError to make it clear the attribute
            # was not found on _NegationClassifier or its body's classifier.
            raise AttributeError(f"'{type(self).__name__}' object and its body's classifier " f"have no attribute '{name}'") from e

@dataclass(frozen=True, eq=False, repr=False)
class _RelativeFeatureCovClusterClassifierTransRot(_BinaryClassifier):
    """Classifies based on the Mahalanobis distance of a relative feature vector
    (including 7D pose) between two objects to a target cluster center and covariance.

    Uses the provided covariance matrix to define the cluster boundary.
    Classification is True if the Mahalanobis distance squared is less than or
    equal to a threshold derived from the Chi-squared distribution.

    The inverse covariance and threshold must be pre-calculated and passed in.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str # Will be "pose" for SE(3) clusters
    cluster_id: int
    trans_center: np.ndarray # Can be 7D for pose
    rot_center: R
    inv_covariance_matrix_trans: np.ndarray # MUST be provided
    inv_covariance_matrix_rot: np.ndarray # MUST be provided
    mahalanobis_threshold_trans: float     # MUST be provided
    mahalanobis_threshold_rot: float     # MUST be provided

    # Feature name constants for convenience
    _pose_feat_name: str = field(default="pose", init=False)
    _trans_feat_name: str = field(default="translation", init=False)
    _quat_feat_name: str = field(default="quaternion", init=False)

    # __post_init__ is removed

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        """Classify based on Mahalanobis distance using pre-calculated covariance."""
        assert obj1.is_instance(self.object1_type)
        assert obj2.is_instance(self.object2_type)
        assert self.feature_name == self._pose_feat_name
        relative_pose = utils.calculate_relative_pose(s, obj1, obj2,
                                                    self._trans_feat_name,
                                                    self._quat_feat_name)
        if relative_pose is None:
            logging.debug(f"Could not compute relative pose for classification between {obj1}, {obj2}. Returning False.")
            return False # Cannot classify if pose cannot be computed

        # Ensure feature is numpy array for Mahalanobis calculation
        relative_pose = np.array(relative_pose, dtype=self.trans_center.dtype)
        relative_trans = relative_pose[:3] # Use only translation part for Mahalanobis
        relative_rot = R.from_quat(relative_pose[3:])

        # Calculate Mahalanobis distance squared
        diff_trans = relative_trans - self.trans_center
        diff_rot = (self.rot_center.inv() * relative_rot).as_rotvec()
        # Perform calculation: diff.T @ inv_cov @ diff
        mahalanobis_dist_sq_trans = diff_trans.T @ self.inv_covariance_matrix_trans @ diff_trans
        mahalanobis_dist_sq_rot = diff_rot.T @ self.inv_covariance_matrix_rot @ diff_rot
        
        # If result is a 1x1 matrix, extract the scalar value
        if isinstance(mahalanobis_dist_sq_trans, np.ndarray) and mahalanobis_dist_sq_trans.size == 1:
            mahalanobis_dist_sq_trans = mahalanobis_dist_sq_trans.item()
        if isinstance(mahalanobis_dist_sq_rot, np.ndarray) and mahalanobis_dist_sq_rot.size == 1:
            mahalanobis_dist_sq_rot = mahalanobis_dist_sq_rot.item()

        # Use the pre-calculated threshold
        return mahalanobis_dist_sq_trans <= self.mahalanobis_threshold_trans and mahalanobis_dist_sq_rot <= self.mahalanobis_threshold_rot

    def __str__(self) -> str:
        # Indicate covariance-based cluster in the name
        return (f"RelCovCluster-{CFG.robo_kitchen_task}-{self.object2_type.name}-in-{self.object1_type.name}-frame-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description referencing Mahalanobis distance
        name1 = CFG.grammar_search_classifier_pretty_str_names[0]
        name2 = CFG.grammar_search_classifier_pretty_str_names[1]
        vars_str = f"{name1}:{self.object1_type.name}, {name2}:{self.object2_type.name}"

        # Adapt feature description based on name
        if self.feature_name == self._pose_feat_name:
             feat_desc = f"RelPose({name1}, {name2})"
        else:
             # Generic diff representation or use feature name directly
             feat_desc = f"Diff({name1}.{self.feature_name}, {name2}.{self.feature_name})"

        # Use the pre-calculated threshold
        body_str = (f"MahaDistSq({feat_desc}, Cluster-{self.feature_name}-ID{self.cluster_id}) "
                    f"<= {self.mahalanobis_threshold:.3f}")
        return vars_str, body_str
@dataclass(frozen=True, eq=False, repr=False)
class _RelativeFeatureCovClusterClassifier(_BinaryClassifier):
    """Classifies based on the Mahalanobis distance of a relative feature vector
    (including 7D pose) between two objects to a target cluster center and covariance.

    Uses the provided covariance matrix to define the cluster boundary.
    Classification is True if the Mahalanobis distance squared is less than or
    equal to a threshold derived from the Chi-squared distribution.

    The inverse covariance and threshold must be pre-calculated and passed in.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str # Will be "pose" for SE(3) clusters
    cluster_center: np.ndarray # Can be 7D for pose
    cluster_cov: np.ndarray # Covariance matrix for the cluster
    diff_fn: Optional[Callable[[Any, Any], Any]]
    cluster_id: int
    inv_covariance_matrix: np.ndarray # MUST be provided
    mahalanobis_threshold: float     # MUST be provided

    # Feature name constants for convenience
    _pose_feat_name: str = field(default="pose", init=False)
    _trans_feat_name: str = field(default="translation", init=False)
    _quat_feat_name: str = field(default="quaternion", init=False)

    # __post_init__ is removed

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        """Classify based on Mahalanobis distance using pre-calculated covariance."""
        assert obj1.is_instance(self.object1_type)
        assert obj2.is_instance(self.object2_type)

        # Calculate the relevant relative feature
        relative_feature = None
        if self.feature_name == self._pose_feat_name:
            relative_feature = utils.calculate_relative_pose(s, obj1, obj2,
                                                       self._trans_feat_name,
                                                       self._quat_feat_name)
            if relative_feature is None:
                logging.debug(f"Could not compute relative pose for classification between {obj1}, {obj2}. Returning False.")
                return False # Cannot classify if pose cannot be computed
        else:
            # If supporting other features, add logic here
            raise ValueError(f"Unsupported feature name: {self.feature_name}")

        # Ensure feature is numpy array for Mahalanobis calculation
        relative_feature = np.array(relative_feature, dtype=self.cluster_center.dtype)
        relative_feature = relative_feature[:3] # Use only translation part for Mahalanobis

        # Calculate Mahalanobis distance squared
        diff = relative_feature - self.cluster_center[:3]
        try:
            # Ensure diff is a column vector for matrix multiplication if it's 1D
            if diff.ndim == 1:
                diff = diff[:, np.newaxis]
            # Perform calculation: diff.T @ inv_cov @ diff
            mahalanobis_dist_sq = diff.T @ self.inv_covariance_matrix @ diff
            # If result is a 1x1 matrix, extract the scalar value
            if isinstance(mahalanobis_dist_sq, np.ndarray) and mahalanobis_dist_sq.size == 1:
                mahalanobis_dist_sq = mahalanobis_dist_sq.item()
        except ValueError as e:
            logging.error(f"Error calculating Mahalanobis distance for {self}: {e}")
            logging.error(f"Shapes: diff.T: {diff.T.shape}, inv_covariance_matrix: {self.inv_covariance_matrix.shape}, diff: {diff.shape}")
            logging.error(f"Relative feature: {relative_feature}, Cluster center: {self.cluster_center}")
            return False

        # Use the pre-calculated threshold
        return mahalanobis_dist_sq <= self.mahalanobis_threshold

    def __str__(self) -> str:
        # Indicate covariance-based cluster in the name
        return (f"RelCovCluster-{CFG.robo_kitchen_task}-{self.object2_type.name}-in-{self.object1_type.name}-frame-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description referencing Mahalanobis distance
        name1 = CFG.grammar_search_classifier_pretty_str_names[0]
        name2 = CFG.grammar_search_classifier_pretty_str_names[1]
        vars_str = f"{name1}:{self.object1_type.name}, {name2}:{self.object2_type.name}"

        # Adapt feature description based on name
        if self.feature_name == self._pose_feat_name:
             feat_desc = f"RelPose({name1}, {name2})"
        else:
             # Generic diff representation or use feature name directly
             feat_desc = f"Diff({name1}.{self.feature_name}, {name2}.{self.feature_name})"

        # Use the pre-calculated threshold
        body_str = (f"MahaDistSq({feat_desc}, Cluster-{self.feature_name}-ID{self.cluster_id}) "
                    f"<= {self.mahalanobis_threshold:.3f}")
        return vars_str, body_str


@dataclass(frozen=True, eq=False, repr=False)
class _RelativeFeatureClusterClassifier(_BinaryClassifier):
    """Classifies based on the Mahalanobis distance of a relative feature vector
    (including 7D pose) between two objects to a target cluster center and covariance.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str # Will be "pose" for SE(3) clusters
    cluster_center: np.ndarray # Can be 7D for pose
    cluster_radius: float
    # diff_fn might not be needed for 'pose' if calculation is explicit
    diff_fn: Optional[Callable[[Any, Any], Any]] # Made optional
    cluster_id: int

    # Cache feature names
    _trans_feat_name: str = field(default="translation", init=False)
    _quat_feat_name: str = field(default="quaternion", init=False)
    _pose_feat_name: str = field(default="pose", init=False)


    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        assert obj1.is_instance(self.object1_type)
        assert obj2.is_instance(self.object2_type)

        # Calculate the relevant relative feature
        if self.feature_name == self._pose_feat_name:
            # Calculate the 7D relative pose
            relative_feature = utils.calculate_relative_pose(s, obj1, obj2, 
                                                       self._trans_feat_name, 
                                                       self._quat_feat_name)
            if relative_feature is None:
                logging.warning(f"Could not compute relative pose for classification between {obj1}, {obj2}. Returning False.")
                return False # Cannot classify if pose cannot be computed
            dist_diff = utils.calculate_se3_distance(relative_feature, self.cluster_center, 
                                                    CFG.clustering_se3_trans_weight, 
                                                    CFG.clustering_se3_rot_weight)
            return dist_diff <= self.cluster_radius
        else:
            # Handle original features (e.g., translation only, rotation only if kept)
            obj1_feat = s.get(obj1, self.feature_name)
            obj2_feat = s.get(obj2, self.feature_name)
            
            # Special handling for local frame translation (if kept as separate feature)
            if self.feature_name == self._trans_feat_name and self._quat_feat_name in obj1.type.feature_names:
                try:
                    obj1_quat = s.get(obj1, self._quat_feat_name)
                    obj1_rot = R.from_quat(obj1_quat)
                    world_diff = np.subtract(obj2_feat, obj1_feat)
                    relative_feature = obj1_rot.inv().apply(world_diff)
                except KeyError:
                    logging.warning(f"Missing quaternion for {obj1}, cannot compute relative translation for classifier.")
                    return False
            elif self.diff_fn is not None:
                 # Use the provided difference function for other features.
                 relative_feature = np.array(self.diff_fn(obj1_feat, obj2_feat), dtype=self.cluster_center.dtype)
            else:
                 logging.error(f"Missing diff_fn for non-pose feature {self.feature_name} in classifier {self}")
                 return False # Cannot compute difference

        # Ensure feature is numpy array for Mahalanobis calculation
        relative_feature = np.array(relative_feature, dtype=self.cluster_center.dtype)

        # Calculate Mahalanobis distance squared
        diff = relative_feature - self.cluster_center
        try:
            if diff.ndim == 1:
                diff = diff[:, np.newaxis] # Ensure column vector
            mahalanobis_dist_sq = diff.T @ self.inv_covariance_matrix @ diff
            if isinstance(mahalanobis_dist_sq, np.ndarray) and mahalanobis_dist_sq.size == 1:
                mahalanobis_dist_sq = mahalanobis_dist_sq.item() # Extract scalar
        except ValueError as e:
            logging.error(f"Error calculating Mahalanobis distance for {self}: {e}")
            logging.error(f"Shapes: diff.T: {diff.T.shape}, inv_covariance_matrix: {self.inv_covariance_matrix.shape}, diff: {diff.shape}")
            logging.error(f"Relative feature: {relative_feature}, Cluster center: {self.cluster_center}")
            return False

        return mahalanobis_dist_sq <= self.mahalanobis_threshold

    def __str__(self) -> str:
        # Keep name format similar, maybe indicate pose explicitly if needed
        prefix = "RelPoseEllipsoidCluster" if self.feature_name == self._pose_feat_name else "RelEllipsoidCluster"
        return (f"{prefix}-{CFG.robo_kitchen_task}-{self.object2_type.name}-in-{self.object1_type.name}-frame-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        name1 = CFG.grammar_search_classifier_pretty_str_names[0]
        name2 = CFG.grammar_search_classifier_pretty_str_names[1]
        vars_str = f"{name1}:{self.object1_type.name}, {name2}:{self.object2_type.name}"
        # Adapt body string for pose
        if self.feature_name == self._pose_feat_name:
             feat_desc = f"RelPose({name1}, {name2})"
        else:
             feat_desc = f"Diff({name1}.{self.feature_name}, {name2}.{self.feature_name})"
        
        body_str = (f"MahaDistSq({feat_desc}, Cluster-{self.feature_name}-ID{self.cluster_id}) "
                    f"<= {self.mahalanobis_threshold:.3f}")
        return vars_str, body_str


@dataclass(frozen=True, eq=False, repr=False)
class _AbsoluteFeatureClusterClassifier(_UnaryClassifier):
    """Classifies based on the Mahalanobis distance of an absolute feature vector
    of an object to a target cluster center and covariance.

    The feature is defined by feature_name for an object of type1.
    The cluster_center is the representative point for this cluster.
    Classification is True if the Mahalanobis distance squared is less than or
    equal to mahalanobis_threshold.
    """
    object_type: Type
    feature_name: str
    cluster_center: np.ndarray
    inv_covariance_matrix: np.ndarray # Store inverse covariance
    mahalanobis_threshold: float    # Store threshold for Mahalanobis distance squared
    cluster_id: int # For unique naming

    def _classify_object(self, s: State, obj: Object) -> bool:
        # Ensure object matches the type this classifier is defined for.
        assert obj.is_instance(self.object_type)
        # Get feature from state.
        obj_feat = np.array(s.get(obj, self.feature_name), dtype=self.cluster_center.dtype)

        # Calculate Mahalanobis distance squared
        diff = obj_feat - self.cluster_center
        try:
             # Ensure diff is a column vector for matrix multiplication if it's 1D
            if diff.ndim == 1:
                diff = diff[:, np.newaxis]
            mahalanobis_dist_sq = diff.T @ self.inv_covariance_matrix @ diff
            # If result is a 1x1 matrix, extract the scalar value
            if isinstance(mahalanobis_dist_sq, np.ndarray) and mahalanobis_dist_sq.size == 1:
                mahalanobis_dist_sq = mahalanobis_dist_sq.item()
        except ValueError as e:
            logging.error(f"Error calculating Mahalanobis distance for {self}: {e}")
            logging.error(f"Shapes: diff.T: {diff.T.shape}, inv_covariance_matrix: {self.inv_covariance_matrix.shape}, diff: {diff.shape}")
            return False # Or handle error differently

        return mahalanobis_dist_sq <= self.mahalanobis_threshold

    def __str__(self) -> str:
        # Generate a unique name based on type, feature, and cluster ID.
        return (f"AbsEllipsoidCluster-{CFG.robo_kitchen_task}-{self.object_type.name}-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description.
        name = CFG.grammar_search_classifier_pretty_str_names[0]
        vars_str = f"{name}:{self.object_type.name}"
        body_str = (f"MahaDistSq({name}.{self.feature_name}, "
                    f"Cluster-{self.feature_name}-ID{self.cluster_id}) <= {self.mahalanobis_threshold:.3f}")
        return vars_str, body_str


################################################################################
#                                 Approach                                     #
################################################################################

class ClusteringSearchInventionApproach(NSRTLearningApproach):
    """An approach that invents predicates via feature clustering in relative frame as predicate
    and then skill learning and operator learning with the predicates."""

    # Caches for expensive computations during beam search
    _atom_dataset_cache: Dict[FrozenSet[Predicate], List[GroundAtomTrajectory]] = {}
    _operator_complexity_cache: Dict[FrozenSet[Predicate], Tuple[int, Set[NSRT]]] = {}
    _segmentation_cache: Dict[FrozenSet[Predicate], int] = {}
    _plan_constraint_cache: Dict[FrozenSet[Predicate], bool] = {}

    @classmethod
    def get_name(cls) -> str:
        return "clustering_invention"

    def _add_additional_remove_effects_operators(self) -> None:
        """Add additional remove effects operators to the loaded operators."""
        # Get a sample state to check which object types actually exist
        env = get_or_create_env(CFG.env)
        ob = env.reset(train_or_test="test", task_idx=0)
        state = env.state_info_to_state(ob["state_info"])
        available_object_types = set()
        for obj in state:
            available_object_types.add(obj.type.name)
        logging.info(f"Object types found in sample state: {available_object_types}")

        updated_nsrts = set()
        for nsrt in self._nsrts:
            new_delete_effects = nsrt.delete_effects
            new_params = nsrt.parameters
            if len(nsrt.add_effects) == 1:
                add_eff = list(nsrt.add_effects)[0]
                if add_eff.entities[1].type.name == "gripper_type":
                    object_in_contact = add_eff.entities[0].type.name
                    # in this case, all other gripper object effects should be added to delete effects
                    for (key1, key2, key3) in CFG.dict_contact_predicate_to_rel_pose_predicates:
                        if key1 == 'InContact' and not key2 == object_in_contact:
                            # Check if this object type actually exists in the current state
                            if key2 not in available_object_types:
                                logging.debug(f"Skipping object type {key2} as it doesn't exist in the current state")
                                continue
                            
                            # this is a gripper object effect that is not the one in contact
                            # we need to add it to the delete effects
                            eff_to_delete = CFG.dict_contact_predicate_to_rel_pose_predicates[key1, key2, key3]
                            for eff in eff_to_delete:
                                # Find the corresponding variables from NSRT parameters
                                # The first variable should be the object type, second should be gripper
                                obj_var = None
                                gripper_var = None
                                for var in nsrt.parameters:
                                    if var.type.name == key2:  # object type
                                        obj_var = var
                                    elif var.type.name == "gripper_type":
                                        gripper_var = var
                                
                                if obj_var is not None and gripper_var is not None:
                                    # Create the LiftedAtom with the correct variables
                                    lifted_atom = LiftedAtom(eff, [obj_var, gripper_var])
                                    new_delete_effects.add(lifted_atom)
                                elif obj_var is None:
                                    extra_param_type = eff.types[0]
                                    new_vars_to_add = utils.create_new_variables([extra_param_type], new_params)
                                    new_params = new_params + new_vars_to_add
                                    obj_var = new_vars_to_add[0]
                                    lifted_atom = LiftedAtom(eff, [obj_var, gripper_var])
                                    new_delete_effects.add(lifted_atom)
                                else:
                                    raise ValueError(f"Could not find object or gripper variable for {eff} in {nsrt.parameters}")
            else:
                logging.warning(f"NSRT {nsrt} has {len(nsrt.add_effects)} add effects, directly adding to updated_nsrts")
            nsrt = nsrt.copy_with(parameters=new_params, delete_effects=new_delete_effects)
            # Add the potentially modified NSRT to the updated set
            updated_nsrts.add(nsrt)
        
        # Update self._nsrts with the modified NSRTs
        self._nsrts = updated_nsrts

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # We need to properly load the learned predicates if they exist
        main_folder = f"{CFG.approach_dir}/"
        all_files = os.listdir(main_folder)
        approach_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith(".NSRTs")]
        contact2rel_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith("_contact2rel_preds.pkl")]
        goal_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith("_gtgoal2dummy_preds.pkl")]

        for file in approach_files:
            with open(file, "rb") as f:
                loaded_nsrts = pkl.load(f)
                self._nsrts.update(loaded_nsrts)
        from predicators.ground_truth_models import get_gt_nsrts
        gt_nsrts = get_gt_nsrts(CFG.env, self._initial_predicates, self._initial_options)
        
        self._nsrts = set(gt_nsrts).union(self._nsrts)

        for file in contact2rel_files:
            with open(file, "rb") as f:
                contact2rel_preds = pkl.load(f)
                for key, value in contact2rel_preds.items():
                    if key not in CFG.dict_contact_predicate_to_rel_pose_predicates:
                        CFG.dict_contact_predicate_to_rel_pose_predicates[key] = value
                    else:
                        CFG.dict_contact_predicate_to_rel_pose_predicates[key].update(value)

        for file in goal_files:
            with open(file, "rb") as f:
                gtgoal2dummy_preds = pkl.load(f)
                for key, value in gtgoal2dummy_preds.items():
                    if key not in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
                        CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key] = value
                    else:
                        CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key].update(value)

        self._add_additional_remove_effects_operators()
        
        if CFG.pretty_print_when_loading:  # pragma: no cover
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            name_map = {}
            logging.info("Invented predicates:")
            for idx, pred in enumerate(sorted(set(preds.values()) - self._initial_predicates)):
                vars_str, body_str = pred.pretty_str()
                logging.info(f"\tP{idx+1}({vars_str}) ≜ {body_str}")
                name_map[body_str] = f"P{idx+1}"
        logging.info("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            if CFG.pretty_print_when_loading:
                logging.info(nsrt.pretty_str(name_map))
            else:
                logging.info(nsrt)
        logging.info("")
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)

        preds, _ = utils.extract_preds_and_types(self._nsrts)
        self._learned_predicates = set(preds.values()) - self._initial_predicates

    def load_learnt_goals(self) -> Set[Predicate]:
        goal_rel_pose_predicates = set()
        negated_goal_predicates = set()

        if CFG.use_learnt_goal_predicates:
            main_folder = f"{CFG.approach_dir}/"
            if os.path.exists(main_folder):
                all_files = os.listdir(main_folder)
                contact2rel_files = [main_folder + f for f in all_files if CFG.robo_kitchen_task not in f and f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith("_contact2rel_preds.pkl")]

                for file in contact2rel_files:
                    with open(file, "rb") as f:
                        contact2rel_preds = pkl.load(f)
                        for key, value in contact2rel_preds.items():
                            if "goal" in key[0]:
                                goal_rel_pose_predicates |= value
            else:
                logging.warning(f"Approach directory {main_folder} does not exist")
            
            if CFG.use_negated_goal_predicates:
                # Generate negated predicates for each goal predicate
                for pred in goal_rel_pose_predicates:
                    negated_classifier = _NegationClassifier(pred)
                    negated_pred_name = f"NOT-{pred.name}"
                    negated_predicate = Predicate(negated_pred_name, pred.types, negated_classifier)
                    negated_goal_predicates.add(negated_predicate)
                
        return goal_rel_pose_predicates | negated_goal_predicates

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    # --- Core Learning Method ---
    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        logging.info("Generating candidate predicates via clustering...")
        # Filter dataset to only keep specific trajectory indices
        if CFG.robo_kitchen_task == "OpenSingleDoor":
            # keep_indices = [0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19, 21, 25, 32, 33, 35, 36, 38, 39, 40, 42, 44, 45, 47, 48, 49]
            keep_indices = [0, 2, 6, 7] # all left cab
            # keep_indices = [0, 2, 5, 6, 7] # all cab
            # keep_indices = [0]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "CloseSingleDoor":
            # keep_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9] # all left close
            # keep_indices = [0, 2, 4, 8, 9] # all microwave
            keep_indices = [1, 3, 6, 7] # all left cab
            # keep_indices = [1]
            # keep_indices = [0, 2, 8, 9] # better microwaves
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPCounterToCab":
            # keep_indices = [4, 6, 7, 12, 23, 33, 39, 44, 45, 46, 48]  # all left cab
            keep_indices = [6, 7, 23, 44, 45, 46]  # all left cab
            # keep_indices = [6, 23]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPCounterToStove":
            remove_indices = [5]
            dataset._trajectories = [dataset._trajectories[i] for i in range(len(dataset._trajectories)) if i not in remove_indices]
        elif CFG.robo_kitchen_task == "TurnOnStove":
            keep_indices = [0, 9, 10, 11, 12, 20, 33, 37, 38, 39, 42, 44, 46] # all counter-clockwise 
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "TurnOffStove":
            keep_indices = [3, 9, 15, 19, 20, 23, 24, 28, 29, 34, 35, 36, 39, 47, 49] #turn off by rotating clockwise
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "CloseDrawer":
            keep_indices = [0, 1, 2, 5, 6] # all left close
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "OpenDrawer":
            keep_indices = [1, 3, 5, 8, 11, 13, 15, 17, 20, 21, 22, 24]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "TurnOnSinkFaucet":
            keep_indices = [0, 1, 4, 5, 9, 18, 21, 23, 24]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        # logging.info(f"Filtered dataset to trajectories (indices: {keep_indices})")
        elif CFG.robo_kitchen_task == "TurnOffSinkFaucet":
            keep_indices = [ 5, 7, 11, 17, 19, 21]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPCabToCounter":
            keep_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        # Clear caches before starting learning
        self._atom_dataset_cache = {}
        self._operator_complexity_cache = {}
        self._segmentation_cache = {}
        self._plan_constraint_cache = {}

        # Add test function call for HDBSCAN if debug flag is set
        if CFG.testing_hdbscan:
            logging.info("Running HDBSCAN test with synthetic data...")
            self._test_clustering_with_dummy_data(
                num_clusters=CFG.test_num_clusters, 
                points_per_cluster=CFG.test_points_per_cluster,
                noise_level=CFG.test_noise_level,
                cluster_separation=CFG.test_cluster_separation
            )
            return  # Skip actual learning if we're just testing

        candidates = {}
        if CFG.predicate_candidates_method == "low_speed":
            logging.info("Generating candidate predicates via low speed method...")
            candidates = self._generate_candidate_predicates(dataset)
            logging.info(f"Generated {len(candidates)} candidate predicates.")
            logging.info(f"Candidate predicates: {candidates}")
            if not candidates:
                logging.warning("No candidate predicates generated. Learning NSRTs with initial predicates only.")
                self._learned_predicates = set()
            else:
                logging.info("Selecting predicates via beam search...")
                self._learned_predicates = self._select_predicates_by_beam_search(candidates, dataset, self._train_tasks)
                logging.info(f"Selected {len(self._learned_predicates)} predicates.")
        elif CFG.predicate_candidates_method == "contact_clustering":
            logging.info("Generating candidate predicates via contact clustering method...")
            og_pred_atom_dataset, cluster_pred_atom_dataset, different_seg_count_trajs, candidates, initial_monitor_preds = self._generate_candidate_predicates_contact_goal_clustering_refactored(dataset)
            self._learned_predicates = set(candidates.keys()) | initial_monitor_preds
        elif CFG.predicate_candidates_method == "motion_analysis_contact": # converting motion analysis to contact predicate 
            logging.info("Generating candidate predicates via motion analysis contact method...")
            og_pred_atom_dataset, cluster_pred_atom_dataset, different_seg_count_trajs, candidates, initial_monitor_preds = self._generate_candidate_predicates_contact_goal_clustering_refactored(dataset)
            self._learned_predicates = set(candidates.keys()) | initial_monitor_preds
        else:
            raise ValueError(f"Invalid predicate candidates method: {CFG.predicate_candidates_method}")
            # self._learned_predicates = self._select_predicates_by_beam_search(candidates, dataset, self._train_tasks)

        # # Save the learned predicates separately for potential reloading
        # save_path = utils.get_approach_save_path_str()
        # learned_preds_path = f"{save_path}_learned_predicates.pkl"
        # # Replace utils.save_to_pickle with direct pkl.dump
        # with open(learned_preds_path, "wb") as f:
        #     pkl.dump(self._learned_predicates, f)

        save_path = utils.get_approach_save_path_str()
        learned_preds_path = f"{save_path}_contact2rel_preds.pkl"
        with open(learned_preds_path, "wb") as f:
            pkl.dump(CFG.dict_contact_predicate_to_rel_pose_predicates, f)

        learned_goal_path = f"{save_path}_gtgoal2dummy_preds.pkl"
        with open(learned_goal_path, "wb") as f:
            pkl.dump(CFG.dict_gt_goal_predicate_to_dummy_goal_predicates, f)

        # Learn NSRTs with the final set of predicates
        # final_predicates = self._get_current_predicates()
        # We need the atom dataset for the final selected predicates
        # atom_dataset_final = self._create_atom_dataset(dataset, final_predicates)
        # annotations = None # Or derive from atom_dataset if needed by _learn_nsrts

        # Segment the trajectories using the final predicates and atom dataset
        # segmented_trajs_final = [
        #     segment_trajectory(ll_traj, final_predicates, atom_seq=atom_seq)
        #     for ll_traj, atom_seq in atom_dataset_final
        trajs = dataset.trajectories
        # Remove trajectories with different segment counts in reverse order
        # to avoid index shifting problems
        for i in sorted(different_seg_count_trajs, reverse=True):
            trajs.pop(i)

        trajs_states_nums = [len(traj.states) for traj in trajs]
        self._metrics['trajs_states_nums'] = trajs_states_nums
        
        # Call learn_nsrts with segmented trajectories
        self._learn_nsrts(
            trajs,
            og_pred_atom_dataset,
            annotations=annotations,
            online_learning_cycle=None,
            passed_in_predicates= self._learned_predicates
        )

    def _get_feature_difference_function(self, feat_name: str) -> Callable:
        if feat_name == "pose":
            return utils.calculate_se3_distance
        else:
            raise ValueError(f"Unsupported feature name: {feat_name}")

    # --- Candidate Generation Functions ---
    def _generate_candidate_predicates(self, dataset: Dataset) -> Dict[Predicate, float]:
        """Generates candidate predicates by clustering relative and absolute features."""
        relative_feature_datasets = self._generate_relative_low_speed_feature_datasets(dataset)
        # absolute_feature_datasets = self._generate_absolute_feature_datasets(dataset)

        candidates: Dict[Predicate, float] = {}
        predicate_counter = 0 # To ensure unique cluster IDs

        # Feature names for special handling



        # Initialize cluster visualization storage attributes
        self._last_cluster_fig = None
        self._last_cluster_ax = None
        self._last_cluster_type1 = None
        self._last_cluster_type2 = None
        self._last_cluster_feat = None
        self._last_cluster_title = None
        self._last_cluster_fname = None

        # Process relative features
        for (type1, type2, feat_name), data in relative_feature_datasets.items():
            logging.debug(f"Clustering relative feature {feat_name} for ({type1.name}, {type2.name}) with {len(data)} points.")

            if not data: continue # Skip if no data collected

            # Save the feature data for analysis and debugging
            feature_key = f"{type1.name}_{type2.name}_{feat_name}"

            # Create directory if it doesn't exist
            os.makedirs("feature_data", exist_ok=True)

            # Save the data to a numpy file
            data_path = f"feature_data/{feature_key}.npy"
            np.save(data_path, np.array(data))

            logging.info(f"Saved {len(data)} data points for feature {feature_key} to {data_path}")

            # Select clustering epsilon based on feature type
            if feat_name == CFG.trans_feat_name:
                epsilon = CFG.clustering_translation_epsilon
            elif feat_name == CFG.quat_feat_name:
                epsilon = CFG.clustering_quaternion_epsilon
            else: #pose_feature
                epsilon = CFG.clustering_epsilon

            # logging.debug(f"Using epsilon: {epsilon:.4f} for feature {feat_name}")
            # Perform clustering
            data_array, labels, unique_labels = self._cluster_feature_dataset(data, epsilon, feat_name)
            diff_fn = self._get_feature_difference_function(feat_name)

            if data_array.size == 0: continue # Skip if clustering returned empty

            min_cluster_size = int(CFG.clustering_min_ratio_of_data * len(data_array))
            logging.debug(f"Using minimum cluster size: {min_cluster_size} ({CFG.clustering_min_ratio_of_data * 100}% of {len(data_array)} data points)")

            # Identify kept clusters based on size
            kept_clusters_info = {}
            discarded_labels = set()

            for k in unique_labels:
                if k == -1: continue # Skip noise points for now
                cluster_points = data_array[labels == k]
                cluster_size = len(cluster_points)
                if cluster_size >= min_cluster_size:
                    # Calculate the mean SE(3) pose for the cluster
                    translations = cluster_points[:, :3]
                    quaternions = cluster_points[:, 3:]

                    # Mean translation is straightforward
                    mean_translation = np.mean(translations, axis=0)

                    # Mean rotation requires specialized handling
                    # try:
                    # Ensure quaternions are valid (non-zero norm) before conversion
                    valid_quats_mask = np.linalg.norm(quaternions, axis=1) > 1e-6
                    if not np.all(valid_quats_mask):
                        raise ValueError("At least one quaternion in cluster is near zero. Skipping this cluster.")

                    # Convert to Rotation objects
                    rotations = R.from_quat(quaternions)
                    # Calculate the mean rotation
                    mean_rotation = rotations.mean()
                    # Convert back to quaternion [qx, qy, qz, qw]
                    mean_quaternion = mean_rotation.as_quat()

                    # Combine mean translation and mean quaternion
                    cluster_center = np.concatenate((mean_translation, mean_quaternion))

                    # logging.warning(f"INCORRECT MEAN CALCULATION:!!!!!!!!!!!!!!!!!!!") # Remove this warning
                    # normalize the quat -- No longer needed as R.mean handles it
                    # cluster_center[3:7] = cluster_center[3:7] / np.linalg.norm(cluster_center[3:7])
                    # difference between cluster_center and cluster_points
                    cluster_center_diff = np.zeros(len(cluster_points))
                    for i in range(len(cluster_points)):
                        diff = utils.calculate_se3_distance(cluster_center, cluster_points[i], 
                                                            CFG.clustering_se3_trans_weight, 
                                                            CFG.clustering_se3_rot_weight)
                        cluster_center_diff[i] = diff

                    # find 95th percentile of cluster_center_diff
                    # cluster_center_diff_90 = np.percentile(cluster_center_diff, 90)
                    # # logging.warning(f"90th percentile of cluster_center_diff: {cluster_center_diff_90:.4f}")
                    # cluster_center_diff_95 = np.percentile(cluster_center_diff, 95)
                    # # logging.warning(f"95th percentile of cluster_center_diff: {cluster_center_diff_95:.4f}")
                    # cluster_center_diff_99 = np.percentile(cluster_center_diff, 99)
                    # logging.warning(f"99th percentile of cluster_center_diff: {cluster_center_diff_99:.4f}")

                    # Store basic info first
                    kept_clusters_info[k] = {'center': cluster_center, 'size': cluster_size, 'points': cluster_points, 'cluster_radius': np.max(cluster_center_diff) } # Store points for cov calculation
                    logging.warning(f"Cluster {k} for {type1.name}-{type2.name}-{feat_name} kept (size {cluster_size} >= {min_cluster_size}).")
                    logging.warning(f"Cluster radius: {np.max(cluster_center_diff)}, Cluster center: {cluster_center}")
                else:
                    discarded_labels.add(k)
                    logging.debug(f"Cluster {k} for {type1.name}-{type2.name}-{feat_name} discarded (size {cluster_size} < {min_cluster_size}).")

            # Now calculate covariance etc. ONLY for kept clusters and add to info dict
            kept_cluster_labels_list = list(kept_clusters_info.keys())
            for cluster_label in kept_cluster_labels_list: # Iterate over keys
                cluster_info = kept_clusters_info[cluster_label]
                cluster_points = cluster_info['points'] # Retrieve stored points
                # compute the SE(3) covariance matrix

            # Now, optionally visualize clusters if in debug mode, passing the *updated* info
            if CFG.clustering_debug and data_array.size > 0: # Check if there is data to plot
                # The kept_clusters_info dict now contains cov matrix and threshold for plot
                self._plot_cluster_results(data_array, labels, unique_labels, kept_clusters_info,
                                           type1.name, type2.name if type2 else None, feat_name)

                # Plot relative trajectories *after* cluster plot, if applicable
                if feat_name == CFG.pose_feature_name and type2 is not None:
                    self._plot_relative_trajectories(dataset, type1.name, type2.name)

            # Sort kept clusters by size (descending) for top_k selection AFTER plotting
            # Filter out any clusters where covariance calculation failed (if needed, though `continue` above handles it)
            # valid_kept_clusters = {k: v for k, v in kept_clusters_info.items() if 'inv_covariance_matrix' in v}
            valid_kept_clusters = kept_clusters_info
            sorted_valid_kept_clusters = sorted(valid_kept_clusters.items(), key=lambda item: item[1]['size'], reverse=True)

            # Create predicates for the top_k *valid* kept clusters
            top_k = min(CFG.clustering_max_clusters, len(sorted_valid_kept_clusters))
            logging.debug(f"Selecting top {top_k} valid kept clusters for {feat_name}:{type1.name}-{type2.name}.")

            for i, (cluster_label, cluster_info) in enumerate(sorted_valid_kept_clusters[:top_k]):
                # logging.debug(f"Creating predicate for kept cluster {cluster_label} (size {cluster_info['size']}, rank {i+1}/{top_k}).")
                # Pass inverse covariance and threshold instead of epsilon
                pred = self._create_predicate_from_relative_cluster( # TODO: THIS IS BROKEN NOW
                    type1, type2, feat_name, cluster_info['center'],
                    cluster_info['cluster_radius'],
                    diff_fn, cluster_label) # Use cluster_label for ID
                candidates[pred] = pred.arity + 1.0
                predicate_counter += 1

        # Rename predicates for PDDL compatibility (reuse from grammar search)
        renamed_candidates = self._rename_predicates_to_remove_incompatible_chars(candidates)
        return renamed_candidates

    def _update_incontact_predicate_using_motion_analysis(self, dataset: Dataset, in_contact_pred: Predicate, gripper_type: Type) -> Dict[Tuple[Type, Type, str], List[np.ndarray]]:
        """Update incontact predicates using motion analysis.
        Incontact seems like a previledged predicate, this function removes it
        and replaces it with a more general predicate that is based on motion analysis.
        It looks at which object is in motion to determine if it is in contact with the gripper.
        """


        # Filter types so things other than gripper and are useful are kept!!!
        # types = {obj.type for traj in dataset.trajectories for obj in traj.states[0]}
        # Example filter (adjust as needed):
        disallowed_type_names = {"gripper_type", "left_finger_type", "right_finger_type", "base_type"} # Added door_type based on usage
        # Dictionary to store motion data for each object in each trajectory
        motion_data = defaultdict(lambda: defaultdict(list))

        gripper_obj = None
        for obj in dataset.trajectories[0].states[0].get_objects(gripper_type):
            gripper_obj = obj
            break
        assert gripper_obj is not None, "No gripper object found in the dataset"
        
        # hack here, since some task the motion stops at the end, so we need 2 change points
        # others achieve the goal and the episode ends, so we need 1 change point detections
        if CFG.robo_kitchen_task == "OpenSingleDoor" \
            or CFG.robo_kitchen_task == "CloseSingleDoor" \
            or CFG.robo_kitchen_task == "CloseDrawer" \
            or CFG.robo_kitchen_task == "OpenDrawer" \
            or CFG.robo_kitchen_task == "TurnOnStove" \
            or CFG.robo_kitchen_task == "TurnOffStove" \
            or CFG.robo_kitchen_task == "TurnOnSinkFaucet" \
            or CFG.robo_kitchen_task == "TurnOffSinkFaucet":
            n_bkps = 1
        else:
            # or CFG.robo_kitchen_task == "PnPCounterToStove":
            # or CFG.robo_kitchen_task == "PnPCounterToCab" \
            # or CFG.robo_kitchen_task == "PnPStoveToCounter" \
            n_bkps = 2

        for i, traj in enumerate(dataset.trajectories):
            logging.debug(f"Processing trajectory {i+1}/{len(dataset.trajectories)} for motion analysis")
            
            # Get all objects in the trajectory
            all_objects = set()
            for state in traj.states:
                all_objects.update(state.data.keys())
            
            # Calculate velocity for each object
            for t in range(len(traj.states) - 1):
                state_t = traj.states[t]
                state_t1 = traj.states[t+1]
                
                for obj in all_objects:
                    if obj not in state_t.data or obj not in state_t1.data or obj.type.name in disallowed_type_names:
                        continue
                        
                    # Get translation data
                    trans_t = state_t.get(obj, CFG.trans_feat_name)
                    trans_t1 = state_t1.get(obj, CFG.trans_feat_name)
                    rot_t = state_t.get(obj, CFG.quat_feat_name)
                    rot_t1 = state_t1.get(obj, CFG.quat_feat_name)
                    
                    if trans_t is not None and trans_t1 is not None and rot_t is not None and rot_t1 is not None:
                        # Calculate velocity (change in position)
                        delta1 = np.linalg.norm(trans_t1 - trans_t)
                        q1 = R.from_quat(rot_t)
                        q2 = R.from_quat(rot_t1)
                        q_diff = q2 * q1.inv()
                        delta2 = q_diff.magnitude()
                        motion_data[i][obj].append((t, delta1, delta2)) # NOTE: adjust weight here
            
            
            # clear in contact set for each state !!!! This makes our method not previledged, good!
            for state in dataset.trajectories[i].states:
                state.items_in_contact = set()
            
            # For each trajectory, find the object with most motion
            max_motion_obj = None
            max_motion = 0
            for obj, motion_list in motion_data[i].items():
                total_motion = sum(vel for _, vel, _ in motion_list)
                if total_motion > max_motion:
                    max_motion = total_motion
                    max_motion_obj = obj
            
            os.makedirs("feature_data", exist_ok=True)
            if max_motion_obj is not None:
                # Find first and last frame of significant motion
                # Compute a dynamic threshold for this object based on its motion statistics
                velocities = [(vel, rot_vel) for _, vel, rot_vel in motion_data[i][max_motion_obj]]
                lin_vel = np.array([vel for vel, _ in velocities])
                rot_vel = np.array([rot_vel for _, rot_vel in velocities])
                algo = rpt.Dynp(model="l1", min_size=10, jump=3).fit(lin_vel)
                if np.max(lin_vel) > CFG.motion_analysis_lin_vel_rot_vel_threshold: # if there is lin motion, use lin vel to find change points
                    logging.warning(f"Using LINEAR velocity to find change points for {max_motion_obj.name}")
                    # data is 10 hz, so min size being 1 sec, jump being 0.3 sec
                    my_bkps = algo.predict(n_bkps=n_bkps)
                    # dynamic_threshold = np.mean(velocities[my_bkps])
                    rpt.show.display(lin_vel, my_bkps, my_bkps, figsize=(10, 6))

                    # save the figure
                    # hopefully the signal has 2 change point, and the velocities above the first one are the ones we want
                    # plot yline of the dynamic threshold
                    plt.savefig(f"feature_data/motion_analysis_traj{i}_obj_lin_{max_motion_obj.name}.png")
                    plt.close()
                else:
                    logging.warning(f"Using ROTATIONAL velocity to find change points for {max_motion_obj.name}")
                    algo = rpt.Dynp(model="l1", min_size=10, jump=3).fit(rot_vel)
                    my_bkps = algo.predict(n_bkps=n_bkps)
                    rpt.show.display(rot_vel, my_bkps, my_bkps, figsize=(10, 6))
                    plt.savefig(f"feature_data/motion_analysis_traj{i}_obj_rot_{max_motion_obj.name}.png")
                    plt.close()
                if n_bkps == 1:
                    motion_frames = range(my_bkps[0]-10, len(lin_vel)) # -10 is a hack , assume 1 sec of contact before the motion
                else:
                    motion_frames = range(my_bkps[0]-10, my_bkps[1]) 
                    # dynamic_threshold = CFG.motion_analysis_contact_threshold

                # motion_frames = [t for t, vel, _ in motion_data[i][max_motion_obj]
                #                  if vel > dynamic_threshold and t > 5]
                # NOTE: we are only looking at motion after 5 steps, since the first few steps are noisy 
                assert len(motion_frames) > 0, "No motion frames found for object"
                first_motion = min(motion_frames)
                last_motion = max(motion_frames)
                    
                # Mark the object as in contact during the motion period
                for t in range(first_motion, last_motion + 1):
                    if t < len(traj.states):
                        
                        dataset.trajectories[i].states[t].items_in_contact = {(gripper_obj, max_motion_obj)}
                        # Update the state to mark the object as in contact
                        # This assumes you have a way to mark objects as in contact
                        # You might need to modify this based on your state representation

    def _generate_relative_low_speed_feature_datasets(self, dataset: Dataset) -> Dict[Tuple[Type, Type, str], List[np.ndarray]]:
        """Extracts relative features constant between consecutive states.
        Includes relative SE(3) pose for types with translation and quaternion.
        """
        feature_data = defaultdict(list)
        feature_changes = defaultdict(list) # Track change magnitudes

        # Filter types so things other than gripper and are useful are kept!!!
        types = {obj.type for traj in dataset.trajectories for obj in traj.states[0]}
        # Optional: Filter types as before
        filtered_types = set()
        # Example filter (adjust as needed):
        allowed_type_names = {"handle", "cabinet"} # Added door_type based on usage
        for type_obj in types:
            if any(name in type_obj.name for name in allowed_type_names):
                filtered_types.add(type_obj)
                logging.info(f"Keeping type for relative features: {type_obj.name}")
            else:
                logging.debug(f"Filtering out type: {type_obj.name}")

        gripper_type = "gripper"
        gripper_type_obj = None
        for type_obj in types:
            if gripper_type in type_obj.name:
                gripper_type_obj = type_obj
                logging.info(f"Keeping gripper type: {type_obj.name}")
                break

        if gripper_type_obj is None:
            logging.warning(f"No gripper type found in the dataset. Skipping relative features.")
            return {}

        type_pairs = list(utils.combinations_no_self_pairs(sorted(list(filtered_types)), 2))
        # Create type pairs that include combinations with gripper
        for type_obj in filtered_types:
            # Add both (gripper, obj) and (obj, gripper) pairs
            type_pairs.append((type_obj, gripper_type_obj))
            logging.info(f"Adding gripper pair: ({gripper_type_obj.name}, {type_obj.name}) and ({type_obj.name}, {gripper_type_obj.name})")

        logging.info(f"Total type pairs for relative features: {len(type_pairs)}")
        # type_pairs = list(product(sorted(list(types)), repeat=2))

        quat_feat_name = "quaternion"
        trans_feat_name = "translation"
        pose_feat_name = "pose" # New combined feature name

        for i, traj in enumerate(dataset.trajectories):
            logging.debug(f"Processing trajectory {i+1}/{len(dataset.trajectories)} for relative features")
            for t in range(len(traj.states) - 1):
                state_t = traj.states[t]
                state_t1 = traj.states[t+1]
                for type1, type2 in type_pairs:
                    objs1 = list(state_t.get_objects(type1))
                    objs2 = list(state_t.get_objects(type2))

                    # --- SE(3) Pose Feature ---
                    # Check if BOTH types have translation and quaternion
                    # has_trans1 = trans_feat_name in type1.feature_names
                    # has_quat1 = quat_feat_name in type1.feature_names
                    # has_trans2 = trans_feat_name in type2.feature_names
                    # has_quat2 = quat_feat_name in type2.feature_names

                    # if has_trans1 and has_quat1 and has_trans2 and has_quat2:
                    # logging.debug(f"Calculating relative pose for ({type1.name}, {type2.name})")
                    for o1 in objs1:
                        # Handle type1 == type2 case
                        obj2_list = objs2 if type1 != type2 else [o for o in objs2 if o != o1]
                        for o2 in obj2_list:
                            rel_pose_t = utils.calculate_relative_pose(state_t, o1, o2, trans_feat_name, quat_feat_name)
                            rel_pose_t1 = utils.calculate_relative_pose(state_t1, o1, o2, trans_feat_name, quat_feat_name)

                            if rel_pose_t is not None and rel_pose_t1 is not None:
                                # Calculate change in relative pose (using SE(3) distance concept)
                                # We need a distance function here, let's define a simple one for constancy check
                                pose_diff_norm = utils.calculate_se3_distance(rel_pose_t, rel_pose_t1, 
                                                                                CFG.clustering_se3_trans_weight, 
                                                                                CFG.clustering_se3_rot_weight)

                                # Add the pose at time t to the dataset
                                feature_key = (type1, type2, pose_feat_name)
                                feature_data[feature_key].append(rel_pose_t)
                                feature_changes[feature_key].append(pose_diff_norm)

        # Filter based on constancy (e.g., keep points below 30th percentile of change)
        final_feature_data = defaultdict(list)
        for feature_key, data_points in feature_data.items():
            changes = np.array(feature_changes[feature_key])
            if len(changes) > 1: # Need at least 2 points to compute percentile
                # get moving average of changes first
                changes_ma = np.convolve(changes, np.ones(CFG.clustering_moving_average_window) / CFG.clustering_moving_average_window, mode='valid')
                constancy_threshold = np.percentile(changes_ma, CFG.clustering_feature_constancy_percentile) # Default 30?
                logging.debug(f"Constancy threshold for {feature_key}: {constancy_threshold:.4f} ({CFG.clustering_feature_constancy_percentile}th percentile)")
                mask = changes <= constancy_threshold
                final_feature_data[feature_key] = [pt for pt, keep in zip(data_points, mask) if keep]
                logging.debug(f"Kept {sum(mask)} / {len(data_points)} points for {feature_key} based on constancy.")
            else:
                logging.debug(f"No data points collected for {feature_key}.")

        return final_feature_data

    def _cluster_feature_dataset(self, feature_data: List[np.ndarray], initial_epsilon: float, feature_name: str) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
        """Performs clustering based on epsilon distance.
        Uses SE(3) metric for 'pose' features, Euclidean otherwise.

        Returns the data array, cluster labels for each point, the set of unique labels,
        and the effective epsilon used for clustering.
        """
        if not feature_data:
            return np.array([]), np.array([]), set()

        data_array = np.array(feature_data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        # Handle case with 0 or 1 data point early
        if data_array.shape[0] < 2:
            labels = np.array([0]) if data_array.shape[0] == 1 else np.array([])
            unique_labels = {0} if data_array.shape[0] == 1 else set()
            return data_array, labels, unique_labels

        # --- Determine Metric and Epsilon ---

        if feature_name == "pose":
            # Use the SE(3) distance function as the metric
            # Define a lambda or wrapper if needed to pass weights, assuming CFG accessible
            metric = lambda p1, p2: utils.calculate_se3_distance(p1, p2, 
                                                            CFG.clustering_se3_trans_weight, 
                                                            CFG.clustering_se3_rot_weight)
            # Use a specific epsilon for SE(3) clustering
            effective_epsilon = CFG.clustering_se3_epsilon # Needs to be defined in CFG
            logging.debug(f"Using SE(3) metric with epsilon: {effective_epsilon:.4f}")
        else:
            raise ValueError(f"Unsupported feature type: {feature_name}")

        if CFG.clustering_algorithm == "hdbscan":
            min_clust_size = max(3, int(0.06 * len(data_array)))
            logging.debug(f"Running HDBSCAN with min_cluster_size: {min_clust_size}, min_samples: {min_clust_size} and metric: {'SE(3)' if callable(metric) else metric}")
            clustering = HDBSCAN(min_cluster_size=min_clust_size,
                                    min_samples=min_clust_size, # Often set to min_cluster_size
                                    metric=metric, # Pass the custom or standard metric
                                    # cluster_selection_epsilon=effective_epsilon, # Optional: for DBSCAN-like flat extraction
                                    allow_single_cluster=False # Default is False
                                ).fit(data_array)

        else: # Agglomerative clustering
            logging.debug(f"Running Agglomerative Clustering with distance_threshold: {effective_epsilon:.4f} and metric: {'SE(3)' if callable(metric) else metric}")
            dists = pdist(data_array, metric=metric)
            dist_matrix = squareform(dists)
            clustering = AgglomerativeClustering(n_clusters=None,
                                                affinity="precomputed", # Pass metric
                                                linkage='single', # Check compatibility with custom metric
                                                distance_threshold=effective_epsilon).fit(dist_matrix)

        labels = clustering.labels_
        unique_labels = set(labels)

        return data_array, labels, unique_labels
    def _plot_cluster_results(self,
                              data_array: np.ndarray,
                              labels: np.ndarray,
                              unique_labels: Set[int],
                              kept_clusters_info: Dict[int, Dict],
                              type1_name: str,
                              type2_name: Optional[str], # None for absolute features
                              feat_name: str,
                              pred: Predicate) -> None:
        """Helper function to visualize clustering results.
        For 'pose' features, plots 3D translation and centroid frames.
        Uses distinct colors for each kept cluster.
        """
        if not CFG.clustering_debug or data_array.size == 0:
            return

        # Imports are assumed present based on original code
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import Ellipse # For 2D
        import matplotlib # Added for colormap access

        # Determine if relative or absolute for titles/filenames
        if type2_name:
            cluster_type_str = f"Relative Cluster: {type2_name} in {type1_name} frame"
            fname_prefix = f"rel_{feat_name}_clusters_{CFG.robo_kitchen_task}_{pred.name}_{type2_name}_in_{type1_name}_frame"
        else:
            cluster_type_str = f"Absolute Cluster: {type1_name}"
            fname_prefix = f"abs_{feat_name}_clusters_{CFG.robo_kitchen_task}_{pred.name}_{type1_name}"

        num_total_clusters = len(unique_labels - {-1})
        num_kept_clusters = len(kept_clusters_info)

        fig = plt.figure(figsize=(15, 12))
        title = (f"{cluster_type_str} ({feat_name})\n"
                 f"MinRatio={CFG.clustering_min_ratio_of_data}, Kept={num_kept_clusters}/{num_total_clusters}")
        fname = f"{fname_prefix}.png"

        # Store figure reference for potential trajectory overlay
        self._last_cluster_fig = fig
        self._last_cluster_type1 = type1_name
        self._last_cluster_type2 = type2_name if type2_name else None
        self._last_cluster_feat = feat_name
        self._last_cluster_fname = fname

        # --- Assign Colors ---
        kept_labels = sorted(list(kept_clusters_info.keys()))
        num_kept = len(kept_labels)
        # Use a colormap suitable for distinct categories
        cmap = matplotlib.colormaps.get_cmap('tab10') # Get the colormap object
        # Map kept cluster labels to colors
        kept_color_map = {label: cmap(i / max(1, num_kept-1)) if num_kept > 1 else cmap(0.0)
                          for i, label in enumerate(kept_labels)}

        colors = []
        for label in labels:
            if label == -1:
                colors.append('black') # Noise
            elif label in kept_color_map:
                colors.append(kept_color_map[label]) # Kept cluster color
            else:
                colors.append('lightgrey') # Discarded cluster

        num_dims = data_array.shape[1]
        ax = None
        is_3d = False

        # --- Setup Plot Axes ---
        if feat_name == "pose":
            # For pose (7D), plot the translational part (first 3 dims)
            if num_dims >= 3:
                ax = fig.add_subplot(111, projection='3d')
                # Scatter plot using only the first 3 dimensions (translation)
                ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=colors, alpha=0.5, s=30, label='Feature Points')
                ax.set_xlabel('Relative Tx')
                ax.set_ylabel('Relative Ty')
                ax.set_zlabel('Relative Tz')
                is_3d = True
                # Optionally add origin marker for relative pose
                ax.scatter([0], [0], [0], c='blue', s=100, marker='x', label='Origin (Frame 1)')

                # Store axis reference for trajectory overlay
                self._last_cluster_ax = ax
            else:
                logging.warning(f"Pose feature has fewer than 3 dimensions ({num_dims}), cannot plot 3D translation.")
                # Fallback to 2D or 1D plot if desired? For now, just skip plotting.
                plt.close(fig)
                return
        elif num_dims == 1:
            ax = fig.add_subplot(111)
            ax.scatter(data_array[:, 0], np.zeros_like(data_array[:, 0]), c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
        elif num_dims == 2:
            ax = fig.add_subplot(111)
            ax.scatter(data_array[:, 0], data_array[:, 1], c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
            ax.set_ylabel(f'{feat_name} dim 2')
            ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for 2D
        elif num_dims >= 3: # Non-pose 3D+ features
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
            ax.set_ylabel(f'{feat_name} dim 2')
            ax.set_zlabel(f'{feat_name} dim 3')
            is_3d = True

        # --- Plot Noise and Discarded First ---
        noise_indices = np.where(labels == -1)[0]
        discarded_indices = np.where(~np.isin(labels, kept_labels + [-1]))[0]

        if ax is not None:
            if feat_name == "pose" and is_3d:
                # Plot noise
                if len(noise_indices) > 0:
                    ax.scatter(data_array[noise_indices, 0], data_array[noise_indices, 1], data_array[noise_indices, 2], c='black', alpha=0.3, s=20, label='Noise Pts')
                # Plot discarded
                if len(discarded_indices) > 0:
                    ax.scatter(data_array[discarded_indices, 0], data_array[discarded_indices, 1], data_array[discarded_indices, 2], c='lightgrey', alpha=0.3, s=20, marker='x', label='Discarded Cluster Pts')
                # Plot origin
                # ax.scatter([0], [0], [0], c='blue', s=100, marker='x', label='Origin (Frame 1)')
            # Add similar plotting logic for 1D/2D/other 3D cases if needed
            # ... (omitted for brevity, focus is on pose) ...

        # --- Plot Kept Clusters with Color Gradient for Distance ---
        # Choose a colormap for distance visualization
        dist_cmap = cm.get_cmap('viridis') # Or 'plasma', 'coolwarm', etc.
        colorbar_added = False # Ensure colorbar is added only once

        # --- Plot Centroids and Boundaries/Frames ---
        centroids_plotted = False
        boundaries_plotted = False
        frames_plotted = False # Track if frames legend is added

        if ax is not None: 
            cluster_counts = defaultdict(int)
            all_centroids = {} # Store centroids for all clusters (kept and discarded)
            for label in labels:
                cluster_counts[label] += 1

            # --- Calculate All Centroids ---
            discarded_centroids_plotted = False # For legend
            for k in unique_labels:
                if k == -1: continue # Skip noise

                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) == 0: continue # Skip empty clusters if they somehow occur

                cluster_points = data_array[cluster_indices]

                # Calculate centroid (handle potential errors for small clusters)
                try:
                    if feat_name == "pose":
                        translations = cluster_points[:, :3]
                        quaternions = cluster_points[:, 3:]
                        # Check for valid quaternions before processing
                        valid_quats_mask = np.linalg.norm(quaternions, axis=1) > 1e-6
                        if not np.any(valid_quats_mask): # If no valid quats, use mean translation only
                            mean_translation = np.mean(translations, axis=0)
                            # Use a default orientation (e.g., identity quaternion)
                            mean_quaternion = np.array([0.0, 0.0, 0.0, 1.0]) 
                        else:
                            # Filter to only valid quaternions for mean calculation
                            valid_quats = quaternions[valid_quats_mask]
                            valid_rots = R.from_quat(valid_quats)
                            mean_rotation = valid_rots.mean()
                            mean_quaternion = mean_rotation.as_quat()
                            mean_translation = np.mean(translations, axis=0) # Mean of all translations

                        centroid = np.concatenate((mean_translation, mean_quaternion))
                    else: # For non-pose features
                        centroid = np.mean(cluster_points, axis=0)

                    all_centroids[k] = centroid # Store calculated centroid

                    # --- Plot Discarded Centroids ---
                    if k not in kept_clusters_info:
                        marker_kwargs_discarded = {'color': 'grey', 's': 50, 'marker': 'o', 'alpha': 0.7}
                        if not discarded_centroids_plotted:
                            marker_kwargs_discarded['label'] = 'Discarded Centroids'

                        if feat_name == "pose" and is_3d:
                            ax.scatter(centroid[0], centroid[1], centroid[2], **marker_kwargs_discarded)
                        # Add plotting for other dimensions/features if needed
                        # ...
                        discarded_centroids_plotted = True

                except (ValueError, LinAlgError) as e:
                    logging.warning(f"Could not calculate or plot centroid for cluster {k} (size {len(cluster_indices)}): {e}")
            # --- End Centroid Calculation and Discarded Plotting ---

            # --- Loop through KEPТ clusters for detailed plotting ---
            for label, info in kept_clusters_info.items():
                # cluster_color = kept_color_map[label] # Color now defined by distance map
                # centroid = info['center'] # Use pre-calculated from kept_clusters_info
                # Get the centroid calculated above to ensure consistency if calculation differs slightly
                if label not in all_centroids:
                    logging.warning(f"Centroid for kept cluster {label} was not calculated? Skipping.")
                    continue
                centroid = all_centroids[label]
                cluster_color = kept_color_map[label] # Get the assigned color for this kept cluster
                count = cluster_counts.get(label, 0)
                label_text = f"Cluster {label}: {count} pts"

                # --- Calculate Distances and Colors for Points in this Kept Cluster ---
                cluster_indices = np.where(labels == label)[0]
                cluster_points_all_dims = data_array[cluster_indices]
                cluster_points_trans = cluster_points_all_dims[:, :3] # For plotting
                cluster_radius = info.get('cluster_radius', 0) # Get radius if available

                point_distances = []
                max_dist = -1.0
                max_dist_idx = -1
                if cluster_radius > 1e-6: # Avoid division by zero
                    for idx, point in enumerate(cluster_points_all_dims):
                        dist = utils.calculate_se3_distance(point, centroid,
                                                            CFG.clustering_se3_trans_weight,
                                                            CFG.clustering_se3_rot_weight)
                        point_distances.append(dist)
                        if dist > max_dist:
                            max_dist = dist
                            max_dist_idx = idx # Store index relative to cluster_points_all_dims
                    # Normalize distances for coloring (0 to 1)
                    normalized_distances = np.array(point_distances) / cluster_radius
                    # Clip values just in case due to float precision
                    normalized_distances = np.clip(normalized_distances, 0.0, 1.0)
                    point_colors = dist_cmap(normalized_distances)
                else:
                    # If radius is near zero, color all points with the base color
                    point_colors = [cluster_color] * len(cluster_indices)
                    normalized_distances = np.zeros(len(cluster_indices)) # For scatter c value

                # --- Plot Kept Cluster Points with Distance Coloring ---
                if ax is not None and feat_name == "pose" and is_3d:
                    scatter_plot = ax.scatter(cluster_points_trans[:, 0], cluster_points_trans[:, 1], cluster_points_trans[:, 2],
                                            c=normalized_distances, cmap=dist_cmap, vmin=0.0, vmax=1.0, # Use normalized distances and colormap
                                            alpha=0.7, s=30)
                    # Add colorbar only once
                    if not colorbar_added:
                        cbar = fig.colorbar(scatter_plot, ax=ax, shrink=0.6, aspect=20)
                        cbar.set_label('Normalized SE(3) Distance to Centroid')
                        colorbar_added = True

                    # --- Plot Furthest Point Marker ---
                    furthest_plotted = False
                    if max_dist_idx != -1:
                        furthest_point_trans = cluster_points_trans[max_dist_idx]
                        logging.info(f"Cluster {label}: Furthest point distance = {max_dist:.4f}")
                        ax.scatter(furthest_point_trans[0], furthest_point_trans[1], furthest_point_trans[2],
                                   c='red', marker='v', s=100, edgecolor='black',
                                   label='Furthest Point' if not furthest_plotted else None)
                        furthest_plotted = True

                # Add similar scatter plot logic for 1D/2D/other 3D cases if needed
                # ... (omitted for brevity) ...

                # --- Plot Centroid Marker ---
                marker_kwargs = {'color': 'magenta', 's': 150, 'marker': '*'} # Use color argument
                # Only add the label once for the first centroid plotted
                if not centroids_plotted:
                    marker_kwargs['label'] = 'Kept Centroids'

                if feat_name == "pose" and is_3d:
                    ax.scatter(centroid[0], centroid[1], centroid[2], **marker_kwargs)
                    ax.text(centroid[0], centroid[1], centroid[2], label_text, fontsize=9)
                elif num_dims == 1:
                    ax.scatter(centroid[0], 0, **marker_kwargs)
                    ax.text(centroid[0], 0.01, label_text, fontsize=9) 
                elif num_dims == 2:
                    ax.scatter(centroid[0], centroid[1], **marker_kwargs)
                    ax.text(centroid[0], centroid[1], label_text, fontsize=9)
                elif num_dims >= 3 and is_3d: # Non-pose 3D
                    ax.scatter(centroid[0], centroid[1], centroid[2], **marker_kwargs)
                    ax.text(centroid[0], centroid[1], centroid[2], label_text, fontsize=9)
                centroids_plotted = True

                # --- Plot Boundaries (Ellipsoids for non-pose) or Frames (for pose) ---
                if feat_name == "pose" and is_3d:
                    # Plot coordinate frame for the centroid pose
                    # try:
                    centroid_trans = centroid[:3]
                    centroid_quat = centroid[3:]
                    # Normalize quaternion to ensure valid rotation
                    q_norm = norm(centroid_quat)
                    if np.isclose(q_norm, 0): raise ValueError("Centroid quaternion norm is zero.")
                    centroid_quat /= q_norm

                    rot_mat = R.from_quat(centroid_quat).as_matrix()
                    axis_len = CFG.clustering_visualization_frame_axis_length # Add to CFG (e.g., 0.05)

                    # Quiver args
                    q_args = {'length': axis_len, 'normalize': False, 'alpha': 0.8}

                    # X-axis (Red)
                    ax.quiver(centroid_trans[0], centroid_trans[1], centroid_trans[2], 
                                rot_mat[0, 0], rot_mat[1, 0], rot_mat[2, 0], 
                                color='r', **q_args, label='Centroid Frame X' if not frames_plotted else None)
                    # Y-axis (Green)
                    ax.quiver(centroid_trans[0], centroid_trans[1], centroid_trans[2], 
                                rot_mat[0, 1], rot_mat[1, 1], rot_mat[2, 1], 
                                color='g', **q_args, label='Centroid Frame Y' if not frames_plotted else None)
                    # Z-axis (Blue)
                    ax.quiver(centroid_trans[0], centroid_trans[1], centroid_trans[2], 
                                rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 2], 
                                color='b', **q_args, label='Centroid Frame Z' if not frames_plotted else None)
                    frames_plotted = True

                    # --- Plot Ellipsoidal Decision Boundary ---
                    # If covariance matrix is available, also plot an ellipsoid representing the Mahalanobis distance boundary
                    if 'inv_covariance_matrix_trans' in info and 'mahalanobis_threshold_trans' in info:
                        # <<< INSERT START >>>
                        # logging.info(f"--- Ellipsoid Plot Debug (Cluster {label}) ---")
                        # logging.info(f"Cluster Info keys: {info.keys()}")
                        # logging.info(f"Has 'cluster_cov': {'cluster_cov' in info}")
                        # logging.info(f"Has 'mahalanobis_threshold': {'mahalanobis_threshold' in info}")
                        # if 'cluster_cov' in info:
                        #     logging.info(f"Cluster Covariance (shape {info['cluster_cov'].shape}):\n{info['cluster_cov']}")
                        # if 'mahalanobis_threshold' in info:
                        #     logging.info(f"Mahalanobis Threshold: {info['mahalanobis_threshold']}")
                        # else:
                        #     logging.warning(f"Mahalanobis Threshold MISSING in info for cluster {label}")
                        # <<< INSERT END >>>

                        cluster_cov = np.linalg.inv(info['inv_covariance_matrix_trans'])
                        trans_cov = cluster_cov
                        mahalanobis_threshold = info['mahalanobis_threshold_trans']

                        # Extract translation part of covariance if dealing with pose
                        assert feat_name == "pose" and trans_cov.shape[0] == 3

                            # Check if covariance is valid for visualization
                        if np.all(np.isfinite(trans_cov)) and not np.any(np.isnan(trans_cov)):
                            try:
                                # Compute eigenvalues and eigenvectors of the covariance matrix
                                eigvals, eigvecs = np.linalg.eigh(trans_cov)
                                # <<< INSERT START >>>
                                logging.info(f"Eigenvalues (eigvals): {eigvals}")
                                logging.info(f"Eigenvectors (eigvecs):\n{eigvecs}")
                                # <<< INSERT END >>>

                                # Ensure positive eigenvalues (should be positive definite)
                                eigvals = np.abs(eigvals)

                                # Scale eigenvalues by Mahalanobis threshold and take square root
                                # as we need standard deviation not variance
                                eigvals_scaled = np.sqrt(mahalanobis_threshold * eigvals)
                                # <<< INSERT START >>>
                                logging.info(f"Scaled Eigenvalues (sqrt(thresh * eigvals)): {eigvals_scaled}")
                                # <<< INSERT END >>>

                                # Create meshgrid of points on a unit sphere
                                u = np.linspace(0, 2 * np.pi, 25)
                                v = np.linspace(0, np.pi, 25)
                                x_unit = np.outer(np.cos(u), np.sin(v))
                                y_unit = np.outer(np.sin(u), np.sin(v))
                                z_unit = np.outer(np.ones_like(u), np.cos(v))

                                # Reshape unit sphere points to apply transformation
                                points = np.stack([x_unit.flatten(), y_unit.flatten(), z_unit.flatten()], axis=1)

                                # Apply eigenvalue scaling (multiply each axis by corresponding eigenvalue)
                                scaled_points = points * eigvals_scaled

                                # Rotate using eigenvectors to align with covariance principal components
                                rotated_points = np.dot(scaled_points, eigvecs.T)

                                # Translate to centroid position
                                ellipsoid_points = rotated_points + centroid_trans

                                # Reshape back to mesh format
                                x_ellipsoid = ellipsoid_points[:, 0].reshape(x_unit.shape)
                                y_ellipsoid = ellipsoid_points[:, 1].reshape(y_unit.shape)
                                z_ellipsoid = ellipsoid_points[:, 2].reshape(z_unit.shape)

                                # Plot ellipsoid as wireframe
                                ellipsoid_label = 'Covariance Ellipsoid (Maha. Thresh.)' if not boundaries_plotted else ""
                                ax.plot_wireframe(
                                    x_ellipsoid, y_ellipsoid, z_ellipsoid,
                                    color='red', alpha=0.2, rstride=4, cstride=4, 
                                    label=ellipsoid_label, linestyle='--'
                                )

                                # Add to legend items
                                # if ellipsoid_label:
                                #     if 'ellipsoid_plotted' not in locals():
                                #         ellipsoid_plotted = True
                                #         handles.append(plt.Line2D([0], [0], linestyle='--', color='red', alpha=0.5,
                                #                                 label='Covariance Ellipsoid (Maha. Thresh.)'))

                            except (np.linalg.LinAlgError, ValueError) as e:
                                logging.warning(f"Could not plot ellipsoid for cluster {label}: {e}")
                        else:
                            logging.warning(f"Invalid covariance for ellipsoid plot in cluster {label}")
                    # --- End Ellipsoid Plotting ---

                else:
                    logging.debug(f"Skipping boundary/frame plot for cluster {label}: Missing info.")

            # --- Finalize Plot ---
            ax.set_title(title)

            # --- Calculate and Store Axis Limits ---
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # ax.set_aspect("equal")
            self._last_cluster_xlim = xlim
            self._last_cluster_ylim = ylim
            if is_3d:
                zlim = ax.get_zlim()
                self._last_cluster_zlim = zlim
            else:
                self._last_cluster_zlim = None # Ensure it's reset for non-3D plots

            # Create legend handles
            # handles = []
            # # Create a single handle for all kept points using a generic marker
            # # handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Kept Cluster Pts', markersize=10, markerfacecolor='gray'))
            # # Handle for discarded points (updated color)
            # if len(discarded_indices) > 0:
            #     handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Discarded Cluster Pts', markersize=10, markerfacecolor='lightgrey', linestyle='None')) # Use marker='x'

            # if len(noise_indices) > 0:
            #     handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise Pts', markersize=10, markerfacecolor='black'))
            # # Handle for centroids (generic marker)
            # if centroids_plotted:
            #     handles.append(plt.Line2D([0], [0], marker='*', color='w', label='Kept Centroids', markersize=10, markerfacecolor='magenta', linestyle='None'))
            # if discarded_centroids_plotted:
            #     handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Discarded Centroids', markersize=8, markerfacecolor='grey', alpha=0.7, linestyle='None'))
            # # Handle for boundaries (generic color)
            # if boundaries_plotted: # Ellipsoid legend
            #     handles.append(plt.Line2D([0], [0], linestyle='--', color='gray', label='Ellipsoid Boundary (Maha. Thresh.)'))
            # if frames_plotted: # Add legend entries for frames if any were plotted
            #      handles.append(plt.Line2D([0],[0], color='r', lw=2, label='Centroid Frame X'))
            #      handles.append(plt.Line2D([0],[0], color='g', lw=2, label='Centroid Frame Y'))
            #      handles.append(plt.Line2D([0],[0], color='b', lw=2, label='Centroid Frame Z'))
            # if boundaries_plotted and feat_name == "pose": # Add legend for pose cluster radius sphere
            #     handles.append(plt.Line2D([0], [0], linestyle='-', color='gray', alpha=0.3, label='Equiv. Trans. Radius (Max Dist)'))
            # # Add legend for furthest point if plotted
            # if furthest_plotted:
            #     handles.append(plt.Line2D([0], [0], marker='v', color='w', label='Furthest Point', markersize=8, markerfacecolor='red', markeredgecolor='black', linestyle='None'))
            # # if feat_name == "pose" and is_3d: # Origin marker for pose - REMOVED
            # #      handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Origin (Frame 1)', markersize=10, markerfacecolor='blue', linestyle='None'))

            # ax.legend(handles=handles)
            # No longer setting axis limits or view init here as it's done above

            # Comment out saving/closing logic if overlay is pending
            # plt.tight_layout()
            # os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
            # plt.savefig(fname)
            # logging.info(f"Cluster visualization saved to feature_data/{fname}")
            # plt.close(fig)
            # self._last_cluster_fig = None
            # self._last_cluster_ax = None
        self._last_cluster_title = title

    def _create_predicate_from_relative_cluster(self,
                                                type1: Type,
                                                type2: Type,
                                                feature_name: str,
                                                cluster_center: np.ndarray,
                                                cluster_radius: float,
                                                diff_fn: Optional[Callable],
                                                cluster_id: int,
                                                cluster_cov: Optional[np.ndarray] = None,
                                                inv_covariance_matrix: Optional[np.ndarray] = None,
                                                mahalanobis_threshold: Optional[float] = None) -> Predicate:
        """Creates a binary predicate from a relative feature cluster (including pose).

        If cluster_cov, inv_covariance_matrix, and mahalanobis_threshold are provided,
        uses _RelativeFeatureCovClusterClassifier. Otherwise, uses
        _RelativeFeatureClusterClassifier based on radius.
        """
        if cluster_cov is not None and inv_covariance_matrix is not None and mahalanobis_threshold is not None:
            # Use the covariance-based classifier, passing pre-calculated values
            classifier = _RelativeFeatureCovClusterClassifier(
                type1, type2, feature_name,
                cluster_center, cluster_cov,
                diff_fn, cluster_id,
                inv_covariance_matrix, # Pass pre-calculated inv cov
                mahalanobis_threshold) # Pass pre-calculated threshold
        else:
            # Use the original radius-based classifier
            classifier = _RelativeFeatureClusterClassifier(
                type1, type2, feature_name,
                cluster_center, cluster_radius,
                diff_fn, cluster_id)

        name = str(classifier)
        types = [type1, type2]
        pred = Predicate(name, types, classifier)
        return pred
    
    def _create_predicate_from_relative_cluster_trans_rot(self,
                                                type1: Type,
                                                type2: Type,
                                                feature_name: str,
                                                trans_center: np.ndarray,
                                                rot_center: R,
                                                inv_covariance_matrix_trans: np.ndarray,
                                                inv_covariance_matrix_rot: np.ndarray,
                                                mahalanobis_threshold_trans: float,
                                                mahalanobis_threshold_rot: float,
                                                cluster_id: int) -> Predicate: 
                                                
        """Creates a binary predicate from a relative feature cluster (including pose).

        If cluster_cov, inv_covariance_matrix, and mahalanobis_threshold are provided,
        uses _RelativeFeatureCovClusterClassifier. Otherwise, uses
        _RelativeFeatureClusterClassifier based on radius.
        """
        if inv_covariance_matrix_trans is not None and inv_covariance_matrix_rot is not None and mahalanobis_threshold_trans is not None and mahalanobis_threshold_rot is not None:
            # Use the covariance-based classifier, passing pre-calculated values
            classifier = _RelativeFeatureCovClusterClassifierTransRot(
                type1, type2, feature_name, cluster_id,
                trans_center, rot_center,
                inv_covariance_matrix_trans,
                inv_covariance_matrix_rot,
                mahalanobis_threshold_trans,
                mahalanobis_threshold_rot) # Pass pre-calculated 
        else:
            raise ValueError("inv_covariance_matrix_trans, inv_covariance_matrix_rot, mahalanobis_threshold_trans, and mahalanobis_threshold_rot must be provided")

        name = str(classifier)
        types = [type1, type2]
        pred = Predicate(name, types, classifier)
        return pred

    def _create_predicate_from_absolute_cluster(self, type1: Type, feature_name: str, cluster_center: np.ndarray, inv_covariance_matrix: np.ndarray, mahalanobis_threshold: float, cluster_id: int) -> Predicate:
        """Creates a unary predicate from an absolute feature cluster."""
        # Note: cluster_info replaced by cluster_center
        classifier = _AbsoluteFeatureClusterClassifier(type1, feature_name, cluster_center, inv_covariance_matrix, mahalanobis_threshold, cluster_id)
        name = str(classifier)
        types = [type1]
        pred = Predicate(name, types, classifier)
        return pred


    def _generate_candidate_predicates_contact_goal_clustering_refactored(self, dataset: Dataset) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float], Set[Predicate]]:
        """Generate candidate predicates based on clustering of contact relative poses."""
        env, in_contact_pred, in_origin_pred, gripper_type = self._prepare_env_and_predicates()
        if not gripper_type:
            logging.warning("Gripper type not found. Cannot generate contact-based predicates.")
            return {}, {} # Return empty dicts if gripper type is not found
        if CFG.predicate_candidates_method == "motion_analysis_contact":
            self._update_incontact_predicate_using_motion_analysis(dataset, in_contact_pred, gripper_type)
        learnt_goal_predicates = self.load_learnt_goals()
        predicates_to_monitor, ground_atom_dataset = self._create_gnd_atom_datasets(dataset, in_contact_pred, in_origin_pred, learnt_goal_predicates)
        all_objs_types = self._find_common_objects_types(ground_atom_dataset)
        relative_pose_dataset_dict, traj_all_objs_all, contact_period_rel_trajs, goal_reached_states, object_type_in_contact_with_gripper_longest_duration, ground_atom_dataset = self._extract_relative_pose_data(ground_atom_dataset, all_objs_types, gripper_type, in_contact_pred, in_origin_pred)
        # single out the object with in contact with gripper for longest duration
        # Ensure all trajectories have the same object with longest contact duration
        assert len(set(object_type_in_contact_with_gripper_longest_duration)) == 1, "All trajectories should have the same object with longest contact duration"
        obj_type_contact_with_gripper = object_type_in_contact_with_gripper_longest_duration[0]
        obj_type_of_reference_best, min_reconstruction_error, list_of_reconstruction_errors = self._select_reference_object(contact_period_rel_trajs)
        # just 1 object does not support contacting with multiple objects 
        self._visualize_contact_period_trajectories(contact_period_rel_trajs, list_of_reconstruction_errors)
        ground_atom_dataset, relative_pose_dataset_dict = self._update_atom_sequences_with_goal_predicates(ground_atom_dataset, traj_all_objs_all, obj_type_of_reference_best, obj_type_contact_with_gripper, goal_reached_states, relative_pose_dataset_dict)
        renamed_cluster_candidates = self._add_goal_states_to_relative_pose_and_cluster(dataset, relative_pose_dataset_dict)
        
        
        return self._postprocess_cluster_predicates(env, dataset, ground_atom_dataset, predicates_to_monitor, renamed_cluster_candidates, obj_type_of_reference_best,obj_type_contact_with_gripper, learnt_goal_predicates)
        

    
    def _postprocess_cluster_predicates(self, env, dataset: Dataset, ground_atom_dataset: List[GroundAtomTrajectory], predicates_to_monitor: Set[Predicate], renamed_cluster_candidates: Dict[Predicate, float], obj_type_of_reference_best: Type, obj_type_contact_with_gripper: Type, learnt_goal_predicates: Set[Predicate]):
        if CFG.reprocess_ground_atom_dataset_using_cluster_predicates: # this turns out to be not good, some traj get not segmented, some traj get segmented too early since cluster is sometimes big.
            kept_preds = set(predicates_to_monitor)
            kept_preds2 = set(renamed_cluster_candidates.keys())
            different_seg_count_trajs = []
            num_seg_1 = []
            num_seg_2 = []
            logging.info("--- Segmentation using ONLY newly generated cluster predicates ---")
            og_pred_atom_dataset = self._create_atom_dataset(dataset, kept_preds)
            cluster_pred_atom_dataset = self._create_atom_dataset(dataset, kept_preds2)
            for i, (traj1_ele, traj2_ele) in enumerate(zip(og_pred_atom_dataset, cluster_pred_atom_dataset)):
                _, atom_seq1 = traj1_ele
                _, atom_seq2 = traj2_ele

                # Print changes in atom sets
                last_atoms1 = None
                seg_count1 = 0
                for t, atoms in enumerate(atom_seq1):
                    current_atoms = frozenset(atoms)
                    if current_atoms != last_atoms1:
                        logging.info(f"Old  Time {t}: {current_atoms if current_atoms else '{}'}")
                        last_atoms1 = current_atoms
                        seg_count1 += 1

                last_atoms2 = None
                seg_count2 = 0
                for t, atoms in enumerate(atom_seq2):
                    current_atoms = frozenset(atoms)
                    if current_atoms != last_atoms2:
                        logging.info(f"New  Time {t}: {current_atoms if current_atoms else '{}'}")
                        last_atoms2 = current_atoms
                        seg_count2 += 1

                if seg_count1 != seg_count2:
                    different_seg_count_trajs.append(i)
                    num_seg_1.append(seg_count1)
                    num_seg_2.append(seg_count2)

            logging.info(f"Trajectories with different segment counts: {different_seg_count_trajs}, num_seg_1: {num_seg_1}, num_seg_2: {num_seg_2}, totoal_num_traj = {len(og_pred_atom_dataset)}")

            # Filter out trajectories with different segment counts from both datasets
            if different_seg_count_trajs:            
                # Reverse sort the indices to safely remove items without affecting other indices
                for idx in sorted(different_seg_count_trajs, reverse=True):
                    if 0 <= idx < len(og_pred_atom_dataset):
                        og_pred_atom_dataset.pop(idx)
                    if 0 <= idx < len(cluster_pred_atom_dataset):
                        cluster_pred_atom_dataset.pop(idx)

                logging.info(f"After filtering: {len(og_pred_atom_dataset)} trajectories remain")

        if CFG.reprocess_ground_atom_dataset_using_cluster_replacement: #replace in contact atoms with rel pose atoms, so easier to do operator learning later
            different_seg_count_trajs = [] 
            for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
                for j, atoms in enumerate(atom_seq):
                    atoms_new = []
                    for atom in atoms:
                        if "RelCovCluster" in atom.predicate.name:
                            atoms_new.append(atom)
                            continue
                        pred = list(CFG.dict_contact_predicate_to_rel_pose_predicates[atom.predicate.name, atom.objects[0].type.name, atom.objects[1].type.name])[0]
                        grounded_pred = GroundAtom(pred, atom.entities)
                        atoms_new.append(grounded_pred)
                    ground_atom_dataset[i][1][j] = set(atoms_new)

        if env.goal_predicates:
            assert len(list(env.goal_predicates)) == 1
            goal_pred = list(env.goal_predicates)[0]
            pred_key = tuple([goal_pred.name] + [t.name for t in goal_pred.types])
            CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[pred_key] = set([DummyPredicate(f"{CFG.robo_kitchen_task}-goal", [obj_type_of_reference_best, obj_type_contact_with_gripper])])
        else:
            raise NotImplementedError("Environment goal predicates not found, did you forget to define it for the task?")
        for pred in predicates_to_monitor:
            pass # the following three lines seems to make dr-unreachable error more likely but don't know why
            # if isinstance(pred, DummyPredicate): # goal predicate
            #     predicates_to_monitor.remove(pred)
            #     pred_new = DummyPredicate(f"{CFG.robo_kitchen_task}-goal", [obj_of_reference_best.type, obj_contact_with_gripper.type])
            #     predicates_to_monitor.add(pred_new)

        # --- End Debugging ---
        if CFG.reprocess_ground_atom_dataset_using_cluster_replacement:
            # if replacing, then goal predicates are gone, need some way to say how to successfully complete the task
            return ground_atom_dataset, ground_atom_dataset, different_seg_count_trajs, renamed_cluster_candidates, learnt_goal_predicates
        elif CFG.reprocess_ground_atom_dataset_using_cluster_predicates:
            return og_pred_atom_dataset, cluster_pred_atom_dataset, different_seg_count_trajs, renamed_cluster_candidates, predicates_to_monitor
        else:
            return ground_atom_dataset, ground_atom_dataset, different_seg_count_trajs, renamed_cluster_candidates, predicates_to_monitor
        
    def _add_goal_states_to_relative_pose_and_cluster(self, dataset: Dataset, relative_pose_dataset_dict: Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]]):
        candidate_cluster_preds: Dict[Predicate, float] = {}
        # predicate_counter = 0 # To ensure unique cluster IDs

        # Initialize cluster visualization storage attributes (copied from _generate_candidate_predicates)
        self._last_cluster_fig = None
        self._last_cluster_ax = None
        self._last_cluster_type1 = None
        self._last_cluster_type2 = None
        self._last_cluster_feat = None
        self._last_cluster_title = None
        self._last_cluster_fname = None

        # Process the collected relative pose data
        for (pred, type1, type2, direction), data in relative_pose_dataset_dict.items():
            feat_name = CFG.pose_feature_name # We are clustering relative SE(3) poses
            logging.debug(f"Clustering relative feature {feat_name} for ({type1.name}, {type2.name}) from {pred.name} with {len(data)} points.")

            # if len(data) < 10: continue # Skip if no data collected

            # Save feature data (optional, copied from _generate_candidate_predicates)
            # feature_key = f"contact_{type1.name}_{type2.name}_{feat_name}"
            # os.makedirs("feature_data", exist_ok=True)
            # data_path = f"feature_data/{feature_key}.npy"
            # np.save(data_path, np.array(data))
            # logging.info(f"Saved {len(data)} contact pose data points for feature {feature_key} to {data_path}")

            # Use SE(3) epsilon
            epsilon = CFG.clustering_se3_epsilon

            # Perform clustering
            data_array, labels, unique_labels = self._cluster_feature_dataset(data, epsilon, feat_name)
            if data_array.size == 0: continue # Skip if clustering returned empty

            # Adjust min cluster size calculation if needed (e.g., minimum 3 points for covariance)
            min_cluster_size = max(3, int(CFG.clustering_min_ratio_of_data * len(data_array)))
            # logging.debug(f"Using minimum cluster size: {min_cluster_size} ({CFG.clustering_min_ratio_of_data * 100}% of {len(data_array)} data points)") # Reduced logging

            # --- Cluster Processing (Copied and adapted from _generate_candidate_predicates) ---
            kept_clusters_info = {}
            discarded_labels = set()

            for k in unique_labels:
                if k == -1: continue # Skip noise points
                cluster_points = data_array[labels == k]
                cluster_size = len(cluster_points)

                if cluster_size >= min_cluster_size:

                    translations = cluster_points[:, :3]
                    quaternions = cluster_points[:, 3:]

                    valid_quats = quaternions
                    rotations = R.from_quat(valid_quats)
                    mean_rotation = rotations.mean()
                    # mean_quaternion = mean_rotation.as_quat()
                    mean_translation = np.mean(translations, axis=0)

                    # cluster_center = np.concatenate((mean_translation, mean_quaternion))

                    # --- Calculate Covariance, Inverse Covariance, and Threshold ---
                    # Use only the translation part for Mahalanobis distance/covariance
                    cluster_translations = cluster_points[:, :3]
                    cluster_quaternions = cluster_points[:, 3:]
                    # try:
                    # Calculate covariance of the translation vectors
                    if cluster_translations.shape[0] < 2: # Need at least 2 points for covariance
                        raise ValueError("Not enough points for covariance calculation.")

                    # Calculate difference from the mean translation
                    num_dims_trans = cluster_translations.shape[1] # Should be 3
                    assert num_dims_trans == 3
                    trans_diff = cluster_translations - mean_translation
                    reg_term_trans = np.eye(num_dims_trans) * CFG.clustering_inv_cov_reg # Use CFG value
                    cluster_cov_trans = np.cov(trans_diff, rowvar=False) + reg_term_trans

                    num_dims_rot = cluster_quaternions.shape[1] - 1 # Should be 3
                    assert num_dims_rot == 3
                    log_deltas = (mean_rotation.inv() * rotations).as_rotvec()
                    if type1.name == "gripper_type" or type2.name == "gripper_type":
                    # allow extra space for relative rotation bw gripper and obj so it doesnt always replan
                        reg = CFG.clustering_inv_cov_reg_rot
                    else: 
                        reg = CFG.clustering_inv_cov_reg_rot_low
                    reg_term_rot = np.eye(3) * reg # Use CFG value
                    cluster_cov_rot = np.cov(log_deltas.T) + reg_term_rot
                    
                    # Convert covariance to degree variation for rotation
                    # Calculate standard deviation in degrees for each rotation axis
                    rot_std_degrees = np.sqrt(np.diag(cluster_cov_rot)) * (180.0 / np.pi)
                    # Calculate the average degree variation across all rotation axes
                    avg_degree_variation = np.mean(rot_std_degrees)
                    # Log the degree variation information
                    logging.debug(f"Cluster {k} rotation degree variations: X={rot_std_degrees[0]:.2f}°, Y={rot_std_degrees[1]:.2f}°, Z={rot_std_degrees[2]:.2f}°, Avg={avg_degree_variation:.2f}°")
                    # Store degree variation info in the cluster info

                    # Calculate inverse covariance with regularization
                    inv_cov_trans = inv(cluster_cov_trans)
                    inv_cov_rot = inv(cluster_cov_rot)

                    # Calculate Mahalanobis threshold
                    threshold_trans = chi2.ppf(CFG.clustering_mahalanobis_confidence, df = num_dims_trans)
                    threshold_rot = chi2.ppf(CFG.clustering_mahalanobis_confidence, df = num_dims_rot)

                    # --- Calculate Radius (Max SE(3) distance) ---
                    # DO NOT CALCULATE RADIUS, USE INVERSE COVARIANCE AND THRESHOLD INSTEAD
                    # (Optional, can still be calculated for reference or radius-based classifier)
                    # cluster_se3_diffs = np.zeros(len(cluster_points))
                    # for i in range(len(cluster_points)):
                    #     diff = utils.calculate_se3_distance(cluster_center, cluster_points[i],
                    #                                         CFG.clustering_se3_trans_weight,
                    #                                         CFG.clustering_se3_rot_weight)
                    #     cluster_se3_diffs[i] = diff
                    # cluster_radius = np.max(cluster_se3_diffs) # Max SE(3) distance

                    # Store calculated info
                    kept_clusters_info[k] = {
                        'trans_center': mean_translation,
                        'rot_center': mean_rotation,
                        'size': cluster_size,
                        'points': cluster_points, # Keep for visualization if needed
                        'inv_covariance_matrix_trans': inv_cov_trans,
                        'inv_covariance_matrix_rot': inv_cov_rot,
                        'mahalanobis_threshold_trans': threshold_trans,
                        'mahalanobis_threshold_rot': threshold_rot,
                    }
                    # logging.info(f"Contact Cluster {k} ({type1.name}-{type2.name}) kept (size {cluster_size}). Radius: {cluster_radius:.4f}, Thresh: {threshold:.4f}") # Reduced logging

                else:
                    discarded_labels.add(k)
                    logging.debug(f"Contact Cluster {k} for {type1.name}-{type2.name}-{feat_name} discarded (size {cluster_size} < {min_cluster_size}).")

            # Optional visualization (now uses stored info)
            if CFG.clustering_debug and data_array.size > 0:
                if direction == "2in1":
                    self._plot_cluster_results(data_array, labels, unique_labels, kept_clusters_info,
                                            type1.name, type2.name, feat_name, pred)
                    # Plot relative trajectories after cluster plot for the same type pair
                    self._plot_relative_trajectories(dataset, type1.name, type2.name, pred)
                    # self._plot_relative_trajectories_segmented (traj_dataset_dict, pred,type1, type2)
                elif direction == "1in2":
                    self._plot_cluster_results(data_array, labels, unique_labels, kept_clusters_info,
                                            type2.name, type1.name, feat_name, pred)
                    # Plot relative trajectories after cluster plot for the same type pair
                    self._plot_relative_trajectories(dataset, type2.name, type1.name, pred)
                    # self._plot_relative_trajectories_segmented (traj_dataset_dict, pred,type2, type1)

            # Sort and select top_k clusters
            valid_kept_clusters = kept_clusters_info
            sorted_valid_kept_clusters = sorted(valid_kept_clusters.items(), key=lambda item: item[1]['size'], reverse=True)
            top_k = min(CFG.clustering_max_clusters, len(sorted_valid_kept_clusters))

            logging.debug(f"Selecting top {top_k} valid contact clusters for {feat_name}:{type1.name}-{type2.name}.")
            for i, (cluster_label, cluster_info) in enumerate(sorted_valid_kept_clusters[:top_k]):
                # Create predicate using the specific relative cluster method
                # Pass the pre-calculated inv_cov and threshold
                pred_generated = self._create_predicate_from_relative_cluster_trans_rot(
                    type1, type2, feat_name,
                    cluster_info['trans_center'],
                    cluster_info['rot_center'],
                    cluster_info['inv_covariance_matrix_trans'],
                    cluster_info['inv_covariance_matrix_rot'],
                    cluster_info['mahalanobis_threshold_trans'],
                    cluster_info['mahalanobis_threshold_rot'],
                    cluster_label, 
                )
                # Add predicate to candidates with cost (e.g., based on arity)
                candidate_cluster_preds[pred_generated] = float(pred_generated.arity) # Example cost
                if (pred.name, type1.name, type2.name) in CFG.dict_contact_predicate_to_rel_pose_predicates:
                    CFG.dict_contact_predicate_to_rel_pose_predicates[(pred.name, type1.name, type2.name)].add(pred_generated)
                else:
                    CFG.dict_contact_predicate_to_rel_pose_predicates[(pred.name, type1.name, type2.name)] = set([pred_generated])

        # Rename predicates for PDDL compatibility
        renamed_cluster_candidates = self._rename_predicates_to_remove_incompatible_chars(candidate_cluster_preds)
        return renamed_cluster_candidates

    def _update_atom_sequences_with_goal_predicates(self, ground_atom_dataset: List[GroundAtomTrajectory], traj_all_objs_all: List[List[Object]], obj_type_of_reference_best: Type, obj_type_contact_with_gripper: Object, goal_reached_states: Dict[int, List[State]], relative_pose_dataset_dict: Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]]):
        for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
            traj_all_objs = traj_all_objs_all[i]
            obj_ref = [o for o in traj_all_objs if o.type == obj_type_of_reference_best][0]
            obj_contact = [o for o in traj_all_objs if o.type == obj_type_contact_with_gripper][0]
            assert obj_ref is not None and obj_contact is not None, "Object of reference or contact with gripper not found"
            for j, atoms in enumerate(atom_seq):
                for k, atom in enumerate(atoms):
                    if isinstance(atom, DummyPredicate):
                        ground_atom_dataset[i][1][j].remove(atom)
                        ground_atom_dataset[i][1][j].add(GroundAtom(DummyPredicate(f"{CFG.robo_kitchen_task}-goal", [obj_type_of_reference_best, obj_type_contact_with_gripper]), [obj_ref, obj_contact]))

            # add stored states before contact lost to relative_pose_dataset_dict
            for state in goal_reached_states[i]:
                rel_pose = utils.calculate_relative_pose(state, obj_ref, obj_contact, CFG.trans_feat_name, CFG.quat_feat_name)
                key = (DummyPredicate(f"{CFG.robo_kitchen_task}-goal"), obj_type_of_reference_best, obj_type_contact_with_gripper, "2in1")
                relative_pose_dataset_dict[key].append(rel_pose)
        return ground_atom_dataset, relative_pose_dataset_dict
    # def _update_atom_sequences_with_goal_predicates(self, ground_atom_dataset: List[GroundAtomTrajectory], obj_of_reference_best: Object):
    def _visualize_contact_period_trajectories(self, contact_period_rel_trajs: Dict[Type, List[List[np.ndarray]]], list_of_reconstruction_errors: List[float]):
        # Visualize the x data for the object of reference
        for j, (o_ref, rel_pose_trajs) in enumerate(contact_period_rel_trajs.items()):
            if len(rel_pose_trajs) == 0: continue
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot each trajectory in a different color
            colors = plt.cm.rainbow(np.linspace(0, 1, len(contact_period_rel_trajs[o_ref])))

            for i, rel_pose_traj in enumerate(rel_pose_trajs):
                x_traj = np.array(rel_pose_traj)[:, :3]  # Get translation part

                # Plot trajectory
                ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], 
                       color=colors[i], linewidth=2, alpha=0.7,
                       label=f'Trajectory {i+1}')

                # Mark start and end points
                ax.scatter(x_traj[0, 0], x_traj[0, 1], x_traj[0, 2], 
                          color=colors[i], marker='o', s=100, label=f'Start {i+1}' if i == 0 else None)
                ax.scatter(x_traj[-1, 0], x_traj[-1, 1], x_traj[-1, 2], 
                          color=colors[i], marker='s', s=100, label=f'End {i+1}' if i == 0 else None)

            ax.set_title(f'Contact Period Relative Trajectories for {o_ref.name}, Reconstruction Error: {list_of_reconstruction_errors[j]:.1f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Add legend
            ax.legend()

            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])

            # Save the visualization
            os.makedirs("feature_data", exist_ok=True)
            plt.savefig(f"feature_data/contact_period_trajectories_{o_ref.name}.png")
            logging.info(f"Saved contact period trajectories visualization to feature_data/contact_period_trajectories_{o_ref.name}.png")
            plt.close(fig)

    def _select_reference_object(self, 
                                contact_period_rel_trajs: Dict[Type, List[List[np.ndarray]]], 
                                ) -> Tuple[Type, float, List[float]]:
        """
        Select the object of reference by learning a DS policy for each object and selecting the one with the lowest reconstruction error.
        """
        obj_type_of_reference_best = None
        min_reconstruction_error = float('inf')
        list_of_reconstruction_errors = []
        black_list = []
        for obj_type, rel_pose_trajs in contact_period_rel_trajs.items():
            if len(rel_pose_trajs) == 0: continue
            x = []
            quat = []
            x_dot = []
            omega = []
            for rel_pose_traj in rel_pose_trajs:
                x_traj = np.array(rel_pose_traj)[:, :3]
                quat_traj = np.array(rel_pose_traj)[:, 3:]
                x_dot_traj, omega_traj = compute_vel_traj(x_traj, np.array([R.from_quat(q).as_matrix() for q in quat_traj]), 1/60)
                x.append(x_traj)
                quat.append(quat_traj)
                x_dot.append(x_dot_traj)
                omega.append(omega_traj)
            # Check if start and end poses are almost the same (indicating no meaningful motion)
            if len(x) > 0 and len(x[0]) > 1:
                start_pos = np.array([traj[0] for traj in x])
                end_pos = np.array([traj[-1] for traj in x])
                start_quat = np.array([traj[0] for traj in quat])
                end_quat = np.array([traj[-1] for traj in quat])
                
                # Calculate average distance between start and end poses
                avg_distance = np.mean([np.linalg.norm(end - start) for start, end in zip(start_pos, end_pos)])
                avg_quat_distance = np.mean([np.linalg.norm((R.from_quat(end) * R.from_quat(start).inv()).as_rotvec()) for start, end in zip(start_quat, end_quat)])
                
                # If average distance is very small, blacklist this object
                if avg_distance < 0.01 and avg_quat_distance < 0.1:  
                    black_list.append(obj_type)
                    logging.info(f"Blacklisting {obj_type.name} due to minimal motion (avg distance: {avg_distance:.4f})")
                    # continue
            unified_config = UnifiedModelConfig(
                mode="se3_lpvds",
                K_candidates=[1]
            )
            ds_policy = DSPolicy(
                x=x,
                x_dot=x_dot,
                quat=quat,
                omega=omega,
                gripper=[],
                unified_config=unified_config,
                dt=1/60
            )
            _, reconstruction_error = ds_policy.compute_reconstruction_error()
            # TODO: Add some basic requirements for the object of reference, so blacklist need more 
            # 1. start pose and end pose of all trajs should be almost the same, otherwise it is not a good reference object
            list_of_reconstruction_errors.append(reconstruction_error)
            if reconstruction_error < min_reconstruction_error and obj_type not in black_list:
                min_reconstruction_error = reconstruction_error
                obj_type_of_reference_best = obj_type
        assert obj_type_of_reference_best is not None, "No object of reference found"
        return obj_type_of_reference_best, min_reconstruction_error, list_of_reconstruction_errors

    def _extract_relative_pose_data(self, ground_atom_dataset: List[GroundAtomTrajectory], all_objs_types: List[Type], gripper_type: Type, in_contact_pred: Predicate, in_origin_pred: Predicate) -> Tuple[Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]], List[List[Object]], Dict[Type, List[List[np.ndarray]]], List[State], List[Type], List[GroundAtomTrajectory]]:
        relative_pose_dataset_dict = defaultdict(list) # Maps (atom_pred, type1, type2) -> List[rel_pose]

        contact_period_rel_trajs = {}
        goal_reached_states = defaultdict(list)
        object_type_in_contact_with_gripper_longest_duration = []
        traj_all_objs_all = []

        # Strong assumption of contacting with only 1 object during the whole trajectory!
        # pose_feat_name = "pose"

        # 1. process of making contact: how to get to grasp (gripper obj centric DS with goal of cluster in step 2)
        # Atom dataset auto split these
        # 2. process of held contact: how to grasp(gripper obj centric cluster) (Obj Obj frame DS)
        # 2.1 gripper obj centric: Already doing with clustering change only flag off
        # 2.2 obj obj frame:(using goal predicate to find the other object)
        # 3. instant of removed contact: achieving relative pose between two object (obj obj frame cluster goal )
        # done
        gripper_objs = [o for o in ground_atom_dataset[0][0].states[0].data.keys() if 'gripper' in o.type.name]
        if len(gripper_objs) == 0:
            raise ValueError("No gripper found in the trajectory")
        gripper_obj = gripper_objs[0] 
        logging.info("Extracting relative poses ...")
        for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
            goal_reached_states[i] = []
            # Get all objects from the first state that match our target types
            traj_all_objs = [o for o in ll_traj.states[0].data.keys() if o.type in all_objs_types]
            # there is only one object of each type in the trajectory since the code below is not designed to handle multiple objects of the same type
            # if there are two cabinets, the dictonary of contact_period rel traj will mess up
            # Check if there are multiple objects of the same type in the trajectory
            # consider which object to keep, prefer the object closer to gripper
            # If there are multiple objects of the same type, select the one closest to the gripper at the end of the trajectory
            for obj_type in all_objs_types:
                objs_of_this_type = [o for o in traj_all_objs if o.type == obj_type]
                if len(objs_of_this_type) > 1: # 0 or 1 is fine, no filtering needed
                    # Compute distances to gripper for all objects of this type at the end state
                    dist_min = np.inf
                    best_obj = None
                    for o_same in objs_of_this_type:
                        trans1 = ll_traj.states[-1].get(o_same, CFG.trans_feat_name)
                        trans2 = ll_traj.states[-1].get(gripper_obj, CFG.trans_feat_name)
                        dist_to_gripper = np.linalg.norm(np.array(trans1) - np.array(trans2))
                        if dist_to_gripper < dist_min:
                            dist_min = dist_to_gripper
                            best_obj = o_same
                    # Remove all other objects of this type 
                    traj_all_objs = [o for o in traj_all_objs if o.type != obj_type]
                    traj_all_objs.append(best_obj) 
            traj_all_objs_all.append(traj_all_objs) # keep this so we can ground it later
            object_type_in_contact_with_gripper_longest_duration.append({})
            if not ll_traj.states: continue # Skip empty trajectories

            if not CFG.remove_inOrigin_pred:
                init_atoms = None
                init_atoms_pred = []
                finish_adding_init_atoms = False
                for t in range(1, len(atom_seq)):
                    atoms_t = atom_seq[t]
                    atoms_tm1 = atom_seq[t - 1]
                    if t == 1 and len(atoms_tm1) > 0 and any(atom.predicate == in_origin_pred for atom in atoms_tm1):
                        init_atoms = atoms_tm1
                        init_atoms_pred = [atom.predicate for atom in atoms_tm1]
                    assert init_atoms is not None, "No InOrigin predicate found in the first state of the trajectory."
                    if len(atoms_t) > 0 and any(atom.predicate not in init_atoms_pred for atom in atoms_t):
                        finish_adding_init_atoms = True
                    if not finish_adding_init_atoms:
                        ground_atom_dataset[i][1][t] = init_atoms
                    elif any(atom.predicate == in_origin_pred for atom in atoms_t):
                        ground_atom_dataset[i][1][t] = set([atom for atom in atoms_t if atom.predicate != in_origin_pred])

            skip_var =max(int(len(atom_seq) / 50),1)
            logging.debug(f"Processing trajectory {i+1}/{len(ground_atom_dataset)} with {len(atom_seq)} atoms, skipping every {skip_var} atoms.")
            achieved_goal = False 
            for t in range(skip_var, len(atom_seq), skip_var): # Start from 1 to compare with t-1, skip every 4, for efficiency
                state_t = ll_traj.states[t]
                atoms_t = atom_seq[t]
                atoms_tm1 = atom_seq[t-skip_var]
                lost_atoms = atoms_tm1 - atoms_t
                # assume when lost contact, we have moved the item to where we want it to be
                # or when the episode end, we have the item at where we want it to be
                # ------ before contact lost, or for last timestep in current traj ------- #
                # ------ add goal predicate for them ------------------------------------- #
                # ------ store states so that we can cluster them later as goal predicate- #
                if not achieved_goal:
                    if t >= len(atom_seq) - skip_var:
                        t_start = t
                        while t_start < len(atom_seq):
                            ground_atom_dataset[i][1][t_start].add(DummyPredicate(f"{CFG.robo_kitchen_task}-goal"))
                            goal_reached_states[i].append(ll_traj.states[t_start])
                            t_start += 1
                        achieved_goal = True
                    else:
                        if any(atom.predicate.name == in_contact_pred.name for atom in lost_atoms):
                            t_start = None
                            for t_test in range(t-skip_var, t): # search for the last state before contact lost
                                if len(atom_seq[t_test]) > len(atom_seq[t_test+1]):
                                    t_start = t_test+1
                                    break
                            logging.debug(f"Contact lost at t={t_start}")
                            assert t_start is not None
                            # if atom.predicate == in_contact_pred: # this is lost gripper with obj
                            #     consistent_contact = False
                            while t_start < len(atom_seq):
                                ground_atom_dataset[i][1][t_start].add(DummyPredicate(f"{CFG.robo_kitchen_task}-goal"))
                                goal_reached_states[i].append(ll_traj.states[t_start])
                                t_start += 1
                            achieved_goal = True

                # ------------------------------------------------------------------------ #

                for atom in atoms_t: # this does not handle multiple objects in contact with the gripper at the same time
                    if CFG.clustering_change_only and atom in atoms_tm1: continue 
                    # this optionally only consider the change of contact, this turns out to be too few points for clustering
                    # so usually all points in contact are used for clustering, i.e. CFG.clustering_change_only is False
                    if hasattr(atom, "predicate") and atom.predicate == in_contact_pred:
                        if atom in atoms_tm1:
                            consistent_contact = True
                        else:
                            consistent_contact = False
                        # Ensure the atom involves the gripper type or handle goals correctly
                        obj1, obj2 = atom.objects
                        assert obj2.type == gripper_type

                        # ------ get relative pose trajs of obj_contact_with_gripper in all other obj's frame ------ #
                        obj_contact_with_gripper = obj1
                        # Exclude robot base from contact duration tracking
                        if "base" not in obj_contact_with_gripper.name.lower():
                            if obj_contact_with_gripper not in object_type_in_contact_with_gripper_longest_duration[i]:
                                object_type_in_contact_with_gripper_longest_duration[i][obj_contact_with_gripper.type] = 0
                            object_type_in_contact_with_gripper_longest_duration[i][obj_contact_with_gripper.type] += 1

                        for obj in traj_all_objs: # go through all object to get obj-obj relative pose
                            if obj.type == gripper_type or obj == obj_contact_with_gripper:
                                continue
                            relative_pose = utils.calculate_relative_pose(state_t, obj, obj_contact_with_gripper, CFG.trans_feat_name, CFG.quat_feat_name)
                            if obj.type not in contact_period_rel_trajs:
                                contact_period_rel_trajs[obj.type] = []
                            if not consistent_contact: # start of contact
                                contact_period_rel_trajs[obj.type].append([relative_pose]) # separate the contact period into different trajectories
                            else:
                                contact_period_rel_trajs[obj.type][-1].append(relative_pose)

                        # -------------------------------------------------------------------------------------------- #
                        # these are used to compute rel pose between GRIPPER and OBJECT for finding end points of DS
                        # Calculate relative pose at the moment of contact (state t)
                        rel_pose_at_contact_obj2_in_obj1_frame = utils.calculate_relative_pose(
                            state_t, obj1, obj2,
                            CFG.trans_feat_name, CFG.quat_feat_name
                        )

                        if rel_pose_at_contact_obj2_in_obj1_frame is not None:
                            key = (atom.predicate, obj1.type, obj2.type, "2in1")
                            relative_pose_dataset_dict[key].append(rel_pose_at_contact_obj2_in_obj1_frame)

                        rel_pose_at_contact_obj1_in_obj2_frame = utils.calculate_relative_pose(
                            state_t, obj2, obj1,
                            CFG.trans_feat_name, CFG.quat_feat_name
                        )

                        if rel_pose_at_contact_obj1_in_obj2_frame is not None:
                            key = (atom.predicate, obj1.type, obj2.type, "1in2")
                            relative_pose_dataset_dict[key].append(rel_pose_at_contact_obj1_in_obj2_frame)
                # find the object in contact with the gripper the longest in each dataset
        for i, obj_type_in_contact_with_gripper_longest_duration in enumerate(object_type_in_contact_with_gripper_longest_duration):
            if len(obj_type_in_contact_with_gripper_longest_duration) == 0: continue
            object_type_in_contact_with_gripper_longest_duration[i] = max(obj_type_in_contact_with_gripper_longest_duration, key=obj_type_in_contact_with_gripper_longest_duration.get)
        logging.info(f"Object in contact with gripper the longest in each dataset: {object_type_in_contact_with_gripper_longest_duration}")


        
        return relative_pose_dataset_dict, traj_all_objs_all, contact_period_rel_trajs, goal_reached_states, object_type_in_contact_with_gripper_longest_duration, ground_atom_dataset
    
    def _find_common_objects_types(self, ground_atom_dataset: List[GroundAtomTrajectory]) -> List[Type]:
        """Find common objects across all trajectories in the dataset."""
        all_objs_types = set([obj.type for obj in ground_atom_dataset[0][0].states[0].data.keys()])
        for traj, _ in ground_atom_dataset[1:]:  # Skip the first one we already processed
            if not traj.states:
                continue  # Skip empty trajectories
            traj_objs_types = set([obj.type for obj in traj.states[0].data.keys()])
            all_objs_types = all_objs_types.intersection(traj_objs_types)  # Keep only objects present in all trajectories
        all_objs_types = list(all_objs_types)  # Convert back to list for further processing

        # Exclude objects whose *type* is explicitly black-listed.  This helps
        # ignore background items (e.g., counters) when deciding on a reference
        # frame.  Extend this set as needed.
        excluded_type_names = {}

        # First apply the original substring filters, then drop any objects
        # whose type is in the blacklist above.
        all_objs_types = [
            o_type for o_type in all_objs_types
            if "finger" not in o_type.name
            and "base"   not in o_type.name
            and o_type not in excluded_type_names
        ]
        logging.info(f"After filtering, {len(all_objs_types)} object types remain") 
        if len(all_objs_types) <= 2:
            raise ValueError(f"Only {len(all_objs_types)} object types remain, which is less than 3. Not enough for finding a reference object.")
        return all_objs_types
    
    def _create_gnd_atom_datasets(self, dataset: Dataset, in_contact_pred: Predicate, in_origin_pred: Predicate, learnt_goal_predicates: Set[Predicate]) -> Tuple[Set[Predicate], List[GroundAtomTrajectory]]:
        """Create ground atom dataset and predicates to monitor."""
        if CFG.remove_inOrigin_pred:
            predicates_to_monitor = {in_contact_pred} | learnt_goal_predicates
        else:
            predicates_to_monitor = {in_contact_pred, in_origin_pred} | learnt_goal_predicates
        ground_atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories, predicates_to_monitor)
        predicates_to_monitor |= {DummyPredicate(f"{CFG.robo_kitchen_task}-goal")}
        return predicates_to_monitor, ground_atom_dataset
    
    def _prepare_env_and_predicates(self):
        """Prepare environment and predicates for clustering."""
        env = get_or_create_env(CFG.env)

        # Identify the InContact predicate and the gripper type
        in_contact_pred = next(p for p in env.predicates if "InContact" in p.name)
        # if not CFG.remove_inOrigin_pred:
        in_origin_pred = next(p for p in env.predicates if "InOrigin" in p.name)
        gripper_type = next(t for t in self._types if "gripper" in t.name) # Assumes gripper type name contains "gripper"
        return env, in_contact_pred, in_origin_pred, gripper_type
    
    # --- Predicate Selection Functions (Beam Search) ---
    def _select_predicates_by_beam_search(self,
                                          candidates: Dict[Predicate, float],
                                          dataset: Dataset,
                                          train_tasks: List[Task]) -> Set[Predicate]:
        """Selects predicates using beam search based on the paper's objective and constraint."""
        # Hyperparameters from CFG
        beam_width = CFG.clustering_search_beam_width
        alpha = CFG.clustering_search_alpha
        max_iterations = CFG.clustering_search_max_iterations # Optional: limit iterations

        # Initialize beam with the empty set
        beam: List[Tuple[float, FrozenSet[Predicate], Set[STRIPSOperator]]] = [(-np.inf, frozenset(), set())] # Score, Predicate Set, Operators

        # Check initial predicates against constraint (if any exist)
        initial_pred_set = frozenset(self._initial_predicates)
        # initial_valid = self._check_plan_length_constraint(initial_pred_set, set(), dataset, [], train_tasks) # Operators not learned yet
        # if not initial_valid:
        #     logging.warning("Initial predicates may violate plan length constraint (if planner were integrated).")
        # Or potentially handle this case more strictly if constraint must hold from start

        best_score = -np.inf
        best_pred_set = frozenset()

        candidate_list = sorted(list(candidates.keys())) # For deterministic iteration

        iteration = 0
        while True: # Loop until convergence or max iterations
            iteration += 1
            if max_iterations is not None and iteration > max_iterations:
                logging.info(f"Beam search reached max iterations ({max_iterations}).")
                break

            successors: List[Tuple[float, FrozenSet[Predicate]]] = []
            processed_sets: Set[FrozenSet[Predicate]] = set(p for _, p, _ in beam)

            # Generate successors by adding one predicate to each set in the beam
            for _, current_preds, _ in beam:
                for cand_pred in candidate_list:
                    if cand_pred.arity == 2 and cand_pred.types[1].name != "gripper_type":
                        continue
                    if cand_pred in current_preds or cand_pred in self._initial_predicates:
                        continue
                    next_pred_set = current_preds | {cand_pred}
                    # Avoid re-evaluating sets already processed in this iteration
                    if next_pred_set in processed_sets:
                        continue
                    processed_sets.add(next_pred_set)

                    # Evaluate the objective function for the successor set
                    # Includes check for the plan length constraint internally
                    combined_preds = initial_pred_set | next_pred_set
                    score, operators = self._evaluate_objective(combined_preds, alpha, dataset, train_tasks)
                    # logging.debug(f"Beam search iteration {iteration}, score: {score:.4f}, num preds: {len(next_pred_set)}")
                    successors.append((score, next_pred_set, operators)) # Store score with the *added* predicates only

            if not successors:
                logging.info("Beam search found no viable successors. Terminating.")
                break # No improvement possible

            # Keep top B successors based on score
            successors.sort(key=lambda x: x[0], reverse=True) # Sort descending by score
            new_beam = successors[:beam_width]

            # Check for convergence (beam hasn't changed or score isn't improving)
            # Simple check: if the best score in the new beam is not better than the previous best
            if new_beam:
                best_score, best_pred_set_added, best_operators = new_beam[0] # Best set in current beam (added preds only)
                logging.info(f"\033[1;36mIteration {iteration} best score: {best_score:.4f}\033[0m")
                logging.info(f"\033[1;32mCurrent best operators: {best_operators}\033[0m")
                logging.info(f"\033[1;33mCurrent best preds: {best_pred_set_added}\033[0m")
                warnings.warn(f"Not doing beam search!!!!!!!!!!!!!!!!!")
                break
            current_best_score_in_beam = new_beam[0][0] if new_beam else -np.inf
            if current_best_score_in_beam <= best_score and iteration > 1 : # Allow first iteration to set baseline
                logging.info("\033[1;35mBeam search converged (no score improvement).\033[0m")
                break

            beam = new_beam

        # Final selection: the best predicate set found that satisfies constraints
        # Need to re-evaluate the best set found to get its final props if needed elsewhere
        final_selected_learned_preds = best_pred_set_added if best_score > -np.inf else frozenset()

        # Return only the *learned* predicates (excluding initial ones)
        return set(final_selected_learned_preds)

    def _evaluate_objective(self,
                            predicates: FrozenSet[Predicate],
                            alpha: float,
                            dataset: Dataset,
                            train_tasks: List[Task]) -> Tuple[float, Set[NSRT]]:
        """Calculates the objective function score for a given predicate set,
           checking constraints. Returns -inf if constraints fail."""

        # Check plan length constraint first (most expensive)
        # Need operators for the constraint check
        atom_dataset = self._create_atom_dataset(dataset, predicates)
        if CFG.clustering_debug:
            for i, (_, atom_seq) in enumerate(atom_dataset):
                print(f"Traj {i}:")
                current_atom_count = 0
                current_atom = atom_seq[0]
                for atom in atom_seq:
                    if atom == current_atom:
                        current_atom_count += 1
                    else:
                        print(f"{current_atom} {current_atom_count}")
                        current_atom = atom
                        current_atom_count = 1

            # Add debug visualization here

            # End debug visualization

        op_term , operators = self._calculate_operator_complexity_term(predicates, dataset, atom_dataset, train_tasks)
        # Now check constraint
        constraint_value = self._check_plan_length_constraint(predicates, operators, dataset, atom_dataset, train_tasks)
        # logging.debug(f"Constraint holds: {constraint_holds}")

        # if constraint_value == 0:
        #     # logging.debug(f"Predicate set failed plan length constraint.")
        #     return -np.inf, operators # Invalid set
        # Calculate segmentation term
        seg_term = self._calculate_segmentation_term(predicates, atom_dataset)             # Cache already handled inside the function call

        # Add the negated absolute value of constraint_value to the score
        score = seg_term - alpha * op_term - CFG.clustering_search_constraint_penalty * abs(constraint_value)
        logging.debug(f"Pred set {predicates}, Seg: {seg_term}, OpComp: {op_term}, Constraint: {constraint_value}, Score: {score:.3f}")
        # logging.debug(f"Pred set size {len(predicates)}, Seg: {seg_term}, OpComp: {op_term}, Score: {score:.3f}")
        return score, operators

    def _calculate_segmentation_term(self,
                                     predicates: FrozenSet[Predicate],
                                     atom_dataset: List[GroundAtomTrajectory]) -> int:
        """Calculates the segmentation term: Σ |ψ(P, τ)|.
        Uses number of segments as |ψ(P, τ)|.
        """
        if predicates in self._segmentation_cache: 
            return self._segmentation_cache[predicates]
        total_segments = 0
        for ll_traj, atom_seq in atom_dataset:
            # Segment trajectory based *only* on the current predicate set
            # Need to ensure atom_seq corresponds *exactly* to predicates,
            # which it should if generated by _create_atom_dataset.
            segments = segment_trajectory(ll_traj, predicates, atom_seq=atom_seq)
            total_segments += len(segments)
        self._segmentation_cache[predicates] = total_segments / len(atom_dataset)
        return self._segmentation_cache[predicates]

    def _calculate_operator_complexity_term(self,
                                            predicates: FrozenSet[Predicate],
                                            dataset: Dataset,
                                            atom_dataset: List[GroundAtomTrajectory],
                                            train_tasks: List[Task]) -> Tuple[int, Set[NSRT]]:
        """Calculates operator complexity |Σ(P, D)| and returns operators."""
        # Use cache if available
        if predicates in self._operator_complexity_cache:
            return self._operator_complexity_cache[predicates]

        # Learn operators using the provided predicates and atom data
        if True:
            # segment_trajectory needs to be called within learn_strips_operators
            # or we need to pre-segment. Let's assume learn_strips_operators handles it.
            # It needs the low-level trajectories too.
            low_level_trajs = [t for t in dataset.trajectories] # Assuming atom_dataset aligns with dataset.trajectories
            # Ensure atom_dataset only contains atoms for 'predicates'
            pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates)
            segmented_trajs = [segment_trajectory(ll_traj, predicates, atom_seq=atom_seq) for (ll_traj, atom_seq) in pruned_atom_data]
            # Print predicates and segment information for each demo
            logging.info(f"Predicates: {', '.join(p.name for p in predicates)}")
            # for i, (_traj, atom_seq) in enumerate(pruned_atom_data):
            # segments = segment_trajectory(ll_traj, predicates, atom_seq=atom_seq)
            # Calculate both segment counts and action counts for each demo
            # segment_lengths = [len(segment) for segment in segmented_trajs]
            # segment_action_counts = [[len(segment.actions) for segment in demo_segments]
            #                         for demo_segments in segmented_trajs]
            # logging.info(f"Segment action counts: \n {segment_action_counts}")
            # for p in predicates:
            #     if p.name [-3:] == 'ID4':
            #         pass

            # TODO: Figure out the right arguments for learn_strips_operators
            # It likely needs the segmented trajectories.
            learned_pnads = learn_strips_operators(
                 trajectories=low_level_trajs, # Pass low-level trajectories
                 train_tasks=train_tasks,
                 predicates=predicates,
                 segmented_trajs=segmented_trajs, # Pass the pre-segmented trajectories
                 verify_harmlessness=False,
                 annotations=None, # No annotations assumed here for invention
                 verbose=False 
             )
            operators = {pnad.op for pnad in learned_pnads}
            print(f"Learned operators: {operators}")
            complexity = len(operators)
            result = (complexity, operators)

        # except (PlanningFailure, PlanningTimeout, TimeoutError, ValueError) as e:
        #     # Handle potential errors during operator learning (e.g., inconsistent data)
        #     logging.warning(f"Operator learning failed for predicate set: {e}")
        #     # Return high complexity or some indicator of failure
        #     result = (np.inf, set()) # Indicate failure with infinite complexity

        # Cache the result
        self._operator_complexity_cache[predicates] = result
        return result

    def _check_plan_length_constraint(self,
                                      predicates: FrozenSet[Predicate],
                                      operators: Set[STRIPSOperator],
                                      dataset: Dataset,
                                      atom_dataset: List[GroundAtomTrajectory],
                                      train_tasks: List[Task]) -> int:
        """Checks plan length difference between demonstrated and optimal plans.
        
        Returns:
            int: Difference in steps (plan_length - demo_length).
                 Positive if plan is longer than demo, negative if plan is shorter.
                 Returns 0 if all trajectories have matching plan lengths or if checks are disabled.
        """
        if predicates in self._plan_constraint_cache:
            return self._plan_constraint_cache[predicates]

        # Initialize cache entry to 0 (no difference)
        self._plan_constraint_cache[predicates] = 0

        if not CFG.clustering_check_plan_length_constraint:
            #  logging.debug("Skipping plan length constraint check (disabled by CFG).")
            return 0  # Skip check if disabled by CFG

        if not operators:  # If operator learning failed, return large negative value
            logging.debug("Cannot check plan length constraint: Operator learning failed.")
            self._plan_constraint_cache[predicates] = -1000  # Significant negative value
            return -1000

        # The 'operators' set already contains STRIPSOperator objects
        strips_ops = operators

        diffs = []  # Store differences for all trajectories

        # Iterate through each demonstration trajectory
        for i, (ll_traj, atom_seq) in enumerate(atom_dataset):
            if not ll_traj.states:
                logging.debug(f"Skipping traj {i}: No states.")
                continue  # Skip trajectories with no states

            init_state = ll_traj.states[0]
            final_state = ll_traj.states[-1]

            # Create initial and goal atom sets using the current predicates
            init_atoms = utils.abstract(init_state, predicates)
            goal_atoms = utils.abstract(final_state, predicates)

            if init_atoms == goal_atoms:
                logging.debug(f"Skipping traj {i}: Init atoms == Goal atoms.")
                continue  # Skip trivial trajectories where start equals goal

            # Get demonstrated plan length (number of segments)
            demo_segments = segment_trajectory(ll_traj, predicates, atom_seq=atom_seq)
            demo_plan_len = len(demo_segments) + 1  # segment does not include last section

            # Create a planning task
            task = Task(init_state, goal_atoms)

            # Run the planner using the learned NSRTs
            plan, _, metrics = run_task_plan_once(
                task=task,
                nsrts=strips_ops,    # Pass NSRTs
                preds=set(predicates),  # Pass predicates
                types=self._types,   # Pass types
                timeout=10.0,   # Pass timeout
                seed=0,      # Pass seed
                task_planning_heuristic=CFG.sesame_task_planning_heuristic,  # Pass heuristic
            )

            # Check planner result
            if plan is None:
                # Planner failed (timeout or unsolvable)
                continue
            else:
                planner_plan_len = len(plan)
                # Calculate difference: positive if plan is longer, negative if shorter
                diff = planner_plan_len - demo_plan_len
                # logging.debug(f"Traj {i}: Demo len={demo_plan_len}, Planner len={planner_plan_len}, Diff={diff}")

                diffs.append(np.abs(diff))

        avg_diff = np.mean(diffs) if len(diffs) > 0 else np.inf
        self._plan_constraint_cache[predicates] = avg_diff
        return avg_diff

    # --- Helper Functions ---
    def _create_atom_dataset(self, dataset: Dataset, predicates: Set[Predicate] | FrozenSet[Predicate]) -> List[GroundAtomTrajectory]:
        """Helper to create ground atoms for evaluation. Uses cache."""
        # Convert to frozenset for caching key
        frozen_preds = frozenset(predicates)
        if frozen_preds in self._atom_dataset_cache:
            return self._atom_dataset_cache[frozen_preds]

        # Important: Ensure this creates atoms *only* for the given predicates.
        atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories, set(predicates))
        self._atom_dataset_cache[frozen_preds] = atom_dataset
        return atom_dataset

    def _rename_predicates_to_remove_incompatible_chars(self, predicates_and_costs: Dict[Predicate, float]) -> Dict[Predicate, float]:
        """Renames predicates to get rid of characters in the name that are
        incompatible with PDDL planners like FD. Reused from grammar search."""
        renamed_predicates: Dict[Predicate, float] = {}
        for p, cost in predicates_and_costs.items():
            # Assumes classifiers are _RelativeFeatureClusterClassifier or _AbsoluteFeatureClusterClassifier
            # Their __str__ methods should already generate compatible names.
            # If other predicate types exist, they might need renaming here.
            new_name = p.name # Already formatted by classifier __str__
            # Basic sanity check/replacement if needed
            new_name = new_name.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "")
            if new_name != p.name:
                renamed_pred = Predicate(new_name, p.types, p._classifier) # pylint: disable=protected-access
                renamed_predicates[renamed_pred] = cost
            else:
                renamed_predicates[p] = cost
        return renamed_predicates 

    def _plot_relative_trajectories(self, dataset: Dataset, type1_name: str, type2_name: str, pred: Predicate) -> None:
        """Plots trajectories of the second object type in the reference frame of the first object type.
        
        For each trajectory in the dataset, transforms positions of type2 objects into
        the reference frame of type1 objects and visualizes these relative motions.
        """
        if not CFG.clustering_debug:
            return

        # Find the object types by name
        type1 = None
        type2 = None
        for obj_type in {obj.type for traj in dataset.trajectories for obj in traj.states[0]}:
            if obj_type.name == type1_name:
                type1 = obj_type
            elif obj_type.name == type2_name:
                type2 = obj_type

        if type1 is None or type2 is None:
            logging.warning(f"Could not find types {type1_name} and/or {type2_name} for trajectory visualization")
            return

        trans_feat_name = "translation"
        quat_feat_name = "quaternion"

        # Check if we have an existing cluster figure to overlay on
        if (hasattr(self, '_last_cluster_fig') and self._last_cluster_fig is not None and
            hasattr(self, '_last_cluster_ax') and self._last_cluster_ax is not None and
            hasattr(self, '_last_cluster_type1') and self._last_cluster_type1 == type1_name and
            hasattr(self, '_last_cluster_type2') and self._last_cluster_type2 == type2_name and
            hasattr(self, '_last_cluster_feat') and self._last_cluster_feat == "pose"): # Ensure it's a pose plot

            # Use the existing figure and axis for overlay
            fig = self._last_cluster_fig
            ax = self._last_cluster_ax
            logging.info(f"Overlaying trajectories on existing cluster visualization for {type1_name}-{type2_name}")
            is_overlay = True
            fname = self._last_cluster_fname.replace("frame", "frame_with_traj")
        else:
            # Create a new figure
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"Trajectories of {type2_name} in {type1_name}'s reference frame")
            ax.set_xlabel('X relative')
            ax.set_ylabel('Y relative')
            ax.set_zlabel('Z relative')
            is_overlay = False
            fname = f"rel_traj_{CFG.robo_kitchen_task}_{pred.name}_{type2_name}_in_{type1_name}_frame.png"

        # Different colors for different trajectories - use brighter colors for trajectories
        colors = plt.cm.rainbow(np.linspace(0, 1, len(dataset.trajectories)))

        # Mark which trajectories are used
        trajectories_plotted = False

        for traj_idx, traj in enumerate(dataset.trajectories):
            # Skip trajectories with too few states
            if len(traj.states) < 2:
                continue

            # Find all objects of the required types in this trajectory
            type1_objs = list(traj.states[0].get_objects(type1))
            type2_objs = list(traj.states[0].get_objects(type2))

            if not type1_objs or not type2_objs:
                continue

            # For simplicity, just use the first object of each type
            # Could be extended to show all pairs
            obj1 = type1_objs[0]
            obj2 = type2_objs[0]

            # Collection for relative positions across time
            relative_positions = []

            for state in traj.states:
                # Calculate relative pose in each state
                rel_pose = utils.calculate_relative_pose(state, obj1, obj2, 
                                                       trans_feat_name, 
                                                       quat_feat_name)
                if rel_pose is not None:
                    # Just extract the translation part (first 3 components)
                    relative_positions.append(rel_pose[:3])

            if relative_positions:
                # Convert to numpy array for plotting
                relative_positions = np.array(relative_positions)

                # Plot the trajectory
                traj_label = f"Traj {traj_idx}" if not trajectories_plotted else None
                ax.plot(relative_positions[:, 0], 
                        relative_positions[:, 1], 
                        relative_positions[:, 2], 
                        '-', color=colors[traj_idx], 
                        linewidth=2,
                        label=traj_label)

                # Mark start and end points
                ax.scatter(relative_positions[0, 0], 
                           relative_positions[0, 1], 
                           relative_positions[0, 2], 
                           color=colors[traj_idx], marker='o', s=100, 
                           label="Start point" if not trajectories_plotted else None)
                ax.scatter(relative_positions[-1, 0], 
                           relative_positions[-1, 1], 
                           relative_positions[-1, 2], 
                           color=colors[traj_idx], marker='s', s=100,
                           label="End point" if not trajectories_plotted else None)

                trajectories_plotted = True

        # Determine final axis limits (considering overlay)
        traj_xlim = ax.get_xlim()
        traj_ylim = ax.get_ylim()
        traj_zlim = ax.get_zlim()

        if is_overlay and hasattr(self, '_last_cluster_xlim'): # Check if stored limits exist
            # Combine cluster and trajectory limits
            final_xlim = (min(traj_xlim[0], self._last_cluster_xlim[0]), 
                          max(traj_xlim[1], self._last_cluster_xlim[1]))
            final_ylim = (min(traj_ylim[0], self._last_cluster_ylim[0]), 
                          max(traj_ylim[1], self._last_cluster_ylim[1]))
            if self._last_cluster_zlim: # Check if cluster plot was 3D
                final_zlim = (min(traj_zlim[0], self._last_cluster_zlim[0]),
                              max(traj_zlim[1], self._last_cluster_zlim[1]))
            else: # Fallback if cluster plot wasn't 3D (shouldn't happen for pose overlay)
                final_zlim = traj_zlim
        else:
            final_xlim = traj_xlim
            final_ylim = traj_ylim
            final_zlim = traj_zlim

        # Apply equal aspect ratio based on the *final* combined range
        # Avoid errors if range is zero
        ax.set_xlim(final_xlim[0], final_xlim[1])
        ax.set_ylim(final_ylim[0], final_ylim[1])
        ax.set_zlim(final_zlim[0], final_zlim[1])
        # ax.set_aspect("equal")

        # Set view angle (consistent for both new and overlaid plots)
        ax.view_init(elev=20., azim=-35) # Example view angle

        # Add legend with a good location
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # If we're overlaying, use the stored title from cluster visualization
        if is_overlay and hasattr(self, '_last_cluster_title'):
            ax.set_title(f"{self._last_cluster_title}\nwith Object Trajectories")

        # Save the visualization
        os.makedirs("feature_data", exist_ok=True)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"feature_data/{fname}")
        logging.info(f"Saved {'combined cluster and' if is_overlay else ''} relative trajectory visualization to feature_data/{fname}")
        plt.close(fig)

        # Clear references
        if is_overlay:
            self._last_cluster_fig = None
            self._last_cluster_ax = None
            self._last_cluster_title = None

    def _test_clustering_with_dummy_data(self, num_clusters=3, points_per_cluster=50, 
                                        noise_level=0.05, cluster_separation=0.5):
        """Test HDBSCAN clustering with synthetic pose data.
        
        Args:
            num_clusters: Number of distinct clusters to generate
            points_per_cluster: Number of points in each cluster
            noise_level: Standard deviation of Gaussian noise added to each cluster
            cluster_separation: Distance between cluster centers
        """
        logging.info(f"Generating synthetic pose data with {num_clusters} clusters, "
                     f"{points_per_cluster} points per cluster, noise level {noise_level}")

        # Import necessary visualization packages
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        import matplotlib
        matplotlib.use('TkAgg')  # Try TkAgg first

        # Set random seed for reproducibility
        np.random.seed(42)

        # Function to generate random rotation quaternion
        def random_quaternion():
            # Generate random rotation axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)

            # Random angle (in radians)
            angle = np.random.uniform(0, 2*np.pi)

            # Convert axis-angle to quaternion
            sin_a = np.sin(angle/2)
            cos_a = np.cos(angle/2)
            qx, qy, qz = axis * sin_a
            qw = cos_a

            # Return in xyzw format
            return np.array([qx, qy, qz, qw])

        # Generate cluster centers with good separation
        centers = []
        for i in range(num_clusters):
            # Position each cluster in a grid pattern
            grid_size = int(np.ceil(np.sqrt(num_clusters)))
            row = i // grid_size
            col = i % grid_size

            # Create translation with separation
            trans = np.array([
                col * cluster_separation - (grid_size-1) * cluster_separation / 2,
                row * cluster_separation - (grid_size-1) * cluster_separation / 2,
                0.0  # Keep Z at zero for clarity
            ])

            # Create a random rotation for each cluster
            quat = random_quaternion()

            # Combine into 7D pose vector [tx, ty, tz, qx, qy, qz, qw]
            center = np.concatenate([trans, quat])
            centers.append(center)

        # Generate data points with noise
        all_data = []
        true_labels = []

        # Generate a single random quaternion to use for all clusters
        # This makes all clusters have the same orientation, varying only in position
        shared_quaternion = random_quaternion()

        # Update all centers to use the same quaternion
        for i in range(len(centers)):
            centers[i][3:] = shared_quaternion

        for cluster_idx, center in enumerate(centers):
            for _ in range(points_per_cluster):
                # Add Gaussian noise to translation
                trans_noise = np.random.normal(0, noise_level, 3)
                noisy_trans = center[:3] + trans_noise

                # Add noise to quaternion (small rotation perturbation)
                # Generate small random rotation
                noise_in_deg = 30
                noise_angle = np.random.normal(0, noise_in_deg * np.pi / 180)  # Smaller noise for rotation
                noise_axis = np.random.randn(3)
                noise_axis = noise_axis / np.linalg.norm(noise_axis)

                # Convert to quaternion
                sin_a = np.sin(noise_angle/2)
                cos_a = np.cos(noise_angle/2)
                noise_quat = np.array([*noise_axis * sin_a, cos_a])  # xyzw format

                # Apply noise rotation to center quaternion using quaternion multiplication
                center_quat = center[3:]

                # Use scipy's Rotation for quaternion multiplication
                center_rot = R.from_quat(center_quat)
                noise_rot = R.from_quat(noise_quat)
                noisy_rot = noise_rot * center_rot
                noisy_quat = noisy_rot.as_quat()

                # Create noisy pose
                noisy_pose = np.concatenate([noisy_trans, noisy_quat])
                all_data.append(noisy_pose)
                true_labels.append(cluster_idx)

        # Add some random noise points
        num_noise_points = int(points_per_cluster * 0.1)  # 10% of points per cluster
        for _ in range(num_noise_points):
            # Random position in the general area
            trans = np.random.uniform(-cluster_separation * grid_size, 
                                      cluster_separation * grid_size, 3)
            quat = random_quaternion()
            noise_point = np.concatenate([trans, quat])
            all_data.append(noise_point)
            true_labels.append(-1)  # -1 for noise points

        all_data = np.array(all_data)
        true_labels = np.array(true_labels)

        # Run clustering
        logging.info("Running HDBSCAN on synthetic data...")
        feat_name = "pose"  # This will use the SE(3) metric

        # Use _cluster_feature_dataset to perform clustering
        data_array, labels, unique_labels = self._cluster_feature_dataset(
            all_data.tolist(), CFG.clustering_se3_epsilon, feat_name)

        # Calculate clustering metrics
        num_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = sum(1 for label in labels if label == -1)

        logging.info(f"HDBSCAN found {num_clusters_found} clusters (ground truth: {num_clusters})")
        logging.info(f"HDBSCAN identified {noise_points} noise points")

        # Create a visualization
        fig = plt.figure(figsize=(20, 15))

        # 3D plot of translations with ground truth labels
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(all_data[:, 0], all_data[:, 1], all_data[:, 2], 
                              c=true_labels, cmap='tab10', s=50, alpha=0.7)
        ax1.set_title('Ground Truth Clusters (Translations)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 3D plot of translations with HDBSCAN labels
        ax2 = fig.add_subplot(222, projection='3d')
        scatter2 = ax2.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], 
                              c=labels, cmap='tab10', s=50, alpha=0.7)
        ax2.set_title(f'HDBSCAN Clusters: {num_clusters_found} found (Translations)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Rotation visualization (Optional)
        # Project quaternions to 3D using PCA if needed

        # Add information table
        params_text = (
            f"Parameters:\n"
            f"Number of clusters: {num_clusters}\n"
            f"Points per cluster: {points_per_cluster}\n"
            f"Noise level: {noise_level}\n"
            f"Cluster separation: {cluster_separation}\n\n"
            f"Results:\n"
            f"Clusters found: {num_clusters_found}\n"
            f"Noise points: {noise_points}/{len(labels)}"
        )

        fig.text(0.1, 0.3, params_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Draw cluster centers
        for i, center in enumerate(centers):
            ax1.scatter([center[0]], [center[1]], [center[2]], 
                       c='black', marker='*', s=200, edgecolor='white')
            ax1.text(center[0], center[1], center[2], f'Center {i}', fontsize=10)

        # Save figure
        plt.tight_layout()
        os.makedirs("feature_data", exist_ok=True)
        plt.savefig("feature_data/hdbscan_test_results.png")
        logging.info("Saved visualization to feature_data/hdbscan_test_results.png")
        plt.show()

        return labels, true_labels
