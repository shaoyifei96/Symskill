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
from predicators.ground_truth_models.robo_kitchen.nsrts import RoboKitchenGroundTruthNSRTFactory
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
from predicators.structs import Dataset, GroundAtomTrajectory, NSRT, LiftedAtom, Object, ParameterizedOption, Predicate, Segment, State, Task, Type, STRIPSOperator, GroundAtom, DummyPredicate, Variable, DummyGroundAtom
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
from scipy.ndimage import uniform_filter1d
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
        relative_pose = utils.calculate_relative_pose_from_state(s, obj1, obj2,
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
        # is_classified = mahalanobis_dist_sq_trans <= self.mahalanobis_threshold_trans and mahalanobis_dist_sq_rot <= self.mahalanobis_threshold_rot
        # color = "\033[92m" if is_classified else "\033[91m"  # Green if True, Red if False
        # print(f"{color} {obj1.name}, {obj2.name}, trans: {mahalanobis_dist_sq_trans},trans_thresh: {self.mahalanobis_threshold_trans}, rot: {mahalanobis_dist_sq_rot}, rot_thresh: {self.mahalanobis_threshold_rot}\033[0m")
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
            relative_feature = utils.calculate_relative_pose_from_state(s, obj1, obj2,
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
            relative_feature = utils.calculate_relative_pose_from_state(s, obj1, obj2, 
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

@dataclass(frozen=True, eq=False, repr=False)
class _DynamicRepositionClassifier(_BinaryClassifier):
    """Dynamic classifier that selects appropriate cluster based on object types."""
    
    object1_type: Type  # object_type
    object2_type: Type  # base_type  
    # Map from (obj1_type_name, task) to the specific classifier
    type_task_to_classifier: Dict[Tuple[str, str], _RelativeFeatureCovClusterClassifierTransRot]
    # Store all the samplers for different combinations
    type_task_to_sampler_data: Dict[Tuple[str, str], Tuple[np.ndarray, R]]
    
    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        # Get the actual type of obj1 (the object we're positioning relative to)
        obj1_type_name = obj1.type.name
        
        # Try to find a matching classifier for any task with this object type
        for (type_name, task), classifier in self.type_task_to_classifier.items():
            if type_name == obj1_type_name:
                return classifier._classify_object(s, obj1, obj2)
        
        # If no specific classifier found, return False
        return False
    
    def get_sampler_data_for_objects(self, obj1: Object) -> Optional[Tuple[np.ndarray, R]]:
        """Get sampler data (trans_center, rot_center) for the given object type."""
        obj1_type_name = obj1.type.name
        
        # Return the first matching sampler data for this object type
        for (type_name, task), sampler_data in self.type_task_to_sampler_data.items():
            if type_name == obj1_type_name:
                return sampler_data
        return None

    def __str__(self) -> str:
        return f"DynamicRepositionCluster[{self.object1_type.name}, {self.object2_type.name}]"
    
    def pretty_str(self) -> Tuple[str, str]:
        name1 = f"?x0:{self.object1_type.name}"
        name2 = f"?x1:{self.object2_type.name}"
        vars_str = f"{name1}, {name2}"
        body_str = f"DynamicRepositionTarget({name1}, {name2})"
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


    def _add_delete_effects(self, all_entries: Dict[str, Set[Predicate]], entry_to_exclude: str, new_params: List[Variable], new_delete_effects: Set[LiftedAtom], task_name: str) -> Tuple[List[Variable], Set[LiftedAtom]]:
        """
        Add delete effects for the incontact predicates and robotbase rel pos preds.
        """
        mode = "robotbase" if "RobotBaseRelCovCluster" in entry_to_exclude else "incontact"
        for key, effects_to_delete in all_entries.items():
            if len(key) == 2: key = key[0]
            for eff in effects_to_delete: # these are necessary for staionary multitask, since if there are no two objects, it cannot ground if adding another detele parameter
                if mode == "incontact" and key == entry_to_exclude: #and task_name == eff.name.split("-")[1]: # incontact predicate
                    continue  # Skip the object that's currently in contact, because it will be deleted by the incontact predicate
                elif mode == "robotbase" and "RobotBaseRelPosPred" in key and key.split("-")[1] == entry_to_exclude.split("-")[1]: #and task_name == eff.name.split("-")[1]:
                    continue # robot base predicate
                # do not exclude for robotbase, since all other robotbase predicates are deleted, the one at first parameter is added, so there is no conflict

                if mode == "robotbase":
                    delete_selectable_vars = new_params[1:] # 0 is new location, 1 is base, 2 is old location
                else:
                    if new_params[0].type.name == "gripper_type": # remove the item param
                        delete_selectable_vars = new_params[0:1] + new_params[2:] # 0 is gripper, 1 is new item
                    else:
                        delete_selectable_vars = new_params[1:] # 0 is new item, 1 is gripper
                # Find the corresponding variables from NSRT parameters
                obj0_var = None
                obj1_var = None
                for var in delete_selectable_vars:
                    if var.type.name == eff.types[0].name:  # object type
                        obj0_var = var
                    elif var.type.name == eff.types[1].name:
                        obj1_var = var

                if obj0_var is None and obj1_var is None:
                    # Need to add new parameters for both object types
                    extra_param_types = [eff.types[0], eff.types[1]]
                    new_vars_to_add = utils.create_new_variables(extra_param_types, new_params)
                    obj0_var = new_vars_to_add[0]
                    obj1_var = new_vars_to_add[1]
                elif obj0_var is None:
                    # Need to add a new parameter for this object type
                    extra_param_type = eff.types[0]
                    new_vars_to_add = utils.create_new_variables([extra_param_type], new_params)
                    obj0_var = new_vars_to_add[0]
                elif obj1_var is None: #ideally this should not happen since for incontact, the gripper is always the second object, for robotbase, it is always object and base type
                    # Need to add a new parameter for this object type
                    extra_param_type = eff.types[1]
                    new_vars_to_add = utils.create_new_variables([extra_param_type], new_params)
                    obj1_var = new_vars_to_add[0]
                else: # both are defined
                    new_vars_to_add = []
                for var in new_vars_to_add:
                    for param in delete_selectable_vars:
                        if var.type.name == param.type.name:
                            raise ValueError(f"Variable {var} of type {var.type.name} already exists in new_params")
                new_params = new_params + new_vars_to_add
                lifted_atom = LiftedAtom(eff, [obj0_var, obj1_var])
                new_delete_effects.add(lifted_atom)
                
                
        ## check check!!
        delete_effect_vars = set()
        for atom in new_delete_effects:
            delete_effect_vars.update(atom.variables)
        
        # If there are variables in delete effects that aren't in parameters, add them
        missing_vars = delete_effect_vars - set(new_params)
        if missing_vars:
            raise ValueError(f"NSRT has variables in delete effects that aren't in parameters: {missing_vars}")

        return new_params, new_delete_effects
    
    def _add_additional_remove_effects_operators(self, available_object_types: Set[Type]) -> None:
        """Add additional remove/delete effects operators to the loaded operators."""
        

        # Pre-process CFG.dict_contact_predicate_to_rel_pose_predicates to filter relevant entries
        incontact_entries = {}
        robotbase_entries = {}
        for (key1, key2, key3), effects in CFG.dict_contact_predicate_to_rel_pose_predicates.items():
            if key1 == 'InContact' and key2 in available_object_types:
                incontact_entries[key2] = effects
            elif "RobotBaseRelPosPred" in key1:
                robotbase_entries[key1, key2] = effects
        logging.info(f"Found {len(incontact_entries)} InContact entries for available object types")
        logging.info(f"Found {len(robotbase_entries)} RobotBaseRelPosPred entries for available object types")

        updated_nsrts = set()
        for nsrt in self._nsrts:
            task_name = nsrt.name.split("-")[0]
            new_delete_effects = set(nsrt.delete_effects)  # Create a copy to avoid shared reference bug
            new_params = list(nsrt.parameters)  # Create a copy to avoid shared reference bug
            
            if len(nsrt.add_effects) == 1:
                add_eff = list(nsrt.add_effects)[0]
                # deal with 2 cases here, if incontact predicate, then we need to delete the other objects that can be in contact with the gripper
                # if robotbase rel pos pred, then we need to delete both the incontact predicates with all objects, and the robotbase rel pos pred with other names

                if add_eff.entities[1].type.name == "gripper_type": # comes into contact, exclude the predicates that keeps the incontact with the gripper
                    object_in_contact = add_eff.entities[0].type.name
                    new_params, new_delete_effects = self._add_delete_effects(incontact_entries, object_in_contact,  new_params, new_delete_effects, task_name)
                elif "RobotBaseRelCovCluster" in str(add_eff): # robot base rel pos pred, exclude the predicates that keeps the incontact with the gripper
                    # delete the robotbase rel pos pred with other names
                    new_params, new_delete_effects = self._add_delete_effects(robotbase_entries, str(add_eff),  new_params, new_delete_effects, task_name)
                    # delete all incontact predicates if moving the base, using str(add_eff) to enforce mode robotbase
                    new_params, new_delete_effects = self._add_delete_effects(incontact_entries, str(add_eff),  new_params, new_delete_effects, task_name)
                else:
                    logging.warning(f"NSRT {nsrt.name} is not gripper related or robotbase rel pos pred, directly adding to updated_nsrts")
            else:
                logging.warning(f"NSRT {nsrt.name} has {len(nsrt.add_effects)} add effects, directly adding to updated_nsrts")
            
            nsrt = nsrt.copy_with(parameters=new_params, delete_effects=new_delete_effects)
            # Add the potentially modified NSRT to the updated set
            updated_nsrts.add(nsrt)
        
        # Update self._nsrts with the modified NSRTs
        self._nsrts = updated_nsrts
    def _get_available_object_types(self) -> Tuple[Set[Type], Set[str]]:
        env = get_or_create_env(CFG.env)
        ob = env.reset(train_or_test="test", task_idx=0)
        state = env.state_info_to_state(ob["state_info"])
        object_type = Object("dummy_object", env.obj_name_to_type["dummy_object"]).type
        available_object_types = set()
        available_object_types_names = set()
        available_object_types.add(object_type)
        available_object_types_names.add(object_type.name)
        for obj in state:
            available_object_types.add(obj.type)
            available_object_types_names.add(obj.type.name)
        logging.info(f"Object types found in sample state: {available_object_types}")
        return available_object_types, available_object_types_names
    
    def _add_base_motion_add_effects(self) -> None:
        """Add base motion add effects to the loaded operators.
        delete effects are added in _add_additional_remove_effects_operators """
        new_nsrts = set()
        # get all possible types of robot base locations
        robot_base_locations_preds = {}
        for pred, obj1_type, obj2_type in CFG.dict_contact_predicate_to_rel_pose_predicates.keys():
            if "RobotBaseRelPosPred" in pred:
                # Create a copy of the set to avoid sharing references
                robot_base_locations_preds[obj1_type] = set(CFG.dict_contact_predicate_to_rel_pose_predicates[pred, obj1_type, obj2_type])
        logging.info(f"Robot base locations: {robot_base_locations_preds}")

        for nsrt in self._nsrts:
            if nsrt.name == "RepositionBase":
                # Keep the original RepositionBase NSRT, add a new NSRT for each task
                for pred, obj1_type, obj2_type in CFG.dict_contact_predicate_to_rel_pose_predicates.keys():
                    val_pred = CFG.dict_contact_predicate_to_rel_pose_predicates[pred, obj1_type, obj2_type]
                    # Handle multiple predicates in val_pred
                    # Check if there are multiple RobotBaseRelCovCluster predicates
                    robot_base_clusters = [p for p in val_pred if "RobotBaseRelCovCluster" in p.name]
                    if len(robot_base_clusters) > 1:
                        raise ValueError("Multiple RobotBaseRelCovCluster predicates not implemented")  # TODO: handle multiple predicates
                    elif len(robot_base_clusters) == 0:
                        continue
                    else:
                        reposition_target_pred = robot_base_clusters[0]
                        task = reposition_target_pred.name.split("-")[1]
                        obj_contact_base_pred = [ v for k, v in CFG.dict_contact_predicate_to_rel_pose_predicates.items() if k[0] == "RobotBaseRelPosObjPred-"+task ]
                        # assert len(obj_contact_base_pred) == 1, "Multiple RobotBaseRelPosObjPred predicates not implemented"
                        assert nsrt.parameters[1].type.name == obj2_type
                        for robot_base_location, pred_delete_effects in robot_base_locations_preds.items():
                            # if robot_base_location == obj1_type: # it is possible to move from the other cabinet location
                            #     continue
                            # Create a separate NSRT for each predicate in val_pred
                            delete_effect = list(pred_delete_effects)[0]
                            from_location_type = delete_effect.types[0]
                            # replace all parameters with new variables
                            new_vars_to_add = utils.create_new_variables(reposition_target_pred.types + [from_location_type])
                            new_nsrt = nsrt.copy_with( # only option is kept
                                name=f"RepositionBase-{task}-from-{robot_base_location}",
                                preconditions={LiftedAtom(delete_effect, [new_vars_to_add[2], new_vars_to_add[1]])},
                                delete_effects={LiftedAtom(delete_effect, [new_vars_to_add[2], new_vars_to_add[1]])},
                                parameters=new_vars_to_add,
                                option_vars=new_vars_to_add,
                                add_effects={LiftedAtom(reposition_target_pred, new_vars_to_add[0:2])},
                                _sampler=RoboKitchenGroundTruthNSRTFactory.create_sampler_with_extra_data(
                                    trans_rot=(reposition_target_pred._classifier.trans_center, reposition_target_pred._classifier.rot_center)),
                            )
                            new_nsrts.add(new_nsrt)
            else:
                new_nsrts.add(nsrt)
        self._nsrts = new_nsrts

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # We need to properly load the learned predicates if they exist
        main_folder = f"{CFG.approach_dir}/"
        all_files = os.listdir(main_folder)
        approach_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith(".NSRTs")]
        contact2rel_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith("_contact2rel_preds.pkl")]
        goal_files = [main_folder + f for f in all_files if f.startswith(f"{CFG.env}__{CFG.approach}") and f.endswith("_gtgoal2dummy_preds.pkl")]

        for file in approach_files:
            if CFG.robo_kitchen_task in CFG.composite_tasks:
                with open(file, "rb") as f:
                    loaded_nsrts = pkl.load(f)
                    self._nsrts.update(loaded_nsrts)
            else: # only load the NSRTs for the current task
                if CFG.robo_kitchen_task in file:
                    with open(file, "rb") as f:
                        loaded_nsrts = pkl.load(f)
                        self._nsrts.update(loaded_nsrts)

        from predicators.ground_truth_models import get_gt_nsrts
        gt_nsrts = get_gt_nsrts(CFG.env, self._initial_predicates, self._initial_options)
        
        self._nsrts = set(gt_nsrts).union(self._nsrts)

        for file in contact2rel_files:
            if CFG.robo_kitchen_task in CFG.composite_tasks:
                with open(file, "rb") as f:
                    contact2rel_preds = pkl.load(f)
                    for key, value in contact2rel_preds.items():
                        if key not in CFG.dict_contact_predicate_to_rel_pose_predicates:
                            CFG.dict_contact_predicate_to_rel_pose_predicates[key] = value
                        else:
                            CFG.dict_contact_predicate_to_rel_pose_predicates[key].update(value)
            else:
                if CFG.robo_kitchen_task in file:
                    with open(file, "rb") as f:
                        contact2rel_preds = pkl.load(f)
                        for key, value in contact2rel_preds.items():
                            if key not in CFG.dict_contact_predicate_to_rel_pose_predicates:
                                CFG.dict_contact_predicate_to_rel_pose_predicates[key] = value
                            else:
                                CFG.dict_contact_predicate_to_rel_pose_predicates[key].update(value)

        for file in goal_files:
            if CFG.robo_kitchen_task in CFG.composite_tasks:
                with open(file, "rb") as f:
                    gtgoal2dummy_preds = pkl.load(f)
                    for key, value in gtgoal2dummy_preds.items():
                        if key not in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
                            CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key] = value
                        else:
                            CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key].update(value)
            else:
                if CFG.robo_kitchen_task in file:
                    with open(file, "rb") as f:
                        gtgoal2dummy_preds = pkl.load(f)
                        for key, value in gtgoal2dummy_preds.items():
                            if key not in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
                                CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key] = value
                            else:
                                CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[key].update(value)

        # add base relative pose predicates as the base motion add effects

        # Get a sample state to check which object types actually exist# Get a sample state to check which object types actually exist

        all_available_object_types, all_available_object_types_names = self._get_available_object_types()


        if CFG.enable_base_ref_obj_precondition:     
            self._add_base_motion_add_effects() 

        self._add_additional_remove_effects_operators(all_available_object_types_names)
        
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
        elif CFG.robo_kitchen_task == "PnPCabToCounter":
            keep_indices = [9, 17, 23, 29, 31, 33, 36]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPCabToCounterTomato":
            # Use similar indices as PnPCabToCounter since it's the same basic action
            keep_indices = [0, 1, 3, 4, 5]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPCounterToStove":
            keep_indices = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
            # don't need 50, just take half to shorten learning time.
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "PnPStoveToCounter":
            keep_indices = [ 0, 1, 3, 5, 6, 7, 8, 9]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
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
        elif CFG.robo_kitchen_task == "MocapOpenLid":
            keep_indices = [0, 1, 2, 3, 4, 5, 6, 7]
            dataset._trajectories = [dataset._trajectories[i] for i in keep_indices if i < len(dataset._trajectories)]
        elif CFG.robo_kitchen_task == "mocap_pour_pot":
            keep_indices = [0, 1, 2, 4]
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
            different_seg_count_trajs = []
            if not candidates:
                logging.warning("No candidate predicates generated. Learning NSRTs with initial predicates only.")
                self._learned_predicates = set()
            else:
                logging.info("Selecting predicates via beam search...")
                self._learned_predicates = self._select_predicates_by_beam_search(candidates, dataset, self._train_tasks)
                logging.info(f"Selected {len(self._learned_predicates)} predicates.")
                og_pred_atom_dataset = self._create_atom_dataset(dataset, self._learned_predicates | self._initial_predicates)
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
        trajs = list(dataset.trajectories)  # Create a copy to avoid modifying original dataset
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


            # Perform clustering
            data_array, labels, unique_labels = self._cluster_feature_dataset(data, CFG.clustering_baseline_epsilon, feat_name)
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
            
            # Compute and print distances between all cluster center pairs
            if len(kept_cluster_labels_list) > 1:
                print(f"\n=== Cluster Center Distances for {type1.name}-{type2.name}-{feat_name} ===")
                for i, label1 in enumerate(kept_cluster_labels_list):
                    for j, label2 in enumerate(kept_cluster_labels_list):
                        if i < j:  # Only compute upper triangle to avoid duplicates
                            center1 = kept_clusters_info[label1]['center']
                            center2 = kept_clusters_info[label2]['center']
                            
                            # Compute linear distance (translation only)
                            trans1, trans2 = center1[:3], center2[:3]
                            linear_distance = np.linalg.norm(trans1 - trans2)
                            
                            # Compute rotational distance (quaternion only)
                            quat1, quat2 = center1[3:], center2[3:]
                            try:
                                from scipy.spatial.transform import Rotation
                                # Normalize quaternions
                                quat1_norm = quat1 / np.linalg.norm(quat1)
                                quat2_norm = quat2 / np.linalg.norm(quat2)
                                
                                rot1 = Rotation.from_quat(quat1_norm)
                                rot2 = Rotation.from_quat(quat2_norm)
                                relative_rot = rot1.inv() * rot2
                                rotational_distance = relative_rot.magnitude()  # Angle in radians
                            except Exception as e:
                                logging.warning(f"Error computing rotational distance: {e}")
                                rotational_distance = float('nan')
                            
                            print(f"Clusters {label1} <-> {label2}:")
                            print(f"  Linear distance:     {linear_distance:.6f}")
                            print(f"  Rotational distance: {rotational_distance:.6f} rad ({np.degrees(rotational_distance):.2f}°)")
                            
                            # Also compute the combined SE3 distance using the existing utility
                            se3_distance = utils.calculate_se3_distance(center1, center2, 
                                                                      CFG.clustering_se3_trans_weight, 
                                                                      CFG.clustering_se3_rot_weight)
                            print(f"  Weighted SE3 distance: {se3_distance:.6f}")
                            print()
            
            for cluster_label in kept_cluster_labels_list: # Iterate over keys
                cluster_info = kept_clusters_info[cluster_label]
                cluster_points = cluster_info['points'] # Retrieve stored points
                # compute the SE(3) covariance matrix



            # Sort kept clusters by size (descending) for top_k selection AFTER plotting
            # Filter out any clusters where covariance calculation failed (if needed, though `continue` above handles it)
            # valid_kept_clusters = {k: v for k, v in kept_clusters_info.items() if 'inv_covariance_matrix' in v}
            valid_kept_clusters = kept_clusters_info
            sorted_valid_kept_clusters = sorted(valid_kept_clusters.items(), key=lambda item: item[1]['size'], reverse=True)

            # Create predicates for the top_k *valid* kept clusters
            top_k = len(sorted_valid_kept_clusters)
            logging.debug(f"Selecting top {top_k} valid kept clusters for {feat_name}:{type1.name}-{type2.name}.")

            for i, (cluster_label, cluster_info) in enumerate(sorted_valid_kept_clusters[:top_k]):
                # logging.debug(f"Creating predicate for kept cluster {cluster_label} (size {cluster_info['size']}, rank {i+1}/{top_k}).")
                # Pass inverse covariance and threshold instead of epsilon
                pred = self._create_predicate_from_relative_cluster( # TODO: THIS IS BROKEN NOW
                    type1, type2, feat_name, cluster_info['center'],
                    cluster_info['cluster_radius'],
                    diff_fn, cluster_label) # Use cluster_label for ID
                candidates[pred] = pred.arity 
                predicate_counter += 1

                # Now, optionally visualize clusters if in debug mode, passing the *updated* info with pred
                if CFG.clustering_debug and data_array.size > 0: # Check if there is data to plot
                    # The kept_clusters_info dict now contains cov matrix and threshold for plot
                    self._plot_cluster_results(data_array, labels, unique_labels, kept_clusters_info,
                                               type1.name, type2.name if type2 else None, feat_name, pred)

                    # Plot relative trajectories *after* cluster plot, if applicable
                    if feat_name == CFG.pose_feature_name and type2 is not None:
                        self._plot_relative_trajectories(dataset, type1.name, type2.name, pred)

        # Rename predicates for PDDL compatibility (reuse from grammar search)
        renamed_candidates = self._rename_predicates_to_remove_incompatible_chars(candidates)
        return renamed_candidates

    def _update_incontact_predicate_using_motion_analysis(self, dataset: Dataset, in_contact_pred: Predicate, gripper_type: Type) -> Tuple[Dict[int, List[Dict]], Dict[int, List[Object]], Object, Dict[int, List[Tuple[int, int]]]]:
        """Update incontact predicates using multi-phase motion analysis.
        This function analyzes motion patterns to identify different phases:
        1. Gripper-only motion (approaching/repositioning)
        2. Gripper+Object motion (manipulation)
        3. Sequential object interactions in long horizon demos
        """
        # Filter types so things other than gripper and are useful are kept!!!
        disallowed_type_names = {"wrist_type", "gripper_type", "left_finger_type", "right_finger_type", "base_type", "drawer_type"}

        # Dictionary to store motion data for each object in each trajectory
        motion_data = defaultdict(lambda: defaultdict(list))
        gripper_motion_data = defaultdict(list)  # Separate tracking for gripper
        
        # Data to return for _extract_relative_pose_data
        trajectory_motion_phases = {}  # trajectory_idx -> List[Dict] (motion phases)
        trajectory_all_objects = {}    # trajectory_idx -> List[Object] 
        contact_lost_periods = {}      # trajectory_idx -> List[Tuple[int, int]] (contact lost periods)

        gripper_obj = None
        for obj in dataset.trajectories[0].states[0].get_objects(gripper_type):
            gripper_obj = obj
            break
        assert gripper_obj is not None, "No gripper object found in the dataset"

        for i, traj in enumerate(dataset.trajectories):
            logging.debug(f"Processing trajectory {i+1}/{len(dataset.trajectories)} for multi-phase motion analysis")
            
            # Get all objects in the trajectory
            all_objects = set()
            for state in traj.states:
                all_objects.update(state.data.keys())
            
            # Filter and store objects for later use (excluding disallowed types)
            filtered_objects = [obj for obj in all_objects if obj.type.name not in disallowed_type_names]
            trajectory_all_objects[i] = filtered_objects
            
            # Calculate velocity for each object INCLUDING gripper
            for t in range(len(traj.states) - 1):
                state_t = traj.states[t]
                state_t1 = traj.states[t+1]
                
                for obj in all_objects:
                    if obj not in state_t.data or obj not in state_t1.data:
                        continue
                        
                    # Get translation and rotation data
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
            
                        if obj == gripper_obj:
                            gripper_motion_data[i].append((t, delta1, delta2))
                        elif obj.type.name not in disallowed_type_names:
                            motion_data[i][obj].append((t, delta1, delta2))
            
            # Clear in contact set for each state
            for state in dataset.trajectories[i].states:
                state.items_in_contact = set()
            
            # Multi-phase motion analysis and collect motion phases
            motion_phases = self._analyze_multi_phase_motion(i, traj, motion_data[i], gripper_motion_data[i], gripper_obj, dataset)
            trajectory_motion_phases[i] = motion_phases
            
            # Extract contact lost periods from motion phases
            contact_lost_periods[i] = self._extract_contact_lost_periods(motion_phases, len(traj.states))
        
        return trajectory_motion_phases, trajectory_all_objects, gripper_obj, contact_lost_periods

    def _analyze_multi_phase_motion(self, traj_idx: int, traj, object_motion_data: Dict, gripper_motion_data: List, gripper_obj, dataset: Dataset) -> List[Dict]:
        """Analyze trajectory for object motion phases and mark contacts accordingly.
        Focuses on object motion analysis while using combined object+gripper velocities for boundary detection.
        """
        os.makedirs("feature_data", exist_ok=True)
        
        if not gripper_motion_data:
            logging.warning(f"No gripper motion data for trajectory {traj_idx}")
            return
        
        if not object_motion_data:
            logging.warning(f"No object motion data for trajectory {traj_idx}")
            return
            
        # Extract gripper velocity profile for boundary detection
        gripper_velocities = [(vel, rot_vel) for _, vel, rot_vel in gripper_motion_data]
        gripper_lin_vel = np.array([vel for vel, _ in gripper_velocities])
        gripper_rot_vel = np.array([rot_vel for _, rot_vel in gripper_velocities])
        
        # Smooth gripper velocities
        gripper_lin_vel_smooth = uniform_filter1d(gripper_lin_vel, size=4)
        gripper_rot_vel_smooth = uniform_filter1d(gripper_rot_vel, size=4)
        
        # Add small noise for robustness
        gripper_lin_vel_smooth += np.random.normal(0, 0.0008, gripper_lin_vel_smooth.shape)
        gripper_rot_vel_smooth += np.random.normal(0, 0.0008, gripper_rot_vel_smooth.shape)
        
        motion_threshold = CFG.motion_analysis_lin_vel_rot_vel_threshold if hasattr(CFG, 'motion_analysis_lin_vel_rot_vel_threshold') else 0.01
        
        # Find motion phases based on object motion but using combined velocities for boundary detection
        object_motion_phases = self._find_object_motion_phases(object_motion_data, gripper_lin_vel_smooth, motion_threshold)
        
        logging.info(f"Trajectory {traj_idx}: Found {len(object_motion_phases)} object motion phases")
        
        # Process object motion phases and mark contacts
        for phase_idx, phase in enumerate(object_motion_phases):
            start_frame = phase['start_frame']
            end_frame = phase['end_frame']
            moving_objects = phase['moving_objects']
            
            logging.debug(f"  Phase {phase_idx}: Object motion (frames {start_frame}-{end_frame})")
            logging.debug(f"    -> Objects in motion: {[obj.name for obj in moving_objects]}")
            

            contact_start = max(0, start_frame ) 
            contact_end = min(len(traj.states) - 1, end_frame )
            
            for t in range(contact_start, contact_end + 1):
                if t < len(traj.states):
                    # Mark contact with ALL moving objects during this phase
                    contacts = {(gripper_obj, obj) for obj in moving_objects}
                    if dataset.trajectories[traj_idx].states[t].items_in_contact is None:
                        dataset.trajectories[traj_idx].states[t].items_in_contact = set()
                    dataset.trajectories[traj_idx].states[t].items_in_contact.update(contacts)
        
        # Visualization with object motion phases
        self._visualize_object_motion_phases(traj_idx, gripper_lin_vel_smooth, object_motion_phases, object_motion_data, gripper_motion_data)
        
        return object_motion_phases

    def _extract_contact_lost_periods(self, motion_phases: List[Dict], traj_length: int) -> List[Tuple[int, int]]:
        """Extract periods between motion phases where contact is lost."""
        contact_lost_periods = []
        
        if not motion_phases:
            return contact_lost_periods
            
        # Add period from start to first motion phase
        if motion_phases[0]['start_frame'] > 0:
            contact_lost_periods.append((0, motion_phases[0]['start_frame'] - 1))
        
        # Add periods between motion phases
        for i in range(len(motion_phases) - 1):
            end_current = motion_phases[i]['end_frame']
            start_next = motion_phases[i + 1]['start_frame']
            if start_next > end_current + 1:
                contact_lost_periods.append((end_current + 1, start_next - 1))
        
        # Add period from last motion phase to end
        if motion_phases[-1]['end_frame'] < traj_length - 1:
            contact_lost_periods.append((motion_phases[-1]['end_frame'] + 1, traj_length - 1))
            
        return contact_lost_periods

    def _find_contiguous_segments(self, motion_mask: np.ndarray, min_length: int = 10) -> List[Tuple[int, int]]:
        """Find contiguous segments where motion_mask is True."""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_moving in enumerate(motion_mask):
            if is_moving and not in_segment:
                # Start of new segment
                start_idx = i
                in_segment = True
            elif not is_moving and in_segment:
                # End of current segment
                if i - start_idx >= min_length:
                    segments.append((start_idx, i - 1))
                in_segment = False
        
        # Handle case where segment continues to end
        if in_segment and len(motion_mask) - start_idx >= min_length:
            segments.append((start_idx, len(motion_mask) - 1))
            
        return segments

    def _fine_tune_motion_boundaries(self, initial_segments: List[Tuple[int, int]], combined_velocities: np.ndarray, search_window: int = 30) -> List[Tuple[int, int]]:
        """Fine-tune motion boundaries by finding minimum combined velocity within search window."""
        if not initial_segments:
            return []
        
        refined_segments = []
        
        for start_frame, end_frame in initial_segments:
            # Fine-tune start boundary
            refined_start = self._find_velocity_minimum_in_window(
                combined_velocities, start_frame, search_window, direction='backward'
            )
            
            # Fine-tune end boundary
            refined_end = self._find_velocity_minimum_in_window(
                combined_velocities, end_frame, search_window, direction='forward'
            )
            
            # Ensure refined boundaries are valid
            refined_start = max(0, min(refined_start, len(combined_velocities) - 1))
            refined_end = max(0, min(refined_end, len(combined_velocities) - 1))
            
            # Only keep segment if it's still meaningful after refinement
            if refined_end > refined_start:
                refined_segments.append((refined_start, refined_end))
                # logging.debug(f"    Refined boundary: {start_frame}-{end_frame} -> {refined_start}-{refined_end}")
        
        return refined_segments

    def _find_velocity_minimum_in_window(self, velocities: np.ndarray, center_frame: int, window_size: int, direction: str) -> int:
        """Find the frame with minimum velocity within a window around center_frame."""
        if direction == 'backward':
            # Search backward from center_frame
            start_idx = max(0, center_frame - window_size)
            end_idx = min(len(velocities), center_frame + 1)
        else:  # direction == 'forward'
            # Search forward from center_frame
            start_idx = max(0, center_frame)
            end_idx = min(len(velocities), center_frame + window_size + 1)
        
        if start_idx >= end_idx:
            return center_frame
        
        # Find the index with minimum velocity in the window
        window_velocities = velocities[start_idx:end_idx]
        min_idx_relative = np.argmin(window_velocities)
        min_idx_absolute = start_idx + min_idx_relative
        
        return min_idx_absolute

    def _combine_overlapping_segments(self, motion_phases: List[Dict], max_gap: int = 10) -> List[Dict]:
        """Combine segments of the same object that have overlap or small gaps."""
        if not motion_phases:
            return []
        
        # Group phases by moving objects (since each phase has only one moving object)
        object_phases = {}
        for phase in motion_phases:
            # Get the object name (there should be only one moving object per phase)
            if phase['moving_objects']:
                obj_name = phase['moving_objects'][0].name
                if obj_name not in object_phases:
                    object_phases[obj_name] = []
                object_phases[obj_name].append(phase)
        
        # Combine overlapping/close segments for each object
        combined_phases = []
        for obj_name, phases in object_phases.items():
            if not phases:
                continue
                
            # Sort phases by start frame
            phases.sort(key=lambda p: p['start_frame'])
            
            if len(phases) == 1:
                # Only one phase for this object, no combining needed
                combined_phases.append(phases[0])
                continue
            
            combined_obj_phases = []
            current_phase = phases[0].copy()
            
            for i in range(1, len(phases)):
                next_phase = phases[i]
                
                # Check if phases overlap or have small gap
                gap = next_phase['start_frame'] - current_phase['end_frame']
                
                if gap <= max_gap:  # Overlapping or small gap (including negative gaps for overlap)
                    # Combine phases by extending the current phase
                    current_phase['end_frame'] = max(current_phase['end_frame'], next_phase['end_frame'])
                    logging.debug(f"    Combined {obj_name} segments: gap={gap}, new range={current_phase['start_frame']}-{current_phase['end_frame']}")
                else:
                    # Gap too large, save current phase and start new one
                    combined_obj_phases.append(current_phase)
                    current_phase = next_phase.copy()
            
            # Add the last phase
            combined_obj_phases.append(current_phase)
            combined_phases.extend(combined_obj_phases)
        
        # Sort final phases by start frame
        combined_phases.sort(key=lambda p: p['start_frame'])
        
        return combined_phases

    def _find_object_motion_phases(self, object_motion_data: Dict, gripper_lin_vel_smooth: np.ndarray, motion_threshold: float) -> List[Dict]:
        """Find motion phases based on object motion, using object velocities primarily and fine-tuning boundaries with combined velocities."""
        if not object_motion_data:
            return []
        
        # Create object-only velocities signal for primary boundary detection
        object_only_velocities = np.zeros_like(gripper_lin_vel_smooth)
        combined_velocities = np.copy(gripper_lin_vel_smooth)  # For fine-tuning
        
        # Add object velocities to both signals
        for obj, motion_list in object_motion_data.items():
            # Create object velocity array aligned with gripper data
            obj_velocities = np.zeros_like(gripper_lin_vel_smooth)
            for t, lin_vel, rot_vel in motion_list:
                if 0 <= t < len(obj_velocities):
                    obj_velocities[t] = lin_vel
            
            # Add to object-only signal for primary detection
            object_only_velocities += obj_velocities
            # Add to combined signal for fine-tuning
            combined_velocities += obj_velocities
        
        # Find initial motion boundaries using object velocities only
        object_in_motion = object_only_velocities > motion_threshold
        initial_motion_segments = self._find_contiguous_segments(object_in_motion, min_length=10)
        # logging.debug(f"    Found {len(initial_motion_segments)} initial object motion segments using object velocities only")
        
        # Fine-tune boundaries using combined velocities within 30 steps
        raw_motion_segments = self._fine_tune_motion_boundaries(
            initial_motion_segments, combined_velocities, search_window=30
        )
        # logging.debug(f"    After fine-tuning with combined velocities: {len(raw_motion_segments)} refined segments")
        
        # For each detected segment, determine which objects are actually moving
        object_motion_phases = []
        for start_frame, end_frame in raw_motion_segments:
            moving_objects = self._find_moving_objects_in_segment(
                object_motion_data, start_frame, end_frame, motion_threshold
            )
            
            # Only create a phase if there are actually moving objects
            if moving_objects:
                object_motion_phases.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'moving_objects': moving_objects
                })
                # logging.debug(f"    Found object motion phase: frames {start_frame}-{end_frame}, objects: {[obj.name for obj in moving_objects]}")
        
        # Combine segments of the same object with overlap or small gaps (<10 frames)
        combined_phases = self._combine_overlapping_segments(object_motion_phases, max_gap=10)
        # logging.debug(f"    After combining overlapping segments: {len(combined_phases)} final phases")
        
        return combined_phases

    def _find_moving_objects_in_segment(self, object_motion_data: Dict, start_frame: int, end_frame: int, threshold: float) -> List:
        """Find objects that are moving significantly during the given time segment.
        Returns list with single object (the one with highest average velocity) if multiple objects are moving.
        """
        moving_objects_with_velocities = []
        
        for obj, motion_list in object_motion_data.items():
            # Calculate average velocity during this segment
            segment_velocities = []
            for t, lin_vel, rot_vel in motion_list:
                if start_frame <= t <= end_frame:
                    segment_velocities.append(lin_vel)
            
            if segment_velocities:
                avg_velocity = np.mean(segment_velocities)
                max_velocity = np.max(segment_velocities)
                
                # Object is considered moving if average velocity exceeds threshold
                # OR if peak velocity is significantly high
                if avg_velocity > threshold or max_velocity > threshold * 2:
                    moving_objects_with_velocities.append((obj, avg_velocity, max_velocity))
                    logging.debug(f"      Object {obj.name}: avg_vel={avg_velocity:.4f}, max_vel={max_velocity:.4f}")
        
        # If multiple objects are moving, pick the one with highest average velocity
        if len(moving_objects_with_velocities) == 0:
            return []
        elif len(moving_objects_with_velocities) == 1:
            return [moving_objects_with_velocities[0][0]]
        else:
            # Sort by average velocity (descending) and pick the top one
            moving_objects_with_velocities.sort(key=lambda x: x[1], reverse=True)
            primary_obj = moving_objects_with_velocities[0]
            logging.debug(f"      Multiple objects moving, selected {primary_obj[0].name} with highest avg_vel={primary_obj[1]:.4f}")
            return [primary_obj[0]]



    def _visualize_alternating_phases(self, traj_idx: int, gripper_velocity: np.ndarray, alternating_phases: List[Dict], object_motion_data: Dict, gripper_motion_data: List):
        """Create visualization showing gripper motion and alternating phases."""
        plt.figure(figsize=(15, 10))
        
        # Extract gripper rotational velocity from gripper_motion_data
        gripper_angular_velocity = np.zeros(len(gripper_velocity))
        for t, linear_vel, angular_vel in gripper_motion_data:
            if t < len(gripper_angular_velocity):
                gripper_angular_velocity[t] = angular_vel
        
        # Plot gripper linear velocity
        plt.subplot(2, 2, 1)
        plt.plot(gripper_velocity, label='Gripper Linear Velocity', color='blue', linewidth=2)
        
        # Highlight alternating phases with different colors
        colors = ['lightcoral', 'lightgreen']  # Red for gripper-only, Green for manipulation
        legend_labels_added = {'gripper_only': False, 'manipulation': False}
        
        for i, phase in enumerate(alternating_phases):
            start, end = phase['start_frame'], phase['end_frame']
            phase_type = phase['type']
            color = colors[0] if phase_type == 'gripper_only' else colors[1]
            alpha = 0.3
            
            # Create descriptive label for legend (only once per type)
            legend_label = None
            if not legend_labels_added[phase_type]:
                if phase_type == 'gripper_only':
                    legend_label = 'Gripper Motion'
                else:  # manipulation
                    moving_objects = phase.get('moving_objects', [])
                    if moving_objects:
                        # Since we now select only the primary object, this should always be length 1
                        obj_names = [obj.name for obj in moving_objects]
                        legend_label = f'Object Motion ({obj_names[0]})'
                    else:
                        legend_label = 'Object Motion'
                legend_labels_added[phase_type] = True
            
            plt.axvspan(start, end, alpha=alpha, color=color, label=legend_label)
        
        # Add vertical lines at refined boundaries
        for i in range(len(alternating_phases) - 1):
            boundary_frame = alternating_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Gripper Linear Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot gripper angular velocity
        plt.subplot(2, 2, 2)
        plt.plot(gripper_angular_velocity, label='Gripper Angular Velocity', color='red', linewidth=2)
        
        # Highlight alternating phases on angular velocity plot
        legend_labels_added_ang = {'gripper_only': False, 'manipulation': False}
        for i, phase in enumerate(alternating_phases):
            start, end = phase['start_frame'], phase['end_frame']
            phase_type = phase['type']
            color = colors[0] if phase_type == 'gripper_only' else colors[1]
            alpha = 0.3
            
            # Create descriptive label for legend (only once per type)
            legend_label = None
            if not legend_labels_added_ang[phase_type]:
                if phase_type == 'gripper_only':
                    legend_label = 'Gripper Motion'
                else:  # manipulation
                    moving_objects = phase.get('moving_objects', [])
                    if moving_objects:
                        obj_names = [obj.name for obj in moving_objects]
                        legend_label = f'Object Motion ({obj_names[0]})'
                    else:
                        legend_label = 'Object Motion'
                legend_labels_added_ang[phase_type] = True
            
            plt.axvspan(start, end, alpha=alpha, color=color, label=legend_label)
        
        # Add vertical lines at refined boundaries
        for i in range(len(alternating_phases) - 1):
            boundary_frame = alternating_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title(f'Gripper Angular Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot object linear velocities
        plt.subplot(2, 2, 3)
        colors_obj = plt.cm.tab10(np.linspace(0, 1, len(object_motion_data)))
        
        for (obj, motion_list), color in zip(object_motion_data.items(), colors_obj):
            if motion_list:  # Only plot if object has motion data
                times = [t for t, _, _ in motion_list]
                velocities = [vel for _, vel, _ in motion_list]
                plt.plot(times, velocities, label=f'{obj.name}', color=color, alpha=0.7)
        
        # Highlight alternating phases on object plot
        for i, phase in enumerate(alternating_phases):
            start, end = phase['start_frame'], phase['end_frame']
            phase_type = phase['type']
            color = colors[0] if phase_type == 'gripper_only' else colors[1]
            plt.axvspan(start, end, alpha=0.2, color=color)
            
            # Add text annotation for phase type with object names
            mid_point = (start + end) / 2
            max_vel = max([max([vel for _, vel, _ in motion_list]) for motion_list in object_motion_data.values() if motion_list] + [0])
            
            # Create informative label
            if phase_type == 'gripper_only':
                label = f'P{i}\ngripper'
            else:  # manipulation phase
                moving_objects = phase.get('moving_objects', [])
                if moving_objects:
                    # Since we now select only the primary object, this should always be length 1
                    obj_names = [obj.name for obj in moving_objects]
                    label = f'P{i}\n{obj_names[0]}'
                else:
                    label = f'P{i}\nmanip'
            
            plt.text(mid_point, max_vel * 0.8, label, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add vertical lines at refined boundaries
        for i in range(len(alternating_phases) - 1):
            boundary_frame = alternating_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Object Linear Velocity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot object angular velocities
        plt.subplot(2, 2, 4)
        
        for (obj, motion_list), color in zip(object_motion_data.items(), colors_obj):
            if motion_list:  # Only plot if object has motion data
                times = [t for t, _, _ in motion_list]
                angular_velocities = [ang_vel for _, _, ang_vel in motion_list]
                plt.plot(times, angular_velocities, label=f'{obj.name}', color=color, alpha=0.7)
        
        # Highlight alternating phases on object angular plot
        for i, phase in enumerate(alternating_phases):
            start, end = phase['start_frame'], phase['end_frame']
            phase_type = phase['type']
            color = colors[0] if phase_type == 'gripper_only' else colors[1]
            plt.axvspan(start, end, alpha=0.2, color=color)
            
            # Add text annotation for phase type with object names
            mid_point = (start + end) / 2
            max_ang_vel = max([max([ang_vel for _, _, ang_vel in motion_list]) for motion_list in object_motion_data.values() if motion_list] + [0])
            
            # Create informative label
            if phase_type == 'gripper_only':
                label = f'P{i}\ngripper'
            else:  # manipulation phase
                moving_objects = phase.get('moving_objects', [])
                if moving_objects:
                    obj_names = [obj.name for obj in moving_objects]
                    label = f'P{i}\n{obj_names[0]}'
                else:
                    label = f'P{i}\nmanip'
            
            plt.text(mid_point, max_ang_vel * 0.8, label, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add vertical lines at refined boundaries
        for i in range(len(alternating_phases) - 1):
            boundary_frame = alternating_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Object Angular Velocity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"feature_data/alternating_phase_motion_analysis_traj{traj_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved alternating phase motion visualization for trajectory {traj_idx}")

    def _visualize_object_motion_phases(self, traj_idx: int, gripper_velocity: np.ndarray, object_motion_phases: List[Dict], object_motion_data: Dict, gripper_motion_data: List):
        """Create visualization showing object motion phases with combined velocity boundary detection."""
        plt.figure(figsize=(15, 10))
        
        # Extract gripper rotational velocity from gripper_motion_data
        gripper_angular_velocity = np.zeros(len(gripper_velocity))
        for t, linear_vel, angular_vel in gripper_motion_data:
            if t < len(gripper_angular_velocity):
                gripper_angular_velocity[t] = angular_vel
        
        # Plot gripper linear velocity
        plt.subplot(2, 2, 1)
        plt.plot(gripper_velocity, label='Gripper Linear Velocity', color='blue', linewidth=2)
        
        # Highlight object motion phases
        colors = ['lightgreen']  # Single color for object motion phases
        
        for i, phase in enumerate(object_motion_phases):
            start, end = phase['start_frame'], phase['end_frame']
            moving_objects = phase.get('moving_objects', [])
            color = colors[0]
            alpha = 0.3
            
            # Create descriptive label for legend (only once)
            legend_label = None
            if i == 0:  # Only add legend for first phase
                if moving_objects:
                    obj_names = [obj.name for obj in moving_objects]
                    legend_label = f'Object Motion ({obj_names[0]})'
                else:
                    legend_label = 'Object Motion'
            
            plt.axvspan(start, end, alpha=alpha, color=color, label=legend_label)
        
        # Add vertical lines at phase boundaries
        for i in range(len(object_motion_phases) - 1):
            boundary_frame = object_motion_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Gripper Linear Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot gripper angular velocity
        plt.subplot(2, 2, 2)
        plt.plot(gripper_angular_velocity, label='Gripper Angular Velocity', color='red', linewidth=2)
        
        # Highlight object motion phases on angular velocity plot
        for i, phase in enumerate(object_motion_phases):
            start, end = phase['start_frame'], phase['end_frame']
            moving_objects = phase.get('moving_objects', [])
            color = colors[0]
            alpha = 0.3
            
            # Create descriptive label for legend (only once)
            legend_label = None
            if i == 0:  # Only add legend for first phase
                if moving_objects:
                    obj_names = [obj.name for obj in moving_objects]
                    legend_label = f'Object Motion ({obj_names[0]})'
                else:
                    legend_label = 'Object Motion'
            
            plt.axvspan(start, end, alpha=alpha, color=color, label=legend_label)
        
        # Add vertical lines at phase boundaries
        for i in range(len(object_motion_phases) - 1):
            boundary_frame = object_motion_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title(f'Gripper Angular Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot object linear velocities
        plt.subplot(2, 2, 3)
        colors_obj = plt.cm.tab10(np.linspace(0, 1, len(object_motion_data)))
        
        for (obj, motion_list), color in zip(object_motion_data.items(), colors_obj):
            if motion_list:  # Only plot if object has motion data
                times = [t for t, _, _ in motion_list]
                velocities = [vel for _, vel, _ in motion_list]
                plt.plot(times, velocities, label=f'{obj.name}', color=color, alpha=0.7)
        
        # Highlight object motion phases on object plot
        for i, phase in enumerate(object_motion_phases):
            start, end = phase['start_frame'], phase['end_frame']
            color = colors[0]
            plt.axvspan(start, end, alpha=0.2, color=color)
            
            # Add text annotation for phase with object names
            mid_point = (start + end) / 2
            max_vel = max([max([vel for _, vel, _ in motion_list]) for motion_list in object_motion_data.values() if motion_list] + [0])
            
            # Create informative label
            moving_objects = phase.get('moving_objects', [])
            if moving_objects:
                obj_names = [obj.name for obj in moving_objects]
                label = f'P{i}\n{obj_names[0]}'
            else:
                label = f'P{i}\nobj'
            
            plt.text(mid_point, max_vel * 0.8, label, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add vertical lines at phase boundaries
        for i in range(len(object_motion_phases) - 1):
            boundary_frame = object_motion_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Object Linear Velocity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot object angular velocities
        plt.subplot(2, 2, 4)
        
        for (obj, motion_list), color in zip(object_motion_data.items(), colors_obj):
            if motion_list:  # Only plot if object has motion data
                times = [t for t, _, _ in motion_list]
                angular_velocities = [ang_vel for _, _, ang_vel in motion_list]
                plt.plot(times, angular_velocities, label=f'{obj.name}', color=color, alpha=0.7)
        
        # Highlight object motion phases on object angular plot
        for i, phase in enumerate(object_motion_phases):
            start, end = phase['start_frame'], phase['end_frame']
            color = colors[0]
            plt.axvspan(start, end, alpha=0.2, color=color)
            
            # Add text annotation for phase with object names
            mid_point = (start + end) / 2
            max_ang_vel = max([max([ang_vel for _, _, ang_vel in motion_list]) for motion_list in object_motion_data.values() if motion_list] + [0])
            
            # Create informative label
            moving_objects = phase.get('moving_objects', [])
            if moving_objects:
                obj_names = [obj.name for obj in moving_objects]
                label = f'P{i}\n{obj_names[0]}'
            else:
                label = f'P{i}\nobj'
            
            plt.text(mid_point, max_ang_vel * 0.8, label, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add vertical lines at phase boundaries
        for i in range(len(object_motion_phases) - 1):
            boundary_frame = object_motion_phases[i]['end_frame']
            plt.axvline(x=boundary_frame, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Object Angular Velocity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"feature_data/object_motion_analysis_traj{traj_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved object motion phase visualization for trajectory {traj_idx}")

    def _visualize_multi_phase_motion(self, traj_idx: int, gripper_velocity: np.ndarray, motion_segments: List[Tuple[int, int]], object_motion_data: Dict):
        """Create visualization showing gripper motion and detected phases."""
        plt.figure(figsize=(12, 8))
        
        # Plot gripper velocity
        plt.subplot(2, 1, 1)
        plt.plot(gripper_velocity, label='Gripper Linear Velocity', color='blue', linewidth=2)
        
        # Highlight motion segments
        for i, (start, end) in enumerate(motion_segments):
            plt.axvspan(start, end, alpha=0.3, color=f'C{i}', label=f'Motion Segment {i}')
        
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title(f'Trajectory {traj_idx}: Gripper Motion Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot object velocities
        plt.subplot(2, 1, 2)
        colors = plt.cm.tab10(np.linspace(0, 1, len(object_motion_data)))
        
        for (obj, motion_list), color in zip(object_motion_data.items(), colors):
            if motion_list:  # Only plot if object has motion data
                times = [t for t, _, _ in motion_list]
                velocities = [vel for _, vel, _ in motion_list]
                plt.plot(times, velocities, label=f'{obj.name}', color=color, alpha=0.7)
        
        # Highlight motion segments on object plot too
        for i, (start, end) in enumerate(motion_segments):
            plt.axvspan(start, end, alpha=0.2, color=f'C{i}')
            
        plt.xlabel('Time Step')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Object Motion During Gripper Motion Phases')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"feature_data/multi_phase_motion_analysis_traj{traj_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved multi-phase motion visualization for trajectory {traj_idx}")

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
        blacklisted_type_names = {"left_finger_type", "right_finger_type", "base_type", "wrist_type","counter_type","surface_type","door_type"}#"thing_type"}
        for type_obj in types:
            if type_obj.name not in blacklisted_type_names:
                filtered_types.add(type_obj)
                logging.info(f"Keeping type for relative features: {type_obj.name}")
            else:
                logging.debug(f"Filtering out blacklisted type: {type_obj.name}")

        # gripper_type = "gripper"
        # gripper_type_obj = None
        # for type_obj in types:
        #     if gripper_type in type_obj.name:
        #         gripper_type_obj = type_obj
        #         logging.info(f"Keeping gripper type: {type_obj.name}")
        #         break

        # if gripper_type_obj is None:
        #     raise ValueError("No gripper type found in the dataset. Skipping relative features.")

        type_pairs = list(utils.combinations_no_self_pairs(sorted(list(filtered_types)), 2))
        # Create type pairs that include combinations with gripper
        # for type_obj in filtered_types:
        #     # Add both (gripper, obj) and (obj, gripper) pairs
        #     type_pairs.append((type_obj, gripper_type_obj))
        #     logging.info(f"Adding gripper pair: ({gripper_type_obj.name}, {type_obj.name}) and ({type_obj.name}, {gripper_type_obj.name})")

        logging.info(f"Total type pairs for relative features: {len(type_pairs)}")


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
                            rel_pose_t = utils.calculate_relative_pose_from_state(state_t, o1, o2, CFG.trans_feat_name, CFG.quat_feat_name)
                            rel_pose_t1 = utils.calculate_relative_pose_from_state(state_t1, o1, o2, CFG.trans_feat_name, CFG.quat_feat_name)

                            if rel_pose_t is not None and rel_pose_t1 is not None:
                                # Calculate change in relative pose (using SE(3) distance concept)
                                # We need a distance function here, let's define a simple one for constancy check
                                pose_diff_norm = utils.calculate_se3_distance(rel_pose_t, rel_pose_t1, 
                                                                                CFG.clustering_se3_trans_weight, 
                                                                                CFG.clustering_se3_rot_weight)

                                # Add the pose at time t to the dataset
                                feature_key = (type1, type2, CFG.pose_feature_name)
                                feature_data[feature_key].append(rel_pose_t)
                                feature_changes[feature_key].append(pose_diff_norm)

        # Filter based on constancy (e.g., keep points below 30th percentile of change)
        final_feature_data = defaultdict(list)
        for feature_key, data_points in feature_data.items():
            changes = np.array(feature_changes[feature_key])
            if len(changes) > 1: # Need at least 2 points to compute percentile
                # get moving average of changes first
                changes_ma = np.convolve(changes, np.ones(CFG.clustering_moving_average_window) / CFG.clustering_moving_average_window, mode='valid')
                percentile_threshold = np.percentile(changes_ma, CFG.clustering_feature_constancy_percentile) # Default 10th percentile
                fixed_threshold = CFG.clustering_constancy_threshold
                # Use the higher threshold (more permissive, keeps more data)
                constancy_threshold = max(percentile_threshold, fixed_threshold)
                logging.debug(f"Constancy threshold for {feature_key}: {constancy_threshold:.4f} (max of {CFG.clustering_feature_constancy_percentile}th percentile: {percentile_threshold:.4f} and fixed: {fixed_threshold:.4f})")
                mask = changes <= constancy_threshold
                filtered_points = [pt for pt, keep in zip(data_points, mask) if keep]
                
                # Additional layer: limit to 500 points maximum
                if len(filtered_points) > 500:
                    # Randomly sample 500 points to maintain diversity
                    indices = np.random.choice(len(filtered_points), 500, replace=False)
                    filtered_points = [filtered_points[i] for i in sorted(indices)]
                    # logging.debug(f"Further reduced from {sum(mask)} to 500 points for {feature_key} via random sampling.")
                
                final_feature_data[feature_key] = filtered_points
                # logging.debug(f"Final count: {len(filtered_points)} points for {feature_key} after all filtering.")
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
        # If there are more than 500 data points, randomly sample to reduce to 500
        if len(data_array) > 500:
            logging.debug(f"Reducing dataset from {len(data_array)} to 500 points via random sampling.")
            indices = np.random.choice(len(data_array), 500, replace=False)
            data_array = data_array[indices]
            # If feature_data is a list, also update it for consistency
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
            effective_epsilon = initial_epsilon
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
                                                linkage='average', # Check compatibility with custom metric
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
            fname_prefix = f"{CFG.robo_kitchen_task}_{type2_name}_in_{type1_name}_frame"
        else:
            cluster_type_str = f"Absolute Cluster: {type1_name}"
            fname_prefix = f"{CFG.robo_kitchen_task}_{type1_name}"

        num_total_clusters = len(unique_labels - {-1})
        num_kept_clusters = len(kept_clusters_info)

        fig = plt.figure(figsize=(15, 12))
        
        # Create shortened title with just task name, type1 name, type2 name, cluster IDs
        cluster_ids = sorted(list(kept_clusters_info.keys()))
        cluster_ids_str = ",".join(map(str, cluster_ids))
        
        if type2_name:
            title = f"{CFG.robo_kitchen_task}, {type1_name}, {type2_name}, {cluster_ids_str}"
        else:
            title = f"{CFG.robo_kitchen_task}, {type1_name}, {cluster_ids_str}"
        
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
                                                cluster_id: int,
                                                name_prefix: str = ""
                                                ) -> Predicate: 
                                                
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

        name = f"{name_prefix}{str(classifier)}"
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
        trajectory_motion_phases, trajectory_all_objects, gripper_obj, contact_lost_periods = None, None, None, None
        if CFG.predicate_candidates_method == "motion_analysis_contact":
            trajectory_motion_phases, trajectory_all_objects, gripper_obj, contact_lost_periods = self._update_incontact_predicate_using_motion_analysis(dataset, in_contact_pred, gripper_type)
        learnt_goal_predicates = self.load_learnt_goals()
        predicates_to_monitor, ground_atom_dataset = self._create_gnd_atom_datasets(dataset, in_contact_pred, in_origin_pred, learnt_goal_predicates)
        all_objs_types, robot_base_obj_type = self._find_common_objects_types(ground_atom_dataset)
        relative_pose_gripper_obj_dataset_dict, traj_all_objs_all, contact_period_obj_obj_rel_trajs, contact_lost_period_rel_pose, ground_atom_dataset = self._extract_relative_pose_data(ground_atom_dataset, all_objs_types, gripper_type, in_contact_pred, in_origin_pred, trajectory_motion_phases, trajectory_all_objects, gripper_obj, contact_lost_periods)            
        
        # Find the most common object type that comes in contact with the gripper
        # if trajectory_all_objects is not None:
            # obj_type_contact_with_gripper = self._find_object_in_contact_with_gripper(trajectory_all_objects)
        # else:
        #     # Fallback: use the first moving object type from contact_period_rel_trajs
        #     if contact_period_obj_obj_rel_trajs:
        #         obj_type_contact_with_gripper = next(iter(contact_period_obj_obj_rel_trajs.keys()))
        #     else:
        #         # Final fallback: use the first non-gripper object type
        #         obj_type_contact_with_gripper = next((obj_type for obj_type in all_objs_types if 'gripper' not in obj_type.name.lower()), all_objs_types[0])
        
        # logging.info(f"Object type in contact with gripper: {obj_type_contact_with_gripper.name}")
        
        # best_reference_per_moving_obj, _ = self._select_reference_object(contact_period_obj_obj_rel_trajs)
        # obj_type_of_reference_best = next(iter(best_reference_per_moving_obj.values()))[0]
        moving_type = next(iter(contact_period_obj_obj_rel_trajs.keys()))
        best_reference_per_moving_obj = {}
        if CFG.use_gt_ref_obj_type and CFG.robo_kitchen_task in CFG.gt_ref_obj_type: # mocap tasks are not using gt ref obj type
            obj_type_of_reference_best_text = CFG.gt_ref_obj_type[CFG.robo_kitchen_task]
            for obj_type in all_objs_types:
                if obj_type.name == obj_type_of_reference_best_text:
                    best_reference_per_moving_obj[moving_type] = obj_type
                    break
        assert best_reference_per_moving_obj[moving_type] is not None, f"Reference object type not found in all_objs_types: {all_objs_types}"
        # logging.error(f"Using ground truth reference object type: {obj_type_of_reference_best.name}")
        # assert obj_type_of_reference_best is not None, f"Reference object type not found in all_objs_types: {all_objs_types}"
        # assert obj_type_of_reference_best.name in gt_ref_obj_type, f"GT reference object type not matching correct solution, gt_ref_obj_type: {gt_ref_obj_type}, obj_type_of_reference_best: {obj_type_of_reference_best.name}"
        
        # self._visualize_contact_period_trajectories(contact_period_rel_trajs, list_of_reconstruction_errors)
        ground_atom_dataset = self._update_atom_sequences_with_goal_predicates(ground_atom_dataset, traj_all_objs_all, best_reference_per_moving_obj)
        relative_pose_all_dict = self._update_rel_pose_dict_with_obj_obj(relative_pose_gripper_obj_dataset_dict, contact_lost_period_rel_pose, best_reference_per_moving_obj)
        
        renamed_cluster_candidates = self._add_goal_states_to_relative_pose_and_cluster(dataset, relative_pose_all_dict)
        
        
        return self._postprocess_cluster_predicates(env, dataset, ground_atom_dataset, predicates_to_monitor, renamed_cluster_candidates, best_reference_per_moving_obj, learnt_goal_predicates)
        
    def _add_base_ref_obj_precondition(self, ground_atom_dataset: List[GroundAtomTrajectory], relative_pose_dataset_dict: Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]],  obj_type_of_reference_best: Type, obj_type_contact_with_gripper: Type, robot_base_obj_type: Type, traj_all_objs_all: List[List[Object]]):
        # RelPosPred
        RelPoseBaseRefObjPred = Predicate("RobotBaseRelPosPred-" + CFG.robo_kitchen_task, [obj_type_of_reference_best, robot_base_obj_type], lambda state, objects: True)
        RelPoseBaseContactObjPred = Predicate("RobotBaseRelPosPredObj-" + CFG.robo_kitchen_task, [obj_type_contact_with_gripper, robot_base_obj_type], lambda state, objects: True)
        # this having a object type since it will need to be used when other tasks load and use the same predicates
        for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
            if not ll_traj.states: continue # Skip empty trajectories
            obj_ref = [o for o in traj_all_objs_all[i] if o.type == obj_type_of_reference_best][0]
            obj_base = [o for o in ll_traj.states[0].data.keys() if o.type == robot_base_obj_type][0]
            obj_contact = [o for o in ll_traj.states[0].data.keys() if o.type == obj_type_contact_with_gripper][0]
            assert obj_ref is not None and obj_base is not None, "Reference or base object not found"
            skip_var =max(int(len(atom_seq) / 50),1)
            logging.debug(f"Processing trajectory {i+1}/{len(ground_atom_dataset)} with {len(atom_seq)} atoms, skipping every {skip_var} atoms.")
            for t in range(len(atom_seq)):# Add the predicate to every ground atom at this timestep
                ground_atom_ref = GroundAtom(RelPoseBaseRefObjPred, [obj_ref, obj_base])
                ground_atom_dataset[i][1][t].add(ground_atom_ref)
                # ground_atom_contact = GroundAtom(RelPoseBaseContactObjPred, [obj_contact, obj_base])
                # ground_atom_dataset[i][1][t].add(ground_atom_contact)

                # Check if there's an InContact predicate with obj_contact
                # has_contact_with_obj = False
                # for atom in atom_seq[t]:
                #     if atom.predicate.name == "InContact" and obj_contact in atom.objects:
                #         has_contact_with_obj = True
                #         break
                
                # Add the reference object precondition
                # if has_contact_with_obj:
                #     ground_atom_dataset[i][1][t].add(ground_atom_ref)
                # else:
                #     ground_atom_dataset[i][1][t].add(ground_atom_contact)
            for t in range(skip_var, len(atom_seq), skip_var): # Start from 1 to compare with t-1, skip every 4, for efficiency
                state_t = ll_traj.states[t]
                
                rel_pose_at_contact_obj2_in_obj1_frame_ref = utils.calculate_relative_pose_from_state(
                    state_t, obj_ref, obj_base,
                    CFG.trans_feat_name, CFG.quat_feat_name
                )
                rel_pose_at_contact_obj2_in_obj1_frame_contact = utils.calculate_relative_pose_from_state(
                    state_t, obj_contact, obj_base,
                    CFG.trans_feat_name, CFG.quat_feat_name
                )

                if rel_pose_at_contact_obj2_in_obj1_frame_ref is not None:
                    key = (RelPoseBaseRefObjPred, obj_type_of_reference_best, robot_base_obj_type, "2in1")
                    relative_pose_dataset_dict[key].append(rel_pose_at_contact_obj2_in_obj1_frame_ref)
                if rel_pose_at_contact_obj2_in_obj1_frame_contact is not None:
                    key = (RelPoseBaseContactObjPred, obj_type_contact_with_gripper, robot_base_obj_type, "2in1")
                    # relative_pose_dataset_dict[key].append(rel_pose_at_contact_obj2_in_obj1_frame_contact)

        return ground_atom_dataset, relative_pose_dataset_dict


    
    def _postprocess_cluster_predicates(self, env, dataset: Dataset, ground_atom_dataset: List[GroundAtomTrajectory], predicates_to_monitor: Set[Predicate], renamed_cluster_candidates: Dict[Predicate, float], best_reference_per_moving_obj: Dict[Type, Type], learnt_goal_predicates: Set[Predicate]):

        if CFG.reprocess_ground_atom_dataset_using_cluster_replacement: 
            #replace in contact atoms with rel pose atoms, so easier to do operator learning later
            different_seg_count_trajs = [] 
            for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
                for j, atoms in enumerate(atom_seq):
                    atoms_new = []
                    for atom in atoms:
                        if "RelCovCluster" in atom.predicate.name:
                            atoms_new.append(atom)
                            continue
                        try:
                            pred = list(CFG.dict_contact_predicate_to_rel_pose_predicates[atom.predicate.name, atom.objects[0].type.name, atom.objects[1].type.name])[0]
                        except KeyError:
                            pred = list(CFG.dict_contact_predicate_to_rel_pose_predicates[atom.predicate.name, atom.predicate.types[0].name, atom.predicate.types[1].name])[0]
                        grounded_pred = GroundAtom(pred, atom.entities)
                        atoms_new.append(grounded_pred)
                    ground_atom_dataset[i][1][j] = set(atoms_new)

       
            logging.warning("goal_predicate not implemented, so can't do online planning, breaking!")
        
        # Collect atoms at the end of each demonstration episode
        if env.goal_predicates:
            # Get final atoms from each trajectory
            final_atoms_per_episode = []
            for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
                if not ll_traj.states or not atom_seq:
                    continue  # Skip empty trajectories
                # Get atoms from the final timestep
                final_atoms = atom_seq[-1] if atom_seq else set()
                final_atoms_per_episode.append(final_atoms)
            
            if final_atoms_per_episode:
                # Count frequency of each atom across all episodes
                atom_counts = {}
                total_episodes = len(final_atoms_per_episode)
                
                for final_atoms in final_atoms_per_episode:
                    for atom in final_atoms:
                        # Create a hashable key for the atom (predicate name + object types)
                        atom_key = (atom.predicate.name, tuple(obj.name for obj in atom.objects), tuple(obj.type.name for obj in atom.objects))
                        atom_counts[atom_key] = atom_counts.get(atom_key, 0) + 1
                
                # Find atoms that appear in most episodes (threshold: at least 80% of episodes)
                threshold = max(1, int(0.8 * total_episodes))
                common_atoms = {atom_key for atom_key, count in atom_counts.items() if count >= threshold}
                
                logging.info(f"Found {len(common_atoms)} atoms that appear in at least {threshold}/{total_episodes} episodes")
                for atom_key, count in atom_counts.items():
                    if atom_key in common_atoms:
                        logging.info(f"  - {atom_key[0]}({', '.join(atom_key[1])}, {', '.join(atom_key[2])}): {count}/{total_episodes} episodes")
                
                # Update goal predicate mapping for each ground truth goal predicate
                pred_key = CFG.robo_kitchen_task 
                
                if common_atoms:
                    # If we found common atoms, save them with full details (predicate, obj_names, obj_types)
                    CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[pred_key] = common_atoms
                    logging.info(f"Updated goal mapping for {CFG.robo_kitchen_task} with {len(common_atoms)} common final atoms")
                else:
                    # No common atoms found, create fallback entries
                    logging.warning(f"No common atoms found with threshold {threshold}/{total_episodes}, creating fallback entries")
                    
                    # Create fallback entries with different levels of specificity
                    fallback_atoms = set()
                    
                    # First fallback: predicate + object names + object types (same as original but with lower threshold)
                    lower_threshold = max(1, int(0.5 * total_episodes))  # 50% threshold
                    lower_common_atoms = {atom_key for atom_key, count in atom_counts.items() if count >= lower_threshold}
                    
                    if lower_common_atoms:
                        fallback_atoms.update(lower_common_atoms)
                        logging.info(f"Added {len(lower_common_atoms)} atoms with lower threshold ({lower_threshold}/{total_episodes})")
                    else:
                        # Second fallback: predicate + object types only (remove specific object names)
                        predicate_type_atoms = set()
                        for atom_key, count in atom_counts.items():
                            pred_name, obj_names, obj_type_names = atom_key
                            # Create key with just predicate and types (empty tuple for obj_names)
                            type_only_key = (pred_name, (), obj_type_names)
                            predicate_type_atoms.add(type_only_key)
                        
                        fallback_atoms.update(predicate_type_atoms)
                        logging.info(f"Added {len(predicate_type_atoms)} predicate+type fallback atoms")
                    
                    CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[pred_key] = fallback_atoms
                    logging.info(f"Updated goal mapping for {CFG.robo_kitchen_task} with {len(fallback_atoms)} fallback atoms")
        
        # --- End Debugging ---
       
        return ground_atom_dataset, ground_atom_dataset, different_seg_count_trajs, renamed_cluster_candidates, learnt_goal_predicates
        
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
            if direction == "1in2":
                continue
            feat_name = CFG.pose_feature_name # We are clustering relative SE(3) poses
            logging.debug(f"Clustering relative feature {feat_name} for ({type1.name}, {type2.name}) from {pred.name} with {len(data)} points.")

            # if len(data) < 10: continue # Skip if no data collected

            # Save feature data (optional, copied from _generate_candidate_predicates)
            # feature_key = f"contact_{type1.name}_{type2.name}_{feat_name}"
            # os.makedirs("feature_data", exist_ok=True)
            # data_path = f"feature_data/{feature_key}.npy"
            # np.save(data_path, np.array(data))
            # logging.info(f"Saved {len(data)} contact pose data points for feature {feature_key} to {data_path}")

            # Perform clustering
            data_array, labels, unique_labels = self._cluster_feature_dataset(data, CFG.clustering_se3_epsilon, feat_name)
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
                    cluster_cov_trans_raw = np.cov(trans_diff, rowvar=False)
                    
                    num_dims_rot = 3 # always 3 for rotation
                    # Rotation regularization
                    log_deltas = (mean_rotation.inv() * rotations).as_rotvec()
                    cluster_cov_rot_raw = np.cov(log_deltas.T)

                    if type1.name == "gripper_type" or type2.name == "gripper_type":
                        base_reg_trans = CFG.clustering_inv_cov_reg_lin
                        base_reg_rot = CFG.clustering_inv_cov_reg_rot_gripper
                    elif type2.name == "base_type": 
                        base_reg_trans = CFG.clustering_inv_cov_reg_lin
                        base_reg_rot = CFG.clustering_inv_cov_reg_rot_base
                    else: # obj obj reg
                        base_reg_trans = CFG.clustering_inv_cov_reg_lin_low
                        base_reg_rot = CFG.clustering_inv_cov_reg_rot_low

                    reg_term_trans = utils.compute_adaptive_reg_term(
                        cluster_cov_trans_raw,
                        base_reg=base_reg_trans,
                        min_reg= 0.001 / 4 # reg adds the std, so divide by 4 to get almost 100% confidence
                    )
                    reg_term_rot = utils.compute_adaptive_reg_term(
                        cluster_cov_rot_raw,
                        base_reg=base_reg_rot,
                        min_reg=0.0005 / 3
                    )
                    cluster_cov_trans = cluster_cov_trans_raw + reg_term_trans
                    cluster_cov_rot = cluster_cov_rot_raw + reg_term_rot
                    # Convert covariance to degree variation for rotation
                    # Calculate standard deviation in degrees for each rotation axis
                    rot_std_degrees = np.sqrt(np.diag(cluster_cov_rot)) * (180.0 / np.pi)
                    # Calculate the average degree variation across all rotation axes
                    avg_degree_variation = np.mean(rot_std_degrees)
                    # Log the degree variation information
                    logging.debug(f"Cluster {k} translation variations: X={cluster_cov_trans[0][0]:.2f}, Y={cluster_cov_trans[1][1]:.2f}, Z={cluster_cov_trans[2][2]:.2f}")
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
                    name_prefix = "RobotBase" if  "RobotBaseRelPosPred" in pred.name else ""
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

    def _update_atom_sequences_with_goal_predicates(self, ground_atom_dataset: List[GroundAtomTrajectory], traj_all_objs_all: List[List[Object]], best_reference_per_moving_obj: Dict[Type, Type]):
        for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
            traj_all_objs = traj_all_objs_all[i]
            # obj_type_of_reference_best = best_reference_per_moving_obj[traj_all_objs[0].type]
            # obj_ref = [o for o in traj_all_objs if o.type == obj_type_of_reference_best][0]
            # assert obj_ref is not None, "Object of reference not found"
            
            # Track all goal/subgoal types found in this trajectory
            # goal_types_found = set()
            
            for j, atoms in enumerate(atom_seq):
                atoms_to_remove = []
                atoms_to_add = []
                
                for atom in atoms:
                    if isinstance(atom, DummyGroundAtom):
                        moving_obj = atom.entities[0]
                        ref_obj_type = best_reference_per_moving_obj[moving_obj.type]
                        ref_obj = [o for o in traj_all_objs if o.type == ref_obj_type]
                        assert len(ref_obj) == 1, "Multiple reference objects found"
                        ref_obj = ref_obj[0]
                        
                        atoms_to_remove.append(atom)
                        
                        # Handle both main goals and subgoals
                        assert "goal" in atom.predicate.name, "Atom is not a goal or subgoal"
                        # goal_types_found.add(atom.name)
                        # Replace with grounded version
                        new_goal_pred = DummyPredicate(atom.predicate.name, [ref_obj_type, moving_obj.type])
                        atoms_to_add.append(GroundAtom(new_goal_pred, [ref_obj, moving_obj]))
                
                # Apply the changes
                for atom in atoms_to_remove:
                    ground_atom_dataset[i][1][j].remove(atom)
                for atom in atoms_to_add:
                    ground_atom_dataset[i][1][j].add(atom)

            # Add all stored states to relative_pose_dataset_dict 
            # (they're already filtered by goal type during the marking phase)
            
                            
        return ground_atom_dataset #, relative_pose_dataset_dict

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
                                contact_period_rel_trajs: Dict[Type, Dict[Type, List[List[np.ndarray]]]], 
                                ) -> Tuple[Type, float, List[float]]:
        """
        Select the object of reference by learning a DS policy for each moving object type and each potential reference object type,
        and selecting the reference object type with the lowest reconstruction error for each moving object type.
        Returns the overall best reference object type.
        """
        best_reference_per_moving_obj = {}  # moving_obj_type -> (best_ref_obj_type, reconstruction_error)
        all_reconstruction_errors = {}
        
        for moving_obj_type, ref_obj_dict in contact_period_rel_trajs.items():
            logging.info(f"Evaluating reference objects for moving object type: {moving_obj_type.name}")
            
            best_ref_obj_type = None
            min_reconstruction_error = float('inf')
            black_list = []
            
            for ref_obj_type, rel_pose_trajs in ref_obj_dict.items():
                if len(rel_pose_trajs) == 0: 
                    continue
                    
                # logging.debug(f"  Testing reference object: {ref_obj_type.name} with {len(rel_pose_trajs)} trajectories")
                
                x = []
                quat = []
                x_dot = []
                omega = []
                
                for rel_pose_traj in rel_pose_trajs:
                    if len(rel_pose_traj) == 0:
                        continue
                    x_traj = np.array(rel_pose_traj)[:, :3]
                    quat_traj = np.array(rel_pose_traj)[:, 3:]
                    x_dot_traj, omega_traj = compute_vel_traj(x_traj, np.array([R.from_quat(q).as_matrix() for q in quat_traj]), 1/10)
                    x.append(x_traj)
                    quat.append(quat_traj)
                    x_dot.append(x_dot_traj)
                    omega.append(omega_traj)
                
                if len(x) == 0:
                    continue
                    
                # Check if start and end poses are almost the same (indicating no meaningful motion)
                if len(x) > 0 and len(x[0]) > 1:
                    start_pos = np.array([traj[0] for traj in x])
                    end_pos = np.array([traj[-1] for traj in x])
                    start_quat = np.array([traj[0] for traj in quat])
                    end_quat = np.array([traj[-1] for traj in quat])
                    
                    # Calculate average distance between start and end poses
                    avg_distance = np.mean([np.linalg.norm(end - start) for start, end in zip(start_pos, end_pos)])
                    avg_quat_distance = np.mean([np.linalg.norm((R.from_quat(end) * R.from_quat(start).inv()).as_rotvec()) for start, end in zip(start_quat, end_quat)])
                    
                    # If average distance is very small, blacklist this reference object for this moving object
                    if avg_distance < 0.01 and avg_quat_distance < 0.1:  
                        black_list.append(ref_obj_type)
                        logging.debug(f"    Blacklisting {ref_obj_type.name} as reference for {moving_obj_type.name} due to minimal motion (avg distance: {avg_distance:.4f})")
                        continue
                
                try:
                    unified_config = UnifiedModelConfig(
                        mode="se3_lpvds",
                        K_candidates=[3]
                    )
                    ds_policy = DSPolicy(
                        x=x,
                        x_dot=x_dot,
                        quat=quat,
                        omega=omega,
                        gripper=[],
                        unified_config=unified_config,
                        dt=1/10
                    )
                    _, reconstruction_error = ds_policy.compute_reconstruction_error()
                    all_reconstruction_errors[(moving_obj_type, ref_obj_type)] = reconstruction_error
                    
                    logging.debug(f"    {ref_obj_type.name} reconstruction error: {reconstruction_error:.6f}")
                    
                    if reconstruction_error < min_reconstruction_error and ref_obj_type not in black_list:
                        min_reconstruction_error = reconstruction_error
                        best_ref_obj_type = ref_obj_type
                        
                except Exception as e:
                    logging.warning(f"    Failed to compute DS policy for {ref_obj_type.name} as reference for {moving_obj_type.name}: {e}")
                    continue
            
            if best_ref_obj_type is not None:
                best_reference_per_moving_obj[moving_obj_type] = best_ref_obj_type
                logging.info(f"  Best reference for {moving_obj_type.name}: {best_ref_obj_type.name} (error: {min_reconstruction_error:.6f})")
            else:
                logging.warning(f"  No suitable reference object found for moving object type: {moving_obj_type.name}")
        
        # Select the overall best reference object type (lowest reconstruction error across all moving objects)
        if not best_reference_per_moving_obj:
            raise ValueError("No suitable reference objects found for any moving object type")
        
        
        return best_reference_per_moving_obj, all_reconstruction_errors

    def _extract_relative_pose_data(self, ground_atom_dataset: List[GroundAtomTrajectory], all_objs_types: List[Type], gripper_type: Type, in_contact_pred: Predicate, in_origin_pred: Predicate, trajectory_motion_phases: Dict[int, List[Dict]] = None, trajectory_all_objects: Dict[int, List[Object]] = None, gripper_obj: Object = None, contact_lost_periods: Dict[int, List[Tuple[int, int]]] = None) -> Tuple[Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]], List[List[Object]], Dict[Type, List[List[np.ndarray]]], List[State], List[Type], List[GroundAtomTrajectory]]:
        relative_pose_gripper_obj_dataset_dict = defaultdict(list) # Maps (atom_pred, type1, type2) -> List[rel_pose]
        contact_period_obj_obj_rel_trajs = {} # Maps (moving_obj_type, reference_obj_type) -> List[List[rel_pose]]
        contact_lost_period_rel_pose = {} # Maps (moving_obj_type, reference_obj_type) -> List[List[rel_pose]]
        # goal_reached_states = defaultdict(list)
        object_type_in_contact_with_gripper_longest_duration = []
        traj_all_objs_all = []

        # Use motion analysis data if available, otherwise fall back to original logic
        if trajectory_motion_phases is not None and trajectory_all_objects is not None and gripper_obj is not None and contact_lost_periods is not None:
            logging.info("Using motion analysis data for relative pose extraction...")
            
            for i, (ll_traj, atom_seq) in enumerate(ground_atom_dataset):
                # goal_reached_states[i] = []
                
                # Get objects from motion analysis
                if i in trajectory_all_objects:
                    traj_all_objs = trajectory_all_objects[i]
                else:
                    # Fall back to original logic if no motion analysis data
                    traj_all_objs = [o for o in ll_traj.states[0].data.keys() if o.type in all_objs_types]
                
                traj_all_objs_all.append(traj_all_objs)
                object_type_in_contact_with_gripper_longest_duration.append({})
                
                if not ll_traj.states: 
                    continue
                
                # Mark subgoals based on contact lost periods
                if i in contact_lost_periods:
                    for idx, (start_period, end_period) in enumerate(contact_lost_periods[i]):
                        if idx == 0: continue
                        idx = idx - 1
                        # Find which object was in motion before this contact lost period
                        moving_obj = self._find_object_in_motion_before_period(trajectory_motion_phases[i], start_period)
                        goal_type = f"{CFG.robo_kitchen_task}-subgoal"
                        
                        # Find the next time this same object starts motion again
                        next_motion_start = self._find_next_motion_start_for_object(trajectory_motion_phases[i], moving_obj, start_period)
                        
                        # Determine the goal end time: either next motion start or trajectory end
                        if next_motion_start != -1:
                            goal_end_time = next_motion_start - 1  # Stop just before next motion starts
                        else:
                            # If no next motion, populate until end of trajectory
                            goal_end_time = len(atom_seq) - 1
                        
                        # logging.debug(f"Marking subgoal from t={start_period} to t={goal_end_time} (next motion at {next_motion_start})")
                        
                        for t_goal in range(start_period, goal_end_time + 1):
                            if t_goal < len(atom_seq):
                                goal_atom = DummyGroundAtom(DummyPredicate(goal_type, [moving_obj.type]), [moving_obj])
                                ground_atom_dataset[i][1][t_goal].add(goal_atom)
                                # goal_reached_states[i].append(ll_traj.states[t_goal])
                        
                        # During contact lost periods, compute relative poses between objects in motion and all other objects
                        if moving_obj:
                            for t in range(start_period, min(goal_end_time + 1, len(ll_traj.states))):
                                state_t = ll_traj.states[t]
                                for obj in traj_all_objs:
                                    if obj.type == gripper_type or obj == moving_obj:
                                        continue
                                    relative_pose = utils.calculate_relative_pose_from_state(
                                        state_t, obj, moving_obj, CFG.trans_feat_name, CFG.quat_feat_name
                                    )
                                    # Use nested dictionary: moving_obj_type -> reference_obj_type -> trajectories
                                    # these should be rather static poses that are easy to cluster
                                    if moving_obj.type not in contact_lost_period_rel_pose:
                                        contact_lost_period_rel_pose[moving_obj.type] = {}
                                    if obj.type not in contact_lost_period_rel_pose[moving_obj.type]:
                                        contact_lost_period_rel_pose[moving_obj.type][obj.type] = []
                                    if t == start_period:  # Start of new period
                                        contact_lost_period_rel_pose[moving_obj.type][obj.type].append([relative_pose])
                                    else:
                                        if contact_lost_period_rel_pose[moving_obj.type][obj.type]:
                                            contact_lost_period_rel_pose[moving_obj.type][obj.type][-1].append(relative_pose)
                
                # Extract relative poses during contact periods from motion phases
                if i in trajectory_motion_phases:
                    for phase in trajectory_motion_phases[i]:
                        start_frame = phase['start_frame']
                        end_frame = phase['end_frame']
                        moving_objects = phase.get('moving_objects', [])
                        
                        # Assume contact between gripper and moving objects during motion phases
                        for moving_obj in moving_objects:
                            for t in range(start_frame, min(end_frame + 1, len(ll_traj.states))):
                                state_t = ll_traj.states[t]
                                
                                # Calculate relative pose between gripper and moving object
                                rel_pose_at_contact_obj2_in_obj1_frame = utils.calculate_relative_pose_from_state(
                                    state_t, moving_obj, gripper_obj, CFG.trans_feat_name, CFG.quat_feat_name
                                )
                                if rel_pose_at_contact_obj2_in_obj1_frame is not None:
                                    key = (in_contact_pred, moving_obj.type, gripper_type, "2in1")
                                    relative_pose_gripper_obj_dataset_dict[key].append(rel_pose_at_contact_obj2_in_obj1_frame)
                                
                                # During contact periods, compute relative poses between objects in motion and all other objects
                                for obj in traj_all_objs:
                                    if obj.type == gripper_type or obj == moving_obj:
                                        continue
                                    relative_pose = utils.calculate_relative_pose_from_state(
                                        state_t, obj, moving_obj, CFG.trans_feat_name, CFG.quat_feat_name
                                    )
                                    # Use nested dictionary: moving_obj_type -> reference_obj_type -> trajectories
                                    if moving_obj.type not in contact_period_obj_obj_rel_trajs:
                                        contact_period_obj_obj_rel_trajs[moving_obj.type] = {}
                                    if obj.type not in contact_period_obj_obj_rel_trajs[moving_obj.type]:
                                        contact_period_obj_obj_rel_trajs[moving_obj.type][obj.type] = []
                                    if t == start_frame:  # Start of new contact period
                                        contact_period_obj_obj_rel_trajs[moving_obj.type][obj.type].append([relative_pose])
                                    else:
                                        if contact_period_obj_obj_rel_trajs[moving_obj.type][obj.type]:
                                            contact_period_obj_obj_rel_trajs[moving_obj.type][obj.type][-1].append(relative_pose)
        
        return relative_pose_gripper_obj_dataset_dict, traj_all_objs_all, contact_period_obj_obj_rel_trajs, contact_lost_period_rel_pose, ground_atom_dataset

    def _find_object_in_motion_before_period(self, motion_phases: List[Dict], period_start: int) -> Object:
        """Find the object that was in motion before the given period start time."""
        # Look for the most recent motion phase that ended before this period
        for phase in reversed(motion_phases):
            if phase['end_frame'] < period_start and phase.get('moving_objects'):
                return phase['moving_objects'][0]  # Return the first (primary) moving object
        raise ValueError("No object in motion before the given period start time")

    def _find_next_motion_start_for_object(self, motion_phases: List[Dict], moving_obj: Object, current_period_start: int) -> int:
        """Find the next time the same object starts motion again after the current period.
        
        Args:
            motion_phases: List of motion phase dictionaries
            moving_obj: The object to find next motion for
            current_period_start: The start of the current period
            
        Returns:
            The frame number when the same object starts motion again, or -1 if not found
        """
        for phase in motion_phases:
            # Look for phases that start after the current period and involve the same object
            if (phase['start_frame'] > current_period_start and 
                phase.get('moving_objects') and 
                any(obj.name == moving_obj.name for obj in phase['moving_objects'])):
                return phase['start_frame']
        
        return -1  # No next motion found for this object

    
    def _find_common_objects_types(self, ground_atom_dataset: List[GroundAtomTrajectory]) -> List[Type]:
        """Find common objects across all trajectories in the dataset."""
        all_objs_types = set()
        # Print each object name and type from the first trajectory
        for obj in ground_atom_dataset[0][0].states[0].data.keys():
            logging.info(f"Object name: {obj.name}, Object type: {obj.type.name}")
            all_objs_types.add(obj.type)
        for traj, _ in ground_atom_dataset[1:]:  # Skip the first one we already processed
            if not traj.states:
                continue  # Skip empty trajectories
            traj_objs_types = set([obj.type for obj in traj.states[0].data.keys()])
            all_objs_types = all_objs_types.intersection(traj_objs_types)  # Keep only objects present in all trajectories
        all_objs_types = list(all_objs_types)  # Convert back to list for further processing
        robot_base_objs_types = [o_type for o_type in all_objs_types if "base_type" == o_type.name]
        robot_base_obj_type = robot_base_objs_types[0] if robot_base_objs_types else None

        all_objs_types = [
            o_type for o_type in all_objs_types
            if "finger" not in o_type.name
            and "base"   not in o_type.name
            and "wrist"  not in o_type.name
            and "base" not in o_type.name
            and "counter" not in o_type.name
        ]
        logging.info(f"After filtering, {len(all_objs_types)} objects remain") 
        if len(all_objs_types) <= 2:
            raise ValueError(f"Only {len(all_objs_types)} objects remain, which is less than 3. Not enough for finding a reference object.")
        return all_objs_types, robot_base_obj_type
    
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
                    # if cand_pred.arity == 2 and cand_pred.types[1].name != "gripper_type":
                    #     continue
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

            # Keep top B successors based on score, excluding -inf scores
            valid_successors = [s for s in successors if s[0] > -np.inf]
            valid_successors.sort(key=lambda x: x[0], reverse=True) # Sort descending by score
            new_beam = valid_successors[:beam_width]

            # Check for convergence (beam hasn't changed or score isn't improving)
            # Simple check: if the best score in the new beam is not better than the previous best
            current_best_score_in_beam = new_beam[0][0] if new_beam else -np.inf
            if current_best_score_in_beam <= best_score and iteration > 1 : # Allow first iteration to set baseline
                logging.info("\033[1;35mBeam search converged (no score improvement).\033[0m")
                break
            
            if new_beam:
                best_score, best_pred_set_added, best_operators = new_beam[0] # Best set in current beam (added preds only)
                logging.info(f"\033[1;36mIteration {iteration} best score: {best_score:.4f}\033[0m")
                logging.info(f"\033[1;32mCurrent best operators: {best_operators}\033[0m")
                logging.info(f"\033[1;33mCurrent best preds: {best_pred_set_added}\033[0m")

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
      
        # Extend predicates with their negations (similar to grammar search approach)
        extended_predicates = set(predicates)
        for predicate in predicates:
            negated_classifier = _NegationClassifier(predicate)
            negated_predicate = Predicate(str(negated_classifier), predicate.types, negated_classifier)
            extended_predicates.add(negated_predicate)
        extended_predicates = frozenset(extended_predicates)
        predicates = extended_predicates
        
        # Check plan length constraint first (most expensive)
        # Need operators for the constraint check
        atom_dataset = self._create_atom_dataset(dataset, predicates)
        # if CFG.clustering_debug:
        #     for i, (_, atom_seq) in enumerate(atom_dataset):
        #         print(f"Traj {i}:")
        #         current_atom_count = 0
        #         current_atom = atom_seq[0]
        #         for atom in atom_seq:
        #             if atom == current_atom:
        #                 current_atom_count += 1
        #             else:
        #                 print(f"{current_atom} {current_atom_count}")
        #                 current_atom = atom
        #                 current_atom_count = 1

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
        score = seg_term - alpha * op_term - constraint_value
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
            for p in predicates:
                if "RelPoseEllipsoidCluster" in p.name:
                    if "gripper_type" in p.types[0].name or "gripper_type" in p.types[1].name:
                        if "thing_type" in p.types[0].name or "thing_type" in p.types[1].name:
                            pass
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
            self._plan_constraint_cache[predicates] = np.inf  # Significant negative value
            return np.inf

        # The 'operators' set already contains STRIPSOperator objects
        strips_ops = operators

        diff_vec = []

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
            demo_plan_len = len(demo_segments)  # segment does not include last section

            # Create a planning task
            task = Task(init_state, goal_atoms)

            # Run the planner using the learned NSRTs
            plan = None
            try:
                plan, _, metrics = run_task_plan_once(
                    task=task,
                    nsrts=strips_ops,    # Pass NSRTs
                    preds=set(predicates),  # Pass predicates
                    types=self._types,   # Pass types
                    timeout=10.0,   # Pass timeout
                    seed=0,      # Pass seed
                    task_planning_heuristic=CFG.sesame_task_planning_heuristic,  # Pass heuristic
                )
            except PlanningFailure as e:
                logging.debug(f"Planning failed for traj {i}: {e}")

            # Check planner result
            if plan is None:
                # Planner failed (timeout or unsolvable) - assume diff of 3
                diff = 3
            else:
                planner_plan_len = len(plan)    
                # number of grounded operators in the plan
                # Calculate difference: positive if plan is longer, negative if shorter
                diff = planner_plan_len - demo_plan_len
                # logging.debug(f"Traj {i}: Demo len={demo_plan_len}, Planner len={planner_plan_len}, Diff={diff}")
            diff_vec.append(diff)
            # Accumulate absolute difference for soft constraint
            # total_diff += abs(diff)
        # if total_diff != 0:
        #     self._plan_constraint_cache[predicates] = np.inf
        #     return np.inf
        # else:
        #     self._plan_constraint_cache[predicates] = 0
        #     return 0
        # Check if any predicate involves gripper and thing types
        for p in predicates:
            if "RelPoseEllipsoidCluster" in p.name:
                if "gripper_type" in p.types[0].name or "gripper_type" in p.types[1].name:
                    if "thing_type" in p.types[0].name or "thing_type" in p.types[1].name:
                        pass

        total_diff = np.sum(np.abs(np.array(diff_vec)))
        self._plan_constraint_cache[predicates] = total_diff
        return self._plan_constraint_cache[predicates]

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
            fname = f"p_{CFG.robo_kitchen_task}_{type2_name}_in_{type1_name}_frame.png"

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
                rel_pose = utils.calculate_relative_pose_from_state(state, obj1, obj2, 
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

    def _update_rel_pose_dict_with_obj_obj(self,  relative_pose_gripper_obj_dataset_dict: Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]], contact_lost_period_obj_obj_rel_trajs: Dict[Type, Dict[Type, List[List[np.ndarray]]]], best_reference_per_moving_obj: Dict[Type, Type]) -> Dict[Tuple[Predicate, Type, Type, str], List[np.ndarray]]:
        """
        Merge gripper-object relative poses with object-object relative poses into a unified dictionary.
        
        Args:
            relative_pose_gripper_obj_dataset_dict: Dictionary containing gripper-object relative poses
                Format: {(predicate, obj_type1, obj_type2, direction): [relative_poses]}
            contact_lost_period_obj_obj_rel_trajs: Dictionary containing object-object relative pose trajectories
                Format: {moving_obj_type: {reference_obj_type: [[trajectory_poses]]}}
        
        Returns:
            Combined dictionary with all relative poses in the same format as relative_pose_gripper_obj_dataset_dict
        """
        from collections import defaultdict
        
        # Start with a copy of the gripper-object data
        relative_pose_all_dict = defaultdict(list)
        
        # Copy gripper-object relative poses
        for key, poses in relative_pose_gripper_obj_dataset_dict.items():
            relative_pose_all_dict[key].extend(poses)
        
        # Add object-object relative poses
        for moving_obj_type, reference_dict in contact_lost_period_obj_obj_rel_trajs.items():
            ref_obj_type_best = best_reference_per_moving_obj[moving_obj_type]
            for ref_obj_type, trajectory_segments in reference_dict.items():
                if ref_obj_type != ref_obj_type_best:
                    continue
                # Create a dummy predicate for object-object relationships
                # This follows the pattern used elsewhere in the code
                obj_obj_pred = DummyPredicate(f"{CFG.robo_kitchen_task}-subgoal")
                
                # Create key in the same format as gripper-object data
                # Use "2in1" direction (moving object relative to reference object)
                key = (obj_obj_pred, ref_obj_type, moving_obj_type, "2in1")
                logging.info(f"Adding key: {key}")        
                # Flatten trajectory segments into individual poses
                for trajectory_segment in trajectory_segments:
                    for pose in trajectory_segment:
                        if pose is not None:
                            relative_pose_all_dict[key].append(pose)
        
        logging.info(f"Merged relative pose data: {len(relative_pose_gripper_obj_dataset_dict)} gripper-object entries + "
                    f"{sum(len(ref_dict) for ref_dict in contact_lost_period_obj_obj_rel_trajs.values())} object-object entries = "
                    f"{len(relative_pose_all_dict)} total entries")
        
        return dict(relative_pose_all_dict)

    