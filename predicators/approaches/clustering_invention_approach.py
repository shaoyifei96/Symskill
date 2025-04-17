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
from sklearn.cluster import AgglomerativeClustering, DBSCAN
# Import HDBSCAN (may need `pip install hdbscan`)
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
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
from predicators.structs import Dataset, GroundAtomTrajectory, NSRT, Object, ParameterizedOption, Predicate, Segment, State, Task, Type, STRIPSOperator
import warnings
from scipy.stats import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse # For 2D ellipses
import numpy.linalg # For eigh
################################################################################
#                          Programmatic classifiers                            #
################################################################################


@dataclass(frozen=True, eq=False, repr=False)
class _RelativeFeatureClusterClassifier(_BinaryClassifier):
    """Classifies based on the Mahalanobis distance of a relative feature vector
    (including 7D pose) between two objects to a target cluster center and covariance.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str # Will be "pose" for SE(3) clusters
    cluster_center: np.ndarray # Can be 7D for pose
    inv_covariance_matrix: np.ndarray # Can be 7x7 for pose
    mahalanobis_threshold: float
    # diff_fn might not be needed for 'pose' if calculation is explicit
    diff_fn: Optional[Callable[[Any, Any], Any]] # Made optional
    cluster_id: int

    # Cache feature names
    _trans_feat_name: str = field(default="translation", init=False)
    _quat_feat_name: str = field(default="quaternion", init=False)
    _pose_feat_name: str = field(default="pose", init=False)

    def _calculate_relative_pose(self, state: State, o1: Object, o2: Object, trans_feat_name: str, quat_feat_name: str) -> Optional[np.ndarray]:
        """Calculates the relative pose of o2 with respect to o1's frame.
        
        Returns a 7D vector [tx, ty, tz, qx, qy, qz, qw] or None if features missing.
        """
        try:
            trans_o1 = state.get(o1, trans_feat_name)
            trans_o2 = state.get(o2, trans_feat_name)
            quat_o1 = state.get(o1, quat_feat_name)
            quat_o2 = state.get(o2, quat_feat_name)

            rot_o1 = Rotation.from_quat(quat_o1)
            rot_o2 = Rotation.from_quat(quat_o2)

            relative_trans_world = np.subtract(trans_o2, trans_o1)
            relative_trans_local = rot_o1.inv().apply(relative_trans_world)

            relative_rot = rot_o1.inv() * rot_o2
            relative_quat = relative_rot.as_quat()

            pose_vec = np.concatenate([relative_trans_local, relative_quat])
            return pose_vec # 7D vector
        except KeyError as e:
            logging.debug(f"Missing feature {e} for relative pose between {o1} and {o2}. Skipping.")
            return None


    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        assert obj1.is_instance(self.object1_type)
        assert obj2.is_instance(self.object2_type)

        # Calculate the relevant relative feature
        if self.feature_name == self._pose_feat_name:
            # Calculate the 7D relative pose
            relative_feature = self._calculate_relative_pose(s, obj1, obj2, 
                                                       self._trans_feat_name, 
                                                       self._quat_feat_name)
            if relative_feature is None:
                logging.warning(f"Could not compute relative pose for classification between {obj1}, {obj2}. Returning False.")
                return False # Cannot classify if pose cannot be computed
        else:
            # Handle original features (e.g., translation only, rotation only if kept)
            obj1_feat = s.get(obj1, self.feature_name)
            obj2_feat = s.get(obj2, self.feature_name)
            
            # Special handling for local frame translation (if kept as separate feature)
            if self.feature_name == self._trans_feat_name and self._quat_feat_name in obj1.type.feature_names:
                try:
                    obj1_quat = s.get(obj1, self._quat_feat_name)
                    obj1_rot = Rotation.from_quat(obj1_quat)
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
        return (f"{prefix}-{self.object1_type.name}-{self.object2_type.name}-"
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
        return (f"AbsEllipsoidCluster-{self.object_type.name}-"
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
    """An approach that invents predicates via feature clustering and beam search
    selection."""

    # Caches for expensive computations during beam search
    _atom_dataset_cache: Dict[FrozenSet[Predicate], List[GroundAtomTrajectory]] = {}
    _operator_complexity_cache: Dict[FrozenSet[Predicate], Tuple[int, Set[NSRT]]] = {}
    _segmentation_cache: Dict[FrozenSet[Predicate], int] = {}
    _plan_constraint_cache: Dict[FrozenSet[Predicate], bool] = {}

    @classmethod
    def get_name(cls) -> str:
        return "clustering_invention"

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # We need to properly load the learned predicates if they exist
        super().load(online_learning_cycle)
        if online_learning_cycle is None: # Only load during offline learning phase
            save_path = utils.get_approach_save_path_str()
            # Load the learned predicates set if available
            learned_preds_path = f"{save_path}_learned_predicates.pkl"
            if utils.file_exists(learned_preds_path):
                self._learned_predicates = utils.load_from_pickle(learned_preds_path)
                logging.info(f"Loaded {len(self._learned_predicates)} learned predicates.")
            else:
                self._learned_predicates = set()
        else: # In online learning, learned predicates are already part of NSRTs
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            self._learned_predicates = set(preds.values()) - self._initial_predicates

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    # --- Core Learning Method ---
    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        logging.info("Generating candidate predicates via clustering...")
        # Filter dataset to only keep specific trajectory indices
        keep_indices = [0, 3, 4, 6, 7, 8]
        dataset._trajectories = [dataset._trajectories[i] for i in keep_indices]

        logging.info(f"Filtered dataset to trajectories (indices: {keep_indices})")
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

        # Save the learned predicates separately for potential reloading
        save_path = utils.get_approach_save_path_str()
        learned_preds_path = f"{save_path}_learned_predicates.pkl"
        # Replace utils.save_to_pickle with direct pkl.dump
        with open(learned_preds_path, "wb") as f:
            pkl.dump(self._learned_predicates, f)

        # Learn NSRTs with the final set of predicates
        final_predicates = self._get_current_predicates()
        # We need the atom dataset for the final selected predicates
        atom_dataset_final = self._create_atom_dataset(dataset, final_predicates)
        annotations = None # Or derive from atom_dataset if needed by _learn_nsrts

        # Segment the trajectories using the final predicates and atom dataset
        segmented_trajs_final = [
            segment_trajectory(ll_traj, final_predicates, atom_seq=atom_seq)
            for ll_traj, atom_seq in atom_dataset_final
        ]

        # Call learn_nsrts with segmented trajectories
        self._learn_nsrts(
            dataset.trajectories,
            annotations=annotations,
            online_learning_cycle=None
        )


    def _get_feature_difference_function(self, feat_name: str) -> Callable:
        if feat_name == "pose":
            return self._calculate_se3_distance
        else:
            return self._get_feature_difference_function(feat_name)

    # --- Candidate Generation Functions ---
    def _generate_candidate_predicates(self, dataset: Dataset) -> Dict[Predicate, float]:
        """Generates candidate predicates by clustering relative and absolute features."""
        relative_feature_datasets = self._generate_relative_feature_datasets(dataset)
        # absolute_feature_datasets = self._generate_absolute_feature_datasets(dataset)

        candidates: Dict[Predicate, float] = {}
        predicate_counter = 0 # To ensure unique cluster IDs

        # Feature names for special handling
        quat_feat_name = "quaternion"
        trans_feat_name = "translation"
        pose_feature_name = "pose"
        
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
            if feat_name == trans_feat_name:
                epsilon = CFG.clustering_translation_epsilon
            elif feat_name == quat_feat_name:
                epsilon = CFG.clustering_quaternion_epsilon
            else: #pose_feature
                epsilon = CFG.clustering_epsilon

            # logging.debug(f"Using epsilon: {epsilon:.4f} for feature {feat_name}")
            # Perform clustering
            data_array, labels, unique_labels, effective_epsilon = self._cluster_feature_dataset(data, epsilon, feat_name)
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
                    cluster_center = np.mean(cluster_points, axis=0)
                    # Store basic info first
                    kept_clusters_info[k] = {'center': cluster_center, 'size': cluster_size, 'points': cluster_points} # Store points for cov calculation
                else:
                    discarded_labels.add(k)
                    logging.debug(f"Cluster {k} for {type1.name}-{type2.name}-{feat_name} discarded (size {cluster_size} < {min_cluster_size}).")

            # Now calculate covariance etc. ONLY for kept clusters and add to info dict
            kept_cluster_labels_list = list(kept_clusters_info.keys())
            for cluster_label in kept_cluster_labels_list: # Iterate over keys
                cluster_info = kept_clusters_info[cluster_label]
                cluster_points = cluster_info['points'] # Retrieve stored points

                # Calculate covariance matrix
                if cluster_points.shape[0] >= 2 and cluster_points.shape[1] > 0: # Need at least 2 points and features
                    # Use rowvar=False because each row is an observation
                    covariance_matrix = np.cov(cluster_points, rowvar=False)
                    # Handle scalar features explicitly (np.cov returns scalar)
                    if covariance_matrix.ndim == 0:
                        covariance_matrix = np.array([[covariance_matrix]])
                    # Add regularization to prevent singular matrix
                    reg_coeff = 1e-6
                    covariance_matrix += np.eye(covariance_matrix.shape[0]) * reg_coeff

                    # Calculate inverse covariance matrix
                    try:
                        inv_covariance_matrix = inv(covariance_matrix)
                        # Check if determinant is near zero (optional sanity check)
                        # _, logdet_cov = np.linalg.slogdet(covariance_matrix)
                        # if logdet_cov < -1e6: # Very small determinant might indicate issues
                        #      logging.warning(f"Cluster {cluster_label} cov matrix determinant is very small ({logdet_cov}). Inverse might be unstable.")

                    except LinAlgError:
                        logging.warning(f"Cluster {cluster_label} for {type1.name}-{type2.name}-{feat_name} has a singular covariance matrix. Skipping predicate creation.")
                        continue
                elif cluster_points.shape[0] == 1 and cluster_points.shape[1] > 0:
                    # Handle single-point cluster: use identity matrix scaled by a small epsilon
                    logging.debug(f"Cluster {cluster_label} has only 1 point. Using scaled identity for covariance.")
                    dims = cluster_points.shape[1]
                    pseudo_variance = 1e-4 # Small variance
                    inv_covariance_matrix = np.eye(dims) / pseudo_variance
                else:
                    logging.warning(f"Cluster {cluster_label} for {type1.name}-{type2.name}-{feat_name} has insufficient points/dims ({cluster_points.shape}) for covariance. Skipping.")
                    continue

                # Store calculated info back into the main dict for plotting
                cluster_info['inv_covariance_matrix'] = inv_covariance_matrix
                cluster_info['covariance_matrix'] = covariance_matrix
                # Initialize dims and threshold with default values
                dims = 1
                base_mahalanobis_threshold = chi2.ppf(CFG.clustering_mahalanobis_confidence, df=dims)

                if data_array.ndim > 1 and data_array.shape[1] > 0: # Case: >= 2D features
                    dims = data_array.shape[1]
                    # Use chi-squared distribution ppf (percent point function) for threshold
                    # Example: 95th percentile -> alpha=0.05
                    confidence_level = CFG.clustering_mahalanobis_confidence
                    base_mahalanobis_threshold = chi2.ppf(confidence_level, df=dims)
                    logging.debug(f"Calculated Mahalanobis threshold: {base_mahalanobis_threshold:.4f} for {dims} dims ({confidence_level*100:.1f}% confidence)")
                elif data_array.ndim == 1 and data_array.shape[0] > 0: # Case: 1D features
                    # Handle scalar features or cases where dims can't be determined
                    dims = 1
                    confidence_level = CFG.clustering_mahalanobis_confidence
                    base_mahalanobis_threshold = chi2.ppf(confidence_level, df=dims) # Default for 1D
                    logging.debug(f"Using 1D Mahalanobis threshold: {base_mahalanobis_threshold:.4f} ({confidence_level*100:.1f}% confidence)")
                else: # Case: data_array is empty or malformed
                    logging.warning(f"Could not determine feature dimension for {feat_name}:{type1.name}-{type2.name} (shape: {data_array.shape}). Using default 1D threshold: {base_mahalanobis_threshold:.4f}")

                cluster_info['mahalanobis_threshold'] = base_mahalanobis_threshold # Use pre-calculated threshold
                # Remove points to save memory if needed, or keep for other analysis
                # del cluster_info['points']

            # Now, optionally visualize clusters if in debug mode, passing the *updated* info
            if CFG.clustering_debug and data_array.size > 0: # Check if there is data to plot
                # The kept_clusters_info dict now contains cov matrix and threshold for plot
                self._plot_cluster_results(data_array, labels, unique_labels, kept_clusters_info,
                                           type1.name, type2.name if type2 else None, feat_name)

                # Plot relative trajectories *after* cluster plot, if applicable
                if feat_name == pose_feature_name and type2 is not None:
                    self._plot_relative_trajectories(dataset, type1.name, type2.name)

            # Sort kept clusters by size (descending) for top_k selection AFTER plotting
            # Filter out any clusters where covariance calculation failed (if needed, though `continue` above handles it)
            valid_kept_clusters = {k: v for k, v in kept_clusters_info.items() if 'inv_covariance_matrix' in v}
            sorted_valid_kept_clusters = sorted(valid_kept_clusters.items(), key=lambda item: item[1]['size'], reverse=True)

            # Create predicates for the top_k *valid* kept clusters
            top_k = min(CFG.clustering_max_clusters, len(sorted_valid_kept_clusters))
            logging.debug(f"Selecting top {top_k} valid kept clusters for {feat_name}:{type1.name}-{type2.name}.")

            for i, (cluster_label, cluster_info) in enumerate(sorted_valid_kept_clusters[:top_k]):
                # logging.debug(f"Creating predicate for kept cluster {cluster_label} (size {cluster_info['size']}, rank {i+1}/{top_k}).")
                # Pass inverse covariance and threshold instead of epsilon
                pred = self._create_predicate_from_relative_cluster(
                    type1, type2, feat_name, cluster_info['center'],
                    cluster_info['inv_covariance_matrix'],
                    cluster_info['mahalanobis_threshold'],
                    diff_fn, cluster_label) # Use cluster_label for ID
                candidates[pred] = pred.arity + 1.0
                predicate_counter += 1


        # Rename predicates for PDDL compatibility (reuse from grammar search)
        renamed_candidates = self._rename_predicates_to_remove_incompatible_chars(candidates)
        return renamed_candidates

    def _generate_relative_feature_datasets(self, dataset: Dataset) -> Dict[Tuple[Type, Type, str], List[np.ndarray]]:
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
                            rel_pose_t = self._calculate_relative_pose(state_t, o1, o2, trans_feat_name, quat_feat_name)
                            rel_pose_t1 = self._calculate_relative_pose(state_t1, o1, o2, trans_feat_name, quat_feat_name)

                            if rel_pose_t is not None and rel_pose_t1 is not None:
                                # Calculate change in relative pose (using SE(3) distance concept)
                                # We need a distance function here, let's define a simple one for constancy check
                                pose_diff_norm = self._calculate_se3_distance(rel_pose_t, rel_pose_t1, 
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
                # Use a threshold relative to the feature type maybe?
                # Using percentile seems reasonable for now.
                # Consider CFG.clustering_feature_constancy_percentile ?
                constancy_threshold = np.percentile(changes, CFG.clustering_feature_constancy_percentile) # Default 30?
                logging.debug(f"Constancy threshold for {feature_key}: {constancy_threshold:.4f} ({CFG.clustering_feature_constancy_percentile}th percentile)")
                mask = changes <= constancy_threshold
                final_feature_data[feature_key] = [pt for pt, keep in zip(data_points, mask) if keep]
                logging.debug(f"Kept {sum(mask)} / {len(data_points)} points for {feature_key} based on constancy.")
            else:
                 logging.debug(f"No data points collected for {feature_key}.")


        return final_feature_data

    def _calculate_relative_pose(self, state: State, o1: Object, o2: Object, trans_feat_name: str, quat_feat_name: str) -> Optional[np.ndarray]:
        """Calculates the relative pose of o2 with respect to o1's frame.
        
        Returns a 7D vector [tx, ty, tz, qx, qy, qz, qw] or None if features missing.
        """
        try:
            trans_o1 = state.get(o1, trans_feat_name)
            trans_o2 = state.get(o2, trans_feat_name)
            quat_o1 = state.get(o1, quat_feat_name)
            quat_o2 = state.get(o2, quat_feat_name)

            rot_o1 = Rotation.from_quat(quat_o1)
            rot_o2 = Rotation.from_quat(quat_o2)

            relative_trans_world = np.subtract(trans_o2, trans_o1)
            relative_trans_local = rot_o1.inv().apply(relative_trans_world)

            relative_rot = rot_o1.inv() * rot_o2
            relative_quat = relative_rot.as_quat()
            # Ensure consistent quaternion representation (e.g., w >= 0) - optional
            # if relative_quat[3] < 0:
            #     relative_quat *= -1

            pose_vec = np.concatenate([relative_trans_local, relative_quat])
            return pose_vec # 7D vector
        except KeyError as e:
            logging.debug(f"Missing feature {e} for relative pose between {o1} and {o2}. Skipping.")
            return None

    def _calculate_se3_distance(self, pose_vec1: np.ndarray, pose_vec2: np.ndarray, 
                                trans_weight: float, rot_weight: float) -> float:
        """Calculates a weighted SE(3) distance between two 7D pose vectors."""
        # Assumes pose_vec is [tx, ty, tz, qx, qy, qz, qw]
        trans1, quat1 = pose_vec1[:3], pose_vec1[3:]
        trans2, quat2 = pose_vec2[:3], pose_vec2[3:]

        # Translational distance (Euclidean)
        trans_dist_sq = np.sum((trans1 - trans2)**2)

        # Rotational distance (angle of relative rotation)
        # Ensure quaternions are valid rotations
        if np.isclose(norm(quat1), 0) or np.isclose(norm(quat2), 0):
            # Handle zero quaternions if they occur, maybe return large distance?
            logging.warning("Encountered near-zero quaternion in SE(3) distance calculation.")
            rot_dist_sq = np.pi**2 # Max possible squared angle
        else:
            try:
                # Normalize quaternions robustly before creating Rotation objects
                quat1_norm = quat1 / norm(quat1)
                quat2_norm = quat2 / norm(quat2)
                # Handle potential numerical instability if quats are *exactly* opposite
                # dot_product = np.dot(quat1_norm, quat2_norm)
                # if np.isclose(dot_product, -1.0):
                #     rot_dist = np.pi # Angle is pi for opposite rotations
                # else:
                rot1 = Rotation.from_quat(quat1_norm)
                rot2 = Rotation.from_quat(quat2_norm)
                relative_rot = rot1.inv() * rot2
                # Use magnitude() which gives the angle in radians
                rot_dist = relative_rot.magnitude() 
                rot_dist_sq = rot_dist**2
            except ValueError as e:
                 logging.error(f"Error calculating rotation distance: {e}. Quats: {quat1}, {quat2}")
                 rot_dist_sq = np.pi**2 # Penalize problematic rotations

        # Weighted combination
        # Using CFG values directly here for simplicity, assuming they are accessible.
        # Ideally, pass them as args or access via self.CFG if inside the class.
        # Requires CFG values: clustering_se3_trans_weight, clustering_se3_rot_weight
        weighted_dist = np.sqrt(CFG.clustering_se3_trans_weight * trans_dist_sq + 
                                CFG.clustering_se3_rot_weight * rot_dist_sq)
        return weighted_dist

    def _cluster_feature_dataset(self, feature_data: List[np.ndarray], initial_epsilon: float, feature_name: str) -> Tuple[np.ndarray, np.ndarray, Set[int], float]:
        """Performs clustering based on epsilon distance.
        Uses SE(3) metric for 'pose' features, Euclidean otherwise.

        Returns the data array, cluster labels for each point, the set of unique labels,
        and the effective epsilon used for clustering.
        """
        if not feature_data:
            return np.array([]), np.array([]), set(), initial_epsilon

        data_array = np.array(feature_data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        
        # Handle case with 0 or 1 data point early
        if data_array.shape[0] < 2:
            labels = np.array([0]) if data_array.shape[0] == 1 else np.array([])
            unique_labels = {0} if data_array.shape[0] == 1 else set()
            return data_array, labels, unique_labels, initial_epsilon

        # --- Determine Metric and Epsilon ---

        if feature_name == "pose":
            # Use the SE(3) distance function as the metric
            # Define a lambda or wrapper if needed to pass weights, assuming CFG accessible
            metric = lambda p1, p2: self._calculate_se3_distance(p1, p2, 
                                                            CFG.clustering_se3_trans_weight, 
                                                            CFG.clustering_se3_rot_weight)
            # Use a specific epsilon for SE(3) clustering
            effective_epsilon = CFG.clustering_se3_epsilon # Needs to be defined in CFG
            logging.debug(f"Using SE(3) metric with epsilon: {effective_epsilon:.4f}")
        else:
            raise ValueError(f"Unsupported feature type: {feature_name}")
            # effective_epsilon = initial_epsilon # Start with provided/feature-specific epsilon

            # # For non-pose features, potentially keep dynamic epsilon logic?
            # # Or just use the passed initial_epsilon which might be feature-specific
            # # Let's use the initial_epsilon passed (e.g., translation or quaternion specific)
            # logging.debug(f"Using Euclidean metric with epsilon: {effective_epsilon:.4f} for {feature_name}")
            # Optional: Re-enable dynamic epsilon calculation for non-pose Euclidean cases if desired
            # if CFG.clustering_algorithm != "dbscan": # Only relevant for Agglomerative
            #     try:
            #         pairwise_distances = pdist(data_array)
            #         dynamic_epsilon = np.percentile(pairwise_distances, 95) * CFG.clustering_agglomerative_ratio
            #         effective_epsilon = dynamic_epsilon
            #         logging.debug(f"Using Dynamic Agglomerative Epsilon: {effective_epsilon:.4f}")
            #     except ValueError as e:
            #         logging.warning(f"Could not calculate dynamic epsilon: {e}")


        # --- Perform Clustering ---
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
            # try:
                 # Use the effective_epsilon determined earlier (SE3 specific or feature specific)
                 # Agglomerative needs precomputed distances if metric is not standard Euclidean/etc.
                 # Or, if metric is callable AND linkage is 'average', 'complete', 'single', it *might* work directly.
                 # Let's try passing the callable metric directly first.
            logging.debug(f"Running Agglomerative Clustering with distance_threshold: {effective_epsilon:.4f} and metric: {'SE(3)' if callable(metric) else metric}")
            dists = pdist(data_array, metric=metric)
            dist_matrix = squareform(dists)
            clustering = AgglomerativeClustering(n_clusters=None,
                                                affinity="precomputed", # Pass metric
                                                linkage='single', # Check compatibility with custom metric
                                                distance_threshold=effective_epsilon).fit(dist_matrix)
            # except ValueError as e:
            #      # If the callable metric doesn't work directly with chosen linkage:
            #      if callable(metric) and "Metric 'function' not valid for linkage" in str(e):
            #           logging.warning(f"Callable metric not directly supported for linkage 'average'. Precomputing distance matrix...")
            #           try:
            #                distance_matrix = pdist(data_array, metric=metric)
            #                # Convert to squareform for AgglomerativeClustering
            #                distance_matrix_sq = squareform(distance_matrix)
            #                logging.debug(f"Precomputed distance matrix shape: {distance_matrix_sq.shape}")
            #                # Use 'precomputed' metric with the distance matrix
            #                clustering = AgglomerativeClustering(n_clusters=None,
            #                                                     metric='precomputed', # Use precomputed
            #                                                     linkage='average', # Linkage works with precomputed
            #                                                     distance_threshold=effective_epsilon).fit(distance_matrix_sq) # Fit the matrix
            #           except Exception as precompute_e:
            #                logging.error(f"Failed to cluster using precomputed SE(3) distances: {precompute_e}")
            #                raise precompute_e
            #      else:
            #          logging.error(f"Agglomerative Clustering failed: {e}")
            #          logging.error(f"Data shape: {data_array.shape}, Epsilon: {effective_epsilon}, Metric: {'SE(3)' if callable(metric) else metric}")
            #          logging.error(f"Sample data point: {data_array[0] if len(data_array) > 0 else 'N/A'}")
            #          raise e

        labels = clustering.labels_
        unique_labels = set(labels)

        return data_array, labels, unique_labels, effective_epsilon

    def _plot_cluster_results(self,
                              data_array: np.ndarray,
                              labels: np.ndarray,
                              unique_labels: Set[int],
                              kept_clusters_info: Dict[int, Dict],
                              type1_name: str,
                              type2_name: Optional[str], # None for absolute features
                              feat_name: str) -> None:
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
            cluster_type_str = f"Relative Cluster: {type1_name}-{type2_name}"
            fname_prefix = f"rel_cluster_{type1_name}_{type2_name}"
        else:
            cluster_type_str = f"Absolute Cluster: {type1_name}"
            fname_prefix = f"abs_cluster_{type1_name}"

        num_total_clusters = len(unique_labels - {-1})
        num_kept_clusters = len(kept_clusters_info)

        fig = plt.figure(figsize=(15, 12))
        title = (f"{cluster_type_str} ({feat_name})\n"
                 f"MinRatio={CFG.clustering_min_ratio_of_data}, Kept={num_kept_clusters}/{num_total_clusters}")
        fname = f"{fname_prefix}_{feat_name}_clusters.png"
        
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


        # --- Plot Centroids and Boundaries/Frames ---
        centroids_plotted = False
        boundaries_plotted = False
        frames_plotted = False # Track if frames legend is added

        if ax is not None: 
            cluster_counts = defaultdict(int)
            for label in labels:
                cluster_counts[label] += 1

            for label, info in kept_clusters_info.items():
                cluster_color = kept_color_map[label] # Get the assigned color for this kept cluster
                centroid = info['center']
                count = cluster_counts.get(label, 0)
                label_text = f"Cluster {label}: {count} pts"

                # --- Plot Centroid Marker ---
                marker_kwargs = {'color': cluster_color, 's': 150, 'marker': '*'} # Use color argument
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
                    
                    rot_mat = Rotation.from_quat(centroid_quat).as_matrix()
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
                    # except Exception as e:
                    #      logging.warning(f"Could not plot coordinate frame for cluster {label}: {e}")

                elif 'covariance_matrix' in info and 'mahalanobis_threshold' in info:
                     # Plot ellipsoidal boundary (original logic for non-pose features)
                     # ... (keep original ellipsoid plotting logic here) ...
                     # Ensure you set boundaries_plotted = True if ellipsoid is drawn
                     # (Code omitted for brevity, but it's the same as before)
                     cov_matrix = info['covariance_matrix']
                     maha_thresh = info['mahalanobis_threshold']
                     legend_label = 'Ellipsoid Boundary' if not boundaries_plotted else ""
                     boundary_color = cluster_color # Use cluster color for boundary
                     logging.debug(f"Plotting ellipsoid for Cluster {label}: Centroid={centroid}, MahaThresh={maha_thresh:.4f}")
                     try:
                         # --- Ellipsoid Plotting Logic (copied from original) ---
                         if num_dims == 1:
                             variance = max(cov_matrix[0, 0], 1e-9)
                             std_dev = np.sqrt(variance)
                             radius = std_dev * np.sqrt(maha_thresh)
                             ax.plot([centroid[0] - radius, centroid[0] + radius], [0, 0], color=boundary_color, linestyle='--', alpha=0.6, label=legend_label)
                         elif num_dims == 2:
                             eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                             eigenvalues = np.maximum(eigenvalues, 1e-9)
                             order = eigenvalues.argsort()[::-1]
                             eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                             width, height = 2 * np.sqrt(maha_thresh * eigenvalues)
                             angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                             ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                                               edgecolor=boundary_color, fc='None', ls='--', alpha=0.6, label=legend_label)
                             ax.add_patch(ellipse)
                         elif num_dims >= 3 and is_3d: # Non-pose 3D
                             cov_3d = cov_matrix[:3, :3]
                             centroid_3d = centroid[:3]
                             eigenvalues, eigenvectors = np.linalg.eigh(cov_3d)
                             eigenvalues = np.maximum(eigenvalues, 1e-9)
                             radii = np.sqrt(maha_thresh * eigenvalues)
                             u = np.linspace(0.0, 2.0 * np.pi, 100)
                             v = np.linspace(0.0, np.pi, 50)
                             x = np.outer(np.cos(u), np.sin(v))
                             y = np.outer(np.sin(u), np.sin(v))
                             z = np.outer(np.ones_like(u), np.cos(v))
                             points = np.stack((x.flatten(), y.flatten(), z.flatten()))
                             scaled_rotated_points = eigenvectors @ np.diag(radii) @ points
                             translated_points = scaled_rotated_points + centroid_3d[:, np.newaxis]
                             x_ell, y_ell, z_ell = translated_points.reshape(3, *x.shape)
                             ax.plot_wireframe(x_ell, y_ell, z_ell, color=boundary_color, alpha=0.2, rstride=4, cstride=4, label=legend_label)
                         # --- End Ellipsoid Plotting Logic ---
                         boundaries_plotted = True
                     except ValueError as e:
                         logging.warning(f"Could not plot ellipsoid for cluster {label}: Value error ({e}). Check eigenvalues/vectors.")
                else:
                     logging.debug(f"Skipping boundary/frame plot for cluster {label}: Missing info.")


            # --- Finalize Plot ---
            ax.set_title(title)

            # --- Calculate and Store Axis Limits ---
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self._last_cluster_xlim = xlim
            self._last_cluster_ylim = ylim
            if is_3d:
                zlim = ax.get_zlim()
                self._last_cluster_zlim = zlim
            else:
                 self._last_cluster_zlim = None # Ensure it's reset for non-3D plots

            # Create legend handles
            handles = []
            # Create a single handle for all kept points using a generic marker
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Kept Cluster Pts', markersize=10, markerfacecolor='gray'))
            # Handle for discarded points (updated color)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Discarded Cluster Pts', markersize=10, markerfacecolor='lightgrey'))

            if -1 in unique_labels:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise Pts', markersize=10, markerfacecolor='black'))
            # Handle for centroids (generic marker)
            if centroids_plotted:
                handles.append(plt.Line2D([0], [0], marker='*', color='w', label='Kept Centroids', markersize=10, markerfacecolor='purple', linestyle='None'))
            # Handle for boundaries (generic color)
            if boundaries_plotted: # Ellipsoid legend
                handles.append(plt.Line2D([0], [0], linestyle='--', color='gray', label='Ellipsoid Boundary (Maha. Thresh.)'))
            if frames_plotted: # Add legend entries for frames if any were plotted
                 handles.append(plt.Line2D([0],[0], color='r', lw=2, label='Centroid Frame X'))
                 handles.append(plt.Line2D([0],[0], color='g', lw=2, label='Centroid Frame Y'))
                 handles.append(plt.Line2D([0],[0], color='b', lw=2, label='Centroid Frame Z'))
            if feat_name == "pose" and is_3d: # Origin marker for pose
                 handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Origin (Frame 1)', markersize=10, markerfacecolor='blue', linestyle='None'))

            ax.legend(handles=handles)
            # No longer setting axis limits or view init here as it's done above

            # Comment out saving/closing logic if overlay is pending
            # plt.tight_layout()
            # os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
            # plt.savefig(fname)
            # logging.info(f"Cluster visualization saved to feature_data/{fname}")
            # plt.close(fig)
            # self._last_cluster_fig = None
            # self._last_cluster_ax = None
        else:
            # Just store the title for later finalization
            self._last_cluster_title = title

    def _create_predicate_from_relative_cluster(self, type1: Type, type2: Type, feature_name: str, cluster_center: np.ndarray, inv_covariance_matrix: np.ndarray, mahalanobis_threshold: float, diff_fn: Optional[Callable], cluster_id: int) -> Predicate:
        """Creates a binary predicate from a relative feature cluster (including pose)."""
        classifier = _RelativeFeatureClusterClassifier(type1, type2, feature_name, 
                                                     cluster_center, inv_covariance_matrix, 
                                                     mahalanobis_threshold, diff_fn, cluster_id)
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

    def _plot_relative_trajectories(self, dataset: Dataset, type1_name: str, type2_name: str) -> None:
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
            fname = self._last_cluster_fname.replace("clusters", "clusters_with_trajectories")
        else:
            # Create a new figure
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"Trajectories of {type2_name} in {type1_name}'s reference frame")
            ax.set_xlabel('X relative')
            ax.set_ylabel('Y relative')
            ax.set_zlabel('Z relative')
            is_overlay = False
            fname = f"rel_traj_{type1_name}_{type2_name}.png"
        
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
                rel_pose = self._calculate_relative_pose(state, obj1, obj2, 
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

        # Set view angle (consistent for both new and overlaid plots)
        ax.view_init(elev=20., azim=-35) # Example view angle
        
        # Add legend with a good location
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # If we're overlaying, use the stored title from cluster visualization
        if is_overlay and hasattr(self, '_last_cluster_title'):
            ax.set_title(f"{self._last_cluster_title}\nWith Object Trajectories")
        
        # Save the visualization
        os.makedirs("feature_data", exist_ok=True)
        plt.tight_layout()
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
                from scipy.spatial.transform import Rotation
                center_rot = Rotation.from_quat(center_quat)
                noise_rot = Rotation.from_quat(noise_quat)
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
        data_array, labels, unique_labels, effective_epsilon = self._cluster_feature_dataset(
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
