"""An approach that invents predicates by clustering features and selecting via
beam search."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
from itertools import combinations_with_replacement, product

import numpy as np
from gym.spaces import Box
# We will use Agglomerative Clustering as described.
# May need `pip install scikit-learn`
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist
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
    between two objects to a target cluster center and covariance.

    The features are defined by feature_name for objects of type1 and type2.
    The diff_fn calculates the difference between the features of the two objects.
    The cluster_center is the representative point for this cluster.
    Classification is True if the Mahalanobis distance squared is less than or
    equal to mahalanobis_threshold.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str
    cluster_center: np.ndarray
    inv_covariance_matrix: np.ndarray # Store inverse covariance
    mahalanobis_threshold: float    # Store threshold for Mahalanobis distance squared
    diff_fn: Callable[[Any, Any], Any] # Function to compute feature difference
    cluster_id: int # For unique naming

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        # Ensure objects match the types this classifier is defined for.
        # Allow parent type matching.
        assert obj1.is_instance(self.object1_type)
        assert obj2.is_instance(self.object2_type)

        # Get features from state.
        obj1_feat = s.get(obj1, self.feature_name)
        obj2_feat = s.get(obj2, self.feature_name)

        # Assumed orientation feature name
        quat_feat_name = "quaternion"

        # Compute the relative feature value.
        if self.feature_name == "translation" and quat_feat_name in obj1.type.feature_names:
            try:
                obj1_quat = s.get(obj1, quat_feat_name)
                obj1_rot = Rotation.from_quat(obj1_quat)
                world_diff = np.subtract(obj2_feat, obj1_feat)
                relative_feature = obj1_rot.inv().apply(world_diff)
            except KeyError:
                # If quaternion is missing, cannot compute local frame translation.
                # Behavior depends on desired handling: either raise error or return False.
                logging.warning(f"Missing quaternion for {obj1}, cannot compute relative translation.")
                return False
        else:
            # Use the provided difference function for other features.
            relative_feature = np.array(self.diff_fn(obj1_feat, obj2_feat), dtype=self.cluster_center.dtype)

        # Calculate Mahalanobis distance squared
        diff = relative_feature - self.cluster_center
        try:
            # Ensure diff is a column vector for matrix multiplication if it's 1D
            if diff.ndim == 1:
                diff = diff[:, np.newaxis]
            # Mahalanobis distance squared: (x - mu)^T * Sigma^-1 * (x - mu)
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
        # Generate a unique name based on types, feature, and cluster ID.
        return (f"RelEllipsoidCluster-{self.object1_type.name}-{self.object2_type.name}-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description.
        name1 = CFG.grammar_search_classifier_pretty_str_names[0]
        name2 = CFG.grammar_search_classifier_pretty_str_names[1]
        vars_str = f"{name1}:{self.object1_type.name}, {name2}:{self.object2_type.name}"
        # Representing the Mahalanobis check symbolically
        body_str = (f"MahaDistSq(Diff({name1}.{self.feature_name}, {name2}.{self.feature_name}), "
                    f"Cluster-{self.feature_name}-ID{self.cluster_id}) <= {self.mahalanobis_threshold:.3f}")
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
        keep_indices = [0, 3, 4, 5, 7, 8]
        dataset._trajectories = [dataset._trajectories[i] for i in keep_indices]
            
        logging.info(f"Filtered dataset to trajectories (indices: {keep_indices})")
        # Clear caches before starting learning
        self._atom_dataset_cache = {}
        self._operator_complexity_cache = {}
        self._segmentation_cache = {}
        self._plan_constraint_cache = {}

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

        # Save the final approach components (including NSRTs with learned predicates)
        save_path = utils.get_approach_save_path_str()
        self._save(save_path, online_learning_cycle=None)


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

        # Process relative features
        for (type1, type2, feat_name), data in relative_feature_datasets.items():
            logging.debug(f"Clustering relative feature {feat_name} for ({type1.name}, {type2.name}) with {len(data)} points.")
            # Skip if we are having gripper type and handle type
            # if ((type1.name == "left_finger_type" and type2.name == "right_finger_type") or \
            #    (type1.name == "right_finger_type" and type2.name == "left_finger_type")) and \
            #    feat_name == quat_feat_name:
            #     logging.debug(f"Skipping relative feature {feat_name} for ({type1.name}, {type2.name}) due to rand jumping between fingers.")
            #     continue
            # # Skip translation features involving fingers with any other object type
            # # But keep the relationship between left and right fingers
            # if (type1.name == "left_finger_type" or type2.name == "right_finger_type" or\
            #     type1.name == "right_finger_type" or type2.name == "left_finger_type") and \
            #     not (type1.name == "left_finger_type" and type2.name == "right_finger_type" and feat_name == trans_feat_name):
            #     logging.debug(f"Skipping relative feature {feat_name} for ({type1.name}, {type2.name}):All finger removed except distance between fingers")
            #     continue

            # Debug mode: Skip everything except specific feature combinations
            # Only keep:
            # 1. Cabinet handle quaternion (with cabinet first)
            # 2. Gripper handle quaternion (with gripper first)
            # 3. Gripper handle translation (with gripper first)
            # 4. Finger finger translation (with left_finger first)
            
            # Check if this is a feature combination we want to keep
            keep_feature = False

            if (type1.name == "handle_type" and type2.name == "gripper_type") and feat_name == quat_feat_name:
                keep_feature = True
                logging.info(f"Keeping handle-gripper quaternion feature")
            
            if (type1.name == "handle_type" and type2.name == "gripper_type") and feat_name == trans_feat_name:
                keep_feature = True
                logging.info(f"Keeping handle-gripper translation feature")
            
            # Cabinet handle quaternion - keep cabinet first
            if (type1.name == "cabinet_type" and type2.name == "door_type") and feat_name == quat_feat_name:
                keep_feature = True
                logging.info(f"Keeping cabinet-handle quaternion feature")
            
            # Gripper handle quaternion - keep gripper first
            elif (type1.name == "gripper_type" and type2.name == "door_type") and feat_name == quat_feat_name:
                keep_feature = True
                logging.info(f"Keeping gripper-handle quaternion feature")
            
            # Gripper handle translation - keep gripper first
            elif (type1.name == "gripper_type" and type2.name == "door_type") and feat_name == trans_feat_name:
                keep_feature = True
                logging.info(f"Keeping gripper-handle translation feature")
            
            # Finger finger translation - keep left_finger first
            elif (type1.name == "left_finger_type" and type2.name == "right_finger_type") and feat_name == trans_feat_name:
                keep_feature = True
                logging.info(f"Keeping finger-finger translation feature")
            
            # Skip all other feature combinations
            if not keep_feature:
                logging.debug(f"Skipping feature {feat_name} for ({type1.name}, {type2.name}) in debug mode")
                continue

            
            if not data: continue # Skip if no data collected

            # Select clustering epsilon based on feature type
            if feat_name == trans_feat_name:
                epsilon = CFG.clustering_translation_epsilon
            elif feat_name == quat_feat_name:
                epsilon = CFG.clustering_quaternion_epsilon
            else:
                epsilon = CFG.clustering_epsilon

            logging.debug(f"Using epsilon: {epsilon:.4f} for feature {feat_name}")
            # Perform clustering
            data_array, labels, unique_labels, effective_epsilon = self._cluster_feature_dataset(data, epsilon)
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

        # Process absolute features 
        # skip absolute features for now since all things are relative
        # for (type1, feat_name), data in absolute_feature_datasets.items():
        #     logging.debug(f"Clustering absolute feature {feat_name} for ({type1.name}) with {len(data)} points.")
        #     # Skip absolute orientation features for fingers as they can be noisy
        #     if (type1.name == "left_finger_type" or type1.name == "right_finger_type") and feat_name == "quaternion":
        #         logging.debug(f"Skipping absolute feature {feat_name} for {type1.name} due to noisy finger orientation.")
        #         continue

        #     if not data: continue

        #     # Select clustering epsilon based on feature type (using defaults if not trans/quat)
        #     if feat_name == trans_feat_name:
        #         epsilon = CFG.clustering_translation_epsilon
        #     elif feat_name == quat_feat_name:
        #         epsilon = CFG.clustering_quaternion_epsilon
        #     else:
        #         epsilon = CFG.clustering_epsilon

        #     logging.debug(f"Using epsilon: {epsilon:.2f} for feature {feat_name}")
        #     clusters = self._cluster_feature_dataset(data, epsilon)
        #     for cluster_id, cluster_info in enumerate(clusters):
        #          logging.debug(f"Cluster {cluster_id} has {len(cluster_info['points'])} points.")
        #          if len(cluster_info['points']) < CFG.clustering_min_samples_per_cluster:
        #              continue
        #          # Use the feature-specific epsilon when creating the predicate
        #          pred = self._create_predicate_from_absolute_cluster(type1, feat_name, cluster_info, epsilon, predicate_counter)
        #          candidates[pred] = pred.arity + 1.0
        #          predicate_counter += 1

        # Rename predicates for PDDL compatibility (reuse from grammar search)
        renamed_candidates = self._rename_predicates_to_remove_incompatible_chars(candidates)
        return renamed_candidates

    def _generate_relative_feature_datasets(self, dataset: Dataset) -> Dict[Tuple[Type, Type, str], List[np.ndarray]]:
        """Extracts relative features constant between consecutive states."""
        feature_data = defaultdict(list)
        feature_changes = defaultdict(list)

        # Corrected approach: Iterate through objects in the initial state and get their types.
        types = {obj.type for traj in dataset.trajectories for obj in traj.states[0]}

        # Use product to get ordered pairs, ensuring both (A, B) and (B, A) are considered.
        # This allows calculating relative features in both A's frame and B's frame.


        ##!!! Keep depending on 
        type_pairs = list(product(sorted(list(types)), repeat=2))
        # type_pairs = list(combinations_with_replacement(sorted(list(types)), 2))

        # Assumed orientation feature name
        quat_feat_name = "quaternion"
        # Assumed translation feature name
        trans_feat_name = "translation"

        for traj in dataset.trajectories:
            for t in range(len(traj.states) - 1):
                state_t = traj.states[t]
                state_t1 = traj.states[t+1]
                for type1, type2 in type_pairs:
                    shared_features = sorted(list(set(type1.feature_names) & set(type2.feature_names)))
                    if not shared_features:
                        # warnings.warn(f"No shared features between {type1.name} and {type2.name}. Skipping.")
                        continue

                    objs1 = list(state_t.get_objects(type1))
                    objs2 = list(state_t.get_objects(type2))

                    for feat_name in shared_features:
                        # Check if we need obj1's orientation (only for translation)
                        needs_quat_for_trans = (feat_name == trans_feat_name and quat_feat_name in type1.feature_names)

                        for o1 in objs1:
                            # If types are the same, avoid comparing object to itself.
                            obj2_list = objs2 if type1 != type2 else [o for o in objs2 if o != o1]
                            for o2 in obj2_list:
                                try:
                                    # Get features at time t
                                    feat_t_o1 = state_t.get(o1, feat_name)
                                    feat_t_o2 = state_t.get(o2, feat_name)
                                    # Get features at time t+1
                                    feat_t1_o1 = state_t1.get(o1, feat_name)
                                    feat_t1_o2 = state_t1.get(o2, feat_name)

                                    # Compute relative feature at time t
                                    if feat_name == trans_feat_name and needs_quat_for_trans:
                                        quat_t_o1 = state_t.get(o1, quat_feat_name)
                                        rot_t_o1 = Rotation.from_quat(quat_t_o1)
                                        diff_t_world = np.subtract(feat_t_o2, feat_t_o1)
                                        rel_feat_t = rot_t_o1.inv().apply(diff_t_world)
                                    else:
                                        diff_fn = self._get_feature_difference_function(feat_name)
                                        rel_feat_t = np.array(diff_fn(feat_t_o1, feat_t_o2))

                                    # Compute relative feature at time t+1
                                    if feat_name == trans_feat_name and needs_quat_for_trans:
                                        quat_t1_o1 = state_t1.get(o1, quat_feat_name)
                                        rot_t1_o1 = Rotation.from_quat(quat_t1_o1)
                                        diff_t1_world = np.subtract(feat_t1_o2, feat_t1_o1)
                                        rel_feat_t1 = rot_t1_o1.inv().apply(diff_t1_world)
                                    else:
                                        # Recompute diff_fn for t+1 in case it's state-dependent (though unlikely here)
                                        diff_fn = self._get_feature_difference_function(feat_name)
                                        rel_feat_t1 = np.array(diff_fn(feat_t1_o1, feat_t1_o2))

                                    # Ensure numpy arrays for norm calculation
                                    rel_feat_t = np.array(rel_feat_t)
                                    rel_feat_t1 = np.array(rel_feat_t1)

                                    # Check for constancy using the feature-specific tolerance
                                    # if np.linalg.norm(rel_feat_t - rel_feat_t1) < tolerance:
                                    feature_data[(type1, type2, feat_name)].append(rel_feat_t)
                                    feature_changes[(type1, type2, feat_name)].append(np.linalg.norm(rel_feat_t - rel_feat_t1))
                                except KeyError as e:
                                     # If a required feature (like quaternion for translation) is missing, skip this pair
                                     # logging.debug(f"Skipping object pair due to missing feature: {e}")
                                     continue
        for feature_key, changes in feature_changes.items():
            # Only keep features that are relatively constant (below 30th percentile of changes)
            # First collect all changes, then filter based on percentile
            # This is done outside the loop to avoid modifying the dictionary during iteration
            percentile_30 = np.percentile(changes, 30)
            bool_mask = changes < percentile_30            
            feature_data[feature_key] = [feat for feat, is_constant in zip(feature_data[feature_key], bool_mask) if is_constant]
            

        return feature_data


    def _generate_absolute_feature_datasets(self, dataset: Dataset) -> Dict[Tuple[Type, str], List[np.ndarray]]:
        """Extracts absolute features constant between consecutive states."""
        feature_data = defaultdict(list)
        types = {obj.type for traj in dataset.trajectories for obj in traj.states[0]}

        for traj in dataset.trajectories:
            for t in range(len(traj.states) - 1):
                state_t = traj.states[t]
                state_t1 = traj.states[t+1]
                for type1 in types:
                     # Consider object might not have features?
                    #  if not hasattr(type1, 'feature_names'): continue
                     objs1 = list(state_t.get_objects(type1))
                     for feat_name in sorted(type1.feature_names):
                          for o1 in objs1:
                               # Check if object exists in the next state
                            #    if o1 not in state_t1:
                            #        continue
                               # Get features at time t and t+1
                               feat_t = np.array(state_t.get(o1, feat_name))
                               feat_t1 = np.array(state_t1.get(o1, feat_name))

                               # Check for constancy
                               if np.linalg.norm(feat_t - feat_t1) < CFG.clustering_feature_constancy_tol:
                                    feature_data[(type1, feat_name)].append(feat_t)
        return feature_data

    def _get_feature_difference_function(self, feature_name: str) -> Callable[[Any, Any], Any]:
        """Returns an appropriate difference function for a given feature."""
        # TODO: Implement more sophisticated difference functions, especially for orientation.
        # This basic version assumes vector subtraction works.
        if "quaternion" == feature_name:
             # For quaternions/rotations, relative rotation is often more meaningful
             def _quat_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
                  # Calculate relative rotation: q_rel = q1.inverse * q2
                  # Return the rotation vector representation of the relative rotation.
                  rot1 = Rotation.from_quat(q1)
                  rot2 = Rotation.from_quat(q2)
                  rel_rot = rot1.inv() * rot2
                  rot_vec = rel_rot.as_rotvec()
                  # Return the 3D rotation vector.
                  return rot_vec
             return _quat_diff
        # Default to simple subtraction for position, velocity, etc.
        return np.subtract


    def _cluster_feature_dataset(self, feature_data: List[np.ndarray], initial_epsilon: float) -> Tuple[np.ndarray, np.ndarray, Set[int], float]:
        """Performs clustering based on epsilon distance using either agglomerative or DBSCAN.

        Returns the data array, cluster labels for each point, the set of unique labels,
        and the effective epsilon used for clustering.
        """
        if not feature_data:
            # Return empty structures and the initial epsilon if no data
            return np.array([]), np.array([]), set(), initial_epsilon

        data_array = np.array(feature_data)
        if data_array.ndim == 1: # Handle scalar features by adding a dimension
            data_array = data_array.reshape(-1, 1)

        # Handle case with 0 or 1 data point early to avoid errors in pdist/clustering
        if data_array.shape[0] < 2:
             labels = np.array([0]) if data_array.shape[0] == 1 else np.array([])
             unique_labels = {0} if data_array.shape[0] == 1 else set()
             # Return initial epsilon if clustering wasn't really performed
             return data_array, labels, unique_labels, initial_epsilon


        # Choose clustering algorithm based on configuration
        effective_epsilon = initial_epsilon # Initialize with the provided epsilon
        if CFG.clustering_algorithm == "dbscan":
            try:
                # Use a ratio of the initial epsilon for DBSCAN
                effective_epsilon = initial_epsilon * CFG.clustering_dbscan_ratio
                logging.debug(f"Using DBSCAN Epsilon: {effective_epsilon:.4f}")
                clustering = DBSCAN(eps=effective_epsilon).fit(data_array)
            except ValueError as e:
                logging.error(f"DBSCAN Clustering failed: {e}")
                logging.error(f"Data shape: {data_array.shape}, Epsilon: {effective_epsilon}")
                logging.error(f"Example data point: {data_array[0] if len(data_array) > 0 else 'N/A'}")
                raise e
        else:  # Default to agglomerative clustering
            try:
                # Dynamically set epsilon based on data range
                pairwise_distances = pdist(data_array)
                # Get the 95th percentile of distances
                dynamic_epsilon = np.percentile(pairwise_distances, 95) * CFG.clustering_agglomerative_ratio
                effective_epsilon = dynamic_epsilon # Use the dynamically calculated one
                logging.debug(f"Using Agglomerative Clustering Epsilon: {effective_epsilon:.4f}")


                # linkage='average' corresponds well to the paper's description
                # distance_threshold ensures clusters stop merging when distance exceeds epsilon
                clustering = AgglomerativeClustering(n_clusters=None,
                                                    affinity='euclidean',
                                                    linkage='average', # Or 'complete', 'ward'
                                                    distance_threshold=effective_epsilon).fit(data_array)
            except ValueError as e:
                logging.error(f"Agglomerative Clustering failed: {e}")
                logging.error(f"Data shape: {data_array.shape}, Epsilon: {effective_epsilon}")
                logging.error(f"Example data point: {data_array[0] if len(data_array) > 0 else 'N/A'}")
                raise e

        labels = clustering.labels_
        unique_labels = set(labels)

        # Return raw results including the effective epsilon used
        return data_array, labels, unique_labels, effective_epsilon


    def _plot_cluster_results(self,
                              data_array: np.ndarray,
                              labels: np.ndarray,
                              unique_labels: Set[int],
                              kept_clusters_info: Dict[int, Dict],
                              type1_name: str,
                              type2_name: Optional[str], # None for absolute features
                              feat_name: str) -> None:
        """Helper function to visualize clustering results with ellipsoidal boundaries."""
        if not CFG.clustering_debug or data_array.size == 0:
            return # Skip if debug flag is off or no data

        # We need matplotlib, etc. Already checked in the calling function.
        # from matplotlib.patches import Ellipse is needed for 2D.

        # Determine if relative or absolute for titles/filenames
        if type2_name:
            cluster_type_str = f"Relative Cluster: {type1_name}-{type2_name}"
            fname_prefix = f"rel_cluster_{type1_name}_{type2_name}"
        else:
            cluster_type_str = f"Absolute Cluster: {type1_name}"
            fname_prefix = f"abs_cluster_{type1_name}"

        # Calculate stats for title
        num_total_clusters = len(unique_labels - {-1}) # Exclude noise label if present
        num_kept_clusters = len(kept_clusters_info)

        fig = plt.figure(figsize=(12, 10))
        # Title now reflects Mahalanobis usage, removed effective_epsilon
        title = (f"{cluster_type_str} ({feat_name})\n"
                 f"MinRatio={CFG.clustering_min_ratio_of_data}, Kept={num_kept_clusters}/{num_total_clusters}")
        # Filename doesn't need epsilon anymore
        fname = f"{fname_prefix}_{feat_name}_clusters.png"

        # Determine colors: green for kept, red for discarded, black for noise
        colors = []
        for label in labels:
            if label == -1:
                colors.append('black') # Noise
            elif label in kept_clusters_info:
                colors.append('green') # Kept
            else:
                colors.append('red')   # Discarded

        num_dims = data_array.shape[1]
        ax = None # Initialize ax

        if num_dims == 1:
            ax = fig.add_subplot(111)
            ax.scatter(data_array[:, 0], np.zeros_like(data_array[:, 0]), c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
        elif num_dims == 2:
            ax = fig.add_subplot(111)
            ax.scatter(data_array[:, 0], data_array[:, 1], c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
            ax.set_ylabel(f'{feat_name} dim 2')
        elif num_dims >= 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=colors, alpha=0.7)
            ax.set_xlabel(f'{feat_name} dim 1')
            ax.set_ylabel(f'{feat_name} dim 2')
            ax.set_zlabel(f'{feat_name} dim 3')
            if feat_name == "translation":
                ax.scatter([0], [0], [0], c='blue', s=100, marker='x', label='Origin')

        # Plot centroids and ellipsoidal boundaries for *kept* clusters
        centroids_plotted = False
        boundaries_plotted = False
        if ax is not None: # Ensure ax was created
            cluster_counts = defaultdict(int)
            for label in labels:
                cluster_counts[label] += 1

            for label, info in kept_clusters_info.items():
                centroid = info['center']
                count = cluster_counts.get(label, 0)
                label_text = f"Cluster {label}: {count} pts"

                # Plot centroid marker
                marker_kwargs = {'c': 'purple', 's': 150, 'marker': '*'} # Removed label here
                if not centroids_plotted:
                     marker_kwargs['label'] = 'Kept Centroids' # Add label only for the first one

                if num_dims == 1:
                    ax.scatter(centroid[0], 0, **marker_kwargs)
                    ax.text(centroid[0], 0.01, label_text, fontsize=9) # Slightly offset text
                elif num_dims == 2:
                    ax.scatter(centroid[0], centroid[1], **marker_kwargs)
                    ax.text(centroid[0], centroid[1], label_text, fontsize=9)
                elif num_dims >= 3:
                    ax.scatter(centroid[0], centroid[1], centroid[2], **marker_kwargs)
                    ax.text(centroid[0], centroid[1], centroid[2], label_text, fontsize=9)
                centroids_plotted = True

                # Plot Ellipsoidal Boundary
                if 'covariance_matrix' in info and 'mahalanobis_threshold' in info:
                    cov_matrix = info['covariance_matrix']
                    maha_thresh = info['mahalanobis_threshold']
                    legend_label = 'Ellipsoid Boundary' if not boundaries_plotted else ""
                    logging.debug(f"Plotting ellipsoid for Cluster {label}: Centroid={centroid}, MahaThresh={maha_thresh:.4f}")
                    # logging.debug(f"Covariance Matrix:\n{cov_matrix}") # Optional: uncomment for detailed matrix view

                    try:
                        if num_dims == 1:
                            # Handle potential 0 variance by adding small epsilon
                            variance = max(cov_matrix[0, 0], 1e-9)
                            std_dev = np.sqrt(variance)
                            radius = std_dev * np.sqrt(maha_thresh) # sqrt(thresh * variance)
                            logging.debug(f"  1D Ellipsoid: Radius={radius:.4f}")
                            ax.plot([centroid[0] - radius, centroid[0] + radius], [0, 0], 'k--', alpha=0.6, label=legend_label)
                        elif num_dims == 2:
                            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                            # Clamp small/negative eigenvalues due to numerical issues
                            eigenvalues = np.maximum(eigenvalues, 1e-9)
                            # Order eigenvalues and eigenvectors
                            order = eigenvalues.argsort()[::-1]
                            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                            # Ellipse axes lengths: sqrt(thresh * eigenvalue)
                            width, height = 2 * np.sqrt(maha_thresh * eigenvalues)
                            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                            logging.debug(f"  2D Ellipsoid: Width={width:.4f}, Height={height:.4f}, Angle={angle:.2f}")
                            ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                                              edgecolor='k', fc='None', ls='--', alpha=0.6, label=legend_label)
                            ax.add_patch(ellipse)
                        elif num_dims >= 3:
                            # Use first 3 dimensions for plotting 3D ellipsoid
                            cov_3d = cov_matrix[:3, :3]
                            centroid_3d = centroid[:3]
                            eigenvalues, eigenvectors = np.linalg.eigh(cov_3d)
                            # Clamp small/negative eigenvalues
                            eigenvalues = np.maximum(eigenvalues, 1e-9)
                            # Axes lengths
                            radii = np.sqrt(maha_thresh * eigenvalues)
                            logging.debug(f"  3D Ellipsoid: Radii={radii}")
                            # Generate points on a unit sphere
                            u = np.linspace(0.0, 2.0 * np.pi, 100)
                            v = np.linspace(0.0, np.pi, 50)
                            x = np.outer(np.cos(u), np.sin(v))
                            y = np.outer(np.sin(u), np.sin(v))
                            z = np.outer(np.ones_like(u), np.cos(v))
                            # Scale, rotate, and translate points
                            points = np.stack((x.flatten(), y.flatten(), z.flatten()))
                            scaled_rotated_points = eigenvectors @ np.diag(radii) @ points
                            translated_points = scaled_rotated_points + centroid_3d[:, np.newaxis]
                            # Reshape for plotting
                            x_ell, y_ell, z_ell = translated_points.reshape(3, *x.shape)
                            ax.plot_wireframe(x_ell, y_ell, z_ell, color='k', alpha=0.2, rstride=4, cstride=4, label=legend_label)

                        boundaries_plotted = True

                    except ValueError as e:
                         logging.warning(f"Could not plot ellipsoid for cluster {label}: Value error ({e}). Check eigenvalues/vectors.")
                else:
                     logging.debug(f"Skipping ellipsoid plot for cluster {label}: Missing covariance or threshold info.")
                     logging.debug(f"  Available info keys: {list(info.keys())}")


            ax.set_title(title)
            # Create custom legend handles
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label='Kept Pts', markersize=10, markerfacecolor='green'),
                plt.Line2D([0], [0], marker='o', color='w', label='Discarded Pts', markersize=10, markerfacecolor='red'),
            ]
            if -1 in unique_labels:
                 handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise Pts', markersize=10, markerfacecolor='black'))
            if centroids_plotted:
                handles.append(plt.Line2D([0], [0], marker='*', color='w', label='Kept Centroids', markersize=10, markerfacecolor='purple', linestyle='None'))
            if boundaries_plotted:
                handles.append(plt.Line2D([0], [0], linestyle='--', color='k', label='Ellipsoid Boundary (Maha. Thresh.)'))
            if feat_name == "translation" and num_dims >= 3:
                handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Origin', markersize=10, markerfacecolor='blue', linestyle='None'))

            # Add aspect ratio setting for 2D plots to make ellipses look right
            if num_dims == 2:
                ax.set_aspect('equal', adjustable='box')

            ax.legend(handles=handles)
            plt.tight_layout()
            plt.savefig(fname)
            logging.info(f"Cluster visualization saved to {fname}")
            plt.close(fig) # Close after showing
        else:
            logging.warning(f"Could not plot for {fname}, unsupported dimension: {num_dims}")


    def _create_predicate_from_relative_cluster(self, type1: Type, type2: Type, feature_name: str, cluster_center: np.ndarray, inv_covariance_matrix: np.ndarray, mahalanobis_threshold: float, diff_fn: Callable, cluster_id: int) -> Predicate:
        """Creates a binary predicate from a relative feature cluster."""
        classifier = _RelativeFeatureClusterClassifier(type1, type2, feature_name, cluster_center, inv_covariance_matrix, mahalanobis_threshold, diff_fn, cluster_id)
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
            segment_action_counts = [[len(segment.actions) for segment in demo_segments] 
                                    for demo_segments in segmented_trajs]
            logging.info(f"Segment action counts: \n {segment_action_counts}")
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
        
        min_diff = float('inf')  # Track the minimum difference across all trajectories

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
                
                # Keep track of the minimum difference (most negative)
                # This represents the worst case where the planner found a much shorter path
                if diff < min_diff:
                    min_diff = diff

        # If we didn't process any valid trajectories, return 0
        if min_diff == float('inf'):
            min_diff = 0
            
        self._plan_constraint_cache[predicates] = min_diff
        return min_diff



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