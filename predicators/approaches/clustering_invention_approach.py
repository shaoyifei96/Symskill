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
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import _BinaryClassifier, _ProgrammaticClassifier, _UnaryClassifier
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.planning import PlanningFailure, PlanningTimeout#, task_plan_grounding, run_task_plan
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtomTrajectory, NSRT, Object, ParameterizedOption, Predicate, Segment, State, Task, Type
import warnings
################################################################################
#                          Programmatic classifiers                            #
################################################################################


@dataclass(frozen=True, eq=False, repr=False)
class _RelativeFeatureClusterClassifier(_BinaryClassifier):
    """Classifies based on the minimum distance of a relative feature vector
    between two objects to a target cluster center.

    The features are defined by feature_name for objects of type1 and type2.
    The diff_fn calculates the difference between the features of the two objects.
    The cluster_center is the representative point for this cluster.
    Classification is True if the distance between the diff_fn result and the
    cluster_center is less than or equal to epsilon.
    """
    object1_type: Type
    object2_type: Type
    feature_name: str
    cluster_center: np.ndarray
    epsilon: float
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

        # Calculate the Euclidean distance to the cluster center.
        distance = np.linalg.norm(relative_feature - self.cluster_center)
        return distance <= self.epsilon

    def __str__(self) -> str:
        # Generate a unique name based on types, feature, and cluster ID.
        return (f"RelCluster-{self.object1_type.name}-{self.object2_type.name}-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description.
        name1 = CFG.grammar_search_classifier_pretty_str_names[0]
        name2 = CFG.grammar_search_classifier_pretty_str_names[1]
        vars_str = f"{name1}:{self.object1_type.name}, {name2}:{self.object2_type.name}"
        # Representing the cluster check symbolically is tricky, use descriptive text.
        body_str = (f"Dist(Diff({name1}.{self.feature_name}, {name2}.{self.feature_name}), "
                    f"Cluster-{self.feature_name}-ID{self.cluster_id}) <= {self.epsilon:.3f}")
        return vars_str, body_str


@dataclass(frozen=True, eq=False, repr=False)
class _AbsoluteFeatureClusterClassifier(_UnaryClassifier):
    """Classifies based on the minimum distance of an absolute feature vector
    of an object to a target cluster center.

    The feature is defined by feature_name for an object of type1.
    The cluster_center is the representative point for this cluster.
    Classification is True if the distance between the object's feature and the
    cluster_center is less than or equal to epsilon.
    """
    object_type: Type
    feature_name: str
    cluster_center: np.ndarray
    epsilon: float
    cluster_id: int # For unique naming

    def _classify_object(self, s: State, obj: Object) -> bool:
        # Ensure object matches the type this classifier is defined for.
        assert obj.is_instance(self.object_type)
        # Get feature from state.
        obj_feat = np.array(s.get(obj, self.feature_name), dtype=self.cluster_center.dtype)
        # Calculate the Euclidean distance to the cluster center.
        distance = np.linalg.norm(obj_feat - self.cluster_center)
        return distance <= self.epsilon

    def __str__(self) -> str:
        # Generate a unique name based on type, feature, and cluster ID.
        return (f"AbsCluster-{self.object_type.name}-"
                f"{self.feature_name}-ID{self.cluster_id}")

    def pretty_str(self) -> Tuple[str, str]:
        # Provide a human-readable description.
        name = CFG.grammar_search_classifier_pretty_str_names[0]
        vars_str = f"{name}:{self.object_type.name}"
        body_str = (f"Dist({name}.{self.feature_name}, "
                    f"Cluster-{self.feature_name}-ID{self.cluster_id}) <= {self.epsilon:.3f}")
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
        # Clear caches before starting learning
        self._atom_dataset_cache = {}
        self._operator_complexity_cache = {}
        self._segmentation_cache = {}
        self._plan_constraint_cache = {}

        candidates = self._generate_candidate_predicates(dataset)
        logging.info(f"Generated {len(candidates)} candidate predicates.")
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
        utils.save_to_pickle(self._learned_predicates, learned_preds_path)

        # Learn NSRTs with the final set of predicates
        final_predicates = self._get_current_predicates()
        # We need the atom dataset for the final selected predicates
        atom_dataset_final = self._create_atom_dataset(dataset, final_predicates)
        annotations = None # Or derive from atom_dataset if needed by _learn_nsrts
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None, annotations=annotations, atom_dataset=atom_dataset_final)

        # Save the final approach components (including NSRTs with learned predicates)
        save_path = utils.get_approach_save_path_str()
        self._save(save_path, online_learning_cycle=None)


    # --- Candidate Generation Functions ---
    def _generate_candidate_predicates(self, dataset: Dataset) -> Dict[Predicate, float]:
        """Generates candidate predicates by clustering relative and absolute features."""
        relative_feature_datasets = self._generate_relative_feature_datasets(dataset)
        absolute_feature_datasets = self._generate_absolute_feature_datasets(dataset)

        candidates: Dict[Predicate, float] = {}
        predicate_counter = 0 # To ensure unique cluster IDs

        # Feature names for special handling
        quat_feat_name = "quaternion"
        trans_feat_name = "translation"

        # Process relative features
        for (type1, type2, feat_name), data in relative_feature_datasets.items():
            logging.debug(f"Clustering relative feature {feat_name} for ({type1.name}, {type2.name}) with {len(data)} points.")
            # Skip if we are having gripper type and handle type
            if (type1.name == "gripper_type" and type2.name == "handle_type") or \
               (type1.name == "handle_type" and type2.name == "gripper_type"):
                pass
            if not data: continue # Skip if no data collected

            # Select clustering epsilon based on feature type
            if feat_name == trans_feat_name:
                epsilon = CFG.clustering_translation_epsilon
            elif feat_name == quat_feat_name:
                epsilon = CFG.clustering_quaternion_epsilon
            else:
                epsilon = CFG.clustering_epsilon

            logging.debug(f"Using epsilon: {epsilon:.4f} for feature {feat_name}")
            clusters = self._cluster_feature_dataset(data, epsilon)
            diff_fn = self._get_feature_difference_function(feat_name)
            for cluster_id, cluster_info in enumerate(clusters):
                 logging.debug(f"Cluster {cluster_id} has {len(cluster_info['points'])} points.")
                 # Skip clusters that are too small (optional hyperparameter)
                 if len(cluster_info['points']) < CFG.clustering_min_samples_per_cluster:
                      continue
                 # Use the feature-specific epsilon when creating the predicate
                 pred = self._create_predicate_from_relative_cluster(type1, type2, feat_name, cluster_info, epsilon, diff_fn, predicate_counter)
                 # Cost can be simple (e.g., arity) or more complex
                 candidates[pred] = pred.arity + 1.0
                 predicate_counter += 1

        # Process absolute features
        for (type1, feat_name), data in absolute_feature_datasets.items():
            logging.debug(f"Clustering absolute feature {feat_name} for ({type1.name}) with {len(data)} points.")
            if not data: continue

            # Select clustering epsilon based on feature type (using defaults if not trans/quat)
            if feat_name == trans_feat_name:
                epsilon = CFG.clustering_translation_epsilon
            elif feat_name == quat_feat_name:
                epsilon = CFG.clustering_quaternion_epsilon
            else:
                epsilon = CFG.clustering_epsilon

            logging.debug(f"Using epsilon: {epsilon:.2f} for feature {feat_name}")
            clusters = self._cluster_feature_dataset(data, epsilon)
            for cluster_id, cluster_info in enumerate(clusters):
                 logging.debug(f"Cluster {cluster_id} has {len(cluster_info['points'])} points.")
                 if len(cluster_info['points']) < CFG.clustering_min_samples_per_cluster:
                     continue
                 # Use the feature-specific epsilon when creating the predicate
                 pred = self._create_predicate_from_absolute_cluster(type1, feat_name, cluster_info, epsilon, predicate_counter)
                 candidates[pred] = pred.arity + 1.0
                 predicate_counter += 1

        # Rename predicates for PDDL compatibility (reuse from grammar search)
        renamed_candidates = self._rename_predicates_to_remove_incompatible_chars(candidates)
        return renamed_candidates

    def _generate_relative_feature_datasets(self, dataset: Dataset) -> Dict[Tuple[Type, Type, str], List[np.ndarray]]:
        """Extracts relative features constant between consecutive states."""
        feature_data = defaultdict(list)

        # Corrected approach: Iterate through objects in the initial state and get their types.
        types = {obj.type for traj in dataset.trajectories for obj in traj.states[0]}

        # Use product to get ordered pairs, ensuring both (A, B) and (B, A) are considered.
        # This allows calculating relative features in both A's frame and B's frame.
        type_pairs = list(product(sorted(list(types)), repeat=2))

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

                        # Select the appropriate tolerance based on feature type
                        if feat_name == trans_feat_name:
                            tolerance = CFG.clustering_translation_constancy_tol
                        elif feat_name == quat_feat_name:
                            tolerance = CFG.clustering_quaternion_constancy_tol
                        else:
                            tolerance = CFG.clustering_feature_constancy_tol

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
                                    if np.linalg.norm(rel_feat_t - rel_feat_t1) < tolerance:
                                        feature_data[(type1, type2, feat_name)].append(rel_feat_t)

                                except KeyError as e:
                                     # If a required feature (like quaternion for translation) is missing, skip this pair
                                     # logging.debug(f"Skipping object pair due to missing feature: {e}")
                                     continue

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


    def _cluster_feature_dataset(self, feature_data: List[np.ndarray], epsilon: float) -> List[Dict[str, Any]]:
        """Performs agglomerative clustering based on epsilon distance."""
        if not feature_data:
            return []

        data_array = np.array(feature_data)
        if data_array.ndim == 1: # Handle scalar features by adding a dimension
            data_array = data_array.reshape(-1, 1)

        # linkage='average' corresponds well to the paper's description
        # distance_threshold ensures clusters stop merging when distance exceeds epsilon
        try:
            clustering = AgglomerativeClustering(n_clusters=None,
                                                 affinity='euclidean',
                                                 linkage='average', # Or 'complete', 'ward'
                                                 distance_threshold=epsilon).fit(data_array)
        except ValueError as e:
            logging.error(f"Agglomerative Clustering failed: {e}")
            logging.error(f"Data shape: {data_array.shape}, Epsilon: {epsilon}")
            # Example problematic data point: data_array[0]
            logging.error(f"Example data point: {data_array[0] if len(data_array) > 0 else 'N/A'}")
            raise e
            # return []

        labels = clustering.labels_
        unique_labels = set(labels)
        cluster_results = []

        # Optionally visualize clusters if in debug mode
        if CFG.clustering_debug :  # Only visualize 1D, 2D, or 3D data
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
                
                fig = plt.figure(figsize=(10, 8))
                
                if data_array.shape[1] == 1:  # 1D data
                    plt.scatter(data_array[:, 0], np.zeros_like(data_array[:, 0]), c=labels, cmap='viridis')
                    plt.title(f'Clustering Results (Epsilon={epsilon:.4f}, Clusters={len(unique_labels)})')
                    plt.xlabel('Feature Value')
                elif data_array.shape[1] == 2:  # 2D data
                    plt.scatter(data_array[:, 0], data_array[:, 1], c=labels, cmap='viridis')
                    plt.title(f'Clustering Results (Epsilon={epsilon:.4f}, Clusters={len(unique_labels)})')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                else:  # 3D data 4D data
                    ax = fig.add_subplot(111, projection='3d')
                    # Mark origin (0,0,0) with red
                    # Plot the actual data points
                    ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=labels, cmap='viridis')
                    ax.scatter([0], [0], [0], c='red', s=100, marker='x')
                    ax.set_title(f'Clustering Results (Epsilon={epsilon:.4f}, Clusters={len(unique_labels)})')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    ax.set_zlabel('Feature 3')
                # Plot the centroids of each cluster
                for k in unique_labels:
                    if k == -1:  # Skip noise points
                        continue
                    
                    # Get points in this cluster and calculate centroid
                    cluster_points = data_array[labels == k]
                    if len(cluster_points) == 0:
                        continue
                        
                    centroid = np.mean(cluster_points, axis=0)
                    
                    # Plot the centroid with a different marker and larger size
                    if data_array.shape[1] == 1:  # 1D data
                        plt.scatter(centroid[0], 0, c='black', s=100, marker='*', 
                                   label=f'Centroid {k}' if k == list(unique_labels)[0] else "")
                    elif data_array.shape[1] == 2:  # 2D data
                        plt.scatter(centroid[0], centroid[1], c='black', s=100, marker='*',
                                   label=f'Centroid {k}' if k == list(unique_labels)[0] else "")
                    else:  # 3D data
                        ax.scatter(centroid[0], centroid[1], centroid[2], c='black', s=100, marker='*',
                                  label=f'Centroid {k}' if k == list(unique_labels)[0] else "")
                
                # Add a legend to identify centroids
                if len(unique_labels) > 0 and -1 not in unique_labels:
                    plt.legend(["Centroids"])
                plt.tight_layout()
                plt.savefig(f'cluster_visualization_eps{epsilon:.4f}.png')
                logging.info(f"Cluster visualization saved to cluster_visualization_eps{epsilon:.4f}.png")
                plt.close()
            except Exception as viz_error:
                logging.warning(f"Failed to visualize clusters: {viz_error}")

        for k in unique_labels:
            if k == -1: continue # Noise points if using algorithms like DBSCAN (not Agglomerative)

            cluster_points = data_array[labels == k]
            if len(cluster_points) == 0: continue

            # Represent cluster by its centroid (mean)
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_results.append({
                'center': cluster_center,
                'points': cluster_points, # Keep points for potential filtering by size
                'label': k  # Store the cluster label for reference
            })

        return cluster_results


    def _create_predicate_from_relative_cluster(self, type1: Type, type2: Type, feature_name: str, cluster_info: Dict, epsilon: float, diff_fn: Callable, cluster_id: int) -> Predicate:
        """Creates a binary predicate from a relative feature cluster."""
        classifier = _RelativeFeatureClusterClassifier(type1, type2, feature_name, cluster_info['center'], epsilon, diff_fn, cluster_id)
        name = str(classifier)
        types = [type1, type2]
        pred = Predicate(name, types, classifier)
        return pred

    def _create_predicate_from_absolute_cluster(self, type1: Type, feature_name: str, cluster_info: Dict, epsilon: float, cluster_id: int) -> Predicate:
        """Creates a unary predicate from an absolute feature cluster."""
        classifier = _AbsoluteFeatureClusterClassifier(type1, feature_name, cluster_info['center'], epsilon, cluster_id)
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
        beam: List[Tuple[float, FrozenSet[Predicate]]] = [(-np.inf, frozenset())] # Score, Predicate Set

        # Check initial predicates against constraint (if any exist)
        initial_pred_set = frozenset(self._initial_predicates)
        initial_valid = self._check_plan_length_constraint(initial_pred_set, set(), dataset, [], train_tasks) # Operators not learned yet
        if not initial_valid:
            logging.warning("Initial predicates may violate plan length constraint (if planner were integrated).")
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
            processed_sets: Set[FrozenSet[Predicate]] = set(p for _, p in beam)

            logging.info(f"Beam search iteration {iteration}, beam size {len(beam)}")

            # Generate successors by adding one predicate to each set in the beam
            for _, current_preds in beam:
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
                    score = self._evaluate_objective(combined_preds, alpha, dataset, train_tasks)
                    # logging.debug(f"Beam search iteration {iteration}, score: {score:.4f}, num preds: {len(next_pred_set)}")
                    successors.append((score, next_pred_set)) # Store score with the *added* predicates only

            if not successors:
                logging.info("Beam search found no viable successors. Terminating.")
                break # No improvement possible

            # Keep top B successors based on score
            successors.sort(key=lambda x: x[0], reverse=True) # Sort descending by score
            new_beam = successors[:beam_width]
            logging.debug(f"Beam search iteration {iteration}, new beam: {new_beam}")

            # Check for convergence (beam hasn't changed or score isn't improving)
            # Simple check: if the best score in the new beam is not better than the previous best
            current_best_score_in_beam = new_beam[0][0] if new_beam else -np.inf
            if current_best_score_in_beam <= best_score and iteration > 1 : # Allow first iteration to set baseline
                logging.info("Beam search converged (no score improvement).")
                break

            beam = new_beam
            if beam:
                best_score, best_pred_set_added = beam[0] # Best set in current beam (added preds only)
                logging.info(f"Iteration {iteration} best score: {best_score:.4f}, num preds: {len(best_pred_set_added)}")


        # Final selection: the best predicate set found that satisfies constraints
        # Need to re-evaluate the best set found to get its final props if needed elsewhere
        final_selected_learned_preds = best_pred_set_added if best_score > -np.inf else frozenset()

        logging.info(f"Beam search finished. Final best score: {best_score:.4f}")
        logging.info(f"Selected learned predicates ({len(final_selected_learned_preds)}):")
        for pred in sorted(list(final_selected_learned_preds)):
            logging.info(f"\t{pred.name}")

        # Return only the *learned* predicates (excluding initial ones)
        return set(final_selected_learned_preds)


    def _evaluate_objective(self,
                            predicates: FrozenSet[Predicate],
                            alpha: float,
                            dataset: Dataset,
                            train_tasks: List[Task]) -> float:
        """Calculates the objective function score for a given predicate set,
           checking constraints. Returns -inf if constraints fail."""

        # Check plan length constraint first (most expensive)
        if predicates in self._plan_constraint_cache:
            constraint_holds = self._plan_constraint_cache[predicates]
        else:
            # Need operators for the constraint check
            if predicates in self._operator_complexity_cache:
                _, operators = self._operator_complexity_cache[predicates]
            else:
                # Create atom dataset needed for operator learning
                atom_dataset = self._create_atom_dataset(dataset, predicates)
                _, operators = self._calculate_operator_complexity_term(predicates, dataset, atom_dataset, train_tasks)
            # Now check constraint
            atom_dataset_for_constraint = self._create_atom_dataset(dataset, predicates) # Recalculate or retrieve cache
            constraint_holds = self._check_plan_length_constraint(predicates, operators, dataset, atom_dataset_for_constraint, train_tasks)
            self._plan_constraint_cache[predicates] = constraint_holds

        if not constraint_holds:
            # logging.debug(f"Predicate set failed plan length constraint.")
            return -np.inf # Invalid set

        # If constraint holds, calculate the rest of the objective
        # Create atom dataset (if not already computed for constraint check)
        if predicates in self._atom_dataset_cache:
             atom_dataset = self._atom_dataset_cache[predicates]
        else:
             atom_dataset = self._create_atom_dataset(dataset, predicates)
             self._atom_dataset_cache[predicates] = atom_dataset

        # Calculate segmentation term
        if predicates in self._segmentation_cache:
             seg_term = self._segmentation_cache[predicates]
        else:
             seg_term = self._calculate_segmentation_term(predicates, atom_dataset)
             self._segmentation_cache[predicates] = seg_term

        # Calculate operator complexity term (operators may have been computed for constraint)
        if predicates in self._operator_complexity_cache:
             op_term, _ = self._operator_complexity_cache[predicates]
        else:
             op_term, _ = self._calculate_operator_complexity_term(predicates, dataset, atom_dataset, train_tasks)
             # Cache already handled inside the function call

        score = seg_term - alpha * op_term
        # logging.debug(f"Pred set size {len(predicates)}, Seg: {seg_term}, OpComp: {op_term}, Score: {score:.3f}")
        return score


    def _calculate_segmentation_term(self,
                                     predicates: FrozenSet[Predicate],
                                     atom_dataset: List[GroundAtomTrajectory]) -> int:
        """Calculates the segmentation term: Σ |ψ(P, τ)|.
        Uses number of segments as |ψ(P, τ)|.
        """
        total_segments = 0
        for ll_traj, atom_seq in atom_dataset:
            # Segment trajectory based *only* on the current predicate set
            # Need to ensure atom_seq corresponds *exactly* to predicates,
            # which it should if generated by _create_atom_dataset.
            segments = segment_trajectory(ll_traj, predicates, atom_seq=atom_seq)
            total_segments += len(segments)
        return total_segments


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
        try:
            # segment_trajectory needs to be called within learn_strips_operators
            # or we need to pre-segment. Let's assume learn_strips_operators handles it.
            # It needs the low-level trajectories too.
            low_level_trajs = [t for t in dataset.trajectories] # Assuming atom_dataset aligns with dataset.trajectories
            # Ensure atom_dataset only contains atoms for 'predicates'
            pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates)
            segmented_trajs = [segment_trajectory(ll_traj, predicates, atom_seq=atom_seq) for (ll_traj, atom_seq) in pruned_atom_data]

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
            complexity = len(operators)
            result = (complexity, operators)

        except (PlanningFailure, PlanningTimeout, TimeoutError, ValueError) as e:
            # Handle potential errors during operator learning (e.g., inconsistent data)
            logging.warning(f"Operator learning failed for predicate set: {e}")
            # Return high complexity or some indicator of failure
            result = (np.inf, set()) # Indicate failure with infinite complexity

        # Cache the result
        self._operator_complexity_cache[predicates] = result
        return result


    def _check_plan_length_constraint(self,
                                      predicates: FrozenSet[Predicate],
                                      operators: Set[NSRT],
                                      dataset: Dataset,
                                      atom_dataset: List[GroundAtomTrajectory],
                                      train_tasks: List[Task]) -> bool:
        """Checks if demonstrated plan length equals optimal plan length in the learned domain."""
        warnings.warn("Plan length constraint check is not fully implemented. Returning True.", UserWarning, stacklevel=2)
        return True
        # raise NotImplementedError("Plan length constraint check is not fully implemented.")
        # Placeholder: Return True until planner integration is done.
        # This is a complex integration task involving:
        # 1. Converting learned NSRTs to PDDL (or using a planner that accepts NSRTs).
        # 2. Defining task goals based on final states and predicates.
        # 3. Running a planner for each trajectory.
        # 4. Comparing planner output length to demonstrated segmentation length.
        if not CFG.clustering_check_plan_length_constraint:
             return True # Skip check if disabled by CFG

        if not operators: # If operator learning failed, constraint cannot hold
            return False

        logging.warning("Plan length constraint check is not fully implemented. Returning True.")

        # --- Start of Placeholder Implementation Sketch ---
        # try:
        #     strips_ops = {nsrt.to_strips_operator() for nsrt in operators}
        #     domain_pddl = utils.generate_pddl_domain(strips_ops, predicates, self._types)
        # except Exception as e:
        #     logging.error(f"Failed to generate PDDL domain: {e}")
        #     return False # Cannot check constraint if domain fails

        # for i, (ll_traj, atom_seq) in enumerate(atom_dataset):
        #     if not ll_traj.states: continue
        #     init_state = ll_traj.states[0]
        #     final_state = ll_traj.states[-1]

        #     # Create initial and goal atom sets
        #     init_atoms = utils.abstract(init_state, predicates)
        #     goal_atoms = utils.abstract(final_state, predicates)
        #     if init_atoms == goal_atoms: continue # Skip trivial trajectories

        #     # Create a PDDL problem file or task structure
        #     objects = set(init_state)
        #     problem_pddl = utils.generate_pddl_problem(objects, init_atoms, goal_atoms, f"traj{i}_problem")

        #     # Run planner (needs planner integration, e.g., Pyperplan or Fast Downward)
        #     # planner_output = run_planner(domain_pddl, problem_pddl)
        #     # planner_plan_len = len(planner_output)
        #     planner_plan_len = np.inf # Placeholder

        #     # Get demonstrated plan length (number of segments)
        #     demo_segments = segment_trajectory(ll_traj, predicates, atom_seq=atom_seq)
        #     demo_plan_len = len(demo_segments)

        #     if planner_plan_len < demo_plan_len:
        #         logging.debug(f"Constraint violation on traj {i}: Planner len {planner_plan_len} < Demo len {demo_plan_len}")
        #         return False # Constraint violated

        # except Exception as e:
        #      logging.error(f"Error during plan length constraint check: {e}")
        #      return False # Fail safe
        # --- End of Placeholder Implementation Sketch ---

        return True

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