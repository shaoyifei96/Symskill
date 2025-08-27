"""A RoboKitchen-specific perceiver."""

from typing import Set
import numpy as np
from predicators.settings import CFG
from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video
import logging

class RoboKitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "robo_kitchen"

    def _build_predicate_key(self, predicate_name: str, entities: list) -> tuple:
        """Build a key for predicate lookups based on predicate name and entity types."""
        if len(entities) == 1:
            return (predicate_name, entities[0].type.name)
        else:
            return (predicate_name, entities[0].type.name, entities[1].type.name)

    def _extract_name_without_number(self, name: str) -> str:
        """Extract object name without trailing number (e.g., 'door_1' -> 'door')."""
        if "_" in name and name.split("_")[-1].isdigit():
            return name.split("_")[0]
        return name

    def _find_matching_object_by_name(self, target_name: str, candidates: list):
        """Find object from candidates that matches target name, with fallback to partial matching."""
        # Try exact match first
        for candidate in candidates:
            if target_name == candidate.name:
                return candidate
        
        # Try partial match without numbers
        target_name_base = self._extract_name_without_number(target_name)
        for candidate in candidates:
            candidate_name_base = self._extract_name_without_number(candidate.name)
            if target_name_base in candidate_name_base:
                return candidate
        
        return None

    def _find_matching_object_by_number_suffix(self, reference_obj, candidates: list):
        """Find object from candidates that has the same number suffix as reference object."""
        if not reference_obj.name.split("_")[-1].isdigit():
            return None
            
        reference_number = reference_obj.name.split("_")[-1]
        for candidate in candidates:
            if (candidate.name.split("_")[-1].isdigit() and 
                candidate.name.split("_")[-1] == reference_number):
                return candidate
        
        return None

    def _find_closest_object(self, reference_obj, candidates: list, state) -> object:
        """Find the closest object to reference_obj from candidates based on position."""
        if not candidates:
            return None
            
        # Special case for containers
        if all(obj.type.name == "container_type" for obj in candidates):
            for candidate in candidates:
                if candidate.name == "container":
                    return candidate
        
        # Find closest by distance
        min_dist = float("inf")
        closest_obj = None
        reference_pos = state.get(reference_obj, "translation")
        
        for candidate in candidates:
            candidate_pos = state.get(candidate, "translation")
            dist = np.linalg.norm(candidate_pos - reference_pos)
            logging.info(f"Distance from {candidate.name} to {reference_obj.name}: {dist}")
            if dist < min_dist:
                min_dist = dist
                closest_obj = candidate
                
        return closest_obj

    def _match_objects_for_predicate(self, goal_atom, type1_objs: list, type2_objs: list, state) -> tuple:
        """Match type1 and type2 objects for a predicate based on names and proximity."""
        assert len(type1_objs) >= 1 and len(type2_objs) >= 1
        
        # Simple case: only one object of each type
        if len(type1_objs) == 1 and len(type2_objs) == 1:
            return type1_objs[0], type2_objs[0]
        
        # Find type2 object (usually the object in motion)
        goal_obj_2_name = goal_atom.entities[0].name  # predicate usually has object as first entity
        type2_obj_final = self._find_matching_object_by_name(goal_obj_2_name, type2_objs)
        assert type2_obj_final is not None, f"Could not find matching type2 object for {goal_obj_2_name}"
        
        # Find matching type1 object
        # Try to match by number suffix first
        type1_obj_final = self._find_matching_object_by_number_suffix(type2_obj_final, type1_objs)
        
        # If no number match, find closest object
        if type1_obj_final is None:
            type1_obj_final = self._find_closest_object(type2_obj_final, type1_objs, state)
        
        assert type1_obj_final is not None, f"Could not find matching type1 object for {type2_obj_final.name}"
        return type1_obj_final, type2_obj_final

    def _convert_single_goal_atom(self, goal_atom, state):
        """Convert a single goal atom to relative pose predicate if needed."""
        if "goal" in goal_atom.predicate.name:
            raise NotImplementedError("Not implemented properly! when saving goal, it is not converted to the right types")
        
        if CFG.predicate_candidates_method != "low_speed":
            # Build key for predicate lookup
            gt_goal_key = self._build_predicate_key(goal_atom.predicate.name, goal_atom.entities)
            
            # Check if this predicate needs conversion
            if gt_goal_key not in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
                # For predicates not in the conversion dict (like MicrowaveOn), keep as is
                return goal_atom
            
            # Get dummy goal predicate and corresponding relative pose predicate
            dummy_goal_pred = list(CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[gt_goal_key])[0]
            rel_pose_key = (dummy_goal_pred.name, dummy_goal_pred.types[0].name, dummy_goal_pred.types[1].name)
            proper_goal_preds = CFG.dict_contact_predicate_to_rel_pose_predicates[rel_pose_key]
            proper_goal_pred = list(proper_goal_preds)[0]
        else:
            proper_goal_pred = goal_atom.predicate
        
        # Find objects of the required types
        type1_objs = [obj for obj in state if obj.type == proper_goal_pred.types[0]]
        type2_objs = [obj for obj in state if obj.type == proper_goal_pred.types[1]]
        
        # Match the appropriate objects
        type1_obj_final, type2_obj_final = self._match_objects_for_predicate(
            goal_atom, type1_objs, type2_objs, state)
        
        # Create and return the relative pose predicate atom
        return GroundAtom(proper_goal_pred, [type1_obj_final, type2_obj_final])

    def _construct_goal_from_stored_atoms(self, state, goal_desc: str) -> Set[GroundAtom]:
        """Construct goal from stored common atoms using object names to find objects in current state."""
        stored_atoms = CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[goal_desc]
        goal = set()
        
        # Get predicate name to predicate mapping
        pred_name_to_pred = RoboKitchenEnv.create_predicates()
        
        # Create a mapping from object name to object in current state
        name_to_obj = {obj.name: obj for obj in state}
        
        for atom_key in stored_atoms:
            # atom_key format: (predicate_name, (obj_names...), (obj_type_names...))
            pred_name, obj_names, obj_type_names = atom_key
            
            # Try to find the predicate
            # if pred_name in pred_name_to_pred:
            for pred_name_key, type1, type2 in CFG.dict_contact_predicate_to_rel_pose_predicates.keys():
                if obj_type_names == (type1, type2):
                    predicate = CFG.dict_contact_predicate_to_rel_pose_predicates[pred_name_key, type1, type2]
                    assert len(predicate) == 1, f"Multiple predicates found for {pred_name_key}, {type1}, {type2}"
                    predicate = list(predicate)[0]
                    break
            else:
                raise NotImplementedError(f"Unrecognized goal: {goal_desc} (This goal is what the algorithm sees online to convert each goal description to something it understands as relative pose predicates it met during training. e.g. InContainer(tomato, plate) -> RelPose(tomato, plate). Since no goal predicate is specified during training, so we need to save a goal dict for each goal description)")            
            
            # Check if this is a fallback entry (empty obj_names tuple)
            if not obj_names:  # Empty tuple means this is a predicate+type only entry
                # Match by object types only
                objects_by_type = []
                found_all_types = True
                
                for obj_type_name in obj_type_names:
                    # Find objects of this type in the current state
                    matching_objs = [obj for obj in state if obj.type.name == obj_type_name]
                    if matching_objs:
                        # Use the first object of this type
                        objects_by_type.append(matching_objs[0])
                    else:
                        # No objects of this type found
                        found_all_types = False
                        break
                
                if found_all_types and len(objects_by_type) == len(obj_type_names):
                    # Create the ground atom using objects matched by type
                    goal_atom = GroundAtom(predicate, objects_by_type)
                    goal.add(goal_atom)
            else:
                # This is a normal entry with specific object names, try name matching first
                objects = []
                found_all_objects = True
                
                for obj_name in obj_names:
                    if obj_name in name_to_obj:
                        objects.append(name_to_obj[obj_name])
                    else:
                        # Object not found by name, skip this atom
                        found_all_objects = False
                        break
                
                if found_all_objects and len(objects) == len(obj_names):
                    # Create the ground atom using objects matched by name
                    goal_atom = GroundAtom(predicate, objects)
                    goal.add(goal_atom)
                else:
                    # Fallback: try to match by object types only
                    objects_by_type = []
                    found_all_types = True
                    
                    for obj_type_name in obj_type_names:
                        # Find objects of this type in the current state
                        matching_objs = [obj for obj in state if obj.type.name == obj_type_name]
                        if matching_objs:
                            # Use the first object of this type
                            objects_by_type.append(matching_objs[0])
                        else:
                            # No objects of this type found
                            found_all_types = False
                            break
                    
                    if found_all_types and len(objects_by_type) == len(obj_type_names):
                        # Create the ground atom using objects matched by type
                        goal_atom = GroundAtom(predicate, objects_by_type)
                        goal.add(goal_atom)
                    
        return goal

    def _convert_goal_for_clustering_reprocess(self, state, goal: Set[GroundAtom]) -> Set[GroundAtom]:
        """Convert goal atoms to relative pose predicates if clustering reprocess is enabled."""
        if (len(list(state)) == 0 or 
            not (CFG.reprocess_ground_atom_dataset_using_cluster_replacement or 
                 CFG.reprocess_ground_atom_dataset_using_cluster_predicates)):
            return goal
        
        new_goal = set()
        for goal_atom in goal:
            converted_atom = self._convert_single_goal_atom(goal_atom, state)
            new_goal.add(converted_atom)
        
        return new_goal

    def reset(self, env_task: EnvironmentTask) -> Task:
        print(f"DEBUG: init_obs: {env_task.init_obs}")
        state = self._observation_to_state(env_task.init_obs)

        pred_name_to_pred = RoboKitchenEnv.create_predicates()

        Dummy = pred_name_to_pred["Dummy"]
        DoorOpen = pred_name_to_pred["DoorOpen"]
        DoorClosed = pred_name_to_pred["DoorClosed"]
        DrawerClosed = pred_name_to_pred["DrawerClosed"]
        DrawerOpen = pred_name_to_pred["DrawerOpen"]
        OnSurface = pred_name_to_pred["OnSurface"]
        OnCounter = pred_name_to_pred["OnCounter"]
        InContainer = pred_name_to_pred["InContainer"]
        InCookware = pred_name_to_pred["InCookware"]
        KnobTurnedOn = pred_name_to_pred["KnobTurnedOn"]
        MicrowaveOn = pred_name_to_pred["MicrowaveOn"]
        StoveOn = pred_name_to_pred["StoveOn"]
        StoveOff = pred_name_to_pred["StoveOff"]
        SinkFaucetOn = pred_name_to_pred["SinkFaucetOn"]
        SinkFaucetOff = pred_name_to_pred["SinkFaucetOff"]
        LidOnDishrack = pred_name_to_pred["LidOnDishrack"]

        # handle = RoboKitchenEnv.object_name_to_object("handle")
        # left_handle = RoboKitchenEnv.object_name_to_object("left_door_handle")
        # right_handle = RoboKitchenEnv.object_name_to_object("right_door_handle")
        door = RoboKitchenEnv.object_name_to_object("door")
        left_door = RoboKitchenEnv.object_name_to_object("leftdoor")
        right_door = RoboKitchenEnv.object_name_to_object("rightdoor")
        cabinet = RoboKitchenEnv.object_name_to_object("cabinet")
        obj = RoboKitchenEnv.object_name_to_object("obj")
        bottom = RoboKitchenEnv.object_name_to_object("bottom")
        counter = RoboKitchenEnv.object_name_to_object("counter")
        stove = RoboKitchenEnv.object_name_to_object("stovetop")
        knob = RoboKitchenEnv.object_name_to_object("knob")
        microwave = RoboKitchenEnv.object_name_to_object("microwave")
        drawer = RoboKitchenEnv.object_name_to_object("drawer")
        drawer_inner_box = RoboKitchenEnv.object_name_to_object("drawer_inner_box")
        tomato = RoboKitchenEnv.object_name_to_object("tomato")
        cheese = RoboKitchenEnv.object_name_to_object("cheese")
        plate = RoboKitchenEnv.object_name_to_object("plate")
        container = RoboKitchenEnv.object_name_to_object("container")
        obj_container = RoboKitchenEnv.object_name_to_object("obj_container")
        sink_faucet_handle = RoboKitchenEnv.object_name_to_object("sink_faucet_handle")
        sink = RoboKitchenEnv.object_name_to_object("sink")
        lid = RoboKitchenEnv.object_name_to_object("lid")
        dish_rack = RoboKitchenEnv.object_name_to_object("dishrack")
        banana = RoboKitchenEnv.object_name_to_object("banana")
        pan = RoboKitchenEnv.object_name_to_object("pan")

        goal_desc = env_task.goal_description
        # if goal_desc == 'OpenSingleDoor':
        #     goal = {
        #         # GroundAtom(DoorOpen, [handle, cabinet]),
        #         GroundAtom(DoorOpen, [door, cabinet]),
        #     }
        # elif goal_desc == 'OpenDoubleDoor':
        #     goal = {
        #         # GroundAtom(DoorOpen, [left_handle, cabinet]),
        #         # GroundAtom(DoorOpen, [right_handle, cabinet]),
        #         GroundAtom(DoorOpen, [left_door, cabinet]),
        #         GroundAtom(DoorOpen, [right_door, cabinet]),
        #     }
        # elif goal_desc == "CloseSingleDoor":
        #     goal = {
        #         GroundAtom(DoorClosed, [door, cabinet]),
        #     }
        # elif goal_desc == "CloseDoubleDoor":
        #     goal = {
        #         GroundAtom(DoorClosed, [left_door, cabinet]),
        #         GroundAtom(DoorClosed, [right_door, cabinet]),
        #     }
        # elif goal_desc == 'PnPCounterToCab':
        #     goal = {
        #         GroundAtom(OnSurface, [obj, bottom]),
        #         GroundAtom(GripperFarFromObj, [obj, obj]),
        #     }
        # elif goal_desc == 'PnPCabToCounter':
        #     goal = {
        #         GroundAtom(OnCounter, [obj, counter]),
        #     }
        # elif goal_desc == 'PnPCounterToStove':
        #     goal = {
        #         GroundAtom(InContainer, [obj, container]),
        #         GroundAtom(GripperFarFromObj, [obj, obj]),
        #     }
        # elif goal_desc == 'StoreFruit':
        #     goal = {
        #         GroundAtom(OnSurface, [obj, bottom]),
        #     }
        # elif goal_desc == 'StoreFruitFull':
        #     goal = {
        #         GroundAtom(OnSurface, [obj, bottom]),
        #         GroundAtom(DoorClosed, [door, cabinet]),
        #     }
        # elif goal_desc == 'TurnOnMicrowave':
        #     goal = {
        #         GroundAtom(MicrowaveOn, [microwave])
        #     }
        # elif goal_desc == 'TurnOnStove':
        #     goal = {
        #         GroundAtom(StoveOn, [stove]),
        #     }
        # elif goal_desc == 'CloseDrawer':
        #     goal = {
        #         GroundAtom(DrawerClosed, [drawer_inner_box, drawer]),
        #     }
        # elif goal_desc == 'PnPStoveToCounter':
        #     goal = {
        #         GroundAtom(InContainer, [obj, container]),
        #     }
        # elif goal_desc == 'OpenDrawer':
        #     goal = {
        #         GroundAtom(DrawerOpen, [drawer_inner_box, drawer]),
        #     }
        # elif goal_desc == 'TurnOffStove':
        #     goal = {
        #         GroundAtom(StoveOff, [stove]),
        #     }
        # elif goal_desc == 'CookCheeseAndTomatoes':
        #     goal = {
        #         GroundAtom(InContainer, [tomato, plate]),
        #         GroundAtom(InContainer, [cheese, plate]),
        #     }
        # elif goal_desc == 'TurnOnSinkFaucet':
        #     goal = {
        #         GroundAtom(SinkFaucetOn, [sink_faucet_handle]),
        #     }
        # elif goal_desc == 'TurnOffSinkFaucet':
        #     goal = {
        #         GroundAtom(SinkFaucetOff, [sink_faucet_handle]),
        #     }
        # elif goal_desc == 'PnPCabToCounterTomato':
        #     goal = {
        #         GroundAtom(InContainer, [tomato, plate]),
        #     }
        # else:
        if goal_desc in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
            # Use the stored common atoms to construct the goal
            goal = self._construct_goal_from_stored_atoms(state, goal_desc)
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc} (This goal is what the algorithm sees online to convert each goal description to something it understands as relative pose predicates it met during training. e.g. InContainer(tomato, plate) -> RelPose(tomato, plate). Since no goal predicate is specified during training, so we need to save a goal dict for each goal description)")

        # Convert task.goal to predicate goal if using clustering reprocess
        goal = self._convert_goal_for_clustering_reprocess(state, goal)

        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        # Get contact set from observation, or use empty set if not provided
        contact_set = obs.get("contact_set", set())

        # Convert state_info to state, passing in the contact set
        state = RoboKitchenEnv.state_info_to_state(obs["state_info"], contact_set)

        assert state.simulator_state is not None
        state.simulator_state["images"] = obs["obs_images"]
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for kitchen")
