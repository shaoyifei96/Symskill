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
        if goal_desc == 'OpenSingleDoor':
            goal = {
                # GroundAtom(DoorOpen, [handle, cabinet]),
                GroundAtom(DoorOpen, [door, cabinet]),
            }
        elif goal_desc == 'OpenDoubleDoor':
            goal = {
                # GroundAtom(DoorOpen, [left_handle, cabinet]),
                # GroundAtom(DoorOpen, [right_handle, cabinet]),
                GroundAtom(DoorOpen, [left_door, cabinet]),
                GroundAtom(DoorOpen, [right_door, cabinet]),
            }
        elif goal_desc == "CloseSingleDoor":
            goal = {
                GroundAtom(DoorClosed, [door, cabinet]),
            }
        elif goal_desc == "CloseDoubleDoor":
            goal = {
                GroundAtom(DoorClosed, [left_door, cabinet]),
                GroundAtom(DoorClosed, [right_door, cabinet]),
            }
        elif goal_desc == 'PnPCounterToCab':
            goal = {
                GroundAtom(OnSurface, [obj, bottom]),
            }
        elif goal_desc == 'PnPCabToCounter':
            goal = {
                GroundAtom(OnCounter, [obj, counter]),
            }
        elif goal_desc == 'PnPCounterToStove':
            goal = {
                GroundAtom(InContainer, [obj, container]),
            }
        elif goal_desc == 'StoreFruit':
            goal = {
                GroundAtom(OnSurface, [obj, bottom]),
            }
        elif goal_desc == 'StoreFruitFull':
            goal = {
                GroundAtom(OnSurface, [obj, bottom]),
                GroundAtom(DoorClosed, [door, cabinet]),
            }
        elif goal_desc == 'TurnOnMicrowave':
            goal = {
                GroundAtom(MicrowaveOn, [microwave])
            }
        elif goal_desc == 'TurnOnStove':
            goal = {
                GroundAtom(StoveOn, [stove]),
            }
        elif goal_desc == 'CloseDrawer':
            goal = {
                GroundAtom(DrawerClosed, [drawer_inner_box, drawer]),
            }
        elif goal_desc == 'PnPStoveToCounter':
            goal = {
                GroundAtom(InContainer, [obj, container]),
            }
        elif goal_desc == 'OpenDrawer':
            goal = {
                GroundAtom(DrawerOpen, [drawer_inner_box, drawer]),
            }
        elif goal_desc == 'TurnOffStove':
            goal = {
                GroundAtom(StoveOff, [stove]),
            }
        elif goal_desc == 'CookCheeseAndTomatoes':
            goal = {
                GroundAtom(InContainer, [tomato, plate]),
                GroundAtom(InContainer, [cheese, plate]),
            }
        elif goal_desc == 'TurnOnSinkFaucet':
            goal = {
                GroundAtom(SinkFaucetOn, [sink_faucet_handle]),
            }
        elif goal_desc == 'TurnOffSinkFaucet':
            goal = {
                GroundAtom(SinkFaucetOff, [sink_faucet_handle]),
            }
        elif goal_desc == 'PnPCabToCounterTomato':
            goal = {
                GroundAtom(InContainer, [tomato, plate]),
            }
        elif goal_desc == 'MocapOpenLid':
            goal = {
                GroundAtom(LidOnDishrack, [lid, dish_rack]),
            }
        elif goal_desc == 'MocapPourWater':
            goal = {
                GroundAtom(InContainer, [tomato, plate]),
            }
        elif goal_desc == 'MocapOpenLidPourWater':
            goal = {
                # GroundAtom(LidOnDishrack, [lid, dish_rack]),
                GroundAtom(InContainer, [tomato, plate]),
            }
        elif goal_desc == 'MocapPnPBanana':
            goal = {
                GroundAtom(InCookware, [banana, pan]),
            }
        elif goal_desc == 'MocapOpenLidPnPBanana':
            goal = {
                GroundAtom(InCookware, [banana, pan]),
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc} (This goal is what the algorithm sees online to convert each goal description to something it understands as relative pose predicates it met during training. e.g. InContainer(tomato, plate) -> RelPose(tomato, plate). Since no goal predicate is specified during training, so we need to save a goal dict for each goal description)")

        # convert task.goal to predicate goal if using clustering reprocess
        for key, value in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates.items():
            print(f"DEBUG: key: {key}, value: {value}")
            for val in value:
                print(type(val ))
        if  len(list(state)) > 0 and (CFG.reprocess_ground_atom_dataset_using_cluster_replacement or CFG.reprocess_ground_atom_dataset_using_cluster_predicates):
            new_goal = set()
            for g in goal:
                if not isinstance(g, str) and"goal" in g.predicate.name:
                    raise NotImplementedError("Not implemented properly! when saving goal, it is not converted to the right types")
                    # Build key based on number of entities
                    if len(g.entities) == 1:
                        rel_pose_key = (g.predicate.name, g.entities[0].type.name)
                    else:
                        rel_pose_key = (g.predicate.name, g.entities[0].type.name, g.entities[1].type.name)
                    rel_pose_preds = CFG.dict_contact_predicate_to_rel_pose_predicates[rel_pose_key]
                    rel_pose_pred = list(rel_pose_preds)[0]
                    rel_pose_pred_atom = GroundAtom(rel_pose_pred, g.entities)
                    new_goal.add(rel_pose_pred_atom)
                else:
                    # Build key based on number of entities
                    if len(g.entities) == 1:
                        gt_goal_key = (g.predicate.name, g.entities[0].type.name)
                    else:
                        gt_goal_key = (g.predicate.name, g.entities[0].type.name, g.entities[1].type.name)

                    # Check if this predicate needs conversion (only for 2-entity predicates currently)
                    if gt_goal_key in CFG.dict_gt_goal_predicate_to_dummy_goal_predicates:
                        dummy_goal_pred = list(CFG.dict_gt_goal_predicate_to_dummy_goal_predicates[gt_goal_key])[0]
                        rel_pose_preds = CFG.dict_contact_predicate_to_rel_pose_predicates[(dummy_goal_pred.name, dummy_goal_pred.types[0].name, dummy_goal_pred.types[1].name)]
                        rel_pose_pred = list(rel_pose_preds)[0]
                        # TODO: this is problematic. It's unable to get both tomato and cheese
                        type1_objs = [obj for obj in state if obj.type == rel_pose_pred.types[0]]
                        type2_objs = [obj for obj in state if obj.type == rel_pose_pred.types[1]] 
                        # print(f"DEBUG: type1_objs: {type1_objs}")
                        # print(f"DEBUG: type2_objs: {type2_objs}")
                        # type 2 is the object in motion, such as door, or tomato, if object name has a _number at the end, and type_1 object also has a _number at the end, then try to match the object with the same _number at the end. if type 2 object has no _number at the end, then match with the object of type 1 that is closer.
                        assert len(type1_objs) >= 1 and len(type2_objs) >= 1, f"Expected at least one object of type {rel_pose_pred.types[0].name} and {rel_pose_pred.types[1].name}, but found {len(type1_objs)} and {len(type2_objs)} respectively. State objects: {[obj.name for obj in state]}"
                        if len(type1_objs) == 1 and len(type2_objs) == 1:
                            type1_obj_final = type1_objs[0]
                            type2_obj_final = type2_objs[0]
                        else:
                            # start with type 2,
                            goal_obj_2_name = g.entities[0].name # since predicate usually have the object as first entity
                            # try to match the whole name first with number
                            for type2_obj in type2_objs:
                                if goal_obj_2_name == type2_obj.name:
                                    type2_obj_final = type2_obj
                                    break
                            else:
                                # try to match the name without _number at the end
                                goal_obj_2_name_without_number = goal_obj_2_name.split("_")[0] if "_" in goal_obj_2_name and goal_obj_2_name.split("_")[-1].isdigit() else goal_obj_2_name
                                for type2_obj in type2_objs:
                                    type2_obj_name_without_number = type2_obj.name.split("_")[0] if "_" in type2_obj.name and type2_obj.name.split("_")[-1].isdigit() else type2_obj.name
                                    if goal_obj_2_name_without_number in type2_obj_name_without_number:
                                        type2_obj_final = type2_obj
                                        break
                        assert type2_obj_final is not None
                        # try to match the object of type 1 that has the name number at the end 
                        if type2_obj_final.name.split("_")[-1].isdigit():
                            for type1_obj in type1_objs:
                                if type1_obj.name.split("_")[-1].isdigit() and type1_obj.name.split("_")[-1] == type2_obj_final.name.split("_")[-1]:
                                    type1_obj_final = type1_obj
                                    break
                        else:
                            # try to match the object of type 1 that is closer
                            type1_obj_final = None
                            all_containers = [obj.type.name == "container_type" for obj in type1_objs ]
                            if all(all_containers): # special case for obj_container being the current container, container being the goal container
                                # find the container whose name is container, 
                                for type1_obj in type1_objs:
                                    if type1_obj.name == "container":
                                        type1_obj_final = type1_obj
                                        break
                            else:
                                min_dist = float("inf")
                                goal_obj_pos = state.get(type2_obj_final, "translation")
                                for type1_obj in type1_objs:
                                    state_obj_pos = state.get(type1_obj, "translation")
                                    dist = np.linalg.norm(state_obj_pos - goal_obj_pos)
                                    logging.info(f"best type1_obj_final: {type1_obj.name} for type2_obj_final: {type2_obj_final.name} with dist: {dist}")
                                    if dist < min_dist:
                                        min_dist = dist
                                        type1_obj_final = type1_obj
                            assert type1_obj_final is not None
                        # type2_obj = type2_objs[0]
                        rel_pose_pred_atom = GroundAtom(rel_pose_pred, [type1_obj_final, type2_obj_final])
                        new_goal.add(rel_pose_pred_atom)
                    else:
                        # For predicates not in the conversion dict (like MicrowaveOn), keep as is
                        new_goal.add(g)
            goal = new_goal

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
