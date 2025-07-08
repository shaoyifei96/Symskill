"""A RoboKitchen-specific perceiver."""

from typing import Set
from predicators.settings import CFG
from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video


class RoboKitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "robo_kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)

        pred_name_to_pred = RoboKitchenEnv.create_predicates()

        Dummy = pred_name_to_pred["Dummy"]
        DoorOpen = pred_name_to_pred["DoorOpen"]
        DoorClosed = pred_name_to_pred["DoorClosed"]
        DrawerClosed = pred_name_to_pred["DrawerClosed"]
        DrawerOpen = pred_name_to_pred["DrawerOpen"]
        OnSurface = pred_name_to_pred["OnSurface"]
        KnobTurnedOn = pred_name_to_pred["KnobTurnedOn"]
        MicrowaveOn = pred_name_to_pred["MicrowaveOn"]
        StoveOn = pred_name_to_pred["StoveOn"]
        StoveOff = pred_name_to_pred["StoveOff"]

        # handle = RoboKitchenEnv.object_name_to_object("handle")
        # left_handle = RoboKitchenEnv.object_name_to_object("left_door_handle")
        # right_handle = RoboKitchenEnv.object_name_to_object("right_door_handle")
        door = RoboKitchenEnv.object_name_to_object("door")
        left_door = RoboKitchenEnv.object_name_to_object("leftdoor")
        right_door = RoboKitchenEnv.object_name_to_object("rightdoor")
        cabinet = RoboKitchenEnv.object_name_to_object("cabinet")
        obj = RoboKitchenEnv.object_name_to_object("obj")
        bottom = RoboKitchenEnv.object_name_to_object("bottom")
        stove = RoboKitchenEnv.object_name_to_object("stovetop")
        knob = RoboKitchenEnv.object_name_to_object("knob")
        microwave = RoboKitchenEnv.object_name_to_object("microwave")
        drawer = RoboKitchenEnv.object_name_to_object("drawer")
        drawer_inner_box = RoboKitchenEnv.object_name_to_object("drawer_inner_box")
        container = RoboKitchenEnv.object_name_to_object("container")
        obj_container = RoboKitchenEnv.object_name_to_object("obj_container")

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
        elif goal_desc == 'StoreFruit':
            goal = {
                GroundAtom(OnSurface, [obj, bottom]),
                # GroundAtom(DoorClosed, [door, cabinet]),
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
                GroundAtom(OnSurface, [obj, container]),
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
            tomato = RoboKitchenEnv.object_name_to_object("tomato")
            cheese = RoboKitchenEnv.object_name_to_object("cheese")
            plate = RoboKitchenEnv.object_name_to_object("plate")
            goal = {
                GroundAtom(OnSurface, [tomato, plate]),
                GroundAtom(OnSurface, [cheese, plate]),
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")

        # convert task.goal to predicate goal if using clustering reprocess
        if  len(list(state)) > 0 and (CFG.reprocess_ground_atom_dataset_using_cluster_replacement or CFG.reprocess_ground_atom_dataset_using_cluster_predicates):
            new_goal = set()
            for g in goal:
                if "goal" in g.predicate.name:
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
                        type1_objs = [obj for obj in state if obj.type == rel_pose_pred.types[0]]
                        type2_objs = [obj for obj in state if obj.type == rel_pose_pred.types[1]]
                        type1_obj = type1_objs[0]
                        type2_obj = type2_objs[0]
                        rel_pose_pred_atom = GroundAtom(rel_pose_pred, [type1_obj, type2_obj])
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
