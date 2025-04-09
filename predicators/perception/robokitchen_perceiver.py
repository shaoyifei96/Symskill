"""A RoboKitchen-specific perceiver."""

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

        DoorOpen = pred_name_to_pred["DoorOpen"]
        OnSurface = pred_name_to_pred["OnSurface"]

        handle = RoboKitchenEnv.object_name_to_object("handle")
        cabinet = RoboKitchenEnv.object_name_to_object("cabinet")
        obj = RoboKitchenEnv.object_name_to_object("obj")
        bottom = RoboKitchenEnv.object_name_to_object("bottom")
        goal_desc = env_task.goal_description
        if goal_desc == 'OpenSingleDoor':
            goal = {
                GroundAtom(DoorOpen, [handle, cabinet]),
            }
        elif goal_desc == 'PnPCounterToCab':
            goal = {
                GroundAtom(OnSurface, [obj, bottom]),
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")
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
