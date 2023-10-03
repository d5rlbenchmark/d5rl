import numpy as np
from dm_control.composer.observation import observable
from dm_control.composer.task import Task
from dm_control.manipulation.place import arenas
from dm_env import specs

from benchmark.domains.widowx.robot.widowx_arm import WidowXArm
from benchmark.domains.widowx.robot.widowx_hand import WidowXHand

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.001


class BaseTask(Task):
    def __init__(
        self,
        physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
        control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
    ) -> None:
        super().__init__()
        self._arena = arenas.Standard()
        self._arena.mjcf_model.size.nconmax = 200
        self._arena.mjcf_model.size.njmax = 1000

        self._arm = WidowXArm()
        self._hand = WidowXHand()

        self._arm.attach(self._hand)
        self._arena.attach(self._arm)

        self._add_sensors()

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep
        )

    def _add_sensors(self):
        self._hand.mjcf_model.sensor.add(
            "framepos",
            name="end_effector_pos",
            objtype="site",
            objname=self._hand.tool_center_point,
            reftype="site",
            refname=self._arm.base_site,
        )

        self._hand.mjcf_model.sensor.add(
            "framequat",
            name="end_effector_quat",
            objtype="site",
            objname=self._hand.tool_center_point,
            reftype="site",
            refname=self._arm.base_site,
        )

        for name in ["right_finger", "left_finger"]:
            self._hand.mjcf_model.sensor.add(
                "jointpos", name=f"{name}_qpos", joint=name
            )

            self._hand.mjcf_model.sensor.add(
                "jointvel", name=f"{name}_qvel", joint=name
            )

    def action_spec(self, physics):
        """Transformrs from 6d + gripper
        into 3d + gripper.
        """

        # 6D + gripper.
        action_spec_7d = super().action_spec(physics)

        actuator_names = []
        indices = []
        for i, name in enumerate(action_spec_7d.name.split("\t")):
            if "private" not in name:
                actuator_names.append(name)
                indices.append(i)

        return specs.BoundedArray(
            shape=(len(indices),),
            dtype=action_spec_7d.dtype,
            minimum=action_spec_7d.minimum[indices],
            maximum=action_spec_7d.maximum[indices],
            name="\t".join(actuator_names),
        )

    def before_step(self, physics, action, random_state):
        del random_state

        action_spec_7d = super().action_spec(physics)
        full_action = np.zeros(action_spec_7d.shape, dtype=action_spec_7d.dtype)
        indices = []
        for i, name in enumerate(action_spec_7d.name.split("\t")):
            if "private" not in name:
                indices.append(i)
            else:
                full_action[i] = 0

        full_action[indices] = action

        physics.set_control(full_action)

    def initialize_episode(self, physics, random_state):
        self._arm.set_pose(physics)
        self._hand.set_grasp(physics, close_factors=random_state.uniform())

    @property
    def task_observables(self):
        task_observables = super().task_observables

        for sensor in self._hand.mjcf_model.find_all("sensor"):
            obs = observable.MJCFFeature("sensordata", sensor)
            task_observables[sensor.name] = obs
            obs.enabled = True
        """
        for camera in self.root_entity.mjcf_model.find_all('camera'):
            obs = observable.MJCFCamera(camera, height=84, width=84)
            task_observables[camera.name] = obs
            obs.enabled = True
        """
        # task_observables['end_effector_quat'].enabled = False

        return task_observables

    @property
    def root_entity(self):
        return self._arena

    def get_reward(self, physics):
        return 0.0
