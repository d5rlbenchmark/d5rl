import numpy as np
from dm_control import composer
from dm_control.locomotion import arenas

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.001


class Walk(composer.Task):

    def __init__(self,
                 robot,
                 terminate_at_height: float = 0.2,
                 terminate_not_upright: float = 0.5,
                 penalize_angular_acc: bool = False,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP):
        self._floor = arenas.Floor(size=(64, 64))

        self._robot = robot
        self._floor.add_free_entity(self._robot)

        observables = (self._robot.observables.proprioception +
                       self._robot.observables.kinematic_sensors +
                       [self._robot.observables.body_height] +
                       [self._robot.observables.prev_action])
        for observable in observables:
            observable.enabled = True

        self._floor._top_camera.remove()
        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1, 0.75],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=60.0)

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._target_linear_velocity = 0.5
        self._penalize_angular_acc = penalize_angular_acc

        self._terminate_at_height = terminate_at_height
        self._terminate_not_upright = terminate_not_upright

    def get_reward(self, physics):
        """
        From https://arxiv.org/abs/2111.01674
        with minor tweaks.
        """

        velocimeter = physics.bind(self._robot.mjcf_model.sensor.velocimeter)
        velocimeter_data = velocimeter.sensordata

        gyro = physics.bind(self._robot.mjcf_model.sensor.gyro)
        gyro_data = gyro.sensordata

        linear_velocity_tracking = -20 * np.abs(velocimeter_data[0] -
                                                self._target_linear_velocity)

        angular_velocity_penalty = -np.square(gyro_data[2])

        # Penalty for y linear velocity.
        linear_velocity_penalty = -np.square(velocimeter_data[1])

        qvel = physics.bind(self._robot.joints).qvel
        torque = physics.bind(self._robot.actuators).force

        # From https://arxiv.org/abs/2111.01674
        energy_penalty = -0.04 * np.sum(qvel * torque)

        total_reward = (linear_velocity_tracking + angular_velocity_penalty +
                        linear_velocity_penalty + energy_penalty)

        total_reward += 20 * abs(self._target_linear_velocity)

        if self._penalize_angular_acc:
            acc = physics.bind(self._robot.mjcf_model.sensor.accelerometer)
            acc_data = acc.sensordata
            total_reward -= 0.01 * np.sum(np.square(acc_data))

        return total_reward

    def initialize_episode(self, physics, random_state):
        self._robot.reinitialize_pose(physics, random_state)

        self._failure_termination = False

        super().initialize_episode(physics, random_state)

    def before_step(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._terminate_at_height is not None:
            if (physics.bind(self._robot.root_body).xpos[-1] <
                    self._terminate_at_height):
                self._failure_termination = True

        if self._terminate_not_upright is not None:
            xmat = physics.bind(self._robot.root_body).xmat
            if xmat[-1] < self._terminate_not_upright:
                self._failure_termination = True

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor
