import os

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'unitree_sim')
_A1_XML_PATH = os.path.join(ASSETS_DIR, 'a1.xml')

_INIT_QPOS = np.asarray([0.05, 0.8, -1.4] * 4)
_ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)
_UPRIGHT_XPOS = [0.0, 0.0, -0.125]


class A1Observables(base.WalkerObservables):

    @composer.observable
    def joints_vel(self):
        return observable.MJCFFeature('qvel', self._entity.observable_joints)

    @composer.observable
    def body_height(self):
        return observable.MJCFFeature('xpos', self._entity.root_body)[2]

    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity.root_body)

    @composer.observable
    def prev_action(self):
        return observable.Generic(lambda _: self._entity.prev_action)

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel] +
                self._collect_from_attachments('proprioception'))

    @property
    def kinematic_sensors(self):
        return ([
            self.sensors_gyro, self.sensors_accelerometer,
            self.sensors_framequat
        ] + self._collect_from_attachments('kinematic_sensors'))


class A1(base.Walker):
    """A composer entity representing a Jaco arm."""

    def _build(self, name=None):
        """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_A1_XML_PATH)
        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._root_body = self._mjcf_root.find('body', 'a1_torso')

        self._joints = self._mjcf_root.find_all('joint')
        self._actuators = self.mjcf_model.find_all('actuator')

        # Check that joints and actuators match each other.
        assert len(self._joints) == len(self._actuators)
        for joint, actuator in zip(self._joints, self._actuators):
            assert joint == actuator.joint

        # Update actuator limits as in https://arxiv.org/abs/2111.01674
        for i, actuator in enumerate(self._actuators):
            low, high = actuator.ctrlrange
            low = max(low, _INIT_QPOS[i] - _ACTION_OFFSET[i])
            high = min(high, _INIT_QPOS[i] + _ACTION_OFFSET[i])
            actuator.ctrlrange[:] = [low, high]

        self._prev_action = np.zeros(shape=self.action_spec.shape,
                                     dtype=self.action_spec.dtype)

    def initialize_episode(self, physics, random_state):
        self._prev_action = np.zeros_like(self._prev_action)

    def reinitialize_pose(self, physics, random_state):
        qpos, xpos, xquat = self.upright_pose

        self.configure_joints(physics, qpos)
        self.set_pose(physics, position=xpos, quaternion=xquat)
        self.set_velocity(physics,
                          velocity=np.zeros(3),
                          angular_velocity=np.zeros(3))

    def apply_action(self, physics, action, random_state):
        super().apply_action(physics, action, random_state)

        # Updates previous action.
        self._prev_action[:] = action

    @property
    def upright_pose(self):
        return base.WalkerPose(qpos=_INIT_QPOS, xpos=_UPRIGHT_XPOS)

    def _build_observables(self):
        return A1Observables(self)

    @property
    def root_body(self):
        return self._root_body

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def observable_joints(self):
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    @property
    def prev_action(self):
        return self._prev_action
