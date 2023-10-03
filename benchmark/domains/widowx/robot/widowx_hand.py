import collections
import os

from dm_control import composer, mjcf
from dm_control.entities.manipulators import base

_ASSETS_DIR = os.path.dirname(__file__)
_WX250S_XML_HAND_PATH = os.path.join(_ASSETS_DIR, "wx250s_hand.xml")


class WidowXHand(base.RobotHand):
    def _build(self, name=None):

        self._mjcf_root = mjcf.from_path(_WX250S_XML_HAND_PATH)
        if name:
            self._mjcf_root.model = name

        self._tool_center_point = self._mjcf_root.find("site", "gripper_ee")

        self._bodies = self.mjcf_model.find_all("body")
        self._joints = self._mjcf_root.find_all("joint")
        self._finger_actuators = self._mjcf_root.find_all("actuator")

    def _build_observables(self):
        return WidowXHandObservables(self)

    @property
    def tool_center_point(self):
        return self._tool_center_point

    @property
    def joints(self):
        return self._joints

    @property
    def actuators(self):
        return self._finger_actuators

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def set_grasp(self, physics, close_factors):
        # From https://github.com/deepmind/dm_control/
        if not isinstance(close_factors, collections.abc.Iterable):
            close_factors = (close_factors,) * len(self.joints)
        for joint, finger_factor in zip(self.joints, close_factors):
            joint_mj = physics.bind(joint)
            min_value, max_value = joint_mj.range
            joint_mj.qpos = min_value + (max_value - min_value) * finger_factor
            physics.after_reset()

        physics.bind(self.actuators).ctrl = 0


class WidowXHandObservables(composer.Observables):
    pass
