import os
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.entities.manipulators import base

_ASSETS_DIR = os.path.dirname(__file__)
_WX250S_XML_ARM_PATH = os.path.join(_ASSETS_DIR, "wx250s_arm.xml")


class WidowXArm(base.RobotArm):
    def _build(self, name=None):
        self._mjcf_root = mjcf.from_path(_WX250S_XML_ARM_PATH)
        if name:
            self._mjcf_root.model = name

        self._actuators = self._mjcf_root.find_all("actuator")
        self._joints = self._mjcf_root.find_all("joint")
        self._bodies = self.mjcf_model.find_all("body")
        self._base_site = self.mjcf_model.find("site", "base_site")
        self._wrist_site = self.mjcf_model.find("site", "wrist_site")

    def _build_observables(self):
        return WidowXArmObservables(self)

    @property
    def actuators(self):
        return self._actuators

    @property
    def joints(self):
        return self._joints

    @property
    def base_site(self):
        return self._base_site

    @property
    def wrist_site(self):
        return self._wrist_site

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def set_pose(self, physics, qpos: Optional[np.ndarray] = None):
        if qpos is None:
            physics.bind(self.joints).qpos = [1.57, -0.6, 0.6, 0, 1.57, 0]
        physics.bind(self.actuators).ctrl = 0


class WidowXArmObservables(composer.Observables):
    pass
