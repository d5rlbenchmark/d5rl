import os

import numpy as np
from dm_control.composer.initializers import prop_initializer
from dm_control.composer.variation import distributions, rotations
from dm_control.utils import transformations as tr

from benchmark.domains.widowx.tasks.base_task import BaseTask
from benchmark.mujoco.utils import XMLObject, get_object_com

ASSERT_PATH = os.path.join(os.path.dirname(__file__), "..", "objects")


class PickAndPlace(BaseTask):
    def __init__(self):
        super().__init__()

        path = os.path.join(ASSERT_PATH, "Utana_5_Porcelain_Ramekin_Large", "model.xml")
        self._plate = XMLObject(path)
        self._arena.attach(self._plate)

        path = os.path.join(
            ASSERT_PATH, "Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler", "model.xml"
        )
        self._toy = XMLObject(path, scale=0.2)
        self._arena.add_free_entity(self._toy)

        camera_target = self._arm.mjcf_model.find("body", "camera_target")
        camera_target.pos[:] = [0.0, 0.25, 0.0]

        camera = self._arm.mjcf_model.find("camera", "pixels")
        camera.pos[:] = [0.0, 0.75, 0.5]

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

        delta = 0.05
        pos = random_state.uniform(
            low=[-0.2 - delta, 0.2 - delta, 0.0], high=[-0.2 + delta, 0.2 + delta, 0.0]
        )
        euler = random_state.uniform((0, 0, -np.pi), (0, 0, np.pi))
        self._plate.set_pose(physics, position=pos, quaternion=tr.euler_to_quat(euler))
        delta = 0.05
        pos = distributions.Uniform(
            [0.0 + delta, 0.2 - delta, 0.005], [0.2 + delta, 0.2 + delta, 0.025]
        )
        quat = rotations.UniformQuaternion()

        prop_placer = prop_initializer.PropPlacer(
            props=[self._toy],
            position=pos,
            quaternion=quat,
            max_settle_physics_attempts=10,
            settle_physics=True,
        )

        prop_placer(physics, random_state)

    def before_step(self, physics, action, random_state):
        super().before_step(physics, action, random_state)

        toy_pos = get_object_com(physics, self._toy)
        plate_pos = get_object_com(physics, self._plate)

        distance = np.linalg.norm(toy_pos - plate_pos)

        self._is_success = distance < 0.05

    def should_terminate_episode(self, physics):
        return False

    def get_reward(self, physics):
        return float(self._is_success)
