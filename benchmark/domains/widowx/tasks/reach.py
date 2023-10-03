import os
from functools import partial

import numpy as np
from dm_control.composer.initializers import prop_initializer
from dm_control.composer.variation import distributions, rotations

from benchmark.domains.widowx.tasks.base_task import BaseTask
from benchmark.mujoco.utils import XMLObject, get_object_com

ASSERT_PATH = os.path.join(os.path.dirname(__file__), "..", "objects")


class Reach(BaseTask):
    def __init__(self, sparse: bool = False):
        super().__init__()

        self._sparse = sparse

        path = os.path.join(
            ASSERT_PATH, "Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler", "model.xml"
        )
        self._toy = XMLObject(path, scale=0.2)
        self._arena.add_free_entity(self._toy)

        camera_target = self._arm.mjcf_model.find("body", "camera_target")
        camera_target.pos[:] = [0.0, 0.25, 0.0]

        camera = self._arm.mjcf_model.find("camera", "pixels")
        camera.pos[:] = [0.0, 0.75, 0.25]

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

        delta = 0.05
        pos = distributions.Uniform(
            [-0.2 + delta, 0.2 - delta, 0.005], [0.2 + delta, 0.2 + delta, 0.025]
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

    def get_sparse_reward(self, negative_distance):
        sparse_reward = -negative_distance < 0.02
        if hasattr(sparse_reward, "astype"):
            return sparse_reward.astype(np.float32)
        else:
            return float(sparse_reward)

    def get_reward(self, physics):
        toy_pos = get_object_com(physics, self._toy)
        tcp_pos = physics.bind(self._hand.tool_center_point).xpos

        distance = np.linalg.norm(toy_pos - tcp_pos)

        reward = -distance

        if self._sparse:
            return self.get_sparse_reward(reward)
        else:
            return reward


ReachSparse = partial(Reach, sparse=True)
