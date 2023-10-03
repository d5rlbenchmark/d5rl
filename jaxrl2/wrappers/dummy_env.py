from gym import Env
import gym.spaces
import numpy as np


class DummyEnv(Env):
    def __init__(
        self, image_shape=(128, 128, 3, 1), state_shape=(17, 1), action_shape=(7,)
    ) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(0, 255, self.image_shape),
                "state": gym.spaces.Box(-1, 1, self.state_shape),
            }
        )

        self.action_space = gym.spaces.Box(-1, 1, self.action_shape)

    def step(self, action):
        return (
            {
                "pixels": np.zeros(self.image_shape)[None],
                "state": np.zeros(self.state_shape)[None],
            },
            0,
            0,
            {},
        )

    def reset(self):
        return {
            "pixels": np.zeros(self.image_shape)[None],
            "state": np.zeros(self.state_shape)[None],
        }
