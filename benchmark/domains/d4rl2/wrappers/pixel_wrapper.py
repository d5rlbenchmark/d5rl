import gym
import numpy as np
from gym.spaces import Box


def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))


class PixelEnv(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(128, 128, 3),
                                     dtype=np.uint8)

    def observation(self, observation):
        return _process_image(observation['image'])