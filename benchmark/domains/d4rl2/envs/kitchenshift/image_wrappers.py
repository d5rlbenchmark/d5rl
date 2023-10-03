import copy
import os

import gym
import numpy as np
# import xml.etree.ElementTree as ET
from lxml import etree as ET

from .constants import FRANKA_INIT_QPOS, OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES
from .kitchen_base import KitchenBase


class KitchenBaseRGB(KitchenBase):
    """Base kitchen environment with RGB observations"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        for key in ['camera_{}'.format(idx)
                    for idx in self.camera_ids] + ['camera_gripper']:
            self.observation_space[key + "_rgb"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )

    def _get_obs_dict(self, noise_ratio='default', robot_cache_obs=False):
        obs_dict = super()._get_obs_dict(noise_ratio=noise_ratio, 
                        robot_cache_obs=robot_cache_obs)

        rgb = self.render(mode='rgb')
        for k, v in rgb.items():
            obs_dict[k] = v
        return obs_dict



class KitchenBaseRGBD(KitchenBaseRGB):
    """Base kitchen environment with RGB observations"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        for key in ['camera_{}'.format(idx)
                    for idx in self.camera_ids] + ['camera_gripper']:
            self.observation_space[key + "_depth"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width),
                dtype=np.float32,
            )

    def _get_obs_dict(self, noise_ratio='default', robot_cache_obs=False):
        obs_dict = super()._get_obs_dict(noise_ratio=noise_ratio, 
                        robot_cache_obs=robot_cache_obs)

        depth = self.render(mode='depth')
        for k, v in depth.items():
            obs_dict[k] = v
        return obs_dict

