import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
# import mujoco_py
import numpy as np

from dm_control.mujoco import engine
from gym import spaces

from d4rl2.envs.kitchen.RPL.adept_envs.adept_envs.franka.kitchen_multitask_v0 import  KitchenTaskRelaxV1
from d4rl2.envs.kitchen.constants import CAMERAS, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, SUCCESS_FUNCTIONS


class KitchenRPL:
    
    TASK_ELEMENTS = ["microwave", "kettle", "switch", "slide"]
    REMOVE_TASKS_WHEN_COMPLETE = False
        
    def __init__(self, 
                 tasks_to_complete=None,
                 frame_skip=40,
                 height=128,
                 width=128,
                 camera_ids = [0, 1]):
        
        self._env =  KitchenTaskRelaxV1()

        self.camera_ids = camera_ids
        self.height = height
        self.width = width

        if tasks_to_complete is not None:
            self.TASK_ELEMENTS = tasks_to_complete
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        
        self.add_cameras()
        
        obs_space = {}
        obs_space['robot_qp'] = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(9, ),
                                            dtype=np.float32)
        
        for key in ['camera_{}'.format(idx)
                    for idx in self.camera_ids] + ['camera_gripper']:
            obs_space[key + "_rgb"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )
            obs_space[key + "_depth"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_space)
        
    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def add_cameras(self, camera_id=None):
        if camera_id is not None:
            self.camera_ids.append(camera_id)
        self.cameras = dict()
        for camera_id in self.camera_ids:
            camera = engine.MovableCamera(self.sim,
                                          height=self.height,
                                          width=self.width)
            camera.set_pose(**CAMERAS[camera_id])
            self.cameras['camera_{}'.format(camera_id)] = camera
        self.cameras['camera_gripper'] = engine.Camera(
            self.sim,
            height=self.height,
            width=self.width,
            camera_id='gripper_camera_rgb')
        
    def step(self, *args, **kwargs):
        obs, _, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._get_reward_n_score(obs)
        img = self.render(mode='rgb_array')
        obs_dict = dict(robot_qp = info['obs_dict']['qp'], 
                        **img,
                        **reward_dict)
        

        return obs_dict, reward_dict['reward'], done, info

    def reset(self, *args, **kwargs):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        obs = self._env.reset(*args, **kwargs)
        reward_dict = self._get_reward_n_score(obs)
        img = self.render(mode='rgb_array')
        obs_dict = dict(robot_qp = self.obs_dict['qp'], 
                        **img,
                        **reward_dict)
        return obs_dict

    def _get_reward_n_score(self, obs):
        reward_dict = {}
        completions = []
        for element in SUCCESS_FUNCTIONS.keys():
            element_idx = OBS_ELEMENT_INDICES[element]
            element_pos = obs[..., element_idx]
            element_goal = OBS_ELEMENT_GOALS[element]
            #distance = np.linalg.norm(obs_obj - obs_goal)
            #complete = distance < BONUS_THRESH
            complete = SUCCESS_FUNCTIONS[element](element_pos, element_goal)
            reward_dict['reward_' + element] = 1.0 * complete
            if complete:
                completions.append(element)
        reward_dict['reward'] = sum([reward_dict['reward_' + obj] for obj in self.tasks_to_complete])
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            for element in self.tasks_to_complete:
              if element in self.tasks_to_complete and element in completions:
                self.tasks_to_complete.remove(element)
            #[self._current_tasks.remove(element) for element in self._current_tasks if element in completions]
            
            obs_dict = self.obs_dict
        return reward_dict
    

    def render(
        self,
        mode='human',
        depth=False,
        segmentation=False,
    ):
        imgs = {}
        if 'rgb' in mode:
            for camera_id, camera in self.cameras.items():
                img_rgb = camera.render(render_flag_overrides=dict(
                    skybox=False, fog=False, haze=False))
                imgs[camera_id + "_rgb"] = img_rgb
        
        if 'depth' in mode:
            for camera_id, camera in self.cameras.items():
                img_depth = camera.render(depth=True, segmentation=False)
                imgs[camera_id + "_depth"] = np.clip(img_depth, 0.0, 4.0)

        if 'human' in mode:
            self.renderer.render_to_window(
            )  # adept_envs.mujoco_env.MujocoEnv.render
            
        return imgs

