import gym
import numpy as np


class KitchenImageConcatWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        obs_space = {}
#        obs_space['robot_qp'] = gym.spaces.Box(low=-np.inf,
#                                               high=np.inf,
#                                               shape=(9, ),
#                                               dtype=np.float32)
#        obs_space['ee_qp'] = gym.spaces.Box(low=-np.inf,
#                                            high=np.inf,
#                                            shape=(7, ),
#                                            dtype=np.float32)
#        obs_space['ee_forces'] = gym.spaces.Box(low=-np.inf,
#                                                high=np.inf,
#                                                shape=(12, ),
#                                                dtype=np.float32)

        obs_space['states'] = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(9, ),
                                                dtype=np.float32)
        obs_space['pixels'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 3 * len(env.cameras)),
            dtype=np.uint8,
        )

        self.observation_space = gym.spaces.Dict(obs_space)

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)

        obs = dict()
        obs['pixels'] = np.concatenate([obs_dict[cam + '_rgb'] for cam in self.env.cameras], axis=-1)
        obs['states'] = obs_dict['robot_qp']
        # obs['states'] = np.concatenate([obs_dict['robot_qp'],
        #                                 obs_dict['ee_qp'],
        #                                 obs_dict['ee_forces']],
        #                                 axis=-1)
        return obs

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        obs = dict()
        obs['pixels'] = np.concatenate([obs_dict[cam + '_rgb'] for cam in self.env.cameras], axis=-1)
        # obs['pixels'] = np.concatenate([obs_dict[cam + '_rgb'] for cam in self.env.cameras], axis=-1)
        obs['states'] = obs_dict['robot_qp']
        # obs['states'] = np.concatenate([obs_dict['robot_qp'],
        #                                 obs_dict['ee_qp'],
        #                                 obs_dict['ee_forces']],
        #                                 axis=-1)
        self.image_state_obs = obs
        return obs, reward, done, info


    def get_image_obs(self):
        return self.image_state_obs


#class KitchenImageConcatWrapper(gym.Wrapper):
#
#    def __init__(self, env):
#        super().__init__(env)
#
#        obs_space = {}
#        obs_space['robot_qp'] = gym.spaces.Box(low=-np.inf,
#                                               high=np.inf,
#                                               shape=(9, ),
#                                               dtype=np.float32)
#        obs_space['ee_qp'] = gym.spaces.Box(low=-np.inf,
#                                            high=np.inf,
#                                            shape=(7, ),
#                                            dtype=np.float32)
#        obs_space['ee_forces'] = gym.spaces.Box(low=-np.inf,
#                                                high=np.inf,
#                                                shape=(12, ),
#                                                dtype=np.float32)
#
#        obs_space['pixels'] = gym.spaces.Box(
#            low=0,
#            high=255,
#            shape=(128, 128, 9),
#            dtype=np.uint8,
#        )
#
#        self.observation_space = gym.spaces.Dict(obs_space)
#
#    def reset(self):
#        obs = self.env.reset()
#        obs['pixels'] = np.concatenate([
#            obs['camera_0_rgb'], obs['camera_1_rgb'], obs['camera_gripper_rgb']
#        ],
#                                       axis=-1)
#        obs = {k: obs[k] for k in self.observation_space.keys()}
#        return obs
#
#    def step(self, action):
#        obs, reward, done, info = self.env.step(action)
#        obs['pixels'] = np.concatenate([
#            obs['camera_0_rgb'], obs['camera_1_rgb'], obs['camera_gripper_rgb']
#        ],
#                                       axis=-1)
#        obs = {k: obs[k] for k in self.observation_space.keys()}
#        return obs, reward, done, info


class Kitchen2ImageConcatWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        obs_space = {}
        obs_space['robot_qp'] = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(9, ),
                                               dtype=np.float32)
        obs_space['ee_qp'] = gym.spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(7, ),
                                            dtype=np.float32)
        obs_space['ee_forces'] = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(12, ),
                                                dtype=np.float32)

        obs_space['pixels'] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 6),
            dtype=np.uint8,
        )

        self.observation_space = gym.spaces.Dict(obs_space)

    def reset(self):
        obs = self.env.reset()
        obs['pixels'] = np.concatenate([
            obs['camera_0_rgb'], obs['camera_gripper_rgb']
        ],
                                       axis=-1)
        obs = {k: obs[k] for k in self.observation_space.keys()}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['pixels'] = np.concatenate([
            obs['camera_0_rgb'], obs['camera_gripper_rgb']
        ],
                                       axis=-1)
        obs = {k: obs[k] for k in self.observation_space.keys()}
        return obs, reward, done, info



class KitchenStateWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        obs_space = {}
        obs_space['robot_qp'] = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(9, ),
                                               dtype=np.float32)
        obs_space['ee_qp'] = gym.spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(7, ),
                                            dtype=np.float32)
        obs_space['ee_forces'] = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(12, ),
                                                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_space)

    def reset(self):
        obs = self.env.reset()
        obs['pixels'] = np.concatenate([
            obs['camera_0_rgb'], obs['camera_gripper_rgb']
        ],
                                       axis=-1)
        obs = {k: obs[k] for k in self.observation_space.keys()}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['pixels'] = np.concatenate([
            obs['camera_0_rgb'], obs['camera_gripper_rgb']
        ],
                                       axis=-1)
        obs = {k: obs[k] for k in self.observation_space.keys()}
        return obs, reward, done, info
