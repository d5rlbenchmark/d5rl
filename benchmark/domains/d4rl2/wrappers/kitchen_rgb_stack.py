import gym
import numpy as np


class KitchenImageConcatWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        obs_space = {}
        
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

        if self.env.pretrained_encoder is not None:
            obs_space["pretrained_representations"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.pretrained_encoder._embed_dim,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_space)

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        
        obs = dict()
        obs['pixels'] = np.concatenate([
            obs_dict[cam + '_rgb'] for cam in self.env.cameras],
                                       axis=-1)
        obs['states'] = obs_dict['robot_qp']

        obs['pretrained_representations'] = obs_dict['pretrained_representations']



        return obs

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        
        obs = dict()
        obs['pixels'] = np.concatenate([
            obs_dict[cam + '_rgb'] for cam in self.env.cameras],
                                       axis=-1)
        obs['states'] = obs_dict['robot_qp']

        obs['pretrained_representations'] = obs_dict['pretrained_representations']

        self.image_state_obs = obs
        return obs, reward, done, info


    def get_image_obs(self):
        return self.image_state_obs

