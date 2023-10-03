import copy
from collections import deque

import gym
import numpy as np


class AddPreviousActions(gym.ObservationWrapper):

    def __init__(self, env, action_history: int = 1, *args, **kwargs):
        self.actions = deque(maxlen=action_history)
        super().__init__(env, *args, **kwargs)

        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert 'actions' not in env.observation_space.spaces
        new_obs = copy.copy(env.observation_space.spaces)
        low = np.repeat(env.action_space.low, repeats=action_history, axis=0)
        high = np.repeat(env.action_space.high, repeats=action_history, axis=0)
        action_space = gym.spaces.Box(low, high)
        new_obs['actions'] = action_space

        self.observation_space = gym.spaces.Dict(new_obs)

    def reset(self, *args, **kwargs):
        for _ in range(self.actions.maxlen):
            self.actions.append(np.zeros_like(self.action_space.low))
        return super().reset(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.actions.append(copy.deepcopy(action))
        return obs, reward, done, info

    def observation(self, observation):
        observation = copy.copy(observation)
        observation['actions'] = np.concatenate(self.actions)
        return observation
        