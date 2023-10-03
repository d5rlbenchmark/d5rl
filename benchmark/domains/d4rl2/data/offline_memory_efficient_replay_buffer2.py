import collections
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict

import copy
import glob
from gym.utils import seeding
from tqdm import tqdm, trange

from benchmark.domains.d4rl2.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer


class OfflineMemoryEfficientReplayBuffer2(MemoryEfficientReplayBuffer):

    def __init__(self,
                 env,
                 datasets_urls,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 include_pixels: bool = True,
                 debug=False):

        super().__init__(observation_space, action_space, capacity,
                         include_pixels)

        self._env = env

        observation_space = copy.deepcopy(self._env.observation_space.spaces)
        pixel_observation_space = observation_space.pop('pixels')
        self._num_stack = pixel_observation_space.shape[-1]

        # np_random, seed = seeding.np_random(seed)

        self.files = sum([glob.glob(url + '/**/*.pkl', recursive=True) for url in datasets_urls], [])
        print(f'There are {len(self.files)} total episodes in the offline datasets')

        self.load(debug)

    def load(self, debug=False):
        # np_random.shuffle(self.files)
        np.random.shuffle(self.files)
        for ep_idx, file in enumerate(tqdm(self.files, total=len(self.files), desc=f"Loading offline data")):
            #print(file)
            episode = np.load(file, allow_pickle=True)
            episode['reward'] = np.sum([episode['reward_' + task] for
                                       task in self._env.TASK_ELEMENTS], axis = 0)

            frames = collections.deque(maxlen=self._num_stack)
            for _ in range(self._num_stack):
                frames.append(
                    np.concatenate([episode[cam + '_rgb'][0] for cam in self._env.cameras],
                                   axis=-1))

            for t in range(episode['reward'].shape[0] - 1):
                transition = dict()
                transition['observations'] = dict()
                transition['observations']['pixels'] = np.stack(frames,
                                                                axis=-1)

                transition['observations']['states'] = episode['robot_qp'][t]
                #np.concatenate(
                #        [episode['robot_qp'][t],
                #         episode['ee_qp'][t],
                #         episode['ee_forces'][t]],
                #         axis=-1)

                transition['actions'] = np.clip(episode['action'][t + 1], -0.99, 0.99)
                transition['rewards'] = episode['reward'][t + 1]

                frames.append(
                    np.concatenate([
                        episode[cam + '_rgb'][t + 1]  for cam in self._env.cameras],
                                   axis=-1))

                transition['next_observations'] = dict()
                transition['next_observations']['pixels'] = np.stack(frames,
                                                                     axis=-1)

                transition['next_observations']['states'] = episode['robot_qp'][t + 1]
                #np.concatenate(
                #       [episode['robot_qp'][t + 1],
                #         episode['ee_qp'][t + 1],
                #         episode['ee_forces'][t + 1]],
                #         axis=-1)

                transition['masks'] = 0.0
                transition['dones'] = 0.0

                self.insert(transition)

                import pdb; pdb.set_trace()

            if debug and ep_idx > 10:
                break

        print(f"self._size: {self._size}, self._capacity: {self._capacity}")
