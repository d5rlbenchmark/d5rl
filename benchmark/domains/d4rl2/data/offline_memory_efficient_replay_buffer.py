import collections
from typing import Iterable, Optional

import gym
import jax
import numpy as np
from flax.core import frozen_dict

from benchmark.domains.d4rl2.data.memory_efficient_replay_buffer import \
    MemoryEfficientReplayBuffer


class OfflineMemoryEfficientReplayBuffer(MemoryEfficientReplayBuffer):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 offline_generator,
                 include_pixels: bool = True,
                 num_offline_samples: int = 10,
                 debug=False):

        super().__init__(observation_space, action_space, capacity,
                         include_pixels)

        self._offline_generator = offline_generator
        self._num_offline_samples = num_offline_samples
        self.load(debug)

    def load(self, debug=False):
        while self._size < self._capacity:
            if self._size % 1000 == 0:
                print(f"Loaded {self._size}/{self._capacity} transitions.")
            self.insert_offline()

            if debug and self._size > 2000:
                break

    def insert_offline(self):
        t = next(self._offline_generator)
        self.insert(t)

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
               reinsert_offline=True) -> frozen_dict.FrozenDict:

        batch = super().sample(batch_size, keys, indx)
        if reinsert_offline:
            for _ in range(self._num_offline_samples):
                self.insert_offline()
        return batch

    def get_offline_iterator(self):
        while True:
            yield next(self._offline_generator)

    def get_iterator(self,
                     batch_size: int,
                     keys: Optional[Iterable[str]] = None,
                     indx: Optional[np.ndarray] = None,
                     queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
            for _ in range(self._num_offline_samples):
                self.insert_offline()
