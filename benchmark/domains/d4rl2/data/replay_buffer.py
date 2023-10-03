from typing import Optional, Union

import gym
import gym.spaces
import numpy as np

from benchmark.domains.d4rl2.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        try:
            assert dataset_dict.keys() == data_dict.keys()
        except:
            import pdb; pdb.set_trace()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 next_observation_space: Optional[gym.Space] = None):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space,
                                                  capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            mc_returns=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
            # time_step=np.empty((capacity, ), dtype=np.float32),
            # trajectory_id=np.empty((capacity, ), dtype=np.int),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._traj_counter = 0

    def __len__(self) -> int:
        return self._size

    def increment_traj_counter(self):
        self._traj_counter += 1

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_random_trajs(self, batch_size):
        observations_list = []
        actions_list = []
        rewards_list = []

        for i in range(batch_size):
            valid_traj_found = False
            i_sample = 0
            while not valid_traj_found:
                if i_sample > 5:
                    raise ValueError('could not sample trajectory with more than 10 steps!')
                i_sample += 1
                if hasattr(self.np_random, 'integers'):
                    indx = self.np_random.integers(len(self), size=1)
                else:
                    indx = self.np_random.randint(len(self), size=1)
                start_ind, end_ind, consecutive = self.get_traj_start_end_indices(indx)
                if consecutive:
                    valid_traj_found = True
                if (end_ind - start_ind) < 10:
                    valid_traj_found = False


            all_observation_keys = {}
            for key in self.dataset_dict['observations']:
                all_observation_keys[key] = self.dataset_dict['observations'][key][start_ind: end_ind + 1]

            observations_list.append(all_observation_keys)
            actions_list.append(self.dataset_dict['actions'][start_ind: end_ind + 1].squeeze())
            rewards_list.append(self.dataset_dict['rewards'][start_ind: end_ind + 1].squeeze())
        batch = {
            'observations': observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
        }
        return batch

    def get_traj_start_end_indices(self, buffer_index):
        current_traj_ind = self.dataset_dict['trajectory_id'][buffer_index].squeeze()
        matching_inds = np.where(self.dataset_dict['trajectory_id'] == current_traj_ind)[0].astype(np.int)
        step = matching_inds[1:] - matching_inds[:-1]
        if np.all(step == 1):
            consecutive = True
        else:
            consecutive = False
        return matching_inds[0], matching_inds[-1], consecutive
