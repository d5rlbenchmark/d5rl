from typing import Any, Iterable, Optional
from collections import defaultdict, OrderedDict, deque
from flax.core.frozen_dict import FrozenDict
import numpy as np
import os
import gc
import jax
from jaxrl2.data.dataset import Dataset
import tqdm

RETURN_TO_GO_DICT = dict()


def calc_return_to_go(rewards, masks=None, gamma=0.99):
    if masks is None:
        masks = rewards
    global RETURN_TO_GO_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in RETURN_TO_GO_DICT.keys():
        reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0] * len(rewards)
        prev_return = rewards[-1] / (1 - gamma)
        for i in range(len(rewards)):
            reward_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * masks[-i - 1]
            prev_return = reward_to_go[-i - 1]
        RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go


def is_dict_like(x):
    return (
        isinstance(x, dict)
        or isinstance(x, FrozenDict)
        or isinstance(x, defaultdict)
        or isinstance(x, OrderedDict)
    )


def default_remapping():
    return {}


def default_obs_remapping():
    mapping_obs = OrderedDict(
        end_effector_pos="state",
        right_finger_qpos="state",
        right_finger_qvel="state",
        left_finger_qpos="state",
        left_finger_qvel="state",
        pixels="pixels",
        task_id="state",
    )

    return mapping_obs


def remap_dict(d, remapping):
    new_d = {}
    for k in d.keys():
        remapped_k = remapping.get(k, k)
        if remapped_k in new_d.keys():
            new_d[remapped_k] = np.concatenate([new_d[remapped_k], d[k]], axis=-1)
        else:
            new_d[remapped_k] = np.array(d[k])
    return new_d


def npify_dict(d):
    for k in d.keys():
        if is_dict_like(d[k]):
            d[k] = npify_dict(d[k])
        else:
            d[k] = np.array(d[k])
    return d


def append_dicts(dict1, dict2):
    assert set(dict1.keys()) == set(
        dict2.keys()
    ), f"Keys don't match: {dict1.keys()} vs {dict2.keys()}"
    for k in dict1.keys():
        if is_dict_like(dict1[k]):
            dict1[k] = append_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = np.concatenate([dict1[k], dict2[k]], axis=0)
    return dict1

def append_all_dicts(dict_list):
    first_dict = dict_list[0]
    return_dict = {}
    for k in first_dict.keys():
        if is_dict_like(first_dict[k]):
            return_dict[k] = append_all_dicts([x[k] for x in dict_list])
        else:
            return_dict[k] = np.concatenate([x[k] for x in dict_list], axis=0)
    return return_dict


def reformat_nested_dict(
    concat_dict, addition, remapping={}, obs_remapping={}, add_framestack_dim=True
):
    # convert addition to correct format
    new_format_dict = {}
    for k in addition.keys():
        for i in range(len(addition[k])):
            if is_dict_like(addition[k][i]):
                for subk in addition[k][i].keys():
                    if k not in new_format_dict.keys():
                        new_format_dict[k] = {}
                    if subk not in new_format_dict[k].keys():
                        new_format_dict[k][subk] = []
                    new_format_dict[k][subk].append(addition[k][i][subk])
            else:
                if k not in new_format_dict.keys():
                    new_format_dict[k] = []
                new_format_dict[k].append(addition[k][i])
    new_format_dict = npify_dict(new_format_dict)
    # now remap and append
    new_format_dict = remap_dict(new_format_dict, remapping)
    new_format_dict["observations"] = remap_dict(
        new_format_dict["observations"].item(), obs_remapping
    )
    new_format_dict["next_observations"] = remap_dict(
        new_format_dict["next_observations"].item(), obs_remapping
    )

    if add_framestack_dim:
        for k, v in new_format_dict["observations"].items():
            new_format_dict["observations"][k] = v[..., None]
        for k, v in new_format_dict["next_observations"].items():
            new_format_dict["next_observations"][k] = v[..., None]

    return new_format_dict


class EpisodicTransitionDataset(Dataset):
    def __init__(
        self,
        paths: Any,
        remapping=default_remapping(),
        obs_remapping=default_obs_remapping(),
        add_framestack_dim=True,
        max_traj_per_buffer=200,
        filter_success=False,
    ):
        if isinstance(paths, str):
            paths = [paths]
        assert isinstance(paths, list)

        self._paths = paths
        self.episodes = []
        new_format_dicts = []
        self.episode_as_dict = None

        self.episodes_lens = []
        for path in paths:
            assert os.path.exists(path), f"Path {path} does not exist"
            print("Loading data from", path)

            try:
                data = np.load(path, allow_pickle=True).tolist()
            except:
                continue # skip this path

            self.episodes.extend(data)

            succ = []
            num_traj = min(len(data), max_traj_per_buffer)
            for i in tqdm.tqdm(range(num_traj)):
                rews = np.array(data[i]["rewards"])
                if filter_success:
                    if not rews.any():
                        continue
                data[i]["mc_returns"] = calc_return_to_go(rews)
                data[i]["masks"] = np.ones_like(np.array(data[i]["terminals"]))

                succ.append(rews.any())
                self.episodes_lens.append(len(rews))

                new_format_dict = reformat_nested_dict(
                    self.episode_as_dict,
                    data[i],
                    remapping=remapping,
                    obs_remapping=obs_remapping,
                    add_framestack_dim=add_framestack_dim,
                )
                new_format_dicts.append(new_format_dict)

            print("Success rate:", np.mean(succ))
            gc.collect()
        
        self.episode_as_dict = append_all_dicts(new_format_dicts)
        self.episodes_lens = np.array(self.episodes_lens)
        self.episodes = np.array(self.episodes)
        super().__init__(self.episode_as_dict)

        print("Total number of episodes:", len(self.episodes))
        
    def get_random_trajs(self, num_trajs):
        trajs = []
        for _ in range(num_trajs):
            traj = self.episodes[np.random.randint(len(self.episodes))]
            trajs.append(traj)
        
        new_format_dict = {}
        for k in traj.keys():
            new_format_dict[k] = []
            for i in range(len(trajs)):
                new_format_dict[k].append(trajs[i][k])
        
        new_format_dict = npify_dict(new_format_dict)
        
        return new_format_dict

    def get_iterator(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        queue_size: int = 2,
    ):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


def main():
    path = "/nfs/kun2/users/asap7772/binsort_bridge/04_26_collect_multitask/actionnoise0.0_binnoise0.0_policypickplace_sparse0/train/out.npy"
    dataset = EpisodicTransitionDataset(path)
    batch = dataset.sample(13)
    print(jax.tree_util.tree_map(lambda x: x.shape, batch))
    breakpoint()


if __name__ == "__main__":
    main()
