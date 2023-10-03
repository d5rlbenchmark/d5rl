from typing import Dict

import gym
import numpy as np

from jaxrl2.data.dataset import Dataset

import collections ###===### ###---###


def evaluate(agent, env: gym.Env, num_episodes: int, progress_bar=False) -> Dict[str, float]: ###===### ###---###
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    ###===###
    # for _ in range(num_episodes):
    tbar = trange(num_episodes) if progress_bar else range(num_episodes)
    for _ in tbar:
    ###---###
        observation, done = env.reset(), False
        while not done:
            out = agent.eval_actions(observation)
            if isinstance(out, tuple):
                action, agent = out
            else:
                action = out
            observation, _, done, _ = env.step(action)

    return {
        'return_mean': np.mean(env.return_queue),
        'return_max': np.max(env.return_queue),
        'return_min': np.min(env.return_queue),
        'return_std': np.std(env.return_queue),
        'length_mean': np.mean(env.length_queue),
        'length_max': np.max(env.length_queue),
        'length_min': np.min(env.length_queue),
        'length_std': np.std(env.length_queue)
    }

###===###
def evaluate_adroit(agent, env: gym.Env, num_episodes: int, progress_bar=False) -> Dict[str, float]: ###===### ###---###
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    ###===###
    # for _ in range(num_episodes):
    tbar = trange(num_episodes) if progress_bar else range(num_episodes)
    for _ in tbar:
    ###---###
        observation, done = env.reset(), False
        while not done:
            # action = agent.eval_actions(observation)
            out = agent.eval_actions(observation)
            if isinstance(out, tuple):
                action, agent = out
            else:
                action = out
            observation, _, done, _ = env.step(action)

    successes = (np.array(env.return_queue) > 0).astype(np.float32)

    return {
        'return_mean': np.mean(env.return_queue),
        'return_max': np.max(env.return_queue),
        'return_min': np.min(env.return_queue),
        'return_std': np.std(env.return_queue),

        'success_mean': np.mean(successes),
        'success_max': np.max(successes),
        'success_min': np.min(successes),
        'success_std': np.std(successes),

        'length_mean': np.mean(env.length_queue),
        'length_max': np.max(env.length_queue),
        'length_min': np.min(env.length_queue),
        'length_std': np.std(env.length_queue)
    }


def evaluate_kitchen(agent, env: gym.Env, num_episodes: int, progress_bar=False) -> Dict[str, float]: ###===### ###---###
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    ###===###
    # for _ in range(num_episodes):
    objects_maniplated_queue = collections.defaultdict(list)

    tbar = trange(num_episodes) if progress_bar else range(num_episodes)
    for _ in tbar:
    ###---###
        objects_maniplated = collections.defaultdict(list)

        observation, done = env.reset(), False
        while not done:
            # action = agent.eval_actions(observation)
            out = agent.eval_actions(observation)
            if isinstance(out, tuple):
                action, agent = out
            else:
                action = out
            env_step = env.step(action)
            if len(env_step) == 4:
                observation, _, done, info = env_step
            elif len(env_step) == 5:
                observation, _, done, _, info = env_step
            else:
                raise ValueError(f"env_step should be length 4 or 5 but is length {len(env_step)}")

            for key, val in info.items():
                if "reward " in key:
                    objects_maniplated[key[len("reward "):]].append(val)

        total = 0
        for key, val in objects_maniplated.items():
            objects_maniplated_queue[key].append(np.sum(val) > 0)
            total += np.sum(val) > 0
        objects_maniplated_queue["total"].append(total)

    successes = [ep_return > 0 for ep_return in env.return_queue]
    return_info = {
        'return_mean': np.mean(env.return_queue),
        'return_max': np.max(env.return_queue),
        'return_min': np.min(env.return_queue),
        'return_std': np.std(env.return_queue),

        'success_mean': np.mean(successes),
        'success_max': np.max(successes),
        'success_min': np.min(successes),
        'success_std': np.std(successes),

        'length_mean': np.mean(env.length_queue),
        'length_max': np.max(env.length_queue),
        'length_min': np.min(env.length_queue),
        'length_std': np.std(env.length_queue),
    }

    for key, val in objects_maniplated_queue.items():
        return_info[key + "_manipulated_mean"] = np.mean(val)
        return_info[key + "_manipulated_max"] = np.max(val)
        return_info[key + "_manipulated_min"] = np.min(val)
        return_info[key + "_manipulated_std"] = np.std(val)

    return return_info
###---###

def evaluate_log_prob(agent, dataset: Dataset, batch_size: int = 2048) -> float:
    num_iters = len(dataset) // batch_size
    total_log_prob = 0.0
    for j in range(num_iters):
        indx = np.arange(j * batch_size, (j + 1) * batch_size)
        batch = dataset.sample(batch_size, keys=("observations", "actions"), indx=indx)
        log_prob = agent.eval_log_probs(batch)
        total_log_prob += log_prob

    return total_log_prob / num_iters
