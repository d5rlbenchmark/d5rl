import copy
import multiprocessing as mp
import os
from functools import partial

import gym
import imageio
import numpy as np
import tqdm
from jaxrl2.data import ReplayBuffer

from d4rl2.envs.a1.collect.maze.policy import Policy
from d4rl2.envs.a1.collect.saving import save_data


def collect_data(seed,
                 target_env_name,
                 collect_env_name,
                 num_samples,
                 maze_name,
                 exclude_expert=True,
                 noise_std=0.1,
                 max_attempts=10):

    gym_target_env = gym.make(target_env_name)
    gym_target_env.reset()
    target_env = gym_target_env.unwrapped._env

    gym_collect_env = gym.make(collect_env_name)
    gym_collect_env.seed(seed)

    collect_env = gym_collect_env.unwrapped._env
    policy = Policy(collect_env, noise_std=noise_std, seed=seed)

    def get_gym_obs(timestep):
        obs = gym.spaces.flatten(gym_collect_env.unwrapped.observation_space,
                                 timestep.observation)
        return obs

    def get_gym_action(action):
        low = gym_collect_env.unwrapped.action_space.low
        high = gym_collect_env.unwrapped.action_space.high

        return (action - low) / (high - low) * 2 - 1

    observations = []
    actions = []
    rewards = []
    dones = []
    masks = []
    next_observations = []

    i = 0
    pbar = tqdm.tqdm(total=num_samples)
    prev_n = 0

    while True:
        maze_random_state = copy.deepcopy(
            collect_env.task._maze_arena.maze._random_state.get_state())
        random_state = copy.deepcopy(collect_env.random_state.get_state())
        solved = False
        attempts = 0
        while not solved and attempts < max_attempts:
            attempts += 1
            if attempts == max_attempts:
                print("Failed to find a solution.")
            collect_env.task._maze_arena.maze._random_state.set_state(
                maze_random_state)
            collect_env.random_state.set_state(random_state)
            while True:
                timestep = collect_env.reset()
                # Exclude trajectories that go from the eval env start to end.
                if (not exclude_expert or
                    (not (np.allclose(
                        collect_env.task._maze_arena.target_positions[0],
                        target_env.task._maze_arena.target_positions[0])
                          and np.allclose(
                              collect_env.task._maze_arena.spawn_positions[0],
                              target_env.task._maze_arena.spawn_positions[0])))
                    ):
                    break

            obs = get_gym_obs(timestep)
            t = 0
            images = []
            end_marker = len(observations)
            while not timestep.last():
                t += 1
                action = policy(timestep)
                timestep = collect_env.step(action)
                images.append(collect_env.physics.render(camera_id=0))

                action = get_gym_action(action)
                robot_position = timestep.observation[
                    'unitree_a1/body_position']
                done = target_env.task._goal_reached(robot_position)
                if done:
                    reward = 1.0
                    mask = 0.0
                else:
                    reward = 0.0
                    mask = 1.0
                # TODO: Relabel rewards and masks
                next_obs = get_gym_obs(timestep)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(timestep.last())
                masks.append(mask)
                next_observations.append(next_obs)

                obs = next_obs

                if (t == gym_collect_env._max_episode_steps
                        or collect_env.task._is_failure):
                    # The robot has failed to reach the goal, not saving the trajectory.
                    observations = observations[:end_marker]
                    actions = actions[:end_marker]
                    rewards = rewards[:end_marker]
                    dones = dones[:end_marker]
                    masks = masks[:end_marker]
                    next_observations = next_observations[:end_marker]
                    solved = False
                    break
                else:
                    solved = True

        pbar.update(len(observations) - prev_n)
        prev_n = len(observations)

        if len(observations) >= num_samples:
            break

        if solved:
            video_dir = os.path.join('videos', maze_name)
            os.makedirs(video_dir, exist_ok=True)
            imageio.mimsave(os.path.join(video_dir,
                                         f'{seed}_{i}_{attempts}.mp4'),
                            images,
                            fps=20)
            i += 1

    assert len(observations) == len(actions)
    assert len(actions) == len(rewards)
    assert len(rewards) == len(dones)
    assert len(dones) == len(masks)
    assert len(masks) == len(next_observations)

    return (observations, actions, rewards, dones, masks, next_observations)


def collect_parallel(maze_name,
                     num_samples,
                     seed,
                     exclude_expert=True,
                     noise_std=0.1):
    target_env_name = f'a1-{maze_name}-diverse-v0'
    collect_env_name = f'a1-{maze_name}-collect-v0'
    mp.set_start_method('spawn')

    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        collect_data_ = partial(collect_data,
                                target_env_name=target_env_name,
                                collect_env_name=collect_env_name,
                                num_samples=num_samples // num_cores,
                                maze_name=maze_name,
                                exclude_expert=exclude_expert,
                                noise_std=noise_std)

        data = pool.map(collect_data_, [i + seed for i in range(num_cores)])

    total_size = np.sum([len(x[0]) for x in data])

    env = gym.make(target_env_name)
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 total_size)

    for (observations, actions, rewards, dones, masks,
         next_observations) in data:
        for obs, act, rew, done, mask, next_obs in zip(observations, actions,
                                                       rewards, dones, masks,
                                                       next_observations):
            replay_buffer.insert(
                dict(observations=obs,
                     actions=act,
                     rewards=rew,
                     dones=done,
                     masks=mask,
                     next_observations=next_obs))

    dataset_folder = os.path.join(os.path.expanduser('~'), '.d4rl2',
                                  'datasets', 'bc2iql')
    os.makedirs(dataset_folder, exist_ok=True)
    h5path = os.path.join(dataset_folder, f'a1-{maze_name}.hdf5')
    save_data(replay_buffer, h5path)
