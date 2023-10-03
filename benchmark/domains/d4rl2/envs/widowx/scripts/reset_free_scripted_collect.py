import argparse
import os
import os.path as osp
import time

import numpy as np
import roboverse
from roboverse.bullet import object_utils
from roboverse.policies.pick_place import PickPlace
from roboverse.utils import get_timestamp
from tqdm import tqdm

NFS_PATH = '/nfs/kun1/users/avi/scripted_sim_datasets/'
EPSILON = 0.1


def main(args):

    timestamp = get_timestamp()
    if osp.exists(NFS_PATH):
        data_save_path = osp.join(NFS_PATH, args.save_directory)
    else:
        data_save_path = osp.join(__file__, "../..", "data",
                                  args.save_directory)
    data_save_path = osp.abspath(data_save_path)

    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)

    env = roboverse.make(args.env_name, transpose_image=True, gui=args.gui)
    env.reset()

    policy = PickPlace(env, pick_point_z=-0.3)

    forward_paths = []
    reset_paths = []
    n_success = 0
    n_attempts = 0

    for j in tqdm(range(args.num_trajectories)):
        env.reset_robot_only()
        info = env.get_info()

        if info['place_success']:
            # drop_point = np.random.uniform(low=env.object_position_low,
            #                                high=env.object_position_high)
            container_position, original_object_positions = \
                object_utils.generate_object_positions_single(
                    env.object_position_low, env.object_position_high,
                    env.container_position_low, env.container_position_high,
                    min_distance_large_obj=env.min_distance_from_object,
                )
            policy.reset(drop_point=original_object_positions[0])
            path_type = 'reset'
        else:
            policy.reset()
            path_type = 'forward'
            n_attempts += 1

        observations = []
        actions = []
        rewards = []
        terminals = []
        env_infos = []
        next_observations = []

        for _ in range(args.num_timesteps):
            obs = env.get_observation()
            action, agent_info = policy.get_action()
            action += np.random.normal(loc=0, scale=args.noise, size=8)
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            if args.slow:
                time.sleep(0.05)
            next_obs, rew, done, info = env.step(action)

            obs["image"] = np.uint8(obs["image"] * 255.)
            next_obs["image"] = np.uint8(next_obs["image"] * 255.)

            observations.append(obs)
            next_observations.append(next_obs)
            rewards.append(rew)
            terminals.append(done)
            actions.append(action)
            env_infos.append(info)

        if info['place_success'] and path_type == 'forward':
            n_success += 1

        path = dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            env_infos=env_infos,
        )

        if path_type == 'forward':
            forward_paths.append(path)
        else:
            reset_paths.append(path)

    print('fw', len(forward_paths))
    print('rs', len(reset_paths))
    print('success rate', n_success / n_attempts)

    path_fw = osp.join(
        data_save_path,
        "scripted_forward_{}_{}.npy".format(args.env_name, timestamp))
    path_rs = osp.join(
        data_save_path,
        "scripted_reset_{}_{}.npy".format(args.env_name, timestamp))

    np.save(path_fw, forward_paths)
    np.save(path_rs, reset_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("--slow", action='store_true', default=False)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
