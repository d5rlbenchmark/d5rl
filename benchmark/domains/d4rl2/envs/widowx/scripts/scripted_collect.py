import argparse
import functools
import os
import os.path as osp
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

import d4rl2.envs.widowx.roboverse
from d4rl2.envs.widowx.roboverse.bullet import object_utils
from d4rl2.envs.widowx.roboverse.policies import *
from d4rl2.envs.widowx.roboverse.utils import get_timestamp

matplotlib.use('Agg')

EPSILON = 0.1

NFS_PATH = '/nfs/kun2/users/aviralkumar/d4rl2_widowx_data_final/'


def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim, transpose_image):

    def reshape_image(obs, img_dim, transpose_image):
        if transpose_image:
            obs["image"] = np.reshape(obs["image"], (3, img_dim, img_dim))
            obs["image"] = np.transpose(obs["image"], [1, 2, 0])
            obs["image"] = np.uint8(obs["image"] * 255.)
        else:
            obs["image"] = np.reshape(np.uint8(obs["image"] * 255.),
                                      (img_dim, img_dim, 3))
        return obs

    reshape_image(observation, img_dim, transpose_image)
    traj["observations"].append(observation)
    reshape_image(next_observation, img_dim, transpose_image)
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def collect_one_traj(env,
                     policy,
                     num_timesteps,
                     noise,
                     accept_trajectory_key,
                     transpose_image,
                     lower_gripper_noise=False,
                     reset_free=False,
                     task_index=None,
                     traj_id=0):
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim

    observation = env.reset()
    policy.reset()

    time.sleep(1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    image_array = []

    for j in range(num_timesteps):

        observation = env.get_observation()
        action, agent_info = policy.get_action()

        obs_image = env.render_obs()
        image_array.append(obs_image)

        # In case we need to pad actions by 1 for easier realNVP modelling
        env_action_dim = env.action_space.shape[0]
        if env_action_dim - action.shape[0] == 1:
            action = np.append(action, 0)
        if lower_gripper_noise:
            noise_scale = np.array([
                noise if i != policy.gripper_dim else 0.1 * noise
                for i in range(env_action_dim)
            ])
        else:
            noise_scale = noise

        # Adding action noise to allow for somewhat suboptimal data
        action += np.random.normal(scale=noise_scale, size=(env_action_dim, ))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        next_observation, reward, done, info = env.step(action)
        add_transition(traj, observation, action, reward, info, agent_info,
                       done, next_observation, img_dim, transpose_image)

        if info[accept_trajectory_key] and num_steps < 0:
            num_steps = j

        rewards.append(reward)
        if done or agent_info['done']:
            break

        if info[accept_trajectory_key]:
            success = True

    # print("traj['rewards']", traj['rewards'])
    # print("rewards", rewards)

    stacked_images = np.concatenate(image_array, axis=1)
    plt.imshow(stacked_images)
    canvas.draw()

    plt.tight_layout()

    plt.savefig('traj_' + str(traj_id) + '.png')

    return traj, success, num_steps


def obtain_policy_class(policy_name):

    # Drawer policies
    if policy_name == 'main_drawer_open':
        scripted_policy = functools.partial(MultiDrawerOpen,
                                            open_drawer_name='main')
    elif policy_name == 'second_drawer_open':
        scripted_policy = functools.partial(MultiDrawerOpen,
                                            open_drawer_name='second')
    elif policy_name == 'second_drawer_close':
        scripted_policy = functools.partial(MultiDrawerClose,
                                            close_drawer_name='second_drawer')
    elif policy_name == 'main_drawer_close':
        scripted_policy = functools.partial(MultiDrawerClose,
                                            close_drawer_name='main_drawer')
    elif policy_name == 'main_top_close':
        scripted_policy = functools.partial(MultiDrawerClose,
                                            close_drawer_name='main_top')
    elif policy_name == 'second_top_close':
        scripted_policy = functools.partial(MultiDrawerClose,
                                            close_drawer_name='second_top')

    # Grasping policies
    if policy_name == 'grasp_any':
        scripted_policy = functools.partial(PickPlace,
                                            pick_height_thresh=-0.32)
    if policy_name == 'grasp_any_from_main_drawer':
        scripted_policy = functools.partial(PickPlace,
                                            pick_height_thresh=-0.32,
                                            grasp_from_main_drawer=True)
    if policy_name == 'grasp_any_from_second_drawer':
        scripted_policy = functools.partial(PickPlace,
                                            pick_height_thresh=-0.32,
                                            grasp_from_second_drawer=True)

    return scripted_policy


def main(args):
    timestamp = get_timestamp()
    if osp.exists(NFS_PATH):
        data_save_path = osp.join(NFS_PATH, args.save_directory)
    else:
        data_save_path = osp.join(__file__, "../..", "data",
                                  args.save_directory)
    data_save_path = osp.abspath(data_save_path)

    print('Data save path: ', data_save_path, args.save_directory)

    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)

    data = []

    # Obtain which policy classes to run
    policy_classes = [
        obtain_policy_class(policy_name) for policy_name in args.policy_names
    ]
    transpose_image = False

    envs = []

    for i in range(len(args.env_names)):
        # For ComboEnv, we can only pass in gui=True for one of the envs
        # (and it only works if done for the first)
        indiv_env_use_gui = (i == 0) and args.gui

        # Run scripted policies for training
        kwargs = {}
        if args.policy_names[i] == 'main_drawer_close':
            kwargs['main_start_opened'] = True
        if args.policy_names[i] == 'main_top_close':
            kwargs['main_start_top_opened'] = True
        if args.policy_names[i] == 'second_drawer_close':
            kwargs['second_start_opened'] = True
        if args.policy_names[i] == 'second_top_close':
            kwargs['second_start_top_opened'] = True

        if args.policy_names[i] == 'grasp_any_from_main_drawer':
            kwargs['main_start_opened'] = True
        if args.policy_names[i] == 'grasp_any_from_second_drawer':
            kwargs['second_start_opened'] = True

        env = d4rl2.envs.widowx.roboverse.make(args.env_names[i],
                                               gui=indiv_env_use_gui,
                                               transpose_image=transpose_image,
                                               **kwargs)
        envs.append(env)
        assert args.accept_trajectory_keys[i] in env.get_info().keys(), \
            f"""The accept trajectory key must be one of: {env.get_info().keys()}"""

    num_success = 0
    num_saved = 0
    num_attempts = 0

    progress_bar = tqdm(total=args.num_trajectories)

    while num_saved < args.num_trajectories:
        num_attempts += 1

        task_index = None

        assert len(policy_classes) == 1
        policy = policy_classes[0](env, suboptimal=args.suboptimal)
        num_timesteps = args.num_timesteps[0]
        accept_trajectory_key = args.accept_trajectory_keys[0]
        env_num_tasks = args.num_tasks[0]

        traj, success, num_steps = collect_one_traj(env,
                                                    policy,
                                                    num_timesteps,
                                                    args.noise,
                                                    accept_trajectory_key,
                                                    transpose_image,
                                                    args.lower_gripper_noise,
                                                    task_index,
                                                    traj_id=num_attempts)

        if success:
            if args.gui:
                print("num_timesteps: ", num_steps)
            data.append(traj)
            num_success += 1
            num_saved += 1
            progress_bar.update(1)
        elif args.save_all:
            data.append(traj)
            num_saved += 1
            progress_bar.update(1)

        if args.gui:
            print("success rate: {}".format(num_success / (num_attempts)))

    progress_bar.close()
    print("success rate: {}".format(num_success / (num_attempts)))

    print('Data save path: ', data_save_path)
    path = osp.join(
        data_save_path,
        "scripted_{}_{}_{}.npy".format("_".join(args.env_names),
                                       "_".join(args.policy_names), timestamp))
    print(path)
    np.save(path, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                        "--env-names",
                        nargs='+',
                        type=str,
                        required=True)
    parser.add_argument("-pl",
                        "--policy-names",
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument("-a",
                        "--accept-trajectory-keys",
                        nargs='+',
                        type=str,
                        required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t",
                        "--num-timesteps",
                        nargs='+',
                        type=int,
                        required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("-m",
                        "--num-tasks",
                        nargs='+',
                        type=int,
                        required=True)
    parser.add_argument("--lower-gripper-noise",
                        action='store_true',
                        default=False)
    parser.add_argument("--suboptimal", action='store_true', default=False)
    args = parser.parse_args()

    assert (len(args.env_names) == len(args.policy_names) == len(
        args.accept_trajectory_keys) == len(args.num_timesteps) == len(
            args.num_tasks))

    main(args)
