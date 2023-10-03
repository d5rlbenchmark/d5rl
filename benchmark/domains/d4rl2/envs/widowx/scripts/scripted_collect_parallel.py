import argparse
import datetime
import os
import subprocess
import time

from d4rl2.envs.widowx.roboverse.utils import get_timestamp


def get_data_save_directory(args):
    data_save_directory = args.data_save_directory

    data_save_directory += '_{}'.format(args.env_names[0])

    if args.num_trajectories > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajectories / 1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajectories)

    if args.save_all:
        data_save_directory += '_save_all'

    data_save_directory += '_noise_{}'.format(args.noise)

    # Add policy type and suboptimality
    data_save_directory += '_policy_{}'.format(args.policy_names[0])
    data_save_directory += '_suboptimal_{}'.format(args.suboptimal)

    data_save_directory += '_{}'.format(get_timestamp())

    return data_save_directory


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
    parser.add_argument("-d", "--data-save-directory", type=str, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--lower-gripper-noise",
                        action='store_true',
                        default=False)
    parser.add_argument("-m",
                        "--num-tasks",
                        nargs='+',
                        type=int,
                        required=True)
    parser.add_argument("--suboptimal", action='store_true', default=False)
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()

    num_trajectories_per_thread = int(args.num_trajectories /
                                      args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1

    timestamp = get_timestamp()
    save_directory = get_data_save_directory(args)

    script_name = "scripted_collect.py"

    if args.policy_names != []:
        policy_flag_args = ["-pl"] + args.policy_names
    else:
        raise NotImplementedError

    num_timesteps_args = [
        str(num_timestep) for num_timestep in args.num_timesteps
    ]
    num_task_args = [str(num_task) for num_task in args.num_tasks]

    command = (['python', 'scripts/{}'.format(script_name)] +
               policy_flag_args + ['-a'] + args.accept_trajectory_keys +
               ['-e'] + args.env_names +
               ['-n {}'.format(num_trajectories_per_thread)] + ['-t'] +
               num_timesteps_args + ['-m'] + num_task_args +
               ['-d={}'.format(save_directory)])

    if args.save_all:
        command.append('--save-all')
    if args.lower_gripper_noise:
        command.append('--lower-gripper-noise')
    if args.suboptimal:
        command.append('--suboptimal')

    subprocesses = []
    for i in range(args.num_parallel_threads):
        subprocesses.append(subprocess.Popen(command))
        time.sleep(2)

    exit_codes = [p.wait() for p in subprocesses]

    merge_command = [
        'python', 'scripts/combine_trajectories.py',
        '-d{}'.format(save_directory)
    ]

    subprocess.call(merge_command)
