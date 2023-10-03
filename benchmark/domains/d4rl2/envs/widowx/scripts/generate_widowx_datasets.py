import argparse
import os
import random

import h5py
import numpy as np

import d4rl2.envs.widowx.roboverse

np.random.seed(42)

GRASP_DATA_PATH = '/nfs/kun2/users/aviralkumar/grasping_data_final/'
DRAWER_OPEN_DATA_PATH = '/nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/'
AUX_DRAWER_OPEN_DATA_PATH = '/nfs/kun2/users/aviralkumar/d4rl2_drawer_data'
DRAWER_CLOSE_DATA_PATH = '/nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/'


def get_npy_files_in_dir(dirname):
    onlyfiles = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]
    npy_files = [f for f in onlyfiles if '.npy' in f]
    return npy_files


def save_data(data_dict, h5path):
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    with h5py.File(h5path, 'w') as dataset:
        dataset.create_dataset('observations',
                               data=data_dict['observations'],
                               compression='gzip')
        dataset.create_dataset('actions',
                               data=data_dict['actions'],
                               compression='gzip')
        dataset.create_dataset('next_observations',
                               data=data_dict['next_observations'],
                               compression='gzip')
        dataset.create_dataset('rewards',
                               data=data_dict['rewards'],
                               compression='gzip')
        dataset.create_dataset('terminals',
                               data=np.array(
                                   data_dict['terminals']).astype(bool),
                               compression='gzip')
        dataset.create_dataset('trajectory_ends',
                               data=np.array(
                                   data_dict['trajectory_ends']).astype(bool),
                               compression='gzip')


def generate_and_save_stitch_datasets(env, dataset_save_path):
    grasping_data = get_npy_files_in_dir(GRASP_DATA_PATH)
    drawer_open_data = get_npy_files_in_dir(DRAWER_OPEN_DATA_PATH)
    drawer_open_data += get_npy_files_in_dir(AUX_DRAWER_OPEN_DATA_PATH)
    drawer_close_data = get_npy_files_in_dir(DRAWER_CLOSE_DATA_PATH)

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in grasping_data:
        if 'suboptimal_True' in npy_file:
            continue
        elif 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded grasping data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving grasping data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-grasping-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading opening data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_open_data:
        if 'suboptimal_True' in npy_file:
            continue
        elif 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded opening data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving opening data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-drawer-opening-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading closing data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_close_data:
        if 'suboptimal_True' in npy_file:
            continue
        elif 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded closing data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving closing data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-drawer-closing-data.hdf5')
    save_data(data_dict, h5path)


def generate_and_save_stitch_and_expert_datasets(env, dataset_save_path):
    grasping_data = get_npy_files_in_dir(GRASP_DATA_PATH)
    drawer_open_data = get_npy_files_in_dir(DRAWER_OPEN_DATA_PATH)
    drawer_open_data += get_npy_files_in_dir(AUX_DRAWER_OPEN_DATA_PATH)
    drawer_close_data = get_npy_files_in_dir(DRAWER_CLOSE_DATA_PATH)

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in grasping_data:
        if 'suboptimal_True' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded grasping data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving grasping data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-expert-grasping-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading opening data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_open_data:
        if 'suboptimal_True' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded opening data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving opening data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-expert-drawer-opening-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading closing data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_close_data:
        if 'suboptimal_True' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded closing data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving closing data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-stitch-expert-drawer-closing-data.hdf5')
    save_data(data_dict, h5path)


def generate_and_save_adversarial_stitch_and_expert_datasets(
        env, dataset_save_path):
    grasping_data = get_npy_files_in_dir(GRASP_DATA_PATH)
    drawer_open_data = get_npy_files_in_dir(DRAWER_OPEN_DATA_PATH)
    drawer_open_data += get_npy_files_in_dir(AUX_DRAWER_OPEN_DATA_PATH)
    drawer_close_data = get_npy_files_in_dir(DRAWER_CLOSE_DATA_PATH)

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in grasping_data:

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.4 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded grasping data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving grasping data now.....')
    h5path = os.path.join(
        dataset_save_path,
        'widowx-adversarial-stitch-expert-grasping-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading opening data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_open_data:

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.4 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded opening data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving opening data now.....')
    h5path = os.path.join(
        dataset_save_path,
        'widowx-adversarial-stitch-expert-drawer-opening-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading closing data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_close_data:
        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.02' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.2 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.4 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded closing data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving closing data now.....')
    h5path = os.path.join(
        dataset_save_path,
        'widowx-adversarial-stitch-expert-drawer-closing-data.hdf5')
    save_data(data_dict, h5path)


def generate_and_save_adversarial_stitch_datasets(env, dataset_save_path):
    grasping_data = get_npy_files_in_dir(GRASP_DATA_PATH)
    drawer_open_data = get_npy_files_in_dir(DRAWER_OPEN_DATA_PATH)
    drawer_open_data += get_npy_files_in_dir(AUX_DRAWER_OPEN_DATA_PATH)
    drawer_close_data = get_npy_files_in_dir(DRAWER_CLOSE_DATA_PATH)

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in grasping_data:
        if 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.7 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded grasping data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving grasping data now.....')
    h5path = os.path.join(dataset_save_path,
                          'widowx-adversarial-stitch-grasping-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading opening data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_open_data:
        if 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.7 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded opening data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving opening data now.....')
    h5path = os.path.join(
        dataset_save_path,
        'widowx-adversarial-stitch-drawer-opening-data.hdf5')
    save_data(data_dict, h5path)

    print('Next loading closing data...')

    data_dict = {}
    data_dict['observations'] = []
    data_dict['actions'] = []
    data_dict['next_observations'] = []
    data_dict['rewards'] = []
    data_dict['terminals'] = []
    data_dict['trajectory_ends'] = []

    for npy_file in drawer_close_data:
        if 'noise_0.02' in npy_file:
            continue

        trajs = np.load(npy_file, allow_pickle=True)
        num_trajectories = len(trajs)
        print(npy_file, ' : ', num_trajectories)

        if 'noise_0.2' in npy_file and 'suboptimal_False' in npy_file:
            random_traj_indices = np.random.choice(np.arange(len(trajs)),
                                                   size=int(0.7 * len(trajs)),
                                                   replace=False)
            trajs = [trajs[i] for i in random_traj_indices]

        for traj in trajs:
            trajectory_end_index = len(traj['observations'])

            for i in range(len(traj['observations'])):
                data_dict['observations'].append(
                    traj['observations'][i]['image'])
                data_dict['actions'].append(traj['actions'][i])
                data_dict['next_observations'].append(
                    traj['next_observations'][i]['image'])
                data_dict['terminals'].append(traj['terminals'][i])

                reward = env.get_reward(info=traj['env_infos'][i])
                data_dict['rewards'].append(reward)
                if i == trajectory_end_index - 1:
                    data_dict['trajectory_ends'].append(True)
                else:
                    data_dict['trajectory_ends'].append(False)

        print('Total trajectories done so far: ',
              sum(data_dict['trajectory_ends']))

    print('Loaded closing data.....')
    print('Dataset size so far: ', len(data_dict['observations']))
    print('Rewards so far: ', sum(data_dict['rewards']))
    print('Saving closing data now.....')
    h5path = os.path.join(
        dataset_save_path,
        'widowx-adversarial-stitch-drawer-closing-data.hdf5')
    save_data(data_dict, h5path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type',
                        type=str,
                        help='dataset type to obtain',
                        default='stitch')
    parser.add_argument('--dataset_save_path',
                        type=str,
                        help='where to save data')
    parser.add_argument('--reward_type',
                        type=str,
                        default='grasp_treasure',
                        help='how to grasp treasure')
    args = parser.parse_args()

    env = d4rl2.envs.widowx.roboverse.make(
        'Widow250MultiDrawerMultiObjectEnv-v0',
        gui=False,
        transpose_image=True,
        reward_type=args.reward_type)

    dataset_path = os.path.join(
        args.dataset_save_path, '{}_{}'.format(args.dataset_type,
                                               args.reward_type))
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    if args.dataset_type == 'stitch':
        generate_and_save_stitch_datasets(env, dataset_path)
    elif args.dataset_type == 'stitch+expert':
        generate_and_save_stitch_and_expert_datasets(env, dataset_path)
    elif args.dataset_type == 'adversarial_stitch':
        generate_and_save_adversarial_stitch_datasets(env, dataset_path)
    elif args.dataset_type == 'adversarial_stitch+expert':
        generate_and_save_adversarial_stitch_and_expert_datasets(
            env, dataset_path)
