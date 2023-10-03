import h5py


def save_data(replay_buffer, h5path):
    data = replay_buffer.dataset_dict
    with h5py.File(h5path, 'w') as dataset:
        dataset.create_dataset('observations',
                               data=data['observations'],
                               compression='gzip')
        dataset.create_dataset('actions',
                               data=data['actions'],
                               compression='gzip')
        dataset.create_dataset('next_observations',
                               data=data['next_observations'],
                               compression='gzip')
        dataset.create_dataset('rewards',
                               data=data['rewards'],
                               compression='gzip')
        dataset.create_dataset('terminals',
                               data=(1 - data['masks']).astype(bool),
                               compression='gzip')
        dataset.create_dataset('trajectory_ends',
                               data=data['dones'].astype(bool),
                               compression='gzip')
