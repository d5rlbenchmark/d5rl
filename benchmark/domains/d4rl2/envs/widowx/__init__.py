import os

from gym import register

from d4rl2.envs.widowx.roboverse.envs.registration import ENVIRONMENT_SPECS

dataset_types = [
    'stitch', 'adversarial_stitch', 'stitch+expert',
    'adversarial_stitch+expert'
]

dataset_prefixes = {
    'stitch': 'widowx-stitch-',
    'adversarial_stitch': 'widowx-adversarial-stitch-',
    'stitch+expert': 'widowx-stitch-expert-',
    'adversarial_stitch+expert': 'widowx-adversarial-stitch-expert-',
}

# This should register four envs:
# widowx-stitch-v0
# widowx-adversarial-stitch-v0
# widowx-stitch-expert-v0
# widowx-adversarial-stitch-expert-v0

for dataset_type in dataset_types:
    for env in ENVIRONMENT_SPECS:
        if env['id'] == 'Widow250MultiDrawerMultiObjectEvalEnv-v0':
            # Change this to use all the data, setting to only buffer so that it is easy to load for debugging
            env['entry_point'] = "d4rl2.envs.widowx.roboverse.envs.widow250_multidrawer_multiobject:make_offline_env"
            dataset_url = dataset_prefixes[
                dataset_type] + 'drawer-opening-data.hdf5'
            env['id'] = dataset_prefixes[dataset_type] + 'v0'
            env['kwargs'].update({'dataset_url': dataset_url})
            env['max_episode_steps'] = 400
            register(**env)
