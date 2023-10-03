import os

from gym.envs.registration import register

register('RPL_kitchen-v1',
         entry_point="d4rl2.wrappers.offline_env:make_env",
         max_episode_steps=500,
         kwargs=dict(task='RPL_kitchen-v1',
                     tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide'],
                     datasets = ['RPL_data']))