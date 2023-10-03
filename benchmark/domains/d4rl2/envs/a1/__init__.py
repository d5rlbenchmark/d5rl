import os

from gym import register

register('a1-walk-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='walk'))

register('a1-walk_stable-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='walk_stable',
                     dataset_file=os.path.join('a1', 'a1-walk.hdf5')))

register('a1-umaze-diverse-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='umaze-play',
                     dataset_file=os.path.join('a1', 'a1-umaze.hdf5')))

register('a1-medium_maze-diverse-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='medium_maze-play',
                     dataset_file=os.path.join('a1', 'a1-medium_maze.hdf5')))

register('a1-umaze-collect-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='umaze-collect'))

register('a1-medium_maze-collect-v0',
         entry_point="d4rl2.envs.a1.env_utils:make_gym_env",
         max_episode_steps=1000,
         kwargs=dict(task='medium_maze-collect'))
