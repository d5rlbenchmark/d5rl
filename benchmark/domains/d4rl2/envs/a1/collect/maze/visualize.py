import gym
from dm_control import viewer

import d4rl2.envs.a1
from d4rl2.envs.a1.collect.maze.policy import get_policy

gym_env = gym.make('a1-medium_maze-diverse-v0')
env = gym_env.unwrapped._env

policy = get_policy(env)
viewer.launch(env, policy)
