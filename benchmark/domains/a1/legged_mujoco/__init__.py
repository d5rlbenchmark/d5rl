import gym
import numpy as np
from dm_control import composer
from dmcgym import DMCGYM
from gym.envs.registration import register
from gym.wrappers import FlattenObservation

import benchmark.domains.a1.legged_mujoco
from benchmark.domains.a1.legged_mujoco.robots import A1
from benchmark.domains.a1.legged_mujoco.tasks import SimpleRun, Hiking, RunToward
from benchmark.domains.a1.legged_mujoco.wrappers import AddPreviousActions


class ClipActionToRange(gym.ActionWrapper):

    def __init__(self, env, min_action, max_action):
        super().__init__(env)

        min_action = np.asarray(min_action)
        max_action = np.asarray(max_action)

        min_action = min_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        max_action = max_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        min_action = np.maximum(min_action, env.action_space.low)
        max_action = np.minimum(max_action, env.action_space.high)

        self.action_space = gym.spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


def make_env(task_name: str,
             clip_actions: bool = True,
             target_linear_velocity: float = 1.0):
    assert task_name in ['simplerun', 'hike', 'runto']

    kp = 60
    kd = kp * 0.1
    robot = A1(kp=kp, kd=kd)

    if task_name == 'simplerun':
        task = SimpleRun(robot, target_linear_velocity=target_linear_velocity)
    
    if task_name == 'hike':
        task = Hiking(robot)

    if task_name == 'runto':
        task = RunToward(robot)

    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    env = DMCGYM(env)

    env = gym.wrappers.ClipAction(env)  # Just for numerical stability.

    if clip_actions:
        ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4) * 3.0
        INIT_QPOS = benchmark.domains.a1.legged_mujoco.robots.a1.A1._INIT_QPOS
        env = ClipActionToRange(env, INIT_QPOS - ACTION_OFFSET,
                                INIT_QPOS + ACTION_OFFSET)

    env = AddPreviousActions(env, action_history=1)
    env = FlattenObservation(env)

    return env


make_env.metadata = DMCGYM.metadata

# Have the robot's local forward velocity track a target

register(id=f"a1-slow-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=0.5))

register(id=f"a1-medium-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=0.8))

register(id=f"a1-fast-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=1.0))

register(id=f"a1-interpolate-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=0.75))

register(id=f"a1-extrapolate-below-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=0.25))

register(id=f"a1-extrapolate-above-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='simplerun',
            target_linear_velocity=1.25))

# Have the robot walk through rough terrain and inclines/declines
register(id=f"a1-hiking-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=1000,
         kwargs=dict(task_name='hike'))

# Randomize goal locations
register(id=f"a1-runto-v0",
         entry_point="benchmark.domains.a1.legged_mujoco:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='runto'))