import os

import gym.spaces
import jax
import numpy as np
from dm_control.utils.transformations import quat_to_euler
from jaxrl import wrappers
from jaxrl.networks import policies
from jaxrl.networks.common import Model

from d4rl2.envs.a1.collect.maze import search
from d4rl2.envs.a1.env_utils import make_dmc_env

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'models')


@jax.jit
def get_action(actor, obs, rng):
    dist = actor(obs, 1.0)
    rng, key = jax.random.split(rng)
    return dist.sample(seed=key), rng


class Policy(object):

    def __init__(self, env, noise_std: float = 0.05, seed: int = 42):
        self.env = env
        self.noise_std = noise_std

        dmc_walk_env = make_dmc_env('walk')
        self.walk_env = wrappers.DMCEnv(env=dmc_walk_env,
                                        task_kwargs={'random': seed})
        obs = gym.spaces.flatten(self.walk_env.observation_space,
                                 self.walk_env.reset())

        actor_def = policies.NormalTanhPolicy((256, 256, 256), 12)
        actor = Model.create(actor_def, inputs=[jax.random.PRNGKey(42), obs])

        self.forward_actor = actor.load(
            os.path.join(ASSETS_DIR, 'new', 'forward'))
        self.turn_left_actor = actor.load(
            os.path.join(ASSETS_DIR, 'new', 'left'))
        self.turn_right_actor = actor.load(
            os.path.join(ASSETS_DIR, 'new', 'right'))

        self.rng = jax.random.PRNGKey(seed)
        self.waypoints = []

    def __call__(self, time_step):
        if time_step.first():
            self.waypoints = search.get_waypoints(self.env.task._maze_arena)
            self.waypoints = np.asarray(self.waypoints)
            noise = np.random.normal(
                size=self.waypoints.shape) * self.noise_std
            self.waypoints += np.clip(noise, -2 * self.noise_std,
                                      2 * self.noise_std)

        body_position = time_step.observation['unitree_a1/body_position']
        height = time_step.observation['unitree_a1/body_position'][-1]
        del time_step.observation['unitree_a1/body_position']
        time_step.observation['unitree_a1/body_height'] = height

        obs = gym.spaces.flatten(self.walk_env.observation_space,
                                 time_step.observation)

        # target_position = env.task._maze_arena.target_positions[0][:2]
        walker_position = body_position[:2]

        current_waypoint = len(self.waypoints) - 1

        while True:
            target_position = self.waypoints[current_waypoint]
            direction = target_position[:2] - walker_position[:2]
            if np.linalg.norm(direction) < 1.5 or current_waypoint == 0:
                break
            current_waypoint -= 1

        body_quat = time_step.observation['unitree_a1/sensors_framequat']
        body_yaw = quat_to_euler(body_quat)[-1]

        goal_yaw = np.arctan2(direction[1], direction[0])

        diff = (goal_yaw - body_yaw) % (2 * np.pi)

        if diff > np.pi:
            diff = -(2 * np.pi - diff)

        if diff > 0.1:
            action, self.rng = get_action(self.turn_left_actor, obs, self.rng)
        elif diff < -0.1:
            action, self.rng = get_action(self.turn_right_actor, obs, self.rng)
        else:
            action, self.rng = get_action(self.forward_actor, obs, self.rng)

        action = 0.5 * (action + 1)
        action = action * (self.walk_env.action_space.high - self.walk_env.
                           action_space.low) + self.walk_env.action_space.low

        return action
