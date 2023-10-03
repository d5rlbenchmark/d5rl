from math import trunc
from os import truncate

import numpy as np
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings

_NUM_RAYS = 4

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.001


def truncated_normal(random_state, scale, size=()):
    noise = random_state.normal(scale=scale, size=size)
    noise = np.clip(noise, -2 * scale, 2 * scale)
    return noise


class Maze(composer.Task):
    """A base task for maze with goals."""

    def __init__(self,
                 walker,
                 maze_arena,
                 randomize_spawn_rotation=True,
                 physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep=DEFAULT_CONTROL_TIMESTEP):
        """Initializes goal-directed maze task.

    Args:
      walker: The body to navigate the maze.
      maze_arena: The physical maze arena object.
      randomize_spawn_rotation: Flag to randomize orientation of spawning.
      rotation_bias_factor: A non-negative number that concentrates initial
        orientation away from walls. When set to zero, the initial orientation
        is uniformly random. The larger the value of this number, the more
        likely it is that the initial orientation would face the direction that
        is farthest away from a wall.
      physics_timestep: timestep of simulation.
      control_timestep: timestep at which agent changes action.
    """
        self._walker = walker
        self._maze_arena = maze_arena
        self._maze_arena.add_free_entity(self._walker)

        self._randomize_spawn_rotation = randomize_spawn_rotation

        self._discount = 1.0

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        observables = (self._walker.observables.proprioception +
                       self._walker.observables.kinematic_sensors +
                       [self._walker.observables.body_position] +
                       [self._walker.observables.prev_action])
        for observable in observables:
            observable.enabled = True

    @property
    def name(self):
        return 'goal_maze'

    @property
    def root_entity(self):
        return self._maze_arena

    def initialize_episode_mjcf(self, unused_random_state):
        self._maze_arena.regenerate()

    def _respawn(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        self._spawn_position = self._maze_arena.spawn_positions[0]

        if self._randomize_spawn_rotation:
            # Move walker up out of the way before raycasting.
            self._walker.shift_pose(physics, [0.0, 0.0, 100.0])

            distances = []
            geomid_out = np.array([-1], dtype=np.intc)
            for i in range(_NUM_RAYS):
                theta = 2 * np.pi * i / _NUM_RAYS
                pos = np.array(
                    [self._spawn_position[0], self._spawn_position[1], 0.1],
                    dtype=np.float64)
                vec = np.array([np.cos(theta), np.sin(theta), 0],
                               dtype=np.float64)
                dist = mjbindings.mjlib.mj_ray(physics.model.ptr,
                                               physics.data.ptr, pos, vec,
                                               None, 1, -1, geomid_out)
                distances.append(dist)

            max_dist = np.max(distances)
            probs_bool = distances > (max_dist - 0.1)
            probs = probs_bool.astype(np.float32)
            probs /= probs.sum()
            indx = random_state.choice(np.arange(0, len(probs),
                                                 dtype=np.int32),
                                       p=probs)

            rotation = 2 * np.pi * indx / _NUM_RAYS

            noise = truncated_normal(random_state, 0.05)
            rotation += noise

            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

            # Move walker back down.
            self._walker.shift_pose(physics, [0.0, 0.0, -100.0])
        else:
            quat = None

        noise = truncated_normal(random_state, 0.05, size=(3, ))
        spawn_position = self._spawn_position + noise

        self._walker.shift_pose(physics,
                                [spawn_position[0], spawn_position[1], 0.0],
                                quat,
                                rotate_velocity=True)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._respawn(physics, random_state)
        self._is_success = False

        arena_geoms = self._maze_arena.mjcf_model.find_all('geom')
        wall_geoms = [
            x for x in arena_geoms
            if x.contype is None and x.name is not None and 'wall' in x.name
        ]
        self._wall_geom_ids = set(physics.bind(wall_geoms).element_id)

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)

    def _goal_reached(self, robot_position):
        target_position = self._maze_arena.target_positions[0]
        return np.linalg.norm(robot_position[:2] - target_position[:2]) < 0.5

    def _wall_collision(self, physics):
        has_collision = False
        for c in physics.data.contact:
            has_collision = has_collision or (c.geom1 in self._wall_geom_ids or
                                              c.geom2 in self._wall_geom_ids)
        return has_collision

    def after_step(self, physics, random_state):
        super().after_step(physics, random_state)

        robot_position, _ = self._walker.get_pose(physics)
        self._is_success = self._goal_reached(robot_position)
        self._is_failure = self._wall_collision(physics)

    def get_reward(self, physics):
        return 1.0 if self._is_success else 0.0

    def get_discount(self, physics):
        return 0.0 if (self._is_success or self._is_failure) else 1.0

    def should_terminate_episode(self, physics):
        return self._is_success or self._is_failure or super(
        ).should_terminate_episode(physics)
