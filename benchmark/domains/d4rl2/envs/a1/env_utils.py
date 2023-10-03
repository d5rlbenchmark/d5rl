import gym
import labmaze
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures, mazes
from gym.wrappers import RescaleAction

from d4rl2.envs.a1.dmc2gym import DMC2GYM
from d4rl2.envs.a1.robots import A1
from d4rl2.envs.a1.single_precision import SinglePrecision
from d4rl2.envs.a1.tasks import Maze, Walk
from d4rl2.wrappers.offline_env import OfflineEnv

# yapf: disable
MEDIUM_MAZE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

UMAZE_LAYOUT = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
# yapf: enable


def make_dmc_env(task):
    robot = A1()

    if 'walk' in task:
        task = Walk(robot, penalize_angular_acc='stable' in task)
    elif 'maze' in task:
        if 'medium' in task:
            MAZE_LAYOUT = MEDIUM_MAZE_LAYOUT
            start_positions = {(1, 1)}
            goal_positions = {(6, 6)}
        elif 'umaze' in task:
            MAZE_LAYOUT = UMAZE_LAYOUT
            start_positions = {(1, 1)}
            goal_positions = {(3, 1)}
        else:
            raise ValueError()

        STR_MAZE_LAYOUT = ""
        for i in range(len(MAZE_LAYOUT)):
            for j in range(len(MAZE_LAYOUT[i])):
                if (i, j) in start_positions and 'collect' not in task:
                    STR_MAZE_LAYOUT += 'P'
                elif (i, j) in goal_positions and 'collect' not in task:
                    STR_MAZE_LAYOUT += 'G'
                else:
                    STR_MAZE_LAYOUT += [' ', '*'][MAZE_LAYOUT[i][j]]
            STR_MAZE_LAYOUT += '\n'

        maze = labmaze.FixedMazeWithRandomGoals(entity_layer=STR_MAZE_LAYOUT,
                                                num_spawns=1,
                                                num_objects=1)

        skybox_texture = labmaze_textures.SkyBox(style='sky_03')
        wall_textures = labmaze_textures.WallTextures(style='style_01')
        wall_textures._textures = wall_textures._textures[:1]
        floor_textures = labmaze_textures.FloorTextures(style='style_01')
        floor_textures._textures = floor_textures._textures[:1]

        arena = mazes.MazeWithTargets(maze=maze,
                                      xy_scale=1.0,
                                      z_height=1.0,
                                      skybox_texture=skybox_texture,
                                      wall_textures=wall_textures,
                                      floor_textures=floor_textures)

        task = Maze(robot, arena)
    return composer.Environment(task,
                                raise_exception_on_physics_error=False,
                                strip_singleton_obs_buffer_dim=True)


def make_gym_env(task, dataset_file: str = None):
    env = make_dmc_env(task)
    env = DMC2GYM(env)
    env = gym.wrappers.FlattenObservation(env)
    env = RescaleAction(env, -1.0, 1.0)
    env = SinglePrecision(env)
    if dataset_file is not None:
        env = OfflineEnv(env, dataset_file)
    return env