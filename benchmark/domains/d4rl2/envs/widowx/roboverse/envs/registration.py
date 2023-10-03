import gym

from d4rl2.envs.widowx.roboverse.assets.meta_env_object_lists import (
    PICK_PLACE_TEST_TASK_CONTAINERS, PICK_PLACE_TEST_TASK_CONTAINERS_REPEATED,
    PICK_PLACE_TEST_TASK_OBJECTS, PICK_PLACE_TEST_TASK_OBJECTS_REPEATED,
    PICK_PLACE_TRAIN_TASK_CONTAINERS, PICK_PLACE_TRAIN_TASK_OBJECTS)
from d4rl2.envs.widowx.roboverse.assets.shapenet_object_lists import (
    GRASP_TEST_OBJECTS, GRASP_TRAIN_OBJECTS, PICK_PLACE_TEST_OBJECTS,
    PICK_PLACE_TRAIN_OBJECTS, TEST_CONTAINERS, TRAIN_CONTAINERS)

ENVIRONMENT_SPECS = (
    {
        'id': 'Widow250MultiDrawerMultiObjectEnv-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('shed', 'gatorade'),
            'object_scales': (
                0.6,
                0.7,
            ),
            'target_object': 'shed',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.4, 0.4),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    # This is the environment we will use for evaluation, others are used for data generation only
    {
        'id': 'Widow250MultiDrawerMultiObjectEvalEnv-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('shed', 'gatorade'),
            'object_scales': (
                0.6,
                0.7,
            ),
            'target_object': 'shed',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.4, 0.4),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': True,
            'second_start_opened': False,
            'second_start_top_opened': True,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    {
        'id': 'Widow250MultiDrawerMultiObjectEnvBall-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('shed', 'ball'),
            'object_scales': (
                0.6,
                0.7,
            ),
            'target_object': 'shed',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.4, 0.4),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    {
        'id': 'Widow250MultiDrawerMultiObjectEnvBallGatorade-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('gatorade', 'ball', 'shed'),
            'object_scales': (0.7, 0.7, 0.6),
            'object_orientations': ((0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
            'target_object': 'shed',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.4, 0.4),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    {
        'id': 'Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('shed', ),
            'object_scales': (0.6, ),
            'target_object': 'shed',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.45, 0.45),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    {
        'id': 'Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('gatorade', ),
            'object_scales': (0.7, ),
            'target_object': 'gatorade',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.45, 0.45),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
    {
        'id': 'Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0',
        'entry_point':
        'roboverse.envs.widow250_multidrawer_multiobject:Widow250MultiDrawerMultiObjectEnv',
        'kwargs': {
            'main_drawer_pos': (0.45, 0.225, -0.35),  #(0.47, 0.2, -.35),
            'second_drawer_pos': (0.75, 0.225, -0.35),
            'main_drawer_quat': (0, 0, 0.707107, 0.707107),
            'second_drawer_quat': (0, 0, -0.707107, 0.707107),
            'drawer_scales': 0.07,
            'reward_type': 'unlock_and_grasp_treasure',
            'control_mode': 'discrete_gripper',
            'object_position_low': (.56, .15, -.3),
            'object_position_high': (.64, .27, -.3),
            'observation_img_dim': 128,
            'object_names': ('ball', ),
            'object_scales': (0.7, ),
            'target_object': 'ball',
            'target_object_names': ('ball', 'ball'),
            'target_object_scales': (0.45, 0.45),
            'load_tray': False,
            'main_start_opened': False,
            'main_start_top_opened': False,
            'second_start_opened': False,
            'second_start_top_opened': False,
            'use_neutral_action': False,
            'blocking_object_in_tray': True,
        }
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(environment_spec['id']
                    for environment_spec in ENVIRONMENT_SPECS)

    return gym_ids


def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
