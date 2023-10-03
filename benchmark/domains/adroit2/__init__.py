from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv
from benchmark.domains.adroit2.door_v0 import DoorEnvV0
from benchmark.domains.adroit2.hammer_v0 import HammerEnvV0
from benchmark.domains.adroit2.pen_v0 import PenEnvV0
from benchmark.domains.adroit2.relocate_v0 import RelocateEnvV0
from benchmark.domains.adroit2 import infos


# V1 envs
MAX_STEPS = {'hammer': 200, 'relocate': 200, 'door': 200, 'pen': 100}
LONG_HORIZONS = {'hammer': 600, 'pen': 200, 'relocate': 500, 'door': 300}
ENV_MAPPING = {'hammer': 'HammerEnvV0', 'relocate': 'RelocateEnvV0', 'door': 'DoorEnvV0', 'pen': 'PenEnvV0'}
# for agent in ['hammer', 'pen', 'relocate', 'door']:
#     for dataset in ['human', 'expert', 'cloned']:
#         env_name = '%s-%s-v1' % (agent, dataset)
#         register(
#             id=env_name,
#             entry_point='benchmark.domains.adroit2:' + ENV_MAPPING[agent],
#             max_episode_steps=MAX_STEPS[agent],
#             kwargs={
#                 'ref_min_score': infos.REF_MIN_SCORE[env_name],
#                 'ref_max_score': infos.REF_MAX_SCORE[env_name],
#                 'dataset_url': infos.DATASET_URLS[env_name]
#             }
#         )
#
#         if dataset == 'human':
#             longhorizon_env_name = '%s-human-longhorizon-v1' % agent
#             register(
#                 id=longhorizon_env_name,
#                 entry_point='benchmark.domains.adroit2:' + ENV_MAPPING[agent],
#                 max_episode_steps=LONG_HORIZONS[agent],
                # kwargs={
                #     'ref_min_score': infos.REF_MIN_SCORE[env_name],
                #     'ref_max_score': infos.REF_MAX_SCORE[env_name],
                #     'dataset_url': infos.DATASET_URLS[env_name]
                # }
#             )

DOOR_RANDOM_SCORE = -56.512833
DOOR_EXPERT_SCORE = 2880.5693087298737

HAMMER_RANDOM_SCORE = -274.856578
HAMMER_EXPERT_SCORE = 12794.134825156867

PEN_RANDOM_SCORE = 96.262799
PEN_EXPERT_SCORE = 3076.8331017826877

RELOCATE_RANDOM_SCORE = -6.425911
RELOCATE_EXPERT_SCORE = 4233.877797728884

# Swing the door open
register(
    id='door-binary2-v0',
    entry_point='benchmark.domains.adroit2:DoorEnvV0',
    max_episode_steps=200,
    kwargs={"reward_type":"binary"},
)


# Hammer a nail into the board
register(
    id='hammer-binary2-v0',
    entry_point='benchmark.domains.adroit2:HammerEnvV0',
    max_episode_steps=200,
    kwargs={"reward_type":"binary"},
)

# Reposition a pen in hand
register(
    id='pen-binary2-v0',
    entry_point='benchmark.domains.adroit2:PenEnvV0',
    max_episode_steps=100,
    kwargs={"reward_type":"binary"},
)

# Relcoate an object to the target
register(
    id='relocate-binary2-v0',
    entry_point='benchmark.domains.adroit2:RelocateEnvV0',
    max_episode_steps=200,
    kwargs={"reward_type":"binary"},
)
