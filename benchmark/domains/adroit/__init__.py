from gym.envs.registration import register

# Swing the door open
register(
    id="door-binary-v0",
    entry_point="benchmark.domains.adroit.gym_envs:get_door_env",
    max_episode_steps=200,
    kwargs=dict(
        ref_max_score=0,
        ref_min_score=-200,
        dataset_url="http://rail.eecs.berkeley.edu/datasets/finetune_rl/adroit/door_binary.hdf5",
    ),
)
from benchmark.domains.adroit.door_v0 import DoorEnvV0

# Reposition a pen in hand
register(
    id="pen-binary-v0",
    entry_point="benchmark.domains.adroit.gym_envs:get_pen_env",
    max_episode_steps=100,
    kwargs=dict(
        ref_max_score=0,
        ref_min_score=-100,
        dataset_url="http://rail.eecs.berkeley.edu/datasets/finetune_rl/adroit/pen_binary.hdf5",
    ),
)
from benchmark.domains.adroit.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id="relocate-binary-v0",
    entry_point="benchmark.domains.adroit.gym_envs:get_relocate_env",
    max_episode_steps=200,
    kwargs=dict(
        ref_max_score=0,
        ref_min_score=-200,
        dataset_url="http://rail.eecs.berkeley.edu/datasets/finetune_rl/adroit/relocate_binary.hdf5",
    ),
)
from benchmark.domains.adroit.relocate_v0 import RelocateEnvV0
