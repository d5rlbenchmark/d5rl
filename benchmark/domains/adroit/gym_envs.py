from benchmark.domains.adroit.door_v0 import DoorEnvV0
from benchmark.domains.adroit.pen_v0 import PenEnvV0
from benchmark.domains.adroit.relocate_v0 import RelocateEnvV0
from benchmark.domains.finetune_env import FinetuneEnv


def get_door_env(**kwargs):
    env = DoorEnvV0(reward_type="binary")
    return FinetuneEnv(env, **kwargs)


def get_pen_env(**kwargs):
    env = PenEnvV0(reward_type="binary")
    return FinetuneEnv(env, **kwargs)


def get_relocate_env(**kwargs):
    env = RelocateEnvV0(reward_type="binary")
    return FinetuneEnv(env, **kwargs)
