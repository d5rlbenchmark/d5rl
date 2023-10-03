import gym
from gym.wrappers.flatten_observation import FlattenObservation

from jaxrl2.wrappers.jaxrl5_wrappers.pixels import wrap_pixels
from jaxrl2.wrappers.jaxrl5_wrappers.single_precision import SinglePrecision
from jaxrl2.wrappers.jaxrl5_wrappers.universal_seed import UniversalSeed
from jaxrl2.wrappers.jaxrl5_wrappers.wandb_video import WANDBVideo


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env
