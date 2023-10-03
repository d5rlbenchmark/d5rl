import gym
from gym.envs.registration import register

from benchmark.domains.finetune_env import FinetuneEnv
from benchmark.domains.widowx import tasks
from benchmark.mujoco import dmcgym


def snake_to_camel(task_name):
    return "".join(x.capitalize() for x in task_name.split("_"))


def make_env(task_name: str, **kwargs):
    task_name = snake_to_camel(task_name)

    env = tasks.load(task_name)

    env = dmcgym.DMCGYM(env)

    env = gym.wrappers.ClipAction(env)

    return FinetuneEnv(env, **kwargs)


make_env.metadata = dmcgym.DMCGYM.metadata

for task_name in ["reach_sparse", "pick_and_place", "drawer"]:
    register(
        id=f"{task_name}-teleop-v0",
        entry_point="benchmark.domains.widowx:make_env",
        max_episode_steps=200,
        kwargs=dict(
            task_name=task_name,
            ref_max_score=200,
            ref_min_score=0,
            dataset_url=f"http://rail.eecs.berkeley.edu/datasets/finetune_rl/widowx/{task_name}.hdf5",
        ),
    )
