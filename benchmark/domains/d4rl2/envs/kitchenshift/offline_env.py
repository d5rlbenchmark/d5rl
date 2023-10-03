import os

import gym
from gym.core import Wrapper

from benchmark.domains.d4rl2.envs.kitchenshift.kitchen_v1 import Kitchen_v1
from benchmark.domains.d4rl2.envs.kitchenshift.randomized_kitchen import Randomized_Kitchen
from benchmark.domains.d4rl2.envs.kitchenshift.kitchen_rgb_stack import KitchenImageConcatWrapper
from benchmark.domains.d4rl2.wrappers.frame_stack import FrameStack

from benchmark.domains.d4rl2.data import OfflineMemoryEfficientReplayBuffer
from benchmark.domains.d4rl2.data.randomized_kitchen_data_generator import RandomizedKitchenDataGenerator

class OfflineEnv(Wrapper):

    def __init__(self, env: gym.Env, datasets_urls: list):
        super().__init__(env)

        self._datasets_urls = datasets_urls
        self._dataset = None
        self._dataset_dict = None

    def q_learning_dataset(self, include_pixels = False, size = 1e5):
            offline_generator = RandomizedKitchenDataGenerator(
                self._datasets_urls, self.env)
            self._dataset = OfflineMemoryEfficientReplayBuffer(
                self.env.observation_space,
                self.env.action_space,
                int(size),
                offline_generator,
                include_pixels=include_pixels)
            return self._dataset


def make_env(task, tasks_to_complete, datasets: str = None):
    if task == "kitchen-v1":
        env = Kitchen_v1(tasks_to_complete = tasks_to_complete)
    elif task == 'random_kitchen-v1':
        env = Randomized_Kitchen(tasks_to_complete = tasks_to_complete)
        env = FrameStack(KitchenImageConcatWrapper(env), num_stack=1)

    DATASET_DIR = os.environ.get('KITCHEN_DATASETS', None)
    print(DATASET_DIR)
    datasets_urls = [DATASET_DIR + '/' + dataset + '/' for dataset in datasets]
    env = OfflineEnv(env, datasets_urls)
    return env
