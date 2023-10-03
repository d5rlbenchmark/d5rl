import os
import pickle
import numpy as np

import gym
# import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

import jaxrl2.wrappers.combo_wrappers as wrappers
from jaxrl2.wrappers.frame_stack import FrameStack

from collections import defaultdict

from tqdm import tqdm, trange
import cv2
import imageio

# %matplotlib inline
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

from glob import glob
from tqdm import tqdm, trange

tf.config.experimental.set_visible_devices([], "GPU")
from jax.lib import xla_bridge
print('DEVICE:', xla_bridge.get_backend().platform)

import os
import wandb

print("Finished imports.")


def download_data(entity, project, task, description, algorithm, seed, keys=["evaluation/total_manipulated_mean"]):
    group_name = f"{task}_{algorithm}_{description}"
    name = f"seed_{seed}"
    run_id=group_name + "-" + name

    api = wandb.Api(timeout=300)
    print(f"\nDownloading wandb data for {entity}/{project}/{run_id}")
    run = api.run(f"{entity}/{project}/{run_id}")
    # history = run.scan_history(keys=keys + ["_step"])
    history = run.history(keys=keys + ["_step"])

    results = defaultdict(list)
    for row in tqdm(history, desc=f"iterating through history"):
        results["steps"].append(row["_step"])
        for key in keys:
            results[key].append(row[key])

    for key in keys:
        assert len(results[key]) == len(results["steps"])
#     successes = [float(ep_return > 0) for ep_return in returns]
    return {key:np.array(val) for key, val in results.items()}



def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


all_results_standard = recursive_defaultdict()


entity = "iris_intel"
project = "bench_standardkitchen_impala_singleencoder"
tasks = ["standardkitchen_indistribution", "standardkitchen_outofdistribution"]
algorithms = ["ddpm_bc", "bc", "cql", "calql", "td3bc", "iql", "idql", "rlpd"]
description = "proprio"
seeds = [0, 1, 2]


for task in tasks:
    for algorithm in algorithms:
        for seed in seeds:
            results = download_data(entity, project, task, description, algorithm, seed, keys=["evaluation/total_manipulated_mean"])
            all_results_standard[task][algorithm][seed] = results


project = "bench_diversekitchen_impala_singleencoder"
tasks = ["diversekitchen_indistribution-expert_demos", "diversekitchen_indistribution-play_data", "diversekitchen_outofdistribution-play_data"]
algorithms = ["ddpm_bc", "bc", "cql", "calql", "td3bc", "iql", "idql", "rlpd"]
description = "proprio"
seeds = [0, 1, 2]

all_results_diverse = recursive_defaultdict()

for task in tasks:
    for algorithm in algorithms:
        for seed in seeds:
            results = download_data(entity, project, task, description, algorithm, seed, keys=["evaluation/total_manipulated_mean"])
            all_results_diverse[task][algorithm][seed] = results

import pdb; pdb.set_trace()
