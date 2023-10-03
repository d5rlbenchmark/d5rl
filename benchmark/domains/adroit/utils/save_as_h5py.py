import os

import gym
import h5py
import numpy as np

from benchmark.domains import adroit
from benchmark.domains.adroit.utils.load_datasets import get_binary_dataset

env_relocate = gym.make("relocate-binary-v0")
env_door = gym.make("door-binary-v0")
env_pen = gym.make("pen-binary-v0")

relocate_train_dataset, relocate_test_dataset = get_binary_dataset(env_relocate)
door_train_dataset, door_test_dataset = get_binary_dataset(env_door)
pen_train_dataset, pen_test_dataset = get_binary_dataset(env_pen)


def write_h5py_dataset(dataset, filename):
    hfile = h5py.File(
        os.path.expanduser(f"~/.finetuning_benchmark/datasets/{filename}.hdf5"), "w"
    )
    for k in dataset:
        hfile.create_dataset(k, data=dataset[k], compression="gzip")
    hfile.close()


os.makedirs(os.path.expanduser("~/.finetuning_benchmark/datasets"), exist_ok=True)

write_h5py_dataset(relocate_train_dataset, "relocate_binary")
write_h5py_dataset(relocate_test_dataset, "relocate_binary_test")
write_h5py_dataset(door_train_dataset, "door_binary")
write_h5py_dataset(door_test_dataset, "door_binary_test")
write_h5py_dataset(pen_train_dataset, "pen_binary")
write_h5py_dataset(pen_test_dataset, "pen_binary_test")

print("Finished")
