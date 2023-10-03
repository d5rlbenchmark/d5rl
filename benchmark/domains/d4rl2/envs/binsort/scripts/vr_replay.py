import roboverse
import numpy as np
#import skvideo.io
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np
import pdb 
import pybullet as p
import roboverse
import roboverse.bullet as bullet
import math
import roboverse
import os

from tqdm import tqdm
from PIL import Image
import argparse
import time

from roboverse.assets.shapenet_object_lists \
    import GRASP_TRAIN_OBJECTS, GRASP_TEST_OBJECTS, PICK_PLACE_TRAIN_OBJECTS, \
    PICK_PLACE_TEST_OBJECTS, TRAIN_CONTAINERS, TEST_CONTAINERS

fig = plt.figure()

data = np.load(sys.argv[1], allow_pickle=True)
env_name = sys.argv[2]
success_metric_key = sys.argv[3]

success = []
failed_id = []

reward_type = 'sparse'
env = roboverse.make(env_name,
                     gui=False,
                     transpose_image=False,
                     observation_img_dim=48,
                     #target_object=sys.argv[3],
                     control_mode='discrete_gripper')
env.reset()

for i in tqdm(range(len(data))):

    ax = plt.axes(projection='3d')
    traj = data[i]

    container_position = traj["container_position"]
    original_object_positions = traj["original_object_positions"]
    object_names = traj["object_names"]

    env.container_position = container_position
    env.original_object_positions = original_object_positions
    env.object_names = object_names
    ######
    # TODO: Change to new objects that's different from object names
    ######
    env.in_vr_replay = True

    observations = np.array([traj["observations"][j]["state"] for j in range(len(traj["observations"]))])
    ax.plot3D(observations[:, 0], observations[:, 1], observations[:, 2], label="original trajs", color="red")

    actions = traj["actions"]
    first_observation = traj["observations"][0]

    env.reset()

    images = []

    replay_obs_list = []
    obs = env.get_observation()

    replay_obs_list.append(obs["state"])

    print("length of obs: ", len(data[i]["observations"]))
    print("length of action: ", len(data[i]["actions"]))

    for index in range(len(actions)):
        data[i]["observations"][index] = obs
        a = actions[index]
        obs, rew, done, info = env.step(a)
        data[i]["next_observations"][index] = obs
        replay_obs_list.append(obs["state"])
        images.append(Image.fromarray(env.render_obs()))

    print(info[success_metric_key])
    success.append(info[success_metric_key])
    if not info[success_metric_key]:
        failed_id.append(i)

    replay_obs_list = np.array(replay_obs_list)
    ax.plot3D(replay_obs_list[:, 0], replay_obs_list[:, 1], replay_obs_list[:, 2], label="replay", color="blue")

    if not os.path.exists('replay_videos'):
        os.mkdir('replay_videos')

    images[0].save('{}/{}.gif'.format("replay_videos", i),
                format='GIF', append_images=images[1:],
                save_all=True, duration=100, loop=0)
    plt.legend()
    fig.savefig('{}/{}.png'.format("replay_videos", i))
    fig.clf()

path = "{}_replayed.npy".format(sys.argv[1][:-4])
np.save(path, data)

success = np.array(success)
print("mean: ", success.mean())
print("std: ", success.std())
print("all success: ", (success == 1).all())
print("failed id: ")
print(failed_id)

new_data = []

for i in range(len(data)):
    if i not in failed_id:
        new_data.append(data[i])

print(len(new_data))                                                                                                                                                                                                                 

for d in new_data:
    for i in range(len(d['observations'])):
        if (d['observations'][i]['image'] <= 1).all():
            d['observations'][i]['image'] = np.reshape(np.uint8(d['observations'][i]['image'] * 255.),
                                                       (48, 48, 3))
        if (d['next_observations'][i]['image'] <= 1).all():
            d['next_observations'][i]['image'] = np.reshape(np.uint8(d['next_observations'][i]['image'] * 255.),
                                                            (48, 48, 3))

path = "{}_cleaned.npy".format(sys.argv[1][:-4])
np.save(path, new_data)
