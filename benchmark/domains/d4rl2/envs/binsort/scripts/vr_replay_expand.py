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

def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim):
    observation["image"] = np.reshape(np.uint8(observation["image"] * 255.),
                                      (img_dim, img_dim, 3))
    traj["observations"].append(observation)
    next_observation["image"] = np.reshape(
        np.uint8(next_observation["image"] * 255.), (img_dim, img_dim, 3))
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj

def replay_one_traj(env, i,
                    accept_trajectory_key, object_names):
    original_traj = data[i]
    # import pdb; pdb.set_trace()
    # fig = plt.figure()
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim
    env.reset()
    time.sleep(1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    ################
    ax = plt.axes(projection='3d')
    container_position = original_traj["container_position"]
    original_object_positions = original_traj["original_object_positions"]
    # object_scales = dict()
    # object_orientations = dict()
    # for object_name in object_names:
    #     object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
    #     object_scales[object_name] = OBJECT_SCALINGS[object_name]
    # object_names = traj["object_names"]
    env.container_position = container_position
    env.original_object_positions = original_object_positions
    env.object_names = object_names
    env.in_vr_replay = True
    observations = np.array(
        [original_traj["observations"][j]["state"] for j in range(len(original_traj["observations"]))])
    ax.plot3D(observations[:, 0], observations[:, 1], observations[:, 2], label="original trajs", color="red")
    env.reset()

    images = []
    replay_obs_list = []
    actions = original_traj["actions"]
    agent_infos = original_traj["agent_infos"]
    for j in range(len(actions)):
        action = actions[j]
        agent_info = agent_infos[j]

        observation = env.get_observation()
        next_observation, reward, done, info = env.step(action)
        replay_obs_list.append(next_observation["state"])
        images.append(Image.fromarray(env.render_obs()))
        add_transition(traj, observation, action, reward, info, agent_info,
                       done, next_observation, img_dim)

        if info[accept_trajectory_key] and num_steps < 0:
            num_steps = j

        rewards.append(reward)
        if done:
            break

    # replay_obs_list = np.array(replay_obs_list)
    replay_obs_list = np.array(
        [traj["observations"][j]["state"] for j in range(len(traj["observations"]))])
    ax.plot3D(replay_obs_list[:, 0], replay_obs_list[:, 1], replay_obs_list[:, 2], label="replay", color="blue")

    if not os.path.exists('replay_videos'):
        os.mkdir('replay_videos')

    images[0].save('{}/{}_{}_{}.gif'.format("replay_videos", i, object_names[0], object_names[1]),
                   format='GIF', append_images=images[1:],
                   save_all=True, duration=100, loop=0)
    plt.legend()
    fig.savefig('{}/{}_{}_{}.png'.format("replay_videos", i, object_names[0], object_names[1]))
    fig.clf()

    if info[accept_trajectory_key]:
        success = True

    return traj, success, num_steps

fig = plt.figure()
data = np.load(sys.argv[1], allow_pickle=True)
env_name = sys.argv[2]
success_metric_key = sys.argv[3]
new_trajs = []
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

possible_objects = np.array(PICK_PLACE_TRAIN_OBJECTS)

for obj_idx in range(len(possible_objects) // 2):
    object_names = tuple(possible_objects[np.array([obj_idx, obj_idx + 1])])
    for i in tqdm(range(len(data))):
        new_traj, cur_success, num_steps = replay_one_traj(env, i, success_metric_key, object_names)

        if cur_success:
            new_trajs.append(new_traj)
        else:
            failed_id.append(i)

        success.append(cur_success)



path = "{}_replayed.npy".format(sys.argv[1][:-4])
# np.save(path, data)

success = np.array(success)
print("mean: ", success.mean())
print("std: ", success.std())
print("all success: ", (success == 1).all())
print("failed id: ")
print(failed_id)

# new_data = []
# for i in range(len(data)):
#     if i not in failed_id:
#         new_data.append(data[i])
#
# print(len(new_data))
# for d in new_data:
#     for i in range(len(d['observations'])):
#         if (d['observations'][i]['image'] <= 1).all():
#             d['observations'][i]['image'] = np.reshape(np.uint8(d['observations'][i]['image'] * 255.),
#                                                        (48, 48, 3))
#         if (d['next_observations'][i]['image'] <= 1).all():
#             d['next_observations'][i]['image'] = np.reshape(np.uint8(d['next_observations'][i]['image'] * 255.),
#                                                             (48, 48, 3))

path = "{}_expanded.npy".format(sys.argv[1][:-4])
np.save(path, new_trajs)




