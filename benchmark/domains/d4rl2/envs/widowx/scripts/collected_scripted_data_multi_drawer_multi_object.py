import matplotlib

matplotlib.use('Agg')
import argparse
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import roboverse
import roboverse.bullet as bullet
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
from roboverse.bullet import object_utils
from roboverse.bullet.object_utils import load_bullet_object
from roboverse.envs import objects
from roboverse.envs.widow250 import Widow250Env
from roboverse.envs.widow250_drawer import Widow250DrawerEnv
from roboverse.envs.widow250_multidrawer_multiobject import \
    Widow250MultiDrawerMultiObjectEnv
from roboverse.policies.drawer_close import MultiDrawerClose
from roboverse.policies.drawer_close_open_transfer import \
    DrawerCloseOpenTransfer
from roboverse.policies.drawer_open import MultiDrawerOpen
from roboverse.policies.pick_place import PickPlace

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argument parser type')
    parser.add_argument("--scripted_policy", type=str, default='main_drawer')
    args = parser.parse_args()

    kwargs = {}
    if args.scripted_policy == 'main_drawer_close':
        kwargs['main_start_opened'] = True
    if args.scripted_policy == 'main_top_close':
        kwargs['main_start_top_opened'] = True
    if args.scripted_policy == 'second_drawer_close':
        kwargs['second_start_opened'] = True
    if args.scripted_policy == 'second_top_close':
        kwargs['second_start_top_opened'] = True

    if args.scripted_policy == 'grasp_any_from_main_drawer':
        kwargs['main_start_opened'] = True
    if args.scripted_policy == 'grasp_any_from_second_drawer':
        kwargs['second_start_opened'] = True

    env = roboverse.make('Widow250MultiDrawerMultiObjectEnv-v0',
                         gui=False,
                         transpose_image=False,
                         **kwargs)

    # Drawer policies
    if args.scripted_policy == 'main_drawer_open':
        scripted_policy = MultiDrawerOpen(env, open_drawer_name='main')
    elif args.scripted_policy == 'second_drawer_open':
        scripted_policy = MultiDrawerOpen(env, open_drawer_name='second')
    elif args.scripted_policy == 'second_drawer_close':
        scripted_policy = MultiDrawerClose(env,
                                           close_drawer_name='second_drawer')
    elif args.scripted_policy == 'main_drawer_close':
        scripted_policy = MultiDrawerClose(env,
                                           close_drawer_name='main_drawer')
    elif args.scripted_policy == 'main_top_close':
        scripted_policy = MultiDrawerClose(env, close_drawer_name='main_top')
    elif args.scripted_policy == 'second_top_close':
        scripted_policy = MultiDrawerClose(env, close_drawer_name='second_top')

    # Grasping policies
    if args.scripted_policy == 'grasp_any':
        scripted_policy = PickPlace(env, pick_height_thresh=-0.32)
    if args.scripted_policy == 'grasp_any_from_main_drawer':
        scripted_policy = PickPlace(env,
                                    pick_height_thresh=-0.32,
                                    grasp_from_main_drawer=True)
    if args.scripted_policy == 'grasp_any_from_second_drawer':
        scripted_policy = PickPlace(env,
                                    pick_height_thresh=-0.32,
                                    grasp_from_second_drawer=True)

    import time
    env.reset()

    for j in range(1):
        image_array = []
        for i in range(30):
            action, _ = scripted_policy.get_action()
            obs, rew, done, info = env.step(action)

            # print("reward", rew, "info", info)
            obs_image = env.render_obs()
            # print (obs_image.max(), obs_image.min())

            image_array.append(obs_image)
            im = Image.fromarray(obs_image, mode='RGB')
            # im.save('stepfinal_' + str(i) + '_' + str(j) + '.png')

            # import pdb; pdb.set_trace()
            time.sleep(0.1)

        # Plot the trajectory
        fig = plt.figure(figsize=(50, 50))
        canvas = FigureCanvas(fig)
        stacked_images = np.concatenate(image_array, axis=1)
        plt.imshow(stacked_images)
        canvas.draw()

        plt.tight_layout()

        plt.savefig('traj_' + str(j) + '.png')
        env.reset()
