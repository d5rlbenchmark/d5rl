import itertools
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

import d4rl2.envs.widowx.roboverse
import d4rl2.envs.widowx.roboverse.bullet as bullet
from d4rl2.envs.widowx.roboverse.assets.shapenet_object_lists import \
    CONTAINER_CONFIGS
from d4rl2.envs.widowx.roboverse.bullet import object_utils
from d4rl2.envs.widowx.roboverse.bullet.object_utils import load_bullet_object
from d4rl2.envs.widowx.roboverse.envs import objects
from d4rl2.envs.widowx.roboverse.envs.widow250 import Widow250Env
from d4rl2.envs.widowx.roboverse.envs.widow250_drawer import Widow250DrawerEnv
from d4rl2.envs.widowx.roboverse.policies.drawer_close_open_transfer import \
    DrawerCloseOpenTransfer
from d4rl2.envs.widowx.roboverse.policies.drawer_open import MultiDrawerOpen
from d4rl2.wrappers.offline_env import OfflineEnv
from d4rl2.wrappers.pixel_wrapper import PixelEnv

matplotlib.use('Agg')


class Widow250MultiDrawerMultiObjectEnv(Widow250DrawerEnv):
    """Environment consisting of multiple drawers and multiple objects, to randomize them"""

    def __init__(self,
                 main_drawer_pos=(0.85, 0.225, -0.35),
                 main_drawer_quat=(0, 0, 0.707107, 0.707107),
                 second_drawer_pos=(0.15, 0.225, -0.35),
                 second_drawer_quat=(0, 0, -0.707107, 0.707107),
                 main_start_opened=False,
                 main_start_top_opened=False,
                 second_start_opened=False,
                 second_start_top_opened=False,
                 drawer_scales=0.1,
                 target_object_names=(),
                 target_object_scales=(),
                 second_drawer_left_opening=False,
                 reward_type='unlock_and_grasp_treasure',
                 **kwargs):
        self.main_start_top_opened = main_start_top_opened
        self.tray_position = (.8, 0.0, -.37)
        self.second_drawer_pos = second_drawer_pos
        self.second_drawer_quat = second_drawer_quat
        self.second_start_opened = second_start_opened
        self.second_start_top_opened = second_start_top_opened
        self.drawer_scales = drawer_scales
        self.second_drawer_left_opening = second_drawer_left_opening

        self.main_start_opened = main_start_opened
        self.main_start_top_opened = main_start_top_opened

        # Target objects go into the drawer during training
        self.target_object_names = target_object_names
        self.target_object_scales = target_object_scales
        assert len(self.target_object_names) == len(
            self.target_object_scales), "Target objects are the same length"

        super(Widow250MultiDrawerMultiObjectEnv,
              self).__init__(drawer_pos=main_drawer_pos,
                             drawer_quat=main_drawer_quat,
                             left_opening=True,
                             start_opened=main_start_opened,
                             drawer_scale=self.drawer_scales,
                             **kwargs)

        self.reward_type = reward_type
        self.place_success_height_threshold = -0.28
        self.place_success_radius_threshold = 0.07

    def _load_meshes(self, ):
        self.table_id = objects.table()
        self.robot_id = objects.widow250(scale=0.6)

        self.objects = {}
        if self.num_objects > 0:
            object_positions = object_utils.generate_object_positions(
                self.object_position_low,
                self.object_position_high,
                self.num_objects,
            )
            self.original_object_positions = object_positions
            print('Object positions generated', self.num_objects)

        ###################################################
        ##########    ADD FIRST DRAWER ###################
        ###################################################
        self.objects["drawer"] = object_utils.load_object(
            "drawer",
            self.drawer_pos,
            self.drawer_quat,
            scale=self.drawer_scale)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]
        opened_drawer_x_pos = object_utils.close_drawer(
            self.objects['drawer'])[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos

        if self.start_opened:
            object_utils.open_drawer(self.objects['drawer'])

        self.objects["drawer_top"] = object_utils.load_object(
            "drawer_no_handle",
            self.drawer_pos + np.array([0, 0, 0.07]),
            self.drawer_quat,
            scale=self.drawer_scales,
        )

        self.tray_id = objects.tray(base_position=self.tray_position,
                                    scale=0.3)

        ## Add object in the drawer on the left:
        closed_drawer_x_pos_temp = object_utils.open_drawer(
            self.objects['drawer'])[0]

        # Load objects in between opening and closing the drawer.
        num_objects_in_first_drawer = int(len(self.target_object_names) // 2)

        self.object_names_in_main_drawer = []
        # Object inside first drawer
        for idx in range(num_objects_in_first_drawer):
            object_idx_position = self.drawer_pos + np.array(
                [0.05, -0.02, 0.07])
            self.objects[self.target_object_names[idx] +
                         '_in_main_drawer'] = object_utils.load_object(
                             self.target_object_names[idx],
                             object_idx_position,
                             object_quat=(0, 0, 1, 0),
                             scale=self.target_object_scales[idx])
            self.object_names_in_main_drawer.append(
                self.target_object_names[idx] + '_in_main_drawer')
            bullet.step_simulation(self.num_sim_steps_reset)

        opened_drawer_x_pos_temp = object_utils.close_drawer(
            self.objects['drawer'])[0]
        #####################################

        ######################################################
        # Now add the second set of drawers
        #####################################################
        self.objects["second_drawer"] = object_utils.load_object(
            "drawer",
            self.second_drawer_pos,
            self.second_drawer_quat,
            scale=self.drawer_scales)

        # Open and close testing.
        closed_second_drawer_x_pos = object_utils.open_drawer(
            self.objects['second_drawer'])[0]
        opened_second_drawer_x_pos = object_utils.close_drawer(
            self.objects['second_drawer'])[0]

        if self.second_drawer_left_opening:
            self.second_drawer_min_x_pos = opened_second_drawer_x_pos
            self.second_drawer_max_x_pos = closed_second_drawer_x_pos
        else:
            self.second_drawer_min_x_pos = opened_second_drawer_x_pos
            self.second_drawer_max_x_pos = closed_second_drawer_x_pos

        print('Drawer positions: ', self.second_drawer_min_x_pos,
              self.second_drawer_max_x_pos, self.drawer_pos)

        ## Add object in the drawer on the right
        closed_second_drawer_x_pos_temp = object_utils.open_drawer(
            self.objects['second_drawer'])[0]

        self.object_names_in_second_drawer = []
        # Object inside second drawer
        for idx in range(
                len(self.target_object_names) - num_objects_in_first_drawer):
            jdx = idx + num_objects_in_first_drawer
            object_idx_position = self.second_drawer_pos + np.array(
                [-0.05, 0.02, 0.07])
            self.objects[self.target_object_names[jdx] +
                         '_in_second_drawer'] = object_utils.load_object(
                             self.target_object_names[jdx],
                             object_idx_position,
                             object_quat=(0, 0, 1, 0),
                             scale=self.target_object_scales[jdx])
            self.object_names_in_second_drawer.append(
                self.target_object_names[jdx] + '_in_second_drawer')
            bullet.step_simulation(self.num_sim_steps_reset)

        opened_second_drawer_x_pos_temp = object_utils.close_drawer(
            self.objects['second_drawer'])[0]

        # Add the second top drawer
        self.objects["second_drawer_top"] = object_utils.load_object(
            "drawer_no_handle",
            self.second_drawer_pos + np.array([0, 0, 0.07]),
            self.second_drawer_quat,
            scale=self.drawer_scales)

        bullet.step_simulation(self.num_sim_steps_reset)
        ##################################################

        if self.main_start_opened:
            object_utils.open_drawer(self.objects["drawer"])

        if self.main_start_top_opened:
            object_utils.open_drawer(self.objects["drawer_top"],
                                     half_open=True)

        if self.second_start_opened:
            object_utils.open_drawer(self.objects['second_drawer'])

        if self.second_start_top_opened:
            object_utils.open_drawer(self.objects["second_drawer_top"],
                                     half_open=True)

        bullet.step_simulation(self.num_sim_steps_reset)

        ## Now object manipulation starts
        ############### Load blocking objects #####################
        self.blocking_object_name = random.choice([
            'fountain_vase',
        ])

        if self.blocking_object_in_tray:
            object_position = np.random.uniform(low=(.79, -.02, -.34),
                                                high=(.81, .02, -.34))
            # TODO Maybe randomize the quat as well
            object_quat = (0, 0, 1, 0)
        else:
            object_position = np.random.uniform(low=(.59, .19, -.34),
                                                high=(.62, .21, -.34))
            object_quat = (0, 0, 1, 0)

        self.blocking_object = object_utils.load_object(
            self.blocking_object_name,
            object_position,
            object_quat=object_quat,
            scale=0.75)

        bullet.step_simulation(self.num_sim_steps_reset)
        ############################################################

        # Finally load other objects
        print('Object names here: ', self.object_names)
        if len(self.object_names) > 0:
            for object_name, object_position in zip(self.object_names,
                                                    object_positions):
                print('in widowx drawer env: ', object_name, object_position)
                self.objects[object_name] = object_utils.load_object(
                    object_name,
                    object_position,
                    object_quat=self.object_orientations[object_name],
                    scale=self.object_scales[object_name])
                bullet.step_simulation(self.num_sim_steps_reset)

        print('Final object names: ', self.object_names, self.objects)

    def get_info(self):
        info = super(Widow250MultiDrawerMultiObjectEnv, self).get_info()

        # Get drawer info
        drawers = [
            'drawer', 'drawer_top', 'second_drawer', 'second_drawer_top'
        ]
        for drawer_name in drawers:
            info[drawer_name + '_x_pos'] = self.get_drawer_pos(drawer_name)[0]
            if 'second' in drawer_name:
                info[
                    drawer_name +
                    '_opened_percentage'] = self.get_second_drawer_opened_percentage(
                        drawer_name)
            else:
                info[drawer_name +
                     '_opened_percentage'] = self.get_drawer_opened_percentage(
                         drawer_name)
            info[drawer_name + '_closed_success'] = info[drawer_name + "_opened_percentage"] \
                < self.drawer_closed_success_thresh
            info[drawer_name + '_opened_success'] = info[drawer_name + "_opened_percentage"] > \
                self.drawer_opened_success_thresh

        # Get info for objects
        info['place_success'] = False
        info['any_object_place_success'] = False
        for object_name in self.object_names:
            target_point = bullet.get_object_position(self.tray_id)[0]
            target_point[2] = -0.2
            place_success = object_utils.check_in_container(
                object_name, self.objects, target_point,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            info['place_success_' + object_name] = place_success
            info['any_object_place_success'] = (
                info['any_object_place_success'] or place_success)

        for object_name in self.object_names_in_main_drawer:
            target_point = bullet.get_object_position(self.tray_id)[0]
            target_point[2] = -0.2
            place_success = object_utils.check_in_container(
                object_name, self.objects, target_point,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            info['place_success_' + object_name +
                 '_from_main_drawer'] = place_success

        for object_name in self.object_names_in_second_drawer:
            target_point = bullet.get_object_position(self.tray_id)[0]
            target_point[2] = -0.2
            place_success = object_utils.check_in_container(
                object_name, self.objects, target_point,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            info['place_success_' + object_name +
                 '_from_second_drawer'] = place_success

        return info

    def get_second_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(
            self.objects["second_drawer"])
        return handle_pos

    def is_second_drawer_open(self):
        # refers to bottom drawer in the double drawer case
        info = self.get_info()
        print('Second drawer: ', info['second_drawer_opened_success'],
              info['second_drawer_opened_percentage'])
        return info['second_drawer_opened_success']

    def get_second_drawer_opened_percentage(self, drawer_key="second_drawer"):
        # compatible with either drawer or upper_drawer
        drawer_x_pos = self.get_drawer_pos(drawer_key)[0]
        return object_utils.get_drawer_opened_percentage(
            self.second_drawer_left_opening, self.second_drawer_min_x_pos,
            self.second_drawer_max_x_pos, drawer_x_pos)

    def is_second_top_drawer_closed(self):
        info = self.get_info()
        return info['second_drawer_top_closed_success']

    def is_top_drawer_closed(self):
        info = self.get_info()
        return info['drawer_top_closed_success']

    def is_second_drawer_closed(self):
        info = self.get_info()
        return info['second_drawer_closed_success']

    def is_drawer_closed(self):
        info = self.get_info()
        return info['drawer_closed_success']

    def get_drawer_pos(self, drawer_key="drawer"):
        drawer_pos = object_utils.get_drawer_pos(self.objects[drawer_key])
        return drawer_pos

    def get_reward(self, info=False):
        if not info:
            info = self.get_info()

        if self.reward_type == 'unlock_and_grasp_treasure':
            key_reward = 0.0
            for object_name in self.object_names_in_second_drawer:
                key_reward += float(info['place_success_' + object_name +
                                         '_from_second_drawer'])

            drawer_reward = float(info['drawer_opened_success'])
            total_reward = max(key_reward + drawer_reward - 1.0, 0.0)

        elif self.reward_type == 'grasp_treasure':
            total_reward = float(info['drawer_opened_success'])

        return total_reward


def make_offline_env(dataset_url, *args, **kwargs):
    env = Widow250MultiDrawerMultiObjectEnv(*args, **kwargs)
    env = PixelEnv(env)
    env = OfflineEnv(env, dataset_url)
    return env
