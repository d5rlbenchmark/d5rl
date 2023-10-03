import itertools
import random

import numpy as np
from PIL import Image

import d4rl2.envs.widowx.roboverse
import d4rl2.envs.widowx.roboverse.bullet as bullet
from d4rl2.envs.widowx.roboverse.assets.shapenet_object_lists import \
    CONTAINER_CONFIGS
from d4rl2.envs.widowx.roboverse.bullet import object_utils
from d4rl2.envs.widowx.roboverse.bullet.object_utils import load_bullet_object
from d4rl2.envs.widowx.roboverse.envs import objects
from d4rl2.envs.widowx.roboverse.envs.widow250 import Widow250Env


class Widow250DrawerEnv(Widow250Env):

    def __init__(
            self,
            drawer_pos=(0.5, 0.2, -.35),
            drawer_quat=(0, 0, 0.707107, 0.707107),
            left_opening=True,  # False is not supported
            start_opened=False,
            blocking_object_in_tray=True,
            drawer_scale=0.1,
            **kwargs):
        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.left_opening = left_opening
        self.start_opened = start_opened
        self.blocking_object_in_tray = blocking_object_in_tray
        self.drawer_opened_success_thresh = 0.95
        self.drawer_closed_success_thresh = 0.05
        self.drawer_scale = drawer_scale
        # obj_pos_high, obj_pos_low = self.get_obj_pos_high_low()
        super(Widow250DrawerEnv, self).__init__(
            # object_position_high=obj_pos_high,
            # object_position_low=obj_pos_low,
            **kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250(scale=0.6)

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low,
            self.object_position_high,
            self.num_objects,
        )
        self.original_object_positions = object_positions

        self.objects["drawer"] = object_utils.load_object(
            "drawer",
            self.drawer_pos,
            self.drawer_quat,
            scale=self.drawer_scale)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]

        print(self.object_names)
        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            print('in widowx drawer env: ', object_name, object_position)
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

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

    def get_obj_pos_high_low(self):
        obj_pos_high = np.array(self.drawer_pos[:2] + (-.2,)) \
                       + (1 - 2 * (not self.left_opening)) * np.array((0.12, 0, 0))
        obj_pos_low = np.array(self.drawer_pos[:2] + (-.2,)) \
            - (1 - 2 * (not self.left_opening)) * np.array((-0.12, 0, 0))

        # randomization along x-axis
        obj_pos_high[0] = obj_pos_high[0] + 0.015
        obj_pos_low[0] = obj_pos_low[0] - 0.015

        # randomization along y-axis
        # obj_pos_high[1] = obj_pos_high[1] + 0.01
        # obj_pos_low[1] = obj_pos_low[1] - 0.01

        return obj_pos_high, obj_pos_low

    def get_info(self):
        info = super(Widow250DrawerEnv, self).get_info()
        info['drawer_x_pos'] = self.get_drawer_pos()[0]
        info['drawer_opened_percentage'] = \
            self.get_drawer_opened_percentage()
        info['drawer_opened_success'] = info["drawer_opened_percentage"] > \
            self.drawer_opened_success_thresh
        return info

    def get_drawer_handle_pos(self):
        handle_pos = object_utils.get_drawer_handle_pos(self.objects["drawer"])
        return handle_pos

    def is_drawer_open(self):
        # refers to bottom drawer in the double drawer case
        info = self.get_info()
        return info['drawer_opened_success']

    def get_drawer_opened_percentage(self, drawer_key="drawer"):
        # compatible with either drawer or upper_drawer
        drawer_x_pos = self.get_drawer_pos(drawer_key)[0]
        return object_utils.get_drawer_opened_percentage(
            self.left_opening, self.drawer_min_x_pos, self.drawer_max_x_pos,
            drawer_x_pos)

    def get_drawer_pos(self, drawer_key="drawer"):
        drawer_pos = object_utils.get_drawer_pos(self.objects[drawer_key])
        return drawer_pos

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "opening":
            return float(self.is_drawer_open())
        else:
            return super(Widow250DrawerEnv, self).get_reward(info)


class Widow250DrawerRandomizedEnv(Widow250DrawerEnv):
    """Repurposed as a meta-env, where the task_idx = {0, 1}
    0 = right-opening drawer
    1 = left-opening drawer.
    """

    def __init__(
            self,
            num_objects=0,  # unused
            possible_objects=[],  # unused
            init_task_idx=None,
            num_tasks=2,  # unused
            fixed_task=False,
            **kwargs):
        self.task_idx = init_task_idx
        # Implicitly enforced: left_opening == self.task_idx
        self.num_tasks = 2
        self.fixed_task = fixed_task
        self.left_opening = self.set_random_task_idx()
        drawer_pos, drawer_quat = self.set_drawer_pos_and_quat()
        super(Widow250DrawerRandomizedEnv,
              self).__init__(drawer_pos=drawer_pos,
                             drawer_quat=drawer_quat,
                             left_opening=self.task_idx,
                             **kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray_no_divider(base_position=(.61, 0.3,
                                                                  -.4),
                                                   scale=0.775)

        self.objects = {}
        object_positions = object_utils.generate_object_positions(
            self.object_position_low,
            self.object_position_high,
            self.num_objects,
        )
        self.original_object_positions = object_positions

        self.objects["drawer"] = object_utils.load_object("drawer",
                                                          self.drawer_pos,
                                                          self.drawer_quat,
                                                          scale=0.1)
        # Open and close testing.
        closed_drawer_x_pos = object_utils.open_drawer(
            self.objects['drawer'])[0]

        # Load objects in between opening and closing the drawer.
        for object_name, object_position in zip(self.object_names,
                                                object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])

            self.blocking_object = self.objects[object_name]
            bullet.step_simulation(self.num_sim_steps_reset)

        opened_drawer_x_pos = object_utils.close_drawer(
            self.objects['drawer'])[0]

        if self.left_opening:
            self.drawer_min_x_pos = closed_drawer_x_pos
            self.drawer_max_x_pos = opened_drawer_x_pos
        else:
            self.drawer_min_x_pos = opened_drawer_x_pos
            self.drawer_max_x_pos = closed_drawer_x_pos

    def get_obj_pos_high_low(self):
        # obj_pos_high = tuple(
        #     np.array(self.drawer_pos) + ((-1) ** (1 + self.left_opening)) * np.array([0.05, 0.06, 0.1]))
        # obj_pos_low = tuple(
        #     np.array(self.drawer_pos) + ((-1) ** (1 + self.left_opening)) * np.array([0.05, 0.06, 0.1]))
        if self.task_idx == 0:
            obj_pos_high = (0.45, 0.18, -0.35)
            obj_pos_low = (0.45, 0.2, -0.35)
        elif self.task_idx == 1:
            obj_pos_high = (0.75, 0.18, -0.35)
            obj_pos_low = (0.75, 0.2, -0.35)
        else:
            raise NotImplementedError
        return obj_pos_high, obj_pos_low

    def reset_task(self, task_idx):
        if not self.fixed_task:
            self.task_idx = task_idx
        return self.task_idx

    def set_random_task_idx(self):
        if self.task_idx is None:
            return self.reset_task(random.choice(list(range(self.num_tasks))))
        else:
            return self.task_idx

    def set_drawer_pos_and_quat(self):
        drawer_position = [(0.75, 0.225, -0.35),
                           (0.45, 0.225, -0.35)][self.task_idx]
        drawer_quat = [(0, 0, -0.707107, 0.707107),
                       (0, 0, 0.707107, 0.707107)][self.task_idx]
        return drawer_position, drawer_quat

    def reset(self):
        self.left_opening = self.set_random_task_idx() >= (self.num_tasks // 2)
        self.drawer_pos, self.drawer_quat = \
            self.set_drawer_pos_and_quat()
        self.object_position_low, self.object_position_high = \
            self.get_obj_pos_high_low()
        return super(Widow250DrawerRandomizedEnv, self).reset()

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks, ))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector
        return observation


class Widow250DrawerRandomizedOpenPickPlaceEnv(Widow250DrawerRandomizedEnv):

    def __init__(
            self,
            num_objects=1,  # unused
            possible_objects=[],  # unused
            init_task_idx=None,
            num_tasks=4,  # unused
            fixed_task=False,
            **kwargs):
        self.task_idx = init_task_idx
        # Implicitly enforced: left_opening == self.task_idx
        self.num_tasks = 4
        self.fixed_task = fixed_task
        self.left_opening = self.set_random_task_idx() >= (self.num_tasks // 2)
        self.place_success_height_threshold = -0.28
        self.place_success_radius_threshold = self.get_place_success_radius_thresh(
        )
        drawer_pos, drawer_quat = self.set_drawer_pos_and_quat()
        self.obj_pos_high_list = [
            (0.45, 0.18, -0.35),
            (0.625, 0.22, -0.35),
            (0.75, 0.18, -0.35),
            (0.575, 0.22, -0.35),
        ]
        self.obj_pos_low_list = [
            (0.45, 0.2, -0.35),
            (0.625, 0.24, -0.35),
            (0.75, 0.2, -0.35),
            (0.575, 0.24, -0.35),
        ]
        super(Widow250DrawerRandomizedEnv,
              self).__init__(drawer_pos=drawer_pos,
                             drawer_quat=drawer_quat,
                             left_opening=self.task_idx,
                             **kwargs)

    def get_place_success_radius_thresh(self):
        # Allow a larger radius when we are picking from drawer
        # and placing outside the drawer.
        return [0.04, 0.07][self.task_idx % 2]

    def set_drawer_pos_and_quat(self):
        drawer_position = [(0.75, 0.225, -0.35),
                           (0.45, 0.225, -0.35)][self.left_opening]
        drawer_quat = [(0, 0, -0.707107, 0.707107),
                       (0, 0, 0.707107, 0.707107)][self.left_opening]
        return drawer_position, drawer_quat

    def get_obj_pos_high_low(self):
        obj_pos_high = self.obj_pos_high_list[self.task_idx]
        obj_pos_low = self.obj_pos_low_list[self.task_idx]
        return obj_pos_high, obj_pos_low

    def get_target_drop_point(self, drawer_pos):
        if self.task_idx % 2 == 0:
            return drawer_pos
        else:
            return np.array(self.obj_pos_high_list[self.task_idx - 1])

    def get_reward(self, info):
        if not info:
            info = self.get_info()
        if self.reward_type == "opening":
            return float(self.is_drawer_open())
        elif self.reward_type == "pick_place":
            return float(info['place_success_target'])
        else:
            raise NotImplementedError

    def get_info(self):
        info = super(Widow250DrawerRandomizedOpenPickPlaceEnv, self).get_info()

        drawer_pos = self.get_drawer_pos()
        target_drop_point = self.get_target_drop_point(drawer_pos)

        # This needs to be conditioned on the task_idx
        info['place_success'] = False
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, target_drop_point,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success
                info['place_success_object_name'] = object_name

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, target_drop_point,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info


class Widow250DoubleDrawerEnv(Widow250DrawerEnv):

    def __init__(
            self,
            drawer_pos=(0.5, 0.2, -.35),
            drawer_quat=(0, 0, 0.707107, 0.707107),
            left_opening=True,  # False is not supported
            start_opened=False,
            start_top_opened=False,
            **kwargs):
        self.start_top_opened = start_top_opened
        self.tray_position = (.8, 0.0, -.37)

        super(Widow250DoubleDrawerEnv, self).__init__(
            drawer_pos=drawer_pos,
            drawer_quat=drawer_quat,
            left_opening=left_opening,  # False is not supported
            start_opened=start_opened,
            **kwargs,
        )

    def _load_meshes(self):
        super(Widow250DoubleDrawerEnv, self)._load_meshes()
        self.objects["drawer_top"] = object_utils.load_object(
            "drawer_no_handle",
            self.drawer_pos + np.array([0, 0, 0.07]),
            self.drawer_quat,
            scale=0.1)

        self.tray_id = objects.tray(base_position=self.tray_position,
                                    scale=0.3)

        self.blocking_object_name = 'gatorade'

        if self.blocking_object_in_tray:
            object_position = np.random.uniform(low=(.79, .0, -.34),
                                                high=(.81, .0, -.34))
            # TODO Maybe randomize the quat as well
            object_quat = (0, 0, 1, 0)
        else:
            object_position = np.random.uniform(low=(.63, .2, -.34),
                                                high=(.65, .2, -.34))
            object_quat = (0, 0, 1, 0)

        self.blocking_object = object_utils.load_object(
            self.blocking_object_name,
            object_position,
            object_quat=object_quat,
            scale=1.0)

        bullet.step_simulation(self.num_sim_steps_reset)

        if self.start_top_opened:
            object_utils.open_drawer(self.objects["drawer_top"],
                                     half_open=True)

    def get_info(self):
        info = super(Widow250DoubleDrawerEnv, self).get_info()
        info['drawer_top_x_pos'] = self.get_drawer_pos("drawer_top")[0]
        info['drawer_top_opened_percentage'] = \
            self.get_drawer_opened_percentage("drawer_top")
        info['drawer_top_closed_success'] = info["drawer_top_opened_percentage"] \
            < self.drawer_closed_success_thresh
        return info

    def is_top_drawer_closed(self):
        info = self.get_info()
        return info['drawer_top_closed_success']
