import os.path as osp

import numpy as np

import d4rl2.envs.widowx.roboverse.bullet as bullet
from d4rl2.envs.widowx.roboverse.assets.shapenet_object_lists import \
    CONTAINER_CONFIGS
from d4rl2.envs.widowx.roboverse.bullet import control, object_utils
from d4rl2.envs.widowx.roboverse.envs import objects
from d4rl2.envs.widowx.roboverse.envs.widow250 import Widow250Env

from .multi_object import (MultiObjectEnv, MultiObjectMultiContainerEnv,
                           MultiObjectMultiContainerV2Env, MultiObjectV2Env)

OBJECT_IN_GRIPPER_PATH = osp.join(
    osp.dirname(osp.dirname(osp.realpath(__file__))),
    'assets/bullet-objects/bullet_saved_states/objects_in_gripper/')


class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 start_object_in_gripper=False,
                 container_position_z_offset=0.01,
                 **kwargs):
        self.container_name = container_name

        container_config = CONTAINER_CONFIGS[self.container_name]
        self.fixed_container_position = fixed_container_position
        if self.fixed_container_position:
            self.container_position_low = container_config[
                'container_position_default']
            self.container_position_high = container_config[
                'container_position_default']
        else:
            self.container_position_low = container_config[
                'container_position_low']
            self.container_position_high = container_config[
                'container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config[
            'min_distance_from_object']

        self.place_success_height_threshold = container_config[
            'place_success_height_threshold']
        self.place_success_radius_threshold = container_config[
            'place_success_radius_threshold']
        if 'half_extents' in container_config:
            self.container_half_extents = container_config['half_extents']
        else:
            self.container_half_extents = None

        self.start_object_in_gripper = start_object_in_gripper
        self.container_position_z_offset = container_position_z_offset
        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        if self.load_tray:
            self.tray_id = objects.tray_no_divider()
        self.robot_id = objects.widow250()
        self.objects = {}
        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if self.num_objects == 2:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        elif self.num_objects == 1:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_single(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        else:
            raise NotImplementedError

        # TODO(avi) Need to clean up
        self.container_position[
            -1] = self.container_position_z + self.container_position_z_offset
        self.container_id = object_utils.load_object(
            self.container_name, self.container_position,
            self.container_orientation, self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)
        for object_name, object_position in zip(
                self.object_names, self.original_object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name],
                randomize_object_quat=self.randomize_object_quat)
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        super(Widow250PickPlaceEnv, self).reset()
        ee_pos_init, ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_pos_init[2] -= 0.05

        if self.start_object_in_gripper:
            bullet.load_state(
                osp.join(OBJECT_IN_GRIPPER_PATH,
                         'object_in_gripper_reset.bullet'))
            self.is_gripper_open = False

        return self.get_observation()

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        info['place_success_object_name'] = None
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success
                info['place_success_object_name'] = object_name

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info

    def get_obj_positions(self):
        object_infos = {}
        for object_name in self.object_names:
            object_pos, _ = control.get_object_position(
                self.objects[object_name])
            object_infos[object_name] = object_pos
        return object_infos


class Widow250PickPlaceMultiObjectEnv(MultiObjectEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceMultiObjectV2Env(MultiObjectV2Env,
                                        Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceMultiObjectMultiContainerEnv(
        MultiObjectMultiContainerEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceResetFreeEnv(MultiObjectV2Env, Widow250PickPlaceEnv):
    """Reset Free multi object env. Reset task is (self.num_tasks + task_idx) """

    def __init__(self, **kwargs):
        super(Widow250PickPlaceResetFreeEnv,
              self).__init__(object_position_low=[0.45, .14, -0.3],
                             object_position_high=[0.73, 0.35, -0.3],
                             camera_distance=0.37,
                             **kwargs)
        self.tray_position = (np.array(self.object_position_low) +
                              np.array(self.object_position_high)) / 2
        self.tray_half_extents = (np.array(self.object_position_high) -
                                  self.tray_position)[:2]

    def is_reset_task(self):
        return (self.task_idx // self.num_tasks) % 2 == 1

    def get_reward(self, info):
        if self.is_reset_task():
            reward = float(info['reset_success_target'])
        else:
            reward = float(info['place_success_target'])
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        info['place_success_object_name'] = None
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name,
                self.objects,
                self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold,
                container_half_extents=self.container_half_extents)
            if place_success:
                info['place_success'] = place_success
                info['place_success_object_name'] = object_name
        info['reset_success'] = not info['place_success']
        for object_name in self.object_names:
            info['reset_success'] = info['reset_success'] and object_utils.check_under_height_threshold(
                object_name, self.objects, self.place_success_height_threshold) and \
                object_utils.check_in_container(
                    object_name, self.objects, self.tray_position,
                    self.place_success_height_threshold,
                    self.place_success_radius_threshold,
                    container_half_extents=self.tray_half_extents)

        if self.is_reset_task():
            info['reset_success_target'] = not object_utils.check_in_container(
                self.target_object, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold,
                container_half_extents=self.container_half_extents) and object_utils.check_under_height_threshold(
                self.target_object, self.objects, self.place_success_height_threshold) and \
                object_utils.check_in_container(
                    self.target_object, self.objects, self.tray_position,
                    self.place_success_height_threshold,
                    self.place_success_radius_threshold,
                    container_half_extents=self.tray_half_extents)
        else:
            info['reset_success_target'] = False

        if not self.is_reset_task():
            info['place_success_target'] = object_utils.check_in_container(
                self.target_object,
                self.objects,
                self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold,
                container_half_extents=self.container_half_extents)
        else:
            info['place_success_target'] = False

        return info

    def get_observation(self):
        observation = super(MultiObjectV2Env, self).get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks * 2, ))
            if self.task_idx < self.num_tasks * 2:
                one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector

        return observation

    def _load_meshes(self):
        self.table_id = objects.table()
        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()
        self.robot_id = objects.widow250()
        self.objects = {}
        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        assert self.num_objects == 2
        # reverse task
        if self.is_reset_task():
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_reverse(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    self.target_object, self.place_success_radius_threshold,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                    container_half_extents=self.container_half_extents,
                )
        else:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )

        # TODO(avi) Need to clean up
        self.container_position[
            -1] = self.container_position_z + self.container_position_z_offset
        self.container_id = object_utils.load_object(
            self.container_name, self.container_position,
            self.container_orientation, self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        self.objects[self.target_object] = object_utils.load_object(
            self.target_object,
            self.original_object_positions[1],
            object_quat=self.object_orientations[self.target_object],
            scale=self.object_scales[self.target_object],
            randomize_object_quat=self.randomize_object_quat)
        bullet.step_simulation(self.num_sim_steps_reset)

        other_object = self.object_names[
            (self.object_names.index(self.target_object) - 1) * -1]
        self.objects[other_object] = object_utils.load_object(
            other_object,
            self.original_object_positions[0],
            object_quat=self.object_orientations[other_object],
            scale=self.object_scales[other_object],
            randomize_object_quat=self.randomize_object_quat)
        bullet.step_simulation(self.num_sim_steps_reset)


class Widow250PickPlaceResetFreeMultiContainerEnv(
        MultiObjectMultiContainerV2Env, Widow250PickPlaceEnv):
    """Reset Free multi object env. Reset task is (self.num_tasks + task_idx) """

    def __init__(self, **kwargs):
        super(Widow250PickPlaceResetFreeMultiContainerEnv,
              self).__init__(object_position_low=[0.45, .14, -0.3],
                             object_position_high=[0.73, 0.35, -0.3],
                             camera_distance=0.37,
                             **kwargs)
        self.tray_position = (np.array(self.object_position_low) +
                              np.array(self.object_position_high)) / 2
        self.tray_half_extents = (np.array(self.object_position_high) -
                                  self.tray_position)[:2]

    def is_reset_task(self):
        return (self.task_idx // self.num_tasks) % 2 == 1

    def get_reward(self, info):
        if self.is_reset_task():
            reward = float(info['reset_success_target'])
        else:
            reward = float(info['place_success_target'])
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success'] = False
        info['place_success_object_name'] = None
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success
                info['place_success_object_name'] = object_name
        info['reset_success'] = not info['place_success']
        for object_name in self.object_names:
            info['reset_success'] = info['reset_success'] and object_utils.check_under_height_threshold(
                object_name, self.objects, self.place_success_height_threshold) and \
                object_utils.check_in_container(
                    object_name, self.objects, self.tray_position,
                    self.place_success_height_threshold,
                    self.place_success_radius_threshold,
                    container_half_extents=self.tray_half_extents)

        if self.is_reset_task():
            info['reset_success_target'] = not object_utils.check_in_container(
                self.target_object, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold,
                container_half_extents=self.container_half_extents) and object_utils.check_under_height_threshold(
                self.target_object, self.objects, self.place_success_height_threshold) and \
                object_utils.check_in_container(
                    self.target_object, self.objects, self.tray_position,
                    self.place_success_height_threshold,
                    self.place_success_radius_threshold,
                    container_half_extents=self.tray_half_extents)
        else:
            info['reset_success_target'] = False

        if not self.is_reset_task():
            info['place_success_target'] = object_utils.check_in_container(
                self.target_object, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
        else:
            info['place_success_target'] = False

        return info

    def get_observation(self):
        observation = super(MultiObjectMultiContainerV2Env,
                            self).get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks * 2, ))
            if self.task_idx < self.num_tasks * 2:
                one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector

        return observation

    def _load_meshes(self):
        self.table_id = objects.table()
        if self.load_tray:
            self.tray_id = objects.tray_no_divider_scaled()
        self.robot_id = objects.widow250()
        self.objects = {}
        """
        TODO(avi) This needs to be cleaned up, generate function should only
                  take in (x,y) positions instead.
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        assert self.num_objects == 2
        # reverse task
        if self.is_reset_task():
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_reverse(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    self.target_object, self.place_success_radius_threshold,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                    container_half_extents=self.container_half_extents,
                )
        else:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )

        # TODO(avi) Need to clean up
        self.container_position[
            -1] = self.container_position_z + self.container_position_z_offset
        self.container_id = object_utils.load_object(
            self.container_name, self.container_position,
            self.container_orientation, self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        self.objects[self.target_object] = object_utils.load_object(
            self.target_object,
            self.original_object_positions[1],
            object_quat=self.object_orientations[self.target_object],
            scale=self.object_scales[self.target_object],
            randomize_object_quat=self.randomize_object_quat)
        bullet.step_simulation(self.num_sim_steps_reset)

        other_object = self.object_names[
            (self.object_names.index(self.target_object) - 1) * -1]
        self.objects[other_object] = object_utils.load_object(
            other_object,
            self.original_object_positions[0],
            object_quat=self.object_orientations[other_object],
            scale=self.object_scales[other_object],
            randomize_object_quat=self.randomize_object_quat)
        bullet.step_simulation(self.num_sim_steps_reset)
