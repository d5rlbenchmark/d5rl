from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
# from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.envs.multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import *

import numpy as np

class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 config_type='default',
                 **kwargs
                 ):
        self.container_name = container_name
        ct = None
        if config_type == 'default':
            ct = CONTAINER_CONFIGS
        elif config_type == 'default_half':
            ct = CONTAINER_CONFIGS_HALVED
        elif config_type == 'default_third':
            ct = CONTAINER_CONFIGS_THIRD
        elif config_type == 'diverse':
            ct = CONTAINER_CONFIGS_DIVERSE
        elif config_type == 'diverse_half':
            ct = CONTAINER_CONFIGS_DIVERSE_HALVED
        elif config_type == 'diverse_third':
            ct = CONTAINER_CONFIGS_DIVERSE_THIRD
        else:
            assert False, 'Invalid config type'

        container_config = ct[self.container_name]
        print('Container config:', container_config)

        self.fixed_container_position = fixed_container_position
        if self.fixed_container_position:
            self.container_position_low = container_config['container_position_default']
            self.container_position_high = container_config['container_position_default']
        else:
            self.container_position_low = container_config['container_position_low']
            self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']

        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']
        # self.push_success_radius_threshold = container_config['push_success_radius_threshold']
        self.num_objects = len(kwargs['object_names']) if 'object_names' in kwargs else 2
        kwargs['max_reward'] = self.num_objects if kwargs['reward_type'] == 'place_success_all' else 1

        super(Widow250PickPlaceEnv, self).__init__(**kwargs)

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()
        self.objects = {}

        """
        TODO(avi) This needs to be cleaned up, generate function should only 
                  take in (x,y) positions instead. 
        """
        assert self.container_position_low[2] == self.object_position_low[2]

        if not self.in_vr_replay:
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
        self.container_position[-1] = self.container_position_z
        print('Container position:', self.container_position)


        self.x_diff = self.container_position_high[0] - self.container_position[0]
        self.iscontainerleft = abs(self.x_diff) < 0.1
        print('LEFT:', self.iscontainerleft)
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'place_success_all':
            reward = float(info['place_success_all'])
        elif self.reward_type == 'pick_place_left':
            reward = float(info['place_success_target_left'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        elif self.reward_type == 'push':
            reward = float(info['push_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()

        info['place_success_all'] = 0 
        info['place_success'] = False 
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success
                info['place_success_all'] += int(place_success)
        info['all_placed'] = info['place_success_all'] == len(self.object_names)
        
        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        info['place_success_target_left'] = info['place_success_target'] and self.iscontainerleft

        if self.reward_type == 'push':
            info['push_success'] = False
            for object_name in self.object_names:
                push_success = object_utils.check_near_container(
                    object_name, self.objects, self.container_position,
                    self.push_success_radius_threshold
                )
                if push_success:
                    info['push_success'] = push_success

            info['push_success_target'] = object_utils.check_near_container(
                self.target_object, self.objects, self.container_position,
                self.push_success_radius_threshold)
        return info
    
class Widow250PickPlaceEnvDiverseDistractors(Widow250PickPlaceEnv):
    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 disjoint_distractors=True,
                 specific_task_id=False,
                 desired_task_id=(0,0),
                 **kwargs
                 ):
        self.disjoint_distractors=disjoint_distractors
        self.specific_task_id=specific_task_id
        self.desired_task_id=desired_task_id
        self.task_id = (0,0)
        super(Widow250PickPlaceEnvDiverseDistractors, self).__init__(container_name, fixed_container_position, **kwargs)

    def reset(self):
        targ_container = DISJOINT_TARGET_OBJECTS if self.disjoint_distractors else INTERFERING_TARGET_OBJECTS
        dist_container = DISJOINT_DISTRACTOR_OBJECTS if self.disjoint_distractors else INTERFERING_DISTRACTOR_OBJECTS
        
        if self.specific_task_id:
            target_obj = targ_container[self.desired_task_id[0]]
            distractor_obj = dist_container[self.desired_task_id[1]]
            self.task_id = self.desired_task_id
        else:
            target_obj = np.random.choice(targ_container)
            distractor_obj = np.random.choice(dist_container)
            self.task_id = (targ_container.index(target_obj),dist_container.index(distractor_obj))
        
        self.object_names = (target_obj, distractor_obj)
        print('objects in scene', self.object_names)
        print('task id', self.task_id)
        print('disjoint distractors', self.disjoint_distractors)
        
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        return super().reset()
    
    def get_info(self):
        info = super(Widow250PickPlaceEnvDiverseDistractors, self).get_info()
        info['task_id'] = self.task_id
        info['disjoint_distractors'] = self.disjoint_distractors
        
        return info

class Widow250PickPlaceMultiObjectEnv(MultiObjectEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


class Widow250PickPlaceMultiObjectMultiContainerEnv(
    MultiObjectMultiContainerEnv, Widow250PickPlaceEnv):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":

    # Fixed container position
    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed', 'two_handled_vase'),
        object_scales=(0.7, 0.6),
        target_object='shed',
        load_tray=False,
        object_position_low=(.49, .18, -.20),
        object_position_high=(.59, .27, -.20),

        container_name='cube',
        container_position_low=(.72, 0.23, -.20),
        container_position_high=(.72, 0.23, -.20),
        container_position_z=-0.34,
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.05,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)
