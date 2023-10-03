from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
# from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.envs.multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import *

import numpy as np
from roboverse.envs.widow250_pickplace import *

def bin_sort_hash(obj_name):
    if obj_name in BIN_SORT_OBJECTS:
        return BIN_SORT_OBJECTS.index(obj_name)
    else:
        assert False

class Widow250BinSortEnv(Widow250Env):
    
    def __init__(self,
                 container1_name='bowl_small_pos1',
                 container2_name='bowl_small_pos2',
                 fixed_container_position=False,
                 config_type='default',
                 rand_obj=False,
                 bin_obj=False,
                 num_objects=2,
                 obj_scale_default=0.75,
                 obj_orientation_default=(0, 0, 1, 0),
                 trunc=0,
                 specific_task_id=False,
                 desired_task_id=(0,0),
                 **kwargs):
        
        if specific_task_id:
            self.num_objects = len(desired_task_id)
        elif rand_obj:
            self.num_objects = num_objects
        else:
            self.num_objects = 2
            
        self.rand_obj = rand_obj
        self.specific_task_id = specific_task_id
        self.desired_task_id = desired_task_id

        self.bin_obj = bin_obj
        self.trunc = max(min(trunc, len(BIN_SORT_OBJECTS)), self.num_objects)
        print('trunc', self.trunc)

        kwargs['max_reward'] = self.num_objects
        kwargs['object_scales'] = [obj_scale_default] * self.num_objects
        kwargs['object_orientations'] = [obj_orientation_default] * self.num_objects

        if specific_task_id:
            kwargs['object_names'] = tuple([BIN_SORT_OBJECTS[x] for x in desired_task_id])
        elif rand_obj:
            if self.trunc == 0:
                kwargs['object_names'] = tuple(np.random.choice(BIN_SORT_OBJECTS, size=self.num_objects, replace=False))
            else:
                kwargs['object_names'] = tuple(np.random.choice(BIN_SORT_OBJECTS[:self.trunc], size=self.num_objects, replace=False))
        else:
            kwargs['object_names'] = ('ball', 'jar')

        self.container1_name = container1_name
        self.container2_name = container2_name

        ct = None
        if config_type == 'default':
            ct = CONTAINER_CONFIGS_BIN_SORT
        else:
            assert False, 'Invalid config type'

        container_config = ct[self.container1_name]
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

        container2_config = ct[self.container2_name]
        print('Container config:', container2_config)
        if self.fixed_container_position:
            self.container2_position_low = container2_config['container_position_default']
            self.container2_position_high = container2_config['container_position_default']
        else:
            self.container2_position_low = container2_config['container_position_low']
            self.container2_position_high = container2_config['container_position_high']
        self.container2_position_z = container2_config['container_position_z']
        self.container2_orientation = container2_config['container_orientation']
        self.container2_scale = container2_config['container_scale']
        self.min_distance_from_object2 = container2_config['min_distance_from_object']

        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']
        
        
        kwargs['target_object'] = np.random.choice(kwargs['object_names'])
        kwargs['camera_distance'] = 0.4

        super(Widow250BinSortEnv, self).__init__(**kwargs)

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
            positions, self.original_object_positions = \
                object_utils.generate_object_positions_v3(
                    self.object_position_low, self.object_position_high,
                    [self.container_position_low, self.container2_position_low],
                    [self.container_position_high, self.container2_position_high],
                    min_distance_large_obj=self.min_distance_from_object,
                    num_large=2, num_small=self.num_objects,
                )
            self.container_position, self.container2_position = positions
        
        if self.bin_obj:
            pos = self.container_position if bin_sort_hash(self.object_names[-1])%2 == 0 else self.container2_position
            pos[-1] = self.original_object_positions[-1][-1]
            self.original_object_positions[-1] = pos
        
        self.container_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.container1_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        self.container2_position[-1] = self.container_position_z
        self.container_id = object_utils.load_object(self.container2_name,
                                                     self.container2_position,
                                                     self.container2_orientation,
                                                     self.container2_scale)
        bullet.step_simulation(self.num_sim_steps_reset)

        for object_name, object_position in zip(self.object_names,
                                                self.original_object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)
    
    def reset(self):
        if self.specific_task_id:
            self.object_names = tuple([BIN_SORT_OBJECTS[x] for x in self.desired_task_id])
        elif self.rand_obj:
            if self.trunc == 0:
                self.object_names = tuple(np.random.choice(BIN_SORT_OBJECTS, size=self.num_objects, replace=False))
            else:
                self.object_names = tuple(np.random.choice(BIN_SORT_OBJECTS[:self.trunc], size=self.num_objects, replace=False))
        else:
            self.object_names = ('ball', 'jar')
        print('objects in scene', self.object_names)
        
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        return super().reset()


    def get_reward(self, info):
        if self.objects_in_container and 'all_placed' in info and info['all_placed'] and info['place_success_target'] != self.num_objects:
            return info['place_success_target']-self.num_objects # number of objects placed wrong
        return float(info['place_success_target'])

    def get_info(self, debug=False):
        info = super(Widow250BinSortEnv, self).get_info()

        threshold_scale=1

        info['place_success'] = False
        info['place_success_target'] = 0
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            place2_success = object_utils.check_in_container(
                object_name, self.objects, self.container2_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            info['place_success'] = info['place_success'] or place_success or place2_success

        info['all_placed'] = True
        for object_name in self.object_names:
            place_success = object_utils.check_in_container(
                object_name, self.objects, self.container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            place2_success = object_utils.check_in_container(
                object_name, self.objects, self.container2_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)
            info['all_placed'] = info['all_placed'] and (place_success or place2_success)

        for object_name in self.object_names:
            which_container = bin_sort_hash(object_name) % 2
            target_container_position=self.container_position if which_container == 0 else self.container2_position

            curr_place = object_utils.check_in_container(
                object_name, self.objects, target_container_position,
                self.place_success_height_threshold * threshold_scale,
                self.place_success_radius_threshold * threshold_scale)

            info['place_success_target'] = info['place_success_target'] + int(curr_place)
        info['sort_success'] = info['place_success_target'] == len(self.object_names)

        if debug:
            print('place success target', info['place_success_target'])
            print('sort success', info['sort_success'])

        return info
    