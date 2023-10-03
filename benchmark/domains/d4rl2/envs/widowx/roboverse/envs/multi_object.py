import numpy as np

from d4rl2.envs.widowx.roboverse.assets.shapenet_object_lists import (
    CONTAINER_CONFIGS, OBJECT_ORIENTATIONS, OBJECT_SCALINGS, TRAIN_CONTAINERS,
    TRAIN_OBJECTS)


class MultiObjectEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """

    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.task_idx = None
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):
        if len(self.possible_objects) == self.num_objects:
            chosen_obj_idx = np.asarray(range(self.num_objects))
        else:
            chosen_obj_idx = np.random.randint(0,
                                               len(self.possible_objects),
                                               size=self.num_objects)
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[
                object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        if self.task_idx is None:
            task_idx = np.random.randint(self.num_objects)
        else:
            task_idx = self.task_idx
        self.target_object = self.object_names[task_idx]
        return super().reset()

    def reset_task(self, task_idx):
        self.task_idx = task_idx

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks, ))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector
        return observation


class MultiObjectV2Env:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """

    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 init_task_idx=None,
                 num_tasks=32,
                 fixed_task=False,
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.task_idx = init_task_idx
        self.num_tasks = num_tasks
        self.fixed_task = fixed_task
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):
        if self.task_idx is None:
            # randomly sample container and objects from given set
            chosen_obj_idx = np.random.randint(0,
                                               len(self.possible_objects),
                                               size=self.num_objects)
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[0]
        else:
            assert len(self.possible_objects[0]) == 2
            assert self.task_idx < 4 * self.num_tasks
            if self.task_idx < 2 * self.num_tasks:
                chosen_obj_idx = int((self.task_idx % self.num_tasks) / 2)
                target_obj_idx = (self.task_idx % self.num_tasks) % 2
            else:
                chosen_obj_idx = int(
                    ((self.task_idx % self.num_tasks) + self.num_tasks) / 2)
                target_obj_idx = (self.task_idx % self.num_tasks) % 2
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[target_obj_idx]

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[
                object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]

        return super().reset()

    def reset_task(self, task_idx):
        if not self.fixed_task:
            self.task_idx = task_idx
            target_obj_idx = (self.task_idx % self.num_tasks) % 2
            self.target_object = self.object_names[target_obj_idx]

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks, ))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector

        return observation

    def save_state(self, path):
        self._saved_task_idx = self.task_idx
        super().save_state(path)

    def restore_state(self, path):
        self.reset_task(self._saved_task_idx)
        super().restore_state(path)


class MultiObjectMultiContainerEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """

    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 possible_containers=TRAIN_CONTAINERS[:3],
                 init_task_idx=None,
                 num_tasks=32,
                 fixed_task=False,
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.possible_containers = np.asarray(possible_containers)
        self.task_idx = init_task_idx
        self.num_tasks = num_tasks
        self.fixed_task = fixed_task
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):

        if self.task_idx is None:
            # randomly sample container and objects from given set
            chosen_container_idx = np.random.randint(
                0, len(self.possible_containers))
            self.container_name = self.possible_containers[
                chosen_container_idx]
            chosen_obj_idx = np.random.randint(0,
                                               len(self.possible_objects),
                                               size=self.num_objects)
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[0]
        else:
            assert len(self.possible_objects[0]) == 2
            chosen_container_idx = int(self.task_idx / 2)
            chosen_obj_idx = int(self.task_idx / 2)
            target_obj_idx = self.task_idx % 2
            self.container_name = self.possible_containers[
                chosen_container_idx]
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[target_obj_idx]

        container_config = CONTAINER_CONFIGS[self.container_name]
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

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[
                object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]

        return super().reset()

    def reset_task(self, task_idx):
        if not self.fixed_task:
            assert task_idx < self.num_tasks
            self.task_idx = task_idx

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks, ))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector
        return observation


class MultiObjectMultiContainerV2Env:
    """
    Generalization env.
    Has 16 tasks. 8 scenes. Each scene corresponds to two objects and one container.
    """

    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 possible_containers=TRAIN_CONTAINERS[:3],
                 init_task_idx=None,
                 num_tasks=16,
                 fixed_task=False,
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.possible_containers = np.asarray(possible_containers)
        self.task_idx = init_task_idx
        self.num_tasks = num_tasks
        assert self.num_tasks == 16
        self.fixed_task = fixed_task
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):
        if self.task_idx is None:
            # randomly sample container and objects from given set
            chosen_container_idx = np.random.randint(
                0, len(self.possible_containers))
            self.container_name = self.possible_containers[
                chosen_container_idx]
            chosen_obj_idx = np.random.randint(0,
                                               len(self.possible_objects),
                                               size=self.num_objects)
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[0]
        else:
            assert len(self.possible_objects[0]) == 2
            assert self.task_idx < 4 * self.num_tasks
            if self.task_idx < 2 * self.num_tasks:
                chosen_obj_idx = int((self.task_idx % self.num_tasks) / 2)
                target_obj_idx = (self.task_idx % self.num_tasks) % 2
                chosen_container_idx = int(self.task_idx / 2)
            else:
                chosen_obj_idx = int(
                    ((self.task_idx % self.num_tasks) + self.num_tasks) / 2)
                target_obj_idx = (self.task_idx % self.num_tasks) % 2
                chosen_container_idx = int(
                    ((self.task_idx % self.num_tasks) + self.num_tasks) / 2)
            self.object_names = tuple(self.possible_objects[chosen_obj_idx])
            self.target_object = self.object_names[target_obj_idx]
            self.container_name = self.possible_containers[
                chosen_container_idx]

        container_config = CONTAINER_CONFIGS[self.container_name]
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

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[
                object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]

        return super().reset()

    def reset_task(self, task_idx):
        if not self.fixed_task:
            self.task_idx = task_idx
            target_obj_idx = (self.task_idx % self.num_tasks) % 2
            self.target_object = self.object_names[target_obj_idx]

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks, ))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector

        return observation

    def save_state(self, path):
        self._saved_task_idx = self.task_idx
        super().save_state(path)

    def restore_state(self, path):
        self.reset_task(self._saved_task_idx)
        super().restore_state(path)
