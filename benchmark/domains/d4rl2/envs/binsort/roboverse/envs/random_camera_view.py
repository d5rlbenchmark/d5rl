from roboverse.envs.widow250 import Widow250Env
from roboverse.bullet import object_utils
import roboverse.bullet as bullet
from roboverse.envs import objects
# from .multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.envs.multi_object import MultiObjectEnv, MultiObjectMultiContainerEnv
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
from roboverse.envs.widow250_pickplace import Widow250PickPlaceEnv
import numpy as np

class Widow250MultViewPickPlaceEnv(Widow250PickPlaceEnv):
    def __init__(self, noise_pos=0.025, noise_angle=30, num_traj=20, seed=42, only_yaw=True, *args, **kwargs) -> None:
        np.random.seed(seed)
        self.noise_pos = noise_pos
        self.noise_angle = noise_angle 
        
        self.num_traj = num_traj
        self.traj_idx = 0
        self.only_yaw = only_yaw

        super().__init__(*args, **kwargs)
    
    def change_view(self):
        def gen(scale):
            return np.random.uniform(low=-scale, high=scale)
        
        self.camera_target_pos=(
            0.6 + gen(self.noise_pos),
            0.2 + gen(self.noise_pos), 
            -0.28 + gen(self.noise_pos)
        )

        self.camera_distance=0.29 + gen(self.noise_pos)
        self.camera_roll=0.0 if self.only_yaw else 0.0 + gen(self.noise_angle) 
        self.camera_pitch=-40 if self.only_yaw else -40 + gen(self.noise_angle)
        self.camera_yaw=180 + gen(self.noise_angle)

        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance,
                                yaw=self.camera_yaw,
                                pitch=self.camera_pitch,
                                roll=self.camera_roll,
                                up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_dim, self.observation_img_dim)

        print('Changed Camera View')

    def reset(self):
        self.traj_idx += 1
        
        if self.traj_idx % self.num_traj == 0:
            self.change_view()
        
        return super().reset()