import gym
import numpy as np

from dm_control.mujoco import engine
from .adept_envs.simulation.renderer import DMRenderer
from .constants import (BONUS_THRESH, CAMERAS, FRANKA_INIT_QPOS,
                        OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES)
from .mujoco.obs_utils import get_obs_ee, get_obs_forces


class KitchenCameraWrapper(gym.Wrapper):

    def __init__(self, env,
                rot_use_euler=False,
                with_obs_ee=True,
                with_obs_forces=True,
                camera_ids=[0],
                height=128,
                width=128,
                render_size=None):
        super().__init__(env)

        self.with_obs_ee = with_obs_ee
        self.with_obs_forces = with_obs_forces
        self.rot_use_euler = rot_use_euler

        self.render_size = render_size
        self.camera_ids = camera_ids
        self.height = height
        self.width = width

        self.create_renderer()
        self.add_cameras()

    def add_cameras(self, camera_id=None):
        if camera_id is not None:
            self.camera_ids.append(camera_id)
        self.cameras = dict()
        for camera_id in self.camera_ids:
            camera = engine.MovableCamera(self.env.sim,
                                          height=self.height,
                                          width=self.width)
            camera.set_pose(**CAMERAS[camera_id])
            self.cameras['camera_{}'.format(camera_id)] = camera
        self.cameras['camera_gripper'] = engine.Camera(
            self.env.sim,
            height=self.height,
            width=self.width,
            camera_id='gripper_camera_rgb')

    def create_renderer(self):
        self.renderer = DMRenderer(self.env.sim,
                                   camera_settings=dict(
                                       distance=2.9,
                                       lookat=[-0.05, 0.5, 2.0],
                                       azimuth=90,
                                       elevation=-50))
        if hasattr(self.env, 'solver_sim'):
            self.solver_sim_renderer = DMRenderer(self.env.solver_sim,
                                                  camera_settings=dict(
                                                      distance=2.9,
                                                      lookat=[-0.05, 0.5, 2.0],
                                                      azimuth=90,
                                                      elevation=-50))

    def render(
        self,
        mode='human',
        depth=False,
        segmentation=False,
    ):
        imgs = {}
        if 'rgb' in mode:
            # http://www.mujoco.org/book/APIreference.html#mjvOption
            # https://github.com/deepmind/dm_control/blob/9e0fe0f0f9713a2a993ca78776529011d6c5fbeb/dm_control/mujoco/engine.py#L200
            # mjtRndFlag(mjRND_SHADOW=0, mjRND_WIREFRAME=1, mjRND_REFLECTION=2, mjRND_ADDITIVE=3, mjRND_SKYBOX=4, mjRND_FOG=5, mjRND_HAZE=6, mjRND_SEGMENT=7, mjRND_IDCOLOR=8, mjNRNDFLAG=9)

            for camera_id, camera in self.cameras.items():
                img_rgb = camera.render(render_flag_overrides=dict(
                    skybox=False, fog=False, haze=False))
                imgs[camera_id + "_rgb"] = img_rgb
        
        if 'depth' in mode:
            for camera_id, camera in self.cameras.items():
                img_depth = camera.render(depth=True, segmentation=False)
                imgs[camera_id + "_depth"] = np.clip(img_depth, 0.0, 4.0)

        if 'human' in mode:
            self.renderer.render_to_window(
            )  # adept_envs.mujoco_env.MujocoEnv.render
            
        return imgs

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        
        obs_dict = dict()
        obs_dict['robot_qp'] = self.env.obs_dict['qp']
        rgb = self.render(mode='rgb')
        for k, v in rgb.items():
            obs_dict[k] = v
        if self.with_obs_ee:
            ee_qp = get_obs_ee(self.env.sim, self.rot_use_euler)
            obs_dict['ee_qp'] = ee_qp

        if self.with_obs_forces:
            ee_forces = get_obs_forces(self.env.sim)
            obs_dict['ee_forces'] = ee_forces

        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        obs_dict = dict()
        obs_dict['robot_qp'] = info['obs_dict']['qp']
        rgb = self.render(mode='rgb')
        for k, v in rgb.items():
            obs_dict[k] = v
        if self.with_obs_ee:
            ee_qp = get_obs_ee(self.env.sim, self.rot_use_euler)
            obs_dict['ee_qp'] = ee_qp

        if self.with_obs_forces:
            ee_forces = get_obs_forces(self.env.sim)
            obs_dict['ee_forces'] = ee_forces

        return obs_dict, reward, done, info
