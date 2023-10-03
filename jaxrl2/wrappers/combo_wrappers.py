import atexit
import functools
import sys
import threading
import traceback

import gym
import mujoco_py
# import d4rl
import benchmark
from benchmark.domains import adroit2
import metaworld
import numpy as np
from PIL import Image

import os 
#import roboverse


import threading
from dm_control.mujoco import engine

class MetaWorldEnv:

  def __init__(self, name="assembly-v2", action_repeat=2, size=(64, 64)):
      from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
      # render_params={"assembly-v2" : {"elevation": -22.5,
      #                                 "azimuth": 15,
      #                                 "distance": 0.5,
      #                                 "lookat": np.array([-0.15, 0.65, 0.25])}}
      render_params={"elevation": -22.5,
                     "azimuth": 15,
                     "distance": 0.75,
                     "lookat": np.array([-0.15, 0.65, 0.25])}

      self._env = ALL_V2_ENVIRONMENTS[name]()
      self._env.max_path_length = np.inf
      self._env._freeze_rand_vec = False
      self._env._partially_observable = False
      self._env._set_task_called = True

      self.hand_init_pose = self._env.hand_init_pos.copy()
      self.hand_init_pose = np.array([0.1 , 0.5, 0.30])

      self.action_repeat = action_repeat

      self.size = size
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
      # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, 0)

      # self.set_viewer_params(render_params[name])
      self.set_viewer_params(render_params)

      self.observation_space = self.get_observation_space()


  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  # @property
  # def observation_space(self):
  def get_observation_space(self):
        spaces = {}
        spaces['pixels'] = gym.spaces.Box(0, 255, (self.size[0], self.size[1], 3), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

  def set_viewer_params(self, params):
      self.viewer.cam.elevation = params["elevation"]
      self.viewer.cam.azimuth = params["azimuth"]
      self.viewer.cam.distance = params["distance"]
      self.viewer.cam.lookat[:] = params["lookat"][:]

  def step(self, action):
    reward = 0.0
    for _ in range(self.action_repeat):
        state, rew, done, info = self._env.step(action)
        reward += rew
        if done:
            break
    reward = 1.0 * info['success']
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    # obs = {'pixels':img, 'reward':reward}
    obs = {'pixels':img}
    return obs, reward, done, info

  def reset(self):
    self._env.hand_init_pos = self.hand_init_pose + 0.02 * np.random.normal(size = 3)
    _ = self._env.reset()
    for i in range(10):
        state,_,_,_ = self._env.step(np.zeros(self.action_space.shape))
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    # obs = {'image':img, 'reward':0.0}
    obs = {'pixels':img}
    return obs

  def render(self, mode, width = 128, height = 128):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(width, height, depth=False)
      # img = self._env.sim.render(width, height)
      img = img[::-1]
      return img


class AdroitHand:
    def __init__(self, env_name, img_width, img_height, proprio=False, camera_angle="camera2"):
        self._env_name = env_name
        self._env = gym.make(env_name).env
        self._img_width = img_width
        self._img_height = img_height
        self._proprio = proprio
        self._camera_angle = camera_angle

        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, 0)
        self.setup_viewer(self.viewer, camera_angle)

        # self.setup_viewer()

        self.observation_space = self.get_observation_space()

    def setup_viewer(self, viewer, camera_angle):
        #Setup camera in environment
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        # # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, 0)

        if camera_angle == "camera1":
            # Use this
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera2":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.4
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.3
        elif camera_angle == "camera3":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.3
        elif camera_angle == "camera4":
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 30
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera5":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 30
            viewer.cam.distance = 0.3
            viewer.cam.lookat[0] = -0.1
            viewer.cam.lookat[1] = -0.3
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera6":
            viewer.cam.elevation = -50
            viewer.cam.azimuth = 0
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.1
            viewer.cam.lookat[1] = -0.0
            viewer.cam.lookat[2] = 0.4
        else:
            raise ValueError(f"Unsupported camera angle: \"{camera_angle}\".")

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def render(self, *args, **kwargs):
        # image = self._env.sim.render(self._img_width, self._img_height)
        # image = np.flip(image, axis=0)
        # return image
        self.viewer.render(width=self._img_width, height=self._img_height)
        img = self.viewer.read_pixels(self._img_width, self._img_height, depth=False)
        img = img[::-1]
        return img

    def reset(self, *args, **kwargs):
        state = self._env.reset(*args, **kwargs)
        img = self.render()
        # obs = {'state':state, 'image':img}
        obs = {'pixels':img}

        if self._proprio:
            obs["states"] = self.get_proprio() # proprio is called 'states' in jaxrl2

        return obs

    def get_proprio(self):
        qpos = self._env.data.qpos.ravel()
        if "hammer" in self._env_name or "pen" in self._env_name or "relocate" in self._env_name:
            return qpos[:-6]
        elif "door" in self._env_name:
            return qpos[1:-2]
        else:
            raise NotImplementedError(f"Proprio not supported for \"{self._env_name}\" environment.")

    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        img = self.render()
        # obs = {'state':state, 'image':img}
        obs = {'pixels':img}

        if self._proprio:
            obs["states"] = self.get_proprio()

        return obs, reward, done, info

    # @property
    # def observation_space(self):
    def get_observation_space(self):
        spaces = {}
        # for key, value in self._env.observation_spec().items():
        #     spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        # spaces["state"] = self._env.observation_space
        spaces['pixels'] = gym.spaces.Box(0, 255, (self._img_width, self._img_height, 3), dtype=np.uint8)

        if self._proprio:
            # spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, self.get_proprio().shape, dtype=np.float32)
            spaces["states"] = gym.spaces.Box(-np.inf, np.inf, self.get_proprio().shape, dtype=np.float32)

        return gym.spaces.Dict(spaces)


class AdroitHandMultiview:
    def __init__(self, env_name, img_width, img_height, proprio=False, camera_angles=["camera2"]):
        self._env_name = env_name
        self._env = gym.make(env_name).env
        self._img_width = img_width
        self._img_height = img_height
        self._proprio = proprio
        self._camera_angles = camera_angles

        # self.viewers = {}
        # for camera_angle in self._camera_angles:
        #     self.viewers[camera_angle] = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        #     self.setup_viewer(self.viewers[camera_angle], camera_angle)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

        self.observation_space = self.get_observation_space()

    def setup_viewer(self, viewer, camera_angle):
        #Setup camera in environment
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        # # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, 0)

        if camera_angle == "camera1":
            # Use this
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera2":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.4
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.3
        elif camera_angle == "camera3":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 20
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.3
        elif camera_angle == "camera4":
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 30
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.2
            viewer.cam.lookat[1] = -0.2
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera5":
            viewer.cam.elevation = -40
            viewer.cam.azimuth = 30
            viewer.cam.distance = 0.3
            viewer.cam.lookat[0] = -0.1
            viewer.cam.lookat[1] = -0.3
            viewer.cam.lookat[2] = 0.4
        elif camera_angle == "camera6":
            viewer.cam.elevation = -50
            viewer.cam.azimuth = 0
            viewer.cam.distance = 0.5
            viewer.cam.lookat[0] = -0.1
            viewer.cam.lookat[1] = -0.0
            viewer.cam.lookat[2] = 0.4
        else:
            raise ValueError(f"Unsupported camera angle: \"{camera_angle}\".")

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def render(self, *args, **kwargs):
        # image = self._env.sim.render(self._img_width, self._img_height)
        # image = np.flip(image, axis=0)
        # return image
        viewer = self.viewers[self._camera_angles[0]]
        viewer.render(width=self._img_width, height=self._img_height)
        img = viewer.read_pixels(self._img_width, self._img_height, depth=False)
        img = img[::-1]
        return img

    def render_all_cameras(self):
        images = []
        # for camera_angle in self._camera_angles:
        #     viewer = self.viewers[camera_angle]
        #     viewer.render(width=self._img_width, height=self._img_height)
        #     img = viewer.read_pixels(self._img_width, self._img_height, depth=False)
        #     img = img[::-1]
        #     images.append(img)
        for camera_angle in self._camera_angles:
            self.setup_viewer(self.viewer, camera_angle)
            self.viewer.render(width=self._img_width, height=self._img_height)
            img = self.viewer.read_pixels(self._img_width, self._img_height, depth=False)
            img = img[::-1]
            images.append(img)

        images =  np.concatenate(images, axis=-1)
        return images

    def reset(self, *args, **kwargs):
        state = self._env.reset(*args, **kwargs)
        img = self.render_all_cameras()
        # obs = {'state':state, 'image':img}
        obs = {'pixels':img}

        if self._proprio:
            obs["states"] = self.get_proprio() # proprio is called 'states' in jaxrl2

        return obs

    def get_proprio(self):
        qpos = self._env.data.qpos.ravel()
        if "hammer" in self._env_name or "pen" in self._env_name or "relocate" in self._env_name:
            return qpos[:-6]
        elif "door" in self._env_name:
            return qpos[1:-2]
        else:
            raise NotImplementedError(f"Proprio not supported for \"{self._env_name}\" environment.")

    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        img = self.render_all_cameras()
        # obs = {'state':state, 'image':img}
        obs = {'pixels':img}

        if self._proprio:
            obs["states"] = self.get_proprio()

        return obs, reward, done, info

    # @property
    # def observation_space(self):
    def get_observation_space(self):
        spaces = {}
        # for key, value in self._env.observation_spec().items():
        #     spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        # spaces["state"] = self._env.observation_space
        spaces['pixels'] = gym.spaces.Box(0, 255, (self._img_width, self._img_height, 3 * len(self._camera_angles)), dtype=np.uint8)

        if self._proprio:
            # spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, self.get_proprio().shape, dtype=np.float32)
            spaces["states"] = gym.spaces.Box(-np.inf, np.inf, self.get_proprio().shape, dtype=np.float32)

        return gym.spaces.Dict(spaces)


OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }

OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }

BONUS_THRESH = 0.3

CAMERAS = {
    0: dict(distance=2.1, lookat=[-0.4, 0.5, 2.0], azimuth=70,
            elevation=-37.5),
    1: dict(distance=2.2,
            lookat=[-0.2, 0.75, 2.0],
            azimuth=150,
            elevation=-30.0),
    2: dict(distance=4.5, azimuth=-66, elevation=-65),
    3: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),  # original, as in https://relay-policy-learning.github.io/
    4: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70,
            elevation=-50),  # angled up to get a more top-down view
    5: dict(distance=2.65, lookat=[0, 0, 2.0], azimuth=90, elevation=-60
            ),  # similar to appendix D of https://arxiv.org/pdf/1910.11956.pdf
    6: dict(distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-60
            ),  # 3-6 are first person views at different angles and distances
    7: dict(
        distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-45
    ),  # problem w/ POV is that the knobs can be hidden by the hinge drawer and arm
    8: dict(distance=2.9, lookat=[-0.05, 0.5, 2.0], azimuth=90, elevation=-50),
    9: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90,
            elevation=-50),  # move back so less of cabinets
    10: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-35),
    11: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-10),

    12: dict(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60) # LEXA view
}

class Kitchen:
    def __init__(self, task=['microwave'], size=(64, 64), proprio=True, log_only_target_tasks=False):
        # export RELAY_POLICY_REPO="/iris/u/khatch/vd5rl/finetuning_benchmark/benchmark/domains/relay-policy-learning/adept_envs"

        RELAY_POLICY_PATH = os.environ.get('RELAY_POLICY_REPO', None)
        print("RELAY_POLICY_PATH:", RELAY_POLICY_PATH)
        sys.path.append(RELAY_POLICY_PATH)

        import adept_envs
        self._env = gym.make('kitchen_relax_rpl-v1')
        self._task = task
        print("env._task:", self._task)
        self._img_h = size[0]
        self._img_w = size[1]
        self._proprio = proprio
        self.tasks_to_complete = ['bottom burner',
                                  'top burner',
                                  'light switch',
                                  'slide cabinet',
                                  'hinge cabinet',
                                  'microwave',
                                  'kettle']
        self._log_only_target_tasks = log_only_target_tasks

        self.observation_space = self.get_observation_space()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)
        img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        obs_dict = dict(pixels=img)

        if self._proprio:
            obs_dict["states"] = obs[:9]

        reward = sum([reward_dict[obj] for obj in self._task])

        info.update({"reward " + key:float(val) for key, val in reward_dict.items()})
        # info.update({"reward " + key:1 if ("kettle" in key or "burner" in key) else float(val) for key, val in reward_dict.items()})


        # obs_dict.update({"reward " + key:float(val) for key, val in reward_dict.items()})
        return obs_dict, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)

        img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        obs_dict = dict(pixels=img)

        if self._proprio:
            obs_dict["states"] = obs[:9]

        return obs_dict

    def _compute_reward_dict(self, obs):
        reward_dict = {}
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            obs_obj = obs[..., element_idx]
            obs_goal = OBS_ELEMENT_GOALS[element]
            distance = np.linalg.norm(obs_obj - obs_goal)
            complete = distance < BONUS_THRESH
            reward_dict[element] = complete

            obs_dict = self.obs_dict

        if self._log_only_target_tasks:
            reward_dict = {key:reward_dict[key] for key in self._task}

        return reward_dict

    def render(self, mode='human', size=(1920, 2550)):
        if mode =='rgb_array':
            # camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera = engine.MovableCamera(self._env.sim, size[0], size[1])
            # camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35) # original?
            camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60) # Lexa?
            img = camera.render()
            return img
        else:
            # super(KitchenTaskRelaxV1, self).render()
            self._env.render(mode, size)

    # @property
    # def observation_space(self):
    def get_observation_space(self):
        spaces = {}
        spaces['pixels'] = gym.spaces.Box(0, 255, (self._img_h, self._img_w, 3), dtype=np.uint8)

        if self._proprio:
            spaces["states"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

        return gym.spaces.Dict(spaces)


class KitchenMultipleViews(Kitchen):
    def __init__(self, *args, camera_ids=[0, 1], use_wrist_cam=True, **kwargs):
        self.camera_ids = camera_ids
        self._use_wrist_cam = use_wrist_cam
        super().__init__(*args, **kwargs)
        self.add_cameras()
        # self.render = self.render2


    def add_cameras(self, camera_id=None):
        if camera_id is not None:
            self.camera_ids.append(camera_id)
        self.cameras = dict()
        for camera_id in self.camera_ids:
            camera = engine.MovableCamera(self.sim,
                                          height=self._img_h,
                                          width=self._img_w)
            camera.set_pose(**CAMERAS[camera_id])
            self.cameras['camera_{}'.format(camera_id)] = camera

        if self._use_wrist_cam:
            self.cameras['camera_gripper'] = engine.Camera(
                self.sim,
                height=self._img_h,
                width=self._img_w,
                camera_id='gripper_camera_rgb')

    def get_observation_space(self):
        spaces = {}

        num_cams = len(self.camera_ids)
        if self._use_wrist_cam:
            num_cams += 1

        spaces['pixels'] = gym.spaces.Box(0, 255, (self._img_h, self._img_w, 3 * num_cams), dtype=np.uint8)
        # spaces['pixels'] = gym.spaces.Box(0, 255, (self._img_h, self._img_w, 3, dtype=np.uint8)

        if self._proprio:
            spaces["states"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

        return gym.spaces.Dict(spaces)

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)

        # img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        imgs = self.render_extra_views("rgb")
        # keys = sorted(list(imgs.keys()))
        # img = np.concatenate([imgs[key] for key in keys], axis=2)
        img = np.concatenate([imgs[key] for key in sorted(list(imgs.keys()))], axis=-1)
        obs_dict = dict(pixels=img)

        if self._proprio:
            obs_dict["states"] = obs[:9]

        return obs_dict

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)
        # img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        imgs = self.render_extra_views("rgb")
        # keys = sorted(list(imgs.keys()))
        img = np.concatenate([imgs[key] for key in sorted(list(imgs.keys()))], axis=-1)
        obs_dict = dict(pixels=img)

        if self._proprio:
            obs_dict["states"] = obs[:9]

        reward = sum([reward_dict[obj] for obj in self._task])

        info.update({"reward " + key:float(val) for key, val in reward_dict.items()})
        # info.update({"reward " + key:1 if ("kettle" in key or "burner" in key) else float(val) for key, val in reward_dict.items()})


        # obs_dict.update({"reward " + key:float(val) for key, val in reward_dict.items()})
        return obs_dict, reward, done, info


    def render_extra_views(self, mode='rgb', depth=False, segmentation=False):
#         print("In render 2")
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
            self.renderer.render_to_window()  # adept_envs.mujoco_env.MujocoEnv.render

        return imgs

class Gym:
    def __init__(self, name, config, size=(64, 64)):
        self._env = gym.make(name)
        self.size = size
        self.use_transform = config.use_transform
        self.pad = int(config.pad/2)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
        if self.use_transform:
            img = img[self.pad:-self.pad, self.pad:-self.pad, :]
        obs = {'state':state, 'image':img}
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
        if self.use_transform:
            img = img[self.pad:-self.pad, self.pad:-self.pad, :]
        obs = {'state':state, 'image':img}
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])


class DeepMindControl:
    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class Collect:
    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

        # self._ep_idx = 1 ###$$$###

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))

        # transition['ep_idx'] = self._ep_idx ###$$$###
        self._episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode)
                # self._ep_idx += 1 ###$$$###
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self._env.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        # transition['ep_idx'] = self._ep_idx ###$$$###



        base_env = self._env._env._env._env
        if hasattr(base_env, "tasks_to_complete"):
            for task in base_env.tasks_to_complete:
                transition['reward ' + task] = 0.

        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(np.isfinite(env.action_space.low),
                                    np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:
    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs

class Async:
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, ctor, strategy='process'):
        self._strategy = strategy
        if strategy == 'none':
            self._env = ctor()
        elif strategy == 'thread':
            import multiprocessing.dummy as mp
        elif strategy == 'process':
            import multiprocessing as mp
        else:
            raise NotImplementedError(strategy)
        if strategy != 'none':
            self._conn, conn = mp.Pipe()
            self._process = mp.Process(target=self._worker, args=(ctor, conn))
            atexit.register(self.close)
            self._process.start()
        self._obs_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._obs_space:
            self._obs_space = self.__getattr__('observation_space')
        return self._obs_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        if self._strategy == 'none':
            return getattr(self._env, name)
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        blocking = kwargs.pop('blocking', True)
        if self._strategy == 'none':
            return functools.partial(getattr(self._env, name), *args, **kwargs)
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        promise = self._receive
        return promise() if blocking else promise

    def close(self):
        if self._strategy == 'none':
            try:
                self._env.close()
            except AttributeError:
                pass
            return
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        return self.call('step', action, blocking=blocking)


    def reset(self, blocking=True):
        return self.call('reset', blocking=blocking)

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except ConnectionResetError:
            raise RuntimeError('Environment worker crashed.')
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError(f'Received message of unexpected type {message}')

    def _worker(self, ctor, conn):
        try:
            env = ctor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError(f'Received message of unknown type {message}')
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print(f'Error in environment process: {stacktrace}')
            conn.send((self._EXCEPTION, stacktrace))
        conn.close()
