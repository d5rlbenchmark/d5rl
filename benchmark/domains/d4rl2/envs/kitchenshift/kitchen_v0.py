# Modifications, Copyright 2021 KitchenShift
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import gym
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco import engine
from dm_control.utils import inverse_kinematics
from gym import spaces
from gym.utils import seeding

from .adept_envs.franka_robot import Robot
from .adept_envs.simulation.renderer import DMRenderer
from .adept_envs.simulation.sim_robot import _patch_mjlib_accessors
from .constants import (CAMERAS, FRANKA_INIT_QPOS, OBS_ELEMENT_GOALS,
                        OBS_ELEMENT_INDICES)
from .mujoco.mocap_utils import reset_mocap2body_xpos, reset_mocap_welds
from .mujoco.obs_utils import get_obs_ee, get_obs_forces
from .mujoco.rotations import (euler2quat, mat2euler, mat2quat, quat2euler,
                               quat_mul)


class Kitchen_v0(gym.Env):
    """Kitchen manipulation environment in Mujoco. Ported from relay-policy-learning/adept_envs."""

    def __init__(
        self,
        ctrl_mode='absvel',
        compensate_gravity=False,
        frame_skip=40,
        camera_id=6,
        with_obs_ee=False,
        with_obs_forces=False,
        rot_use_euler=False,
    ):
        self.ctrl_mode = ctrl_mode
        self.frame_skip = frame_skip

        # see https://github.com/ARISE-Initiative/robosuite/blob/e0982ca9000fd373bc60781ec9acd1ef29de5beb/robosuite/models/grippers/gripper_tester.py#L195
        # https://github.com/deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/locomotion/soccer/boxhead.py#L36
        # http://www.mujoco.org/forum/index.php?threads/gravitational-matrix-calculation.3404/
        # https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjpid.pyx#L243
        self.compensate_gravity = compensate_gravity

        self.with_obs_ee = with_obs_ee
        self.with_obs_forces = with_obs_forces
        self.rot_use_euler = rot_use_euler  # affects format of with_obs_ee

        self.N_DOF_ROBOT = 9
        self.N_DOF_OBJECT = 21

        # NOTE: this is actually getting multiplied by values in franka_config.xml, so it's 10% of the actual noise specified,
        # see https://github.com/google-research/relay-policy-learning/blob/cd70ac9334f584f86db281a1ffd3e5cbc3e5e293/adept_envs/adept_envs/franka/robot/franka_robot.py#L155
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs

        # self.model_dir = os.path.join(os.path.dirname(__file__), 'assets_old/')
        # self.model_path = os.path.join(self.model_dir, 'franka_kitchen_jntpos_act_ab.xml')
        self.model_dir = os.path.join(os.path.dirname(__file__), 'assets/')
        self.model_path = os.path.join(self.model_dir, 'kitchen.xml')
        self.model_xml = open(self.model_path, 'r').read()

        # mujoco.Physics.from_xml_string messes up asset paths
        # mjcf.from_xml_string doesn't seem to support the same xml parsing as the actual mjlib
        # to circumvent these issues, in order to dynamically change the env and reload the xml,
        # we write the xml string to a temporary xml file located in self.model_dir
        #
        # self.sim = mujoco.Physics.from_xml_string(self.model_xml)
        # self.sim = mjcf.from_xml_string(model_xml, model_dir=self.model_dir)
        # self.sim = mujoco.Physics.from_xml_path(self.model_path)
        # _patch_mjlib_accessors(self.model, self.sim.data, True)
        # print(self.model_xml)
        self.load_sim(self.model_xml)
        self.set_camera_id(camera_id)

        # from adept_env
        self.seed()
        # self.initializing = True
        self.robot = Robot(
            self.N_DOF_ROBOT,
            self.N_DOF_OBJECT,
            calibration_path=os.path.join(os.path.dirname(__file__),
                                          'adept_envs/franka_config.xml'),
        )
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        # # initializing and step from adept_envs.RobotEnv and adept_envs.MujocoEnv
        # observation, _reward, done, _info = self.step(np.zeros(9))
        self.initializing = False
        self.set_init_qpos(FRANKA_INIT_QPOS.copy())
        # self.init_qvel = self.sim.model.key_qvel[0].copy()  # this should be np.zeros(29)

        if self.ctrl_mode == 'absvel':
            action_dim = 9
        elif self.ctrl_mode == 'mocapik':
            # with mocapik, we follow robogym and robosuite by spawning a separate simulator
            self.mocapid = None  # set later since sim is not yet initialized
            self.initial_mocap_quat = np.array(
                [-0.65269804, 0.65364932, 0.27044485, 0.27127002])
            self.fix_gripper_quat = False

            self.binary_gripper = False

            pos_action_dim = 3
            rot_action_dim = 3 if self.rot_use_euler else 4
            gripper_action_dim = 1 if self.binary_gripper else 2
            action_dim = pos_action_dim + rot_action_dim + gripper_action_dim

            self.pos_range = 0.075  # allow larger change
            self.rot_range = 0.075
            self._create_solver_sim()
            # TODO: if using worldgen, then need to propogate changes to solver_sim
        else:
            raise ValueError

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim, ))
        self.act_mid = np.zeros(action_dim)
        self.act_amp = 2.0 * np.ones(action_dim)

        obs_space = {
            k: spaces.Box(low=-np.inf,
                          high=np.inf,
                          shape=v.shape,
                          dtype=np.float32)
            for k, v in self._get_obs_dict().items()
        }
        self.observation_space = spaces.Dict(obs_space)

    def load_sim(self, xml_string):
        with tempfile.NamedTemporaryFile(mode='w+', dir=self.model_dir) as f:
            f.write(xml_string)
            f.flush()
            self.sim = mujoco.Physics.from_xml_path(f.name)

        _patch_mjlib_accessors(self.model, self.sim.data, True)

    def set_camera_id(self, camera_id):
        self.camera_id = camera_id
        self.renderer = DMRenderer(self.sim,
                                   camera_settings=CAMERAS[self.camera_id])
        # self.solver_sim_renderer = DMRenderer(
        #                 self.solver_sim, camera_settings=CAMERAS[self.camera_id])

    def set_init_qpos(self, qpos):
        self.init_qpos = qpos

    @property
    def skip(self):
        """Alias for frame_skip. Needed for MJRL."""
        return self.frame_skip

    @property
    def data(self):
        return self.sim.data

    @property
    def model(self):
        return self.sim.model

    @property
    def physics(self):
        return self.sim

    def _get_obs_dict(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        obs_dict = {
            'robot_qp': qp,
            'robot_qv': qv,
            'obj_qp': obj_qp,
            'obj_qv': obj_qv,
        }
        if self.with_obs_ee:
            obs_dict['ee_qp'] = get_obs_ee(self.sim, self.rot_use_euler)
        if self.with_obs_forces:
            obs_dict['ee_forces'] = get_obs_forces(self.sim)

        # cast to float32
        for k, v in obs_dict.items():
            obs_dict[k] = v.astype(np.float32)
        return obs_dict

    def _create_solver_sim(self):
        model_path = os.path.join(os.path.dirname(__file__),
                                  'assets/kitchen_teleop_solver.xml')
        self.solver_sim = mujoco.Physics.from_xml_path(model_path)
        _patch_mjlib_accessors(self.solver_sim.model, self.solver_sim.data,
                               True)

    # from adept_envs.mujoco_env.MujocoEnv
    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]

        for _ in range(n_frames):
            self.sim.step()

    def step(self, action):
        if self.ctrl_mode == 'absvel':
            self._step_absvel(action)
        elif self.ctrl_mode == 'mocapik':
            self._step_mocapik(action)
        else:
            raise RuntimeError(f"Unsupported ctrl_mode: {self.ctrl_mode}")

        obs = self._get_obs_dict()
        done = False
        reward = 0.0
        env_info = {}

        return obs, done, reward, env_info

    def _step_absvel(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self,
                        a,
                        step_duration=self.frame_skip *
                        self.model.opt.timestep,
                        mode='velact')

    def _step_mocapik(self, a):
        a = np.clip(a, -1.0, 1.0)

        self.solver_sim.data.qpos[:] = self.sim.data.qpos[:].copy()
        self.solver_sim.data.qvel[:] = self.sim.data.qvel[:].copy()

        # # track object state only
        # self.solver_sim.data.qpos[-self.N_DOF_OBJECT:] = self.sim.data.qpos[-self.N_DOF_OBJECT:].copy()
        # self.solver_sim.data.qvel[-self.N_DOF_OBJECT:] = self.sim.data.qvel[-self.N_DOF_OBJECT:].copy()

        self.solver_sim.forward()
        # self.solver_sim.step()
        reset_mocap2body_xpos(self.solver_sim)

        if self.mocapid is None:
            self.mocapid = self.solver_sim.model.body_mocapid[
                self.solver_sim.model.body_name2id('vive_controller')]

        # split action [3-dim Cartesian coordinate, 3-dim euler angle OR 4-dim quarternion, 2-dim gripper joints]
        current_pos = self.solver_sim.data.mocap_pos[self.mocapid, ...].copy()
        new_pos = current_pos + a[:3] * self.pos_range
        self.solver_sim.data.mocap_pos[self.mocapid, ...] = new_pos.copy()

        if self.rot_use_euler:
            rot_a = a[3:6] * self.rot_range
            gripper_a = np.sign(a[6]) if self.binary_gripper else a[6:8]
        else:
            rot_a = quat2euler(a[3:7]) * self.rot_range
            gripper_a = np.sign(a[7]) if self.binary_gripper else a[7:9]

        if self.fix_gripper_quat:
            # fixed to initial
            self.solver_sim.data.mocap_quat[self.mocapid,
                                            ...] = self.initial_mocap_quat
        else:
            current_quat = self.solver_sim.data.mocap_quat[self.mocapid,
                                                           ...].copy()
            new_quat = euler2quat(quat2euler(current_quat) + rot_a)
            self.solver_sim.data.mocap_quat[self.mocapid,
                                            ...] = new_quat.copy()

        # mocap do_simulation w/ solver_sim
        self.solver_sim.data.ctrl[:2] = gripper_a.copy()
        step_duration = self.skip * self.model.opt.timestep
        n_frames = int(step_duration / self.solver_sim.model.opt.timestep)

        with self.solver_sim.model.disable('gravity'):
            for _ in range(n_frames):
                self.solver_sim.step()

        # self.solver_sim_renderer.render_to_window()

        # get robot qpos from solver_sim and swap in gripper_a
        ja = self.solver_sim.data.qpos[:self.N_DOF_ROBOT].copy()
        ja[7:9] = gripper_a

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self,
                        ja,
                        step_duration=self.skip * self.model.opt.timestep,
                        mode='posact')

    def reset(self, objects_done_set=None):
        self.sim.reset()
        self.sim.forward()
        if objects_done_set is not None:
            reset_qpos = self.init_qpos[:].copy()
            for element in objects_done_set:
                reset_qpos[OBS_ELEMENT_INDICES[element]] = OBS_ELEMENT_GOALS[
                    element][:].copy()
            obs = self.reset_model(reset_qpos=reset_qpos)
        else:
            obs = self.reset_model()

        obs = self._get_obs_dict()
        return obs

    def reset_model(self, reset_qpos=None):
        if reset_qpos is None:
            # NOTE: if obj penetration happens, ie. arm going thru hinge, this will NOT resolve it
            # and if sim.step is not called, then obj changes will reset in next call to env.step()
            reset_qpos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_qpos, reset_vel)

        reset_mocap_welds(self.sim)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()
        # self.robot._observation_cache_refresh(self)

        if self.ctrl_mode == 'mocapik':
            self.solver_sim.data.qpos[:] = self.sim.data.qpos[:].copy()
            self.solver_sim.data.qvel[:] = self.sim.data.qvel[:].copy()
            reset_mocap_welds(self.solver_sim)
            reset_mocap2body_xpos(self.solver_sim)
            self.solver_sim.forward()

    def render(
        self,
        mode='human',
        width=480,  # DEFAULT_RENDER_SIZE = 480
        height=480,
        camera_id=None,
        depth=False,
        segmentation=False,
    ):
        if mode == 'rgb_array':
            if camera_id is None:
                camera_id = self.camera_id

            # TODO: cache camera? doesn't seem to affect performance that much
            # also use camera._scene.free()? though it will slow things down
            if camera_id is None:
                camera_id = self.camera_id
            camera = engine.MovableCamera(self.sim, height=height, width=width)
            camera.set_pose(**CAMERAS[camera_id])

            # http://www.mujoco.org/book/APIreference.html#mjvOption
            # https://github.com/deepmind/dm_control/blob/9e0fe0f0f9713a2a993ca78776529011d6c5fbeb/dm_control/mujoco/engine.py#L200
            # mjtRndFlag(mjRND_SHADOW=0, mjRND_WIREFRAME=1, mjRND_REFLECTION=2, mjRND_ADDITIVE=3, mjRND_SKYBOX=4, mjRND_FOG=5, mjRND_HAZE=6, mjRND_SEGMENT=7, mjRND_IDCOLOR=8, mjNRNDFLAG=9)
            if not (depth or segmentation):  # RGB
                img = camera.render(render_flag_overrides=dict(
                    skybox=False, fog=False, haze=False))
            else:
                img = camera.render(depth=depth, segmentation=segmentation)
            return img
        elif mode == 'human':
            self.renderer.render_to_window(
            )  # adept_envs.mujoco_env.MujocoEnv.render
        else:
            raise NotImplementedError(mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
