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

from .adept_envs.simulation.renderer import DMRenderer
from .adept_envs.simulation.sim_robot import _patch_mjlib_accessors
from .constants import (BONUS_THRESH, CAMERAS, FRANKA_INIT_QPOS,
                        OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES)
from .mujoco.mocap_utils import reset_mocap2body_xpos, reset_mocap_welds
from .mujoco.obs_utils import get_obs_ee, get_obs_forces
from .mujoco.robot import Robot
from .mujoco.rotations import (euler2quat, mat2euler, mat2quat, quat2euler,
                               quat_mul)
from .utils import make_rng


class KitchenBase(gym.Env):
    """Kitchen manipulation environment in Mujoco. Ported from relay-policy-learning/adept_envs."""

    TASK_ELEMENTS = ['microwave', 'kettle', 'bottomknob', 'switch']
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True

    def __init__(
        self,
        ctrl_mode='absvel',
        compensate_gravity=True,
        noslip_off=False,
        frame_skip=40,
        camera_ids=[0, 1],
        height=128,
        width=128,
        with_obs_ee=True,
        with_obs_forces=True,
        robot='franka2',
        rot_use_euler=False,
        render_size=None,
        noise_ratio=0.1,
        robot_cache_noise_ratio=0.0,
        object_pos_noise_amp=0.1,
        object_vel_noise_amp=0.1,
        robot_obs_extra_noise_amp=0.1,
        init_random_steps_set=None,
        init_perturb_robot_ratio=None,
        init_perturb_object_ratio=None,
        rng_type='legacy',
    ):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        self.goal = self._get_task_goal()

        self.ctrl_mode = ctrl_mode
        self.frame_skip = frame_skip

        # see https://github.com/ARISE-Initiative/robosuite/blob/e0982ca9000fd373bc60781ec9acd1ef29de5beb/robosuite/models/grippers/gripper_tester.py#L195
        # https://github.com/deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/locomotion/soccer/boxhead.py#L36
        # http://www.mujoco.org/forum/index.php?threads/gravitational-matrix-calculation.3404/
        # https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjpid.pyx#L243
        self.compensate_gravity = compensate_gravity
        self.noslip_off = noslip_off

        self.with_obs_ee = with_obs_ee
        self.with_obs_forces = with_obs_forces
        self.rot_use_euler = rot_use_euler  # affects format of with_obs_ee

        self.robot_name = robot
        self.noise_ratio = noise_ratio  # global noise multiplier, if < 1 then reduces noise
        # be careful when using robot_cache_noise_ratio, since this will affect noise
        # of obs used by the robot controller
        self.robot_cache_noise_ratio = robot_cache_noise_ratio
        self.object_pos_noise_amp = object_pos_noise_amp
        self.object_vel_noise_amp = object_vel_noise_amp
        self.robot_obs_extra_noise_amp = robot_obs_extra_noise_amp

        self.init_random_steps_set = init_random_steps_set
        self.init_perturb_robot_ratio = init_perturb_robot_ratio
        self.init_perturb_object_ratio = init_perturb_object_ratio
        self.rng_type = rng_type

        self.model_dir = os.path.join(os.path.dirname(__file__), 'assets/')
        self.model_path = os.path.join(self.model_dir, 'kitchen.xml')
        self.model_xml = open(self.model_path, 'r').read()

        self.render_size = render_size
        self.camera_ids = camera_ids
        self.height = height
        self.width = width

        if self.noslip_off:
            self.model_xml = self.model_xml.replace(
                '<option timestep="0.002" cone="elliptic" impratio="2" noslip_iterations="20"/>',
                '<option timestep="0.002"/>',
            )

        if self.robot_name == 'franka':
            pass
        elif self.robot_name == 'franka2':
            self.model_xml = self.model_xml.replace(
                '<include file="franka/actuator0.xml"/>',
                '<include file="franka2/actuator0.xml"/>',
            )
            self.model_xml = self.model_xml.replace(
                '<include file="franka/franka_panda.xml"/>',
                '<include file="franka2/franka_panda.xml"/>',
            )
        elif self.robot_name == 'xarm7':
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

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
        self.seed()

        # self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.set_init_qpos(FRANKA_INIT_QPOS.copy())
        # self.init_qvel = self.sim.model.key_qvel[0].copy()  # this should be np.zeros(29)

        if self.ctrl_mode == 'absvel':
            action_dim = self.N_DOF_ROBOT

            self.act_mid = np.zeros(action_dim, dtype=np.float32)
            self.act_amp = 2.0 * np.ones(action_dim, dtype=np.float32)
        elif self.ctrl_mode == 'abspos':
            action_dim = self.N_DOF_ROBOT

            self.act_mid = np.zeros(action_dim, dtype=np.float32)
            self.act_amp = 3.0 * np.ones(action_dim, dtype=np.float32)
        elif self.ctrl_mode == 'relmocapik':
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

            self.pos_range = 0.075
            self.rot_range = 0.075
        elif self.ctrl_mode == 'absmocapik':
            self.mocapid = None  # set later since sim is not yet initialized

            action_dim = self.N_DOF_ROBOT  # xyz (3) + quat (4) + gripper (2) == 9

            self.act_mid = np.zeros(action_dim, dtype=np.float32)
            self.act_amp = 3.0 * np.ones(action_dim, dtype=np.float32)
        elif self.ctrl_mode == 'mixmocapik':

            self.mocapid = None  # set later since sim is not yet initialized
            action_dim = self.N_DOF_ROBOT  # xyz (3) + quat (4) + gripper (2) == 9
            self.act_mid = np.zeros(action_dim, dtype=np.float32)
            self.act_amp = 2.0 * np.ones(action_dim, dtype=np.float32)
            self.pos_range = 0.075
        else:
            raise ValueError(f"Unsupported ctrl_mode: {self.ctrl_mode}")

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim, ))

        obs_space = {}
        obs_space['robot_qp'] = spaces.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(9, ),
                                           dtype=np.float32)
        obs_space['ee_qp'] = spaces.Box(low=-np.inf,
                                        high=np.inf,
                                        shape=(7, ),
                                        dtype=np.float32)
        obs_space['ee_forces'] = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(12, ),
                                            dtype=np.float32)

        self.observation_space = spaces.Dict(obs_space)

    def _create_sim(self, xml_string):
        with tempfile.NamedTemporaryFile(mode='w+', dir=self.model_dir) as f:
            f.write(xml_string)
            f.flush()
            sim = mujoco.Physics.from_xml_path(f.name)

        return sim

    def load_sim(self, xml_string):
        self.sim = self._create_sim(xml_string)
        _patch_mjlib_accessors(self.model, self.sim.data, True)

        self.N_DOF_ROBOT = self.sim.data.model.nu
        self.N_DOF_OBJECT = self.sim.data.model.nq - self.N_DOF_ROBOT
        self.robot = Robot(self.N_DOF_ROBOT,
                           actuator_specs=self.sim.data.model.actuator_user)

        if 'mocap' in self.ctrl_mode:
            self._create_solver_sim(xml_string)

        self.create_renderer()
        self.add_cameras()

    def _create_solver_sim(self, xml_string):
        from lxml import etree as ET

        # returns Element rather than ElementTree like ET.parse, so don't need to getroot()
        parser = ET.XMLParser(remove_blank_text=True, remove_comments=True)
        domain_model_xml_tree = ET.fromstring(xml_string, parser=parser)
        worldbody = domain_model_xml_tree.find('worldbody')

        if self.robot_name == 'franka2':
            fn = f'franka2/actuator0.xml'
            n = domain_model_xml_tree.find(f'include[@file="{fn}"]')
            n.attrib['file'] = 'franka2/teleop_actuator.xml'

            equality = """
            <equality>
                <!-- original constraints -->
                <!-- <weld body1="vive_controller" body2="world" solref="0.02 1" solimp=".7 .95 0.050"/>  -->
                <!-- <weld body1="vive_controller" body2="panda0_link7" solref="0.02 1" solimp="0.7 0.95 0.050"/> -->

                <!-- Set the impedance to constant 0.9, with width 0, seems to reduce penetration (ie. gripper finger w/ microwave handle) -->
                <weld body1="vive_controller" body2="panda0_link7" solref="0.02 1" solimp="0.7 0.9 0"/>

                <!-- from franka_panda_teleop.xml-->
                <!-- <weld body1="vive_controller" body2="panda0_link7" solref="0.01 1" solimp=".25 .25 0.001"/>  -->

                <!-- from Abhishek's code -->
                <!-- <weld body1="vive_controller" body2="panda0_link7" solref="0.02 1" solimp=".4 .85 .1"/> -->
            </equality>
            """
            equality = ET.fromstring(equality, parser=parser)
            i = domain_model_xml_tree.getchildren().index(worldbody)
            domain_model_xml_tree.insert(i - 1, equality)

            controller = """
            <!-- Mocap -->
            <!-- <body name="vive_controller" mocap="true" pos="0 0 2.89" euler="-1.57 0 -.785"> -->
            <body name="vive_controller" mocap="true" pos="-0.440 -0.092 2.026" euler="-1.57 0 -.785">
                <geom type="box" group="2" pos='0 0 .142' size="0.02 0.10 0.03" contype="0" conaffinity="0" rgba=".9 .7 .95 0" euler="0 0 -.785"/>
            </body>
            """
            controller = ET.fromstring(controller, parser=parser)
            worldbody.insert(0, controller)

            # for efficiency, delete some of the unneeded things
            # texplane, MatPlane, light, floor, xaxis, yaxis, cylinder
        else:
            raise NotImplementedError

        domain_model_xml = ET.tostring(
            domain_model_xml_tree,
            encoding='utf8',
            method='xml',
            pretty_print=True,
        ).decode('utf8')

        self.solver_sim = self._create_sim(domain_model_xml)
        _patch_mjlib_accessors(self.solver_sim.model, self.solver_sim.data,
                               True)

    def add_cameras(self, camera_id=None):
        if camera_id is not None:
            self.camera_ids.append(camera_id)
        self.cameras = dict()
        for camera_id in self.camera_ids:
            camera = engine.MovableCamera(self.sim,
                                          height=self.height,
                                          width=self.width)
            camera.set_pose(**CAMERAS[camera_id])
            self.cameras['camera_{}'.format(camera_id)] = camera
        self.cameras['camera_gripper'] = engine.Camera(
            self.sim,
            height=self.height,
            width=self.width,
            camera_id='gripper_camera_rgb')

    def create_renderer(self):
        self.renderer = DMRenderer(self.sim,
                                   camera_settings=dict(
                                       distance=2.9,
                                       lookat=[-0.05, 0.5, 2.0],
                                       azimuth=90,
                                       elevation=-50))
        if hasattr(self, 'solver_sim'):
            self.solver_sim_renderer = DMRenderer(self.solver_sim,
                                                  camera_settings=dict(
                                                      distance=2.9,
                                                      lookat=[-0.05, 0.5, 2.0],
                                                      azimuth=90,
                                                      elevation=-50))

    def set_init_qpos(self, qpos):
        self.init_qpos = qpos

    def set_noise_ratio(self, noise_ratio, robot_cache_noise_ratio=None):
        self.noise_ratio = noise_ratio
        self.robot_cache_noise_ratio = robot_cache_noise_ratio

    def set_init_noise_params(
        self,
        init_random_steps_set,
        init_perturb_robot_ratio,
        init_perturb_object_ratio,
        rng_type,
    ):
        self.init_random_steps_set = init_random_steps_set
        self.init_perturb_robot_ratio = init_perturb_robot_ratio
        self.init_perturb_object_ratio = init_perturb_object_ratio

        if rng_type != self.rng_type:
            self.rng_type = rng_type
            self.seed(seed=self._base_seed)

    @property
    def data(self):
        return self.sim.data

    @property
    def model(self):
        return self.sim.model

    @property
    def physics(self):
        return self.sim

    def _get_obs_dict(self, noise_ratio='default', robot_cache_obs=False):
        if noise_ratio == 'default':
            noise_ratio = self.noise_ratio

#        noise_ratio = None;
# Gather simulated observation
        robot_qp = self.sim.data.qpos[:self.N_DOF_ROBOT].copy()
        robot_qv = self.sim.data.qvel[:self.N_DOF_ROBOT].copy()
        obj_qp = self.sim.data.qpos[-self.N_DOF_OBJECT:].copy()
        obj_qv = self.sim.data.qvel[-self.N_DOF_OBJECT:].copy()
        t = self.sim.data.time

        # Simulate observation noise
        if noise_ratio is not None:
            # currently, robot noise is specified per actuator
            # while object noise is constant across different objects
            robot_qp += (noise_ratio *
                         self.robot.pos_noise_amp[:self.N_DOF_ROBOT] *
                         self.np_random.uniform(
                             low=-1.0, high=1.0, size=self.N_DOF_ROBOT))
            robot_qv += (noise_ratio *
                         self.robot.vel_noise_amp[:self.N_DOF_ROBOT] *
                         self.np_random.uniform(
                             low=-1.0, high=1.0, size=self.N_DOF_ROBOT))
            obj_qp += (noise_ratio * self.object_pos_noise_amp *
                       self.np_random.uniform(
                           low=-1.0, high=1.0, size=self.N_DOF_OBJECT))
            obj_qv += (noise_ratio * self.object_vel_noise_amp *
                       self.np_random.uniform(
                           low=-1.0, high=1.0, size=self.N_DOF_OBJECT))

        obs_dict = {
            'robot_qp': robot_qp,
            'robot_qv': robot_qv,
            'obj_qp': obj_qp,
            'obj_qv': obj_qv,
        }

        # using np_random2 randomstate for these to preserve past behavior
        if self.with_obs_ee:
            ee_qp = get_obs_ee(self.sim, self.rot_use_euler)
            if noise_ratio is not None:
                ee_qp += (noise_ratio * self.robot_obs_extra_noise_amp *
                          self.np_random2.uniform(
                              low=-1.0, high=1.0, size=ee_qp.shape))
            obs_dict['ee_qp'] = ee_qp

        if self.with_obs_forces:
            ee_forces = get_obs_forces(self.sim)
            if noise_ratio is not None:
                ee_forces += (noise_ratio * self.robot_obs_extra_noise_amp *
                              self.np_random2.uniform(
                                  low=-1.0, high=1.0, size=ee_forces.shape))
            obs_dict['ee_forces'] = ee_forces

        if robot_cache_obs:
            if self.robot_cache_noise_ratio is not None:
                _robot_qp = self.sim.data.qpos[:self.N_DOF_ROBOT].copy()
                _robot_qv = self.sim.data.qvel[:self.N_DOF_ROBOT].copy()

                _robot_qp += (self.robot_cache_noise_ratio *
                              self.robot.pos_noise_amp[:self.N_DOF_ROBOT] *
                              self.np_random2.uniform(
                                  low=-1.0, high=1.0, size=self.N_DOF_ROBOT))
                _robot_qv += (self.robot_cache_noise_ratio *
                              self.robot.vel_noise_amp[:self.N_DOF_ROBOT] *
                              self.np_random2.uniform(
                                  low=-1.0, high=1.0, size=self.N_DOF_ROBOT))

                self.robot.cache_obs(_robot_qp, _robot_qv)
            else:
                self.robot.cache_obs(robot_qp, robot_qv)


#                self.robot.cache_obs(self.sim.data.qpos[: self.N_DOF_ROBOT].copy(),
#                                     self.sim.data.qvel[: self.N_DOF_ROBOT].copy())

# cast to float32
        for k, v in obs_dict.items():
            obs_dict[k] = v.astype(np.float32)
        return obs_dict

    def _get_task_goal(self):
        goal = np.zeros((30, ))
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            goal[element_idx] = element_goal
        return goal

    def _get_reward_n_score(self, obs_dict):
        next_obj_obs = obs_dict['obj_qp']
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(next_obj_obs[..., element_idx -
                                                   self.N_DOF_ROBOT] -
                                      self.goal[element_idx])
            complete = distance < BONUS_THRESH[element]
            if complete:
                completions.append(element)
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        return bonus

    # from adept_envs.mujoco_env.MujocoEnv
    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]

        for _ in range(n_frames):
            self.sim.step()

    def step(self, action):
        # if getting dm_control.rl.control.PhysicsError: Physics state is invalid. Warning(s) raised: mjWARN_BADCTRL
        # then probably passing in nans, https://github.com/deepmind/dm_control/issues/99
        env_info = {}

        if self.ctrl_mode == 'absvel':
            self._step_absvel(action)
        elif self.ctrl_mode == 'abspos':
            self._step_abspos(action)
        elif self.ctrl_mode == 'absmocapik':
            self._step_absmocapik(action)
        elif self.ctrl_mode == 'relmocapik':
            self._step_relmocapik(action)
        elif self.ctrl_mode == 'mixmocapik':
            jarel = self._step_mixmocapik(action)
            env_info['action'] = jarel.copy() / self.act_amp
        else:
            raise RuntimeError(f"Unsupported ctrl_mode: {self.ctrl_mode}")

        obs_dict = self._get_obs_dict(robot_cache_obs=True)
        reward = self._get_reward_n_score(obs_dict)
        #        obs = {k: obs_dict[k] for k in self.observation_space.keys()}
        done = False
        #        env_info = {k: obs_dict[k] for k in obs_dict.keys() if 'rgb' not in k}

        return obs_dict, reward, done, env_info

    def _step_absvel(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a * self.act_amp  # mean center and scale

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self, a, self.frame_skip, mode='velact')

    def _step_abspos(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a * self.act_amp  # mean center and scale

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self, a, self.frame_skip, mode='posact')

    def _reset_solver_sim(self, qpos, qvel):
        self.solver_sim.data.qpos[:] = qpos[:].copy()
        self.solver_sim.data.qvel[:] = qvel[:].copy()

        # # track object state only
        # self.solver_sim.data.qpos[-self.N_DOF_OBJECT:] = self.sim.data.qpos[-self.N_DOF_OBJECT:].copy()
        # self.solver_sim.data.qvel[-self.N_DOF_OBJECT:] = self.sim.data.qvel[-self.N_DOF_OBJECT:].copy()

        self.solver_sim.forward()
        # self.solver_sim.step()
        reset_mocap2body_xpos(self.solver_sim)

        if self.mocapid is None:
            self.mocapid = self.solver_sim.model.body_mocapid[
                self.solver_sim.model.body_name2id('vive_controller')]

    def _apply_solver_sim(self):
        step_duration = self.frame_skip * self.model.opt.timestep
        n_frames = int(step_duration / self.solver_sim.model.opt.timestep)

        with self.solver_sim.model.disable('gravity'):
            for _ in range(n_frames):
                self.solver_sim.step()

        ja = self.solver_sim.data.qpos[:self.N_DOF_ROBOT].copy()
        return ja

    def _step_absmocapik(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mid + a * self.act_amp  # mean center and scale

        self._reset_solver_sim(self.sim.data.qpos, self.sim.data.qvel)

        pos_a = a[0:3]
        quat_a = a[3:7]
        gripper_a = a[7:9]

        self.solver_sim.data.mocap_pos[self.mocapid, ...] = pos_a.copy()
        self.solver_sim.data.mocap_quat[self.mocapid, ...] = quat_a.copy()
        self.solver_sim.data.ctrl[:2] = gripper_a.copy()

        # mocap do_simulation w/ solver_sim
        # get robot qpos from solver_sim and swap in gripper_a
        ja = self._apply_solver_sim()
        ja[7:9] = gripper_a

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self, ja, self.frame_skip, mode='posact')

    def _step_relmocapik(self, a):
        a = np.clip(a, -1.0, 1.0)

        self._reset_solver_sim(self.sim.data.qpos, self.sim.data.qvel)

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

        self.solver_sim.data.ctrl[:2] = gripper_a.copy()

        # mocap do_simulation w/ solver_sim
        # get robot qpos from solver_sim and swap in gripper_a
        ja = self._apply_solver_sim()
        ja[7:9] = gripper_a

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        self.robot.step(self, ja, self.frame_skip, mode='posact')

    def _step_mixmocapik(self, a):
        a = np.clip(a, -1.0, 1.0)

        self._reset_solver_sim(self.sim.data.qpos, self.sim.data.qvel)

        # split action [3-dim Cartesian coordinate, 3-dim euler angle OR 4-dim quarternion, 2-dim gripper joints]
        current_pos = self.solver_sim.data.mocap_pos[self.mocapid, ...].copy()
        new_pos = current_pos + a[:3] * self.pos_range
        self.solver_sim.data.mocap_pos[self.mocapid, ...] = new_pos.copy()

        quat_a = a[3:7]
        gripper_a = a[7:9]

        self.solver_sim.data.mocap_quat[self.mocapid, ...] = quat_a.copy()
        self.solver_sim.data.ctrl[:2] = gripper_a.copy()

        # mocap do_simulation w/ solver_sim
        # get robot qpos from solver_sim and swap in gripper_a
        ja = self._apply_solver_sim()
        ja[7:9] = gripper_a

        if self.compensate_gravity:
            self.sim.data.qfrc_applied[:9] = self.sim.data.qfrc_bias[:9]

        jarel = (ja - self.robot.last_qpos[:self.robot.n_jnt]) / (
            self.frame_skip * self.model.opt.timestep)
        jarel = np.clip(jarel, -self.act_amp, self.act_amp)
        jarel = jarel.astype(np.float32)

        self.robot.step(self, jarel, self.frame_skip, mode='velact')

        return jarel

    def reset(self, objects_done_set=None):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)

        self.sim.reset()
        self.sim.forward()

        if objects_done_set is not None:
            reset_qpos = self.init_qpos[:].copy()
            for element in objects_done_set:
                reset_qpos[OBS_ELEMENT_INDICES[element]] = OBS_ELEMENT_GOALS[
                    element][:].copy()
        else:
            reset_qpos = None

        self.reset_model(reset_qpos=reset_qpos)
        obs_dict = self._get_obs_dict(robot_cache_obs=True)

        if self.init_random_steps_set is not None:
            if self.rng_type != 'generator':
                raise RuntimeError(
                    "Can only use rng_type=='generator' with init_random_steps_set"
                )

            t = self.np_random2.choice(self.init_random_steps_set)
            for _ in range(t):
                self.sim.step()

        return obs_dict

    def reset_model(self, reset_qpos=None):
        if reset_qpos is None:
            # NOTE: if obj penetration happens, ie. arm going thru hinge, this will NOT resolve it
            # and if sim.step is not called, then obj changes will reset in next call to env.step()
            reset_qpos = self.init_qpos[:].copy()
        else:
            reset_qpos = reset_qpos[:].copy()
        reset_qvel = self.init_qvel[:].copy()

        if self.init_perturb_robot_ratio is not None:
            init_robot_noise = self.init_perturb_robot_ratio * self.np_random2.uniform(
                low=self.robot.robot_pos_bound[:self.N_DOF_ROBOT, 0],
                high=self.robot.robot_pos_bound[:self.N_DOF_ROBOT, 1],
            )
            reset_qpos[:self.N_DOF_ROBOT] += init_robot_noise
        if self.init_perturb_object_ratio is not None:
            reset_qpos[-self.N_DOF_OBJECT:] += self.np_random2.uniform(
                low=-self.init_perturb_object_ratio,
                high=self.init_perturb_object_ratio)
        # Not adding perturbation to reset_qvel to ensure objects will be static

        # moved robot.reset() function contents to here
        # self.robot.reset(self, reset_qpos, reset_qvel)
        reset_qpos[:self.N_DOF_ROBOT] = self.robot.enforce_position_limits(
            reset_qpos[:self.N_DOF_ROBOT])
        # reset_qvel[: self.N_DOF_ROBOT] = self.robot.enforce_velocity_limits(
        #     reset_qvel[: self.N_DOF_ROBOT]
        # )

        self.sim.reset()
        # reset robot
        self.sim.data.qpos[:self.N_DOF_ROBOT] = reset_qpos[:self.
                                                           N_DOF_ROBOT].copy()
        self.sim.data.qvel[:self.N_DOF_ROBOT] = reset_qvel[:self.
                                                           N_DOF_ROBOT].copy()
        # reset objects
        self.sim.data.qpos[-self.N_DOF_OBJECT:] = reset_qpos[
            -self.N_DOF_OBJECT:].copy()
        self.sim.data.qvel[-self.N_DOF_OBJECT:] = reset_qvel[
            -self.N_DOF_OBJECT:].copy()
        self.sim.forward()

        reset_mocap_welds(self.sim)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

        if 'mocap' in self.ctrl_mode:
            self.solver_sim.data.qpos[:] = self.sim.data.qpos[:].copy()
            self.solver_sim.data.qvel[:] = self.sim.data.qvel[:].copy()
            reset_mocap_welds(self.solver_sim)
            reset_mocap2body_xpos(self.solver_sim)
            self.solver_sim.forward()

    def render(
        self,
        mode='human',
        depth=False,
        segmentation=False,
    ):
        imgs = {}
        if mode == 'rgb':
            # http://www.mujoco.org/book/APIreference.html#mjvOption
            # https://github.com/deepmind/dm_control/blob/9e0fe0f0f9713a2a993ca78776529011d6c5fbeb/dm_control/mujoco/engine.py#L200
            # mjtRndFlag(mjRND_SHADOW=0, mjRND_WIREFRAME=1, mjRND_REFLECTION=2, mjRND_ADDITIVE=3, mjRND_SKYBOX=4, mjRND_FOG=5, mjRND_HAZE=6, mjRND_SEGMENT=7, mjRND_IDCOLOR=8, mjNRNDFLAG=9)

            for camera_id, camera in self.cameras.items():
                img_rgb = camera.render(render_flag_overrides=dict(
                    skybox=False, fog=False, haze=False))
                imgs[camera_id + "_rgb"] = img_rgb
        
        elif mode == 'depth':    
            for camera_id, camera in self.cameras.items():
                img_depth = camera.render(depth=True, segmentation=False)
                imgs[camera_id + "_depth"] = np.clip(img_depth, 0.0, 4.0)        
        elif mode == 'human':
            self.renderer.render_to_window(
            )  # adept_envs.mujoco_env.MujocoEnv.render
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

        return imgs

    def seed(self, seed=None):
        self._base_seed = seed

        if self.rng_type == 'legacy':
            self.np_random, seed = seeding.np_random(seed)
            self.np_random2, _ = seeding.np_random(
                seed + 1 if seed is not None else seed)
            # a separate generator is used to preserve behavior with the original adept_envs,
            # this is important for consistently generating demonstration trajectories from mocap demos.
            # also see https://github.com/openai/gym/blob/4ede9280f9c477f1ca09929d10cdc1e1ba1129f1/gym/utils/seeding.py#L24
            # for more info on random seeding
            return [seed]
        elif self.rng_type == 'generator':
            # Careful, generator API is slightly different from the original random API
            self.np_random = make_rng(seed)
            self.np_random2 = make_rng(seed + 1 if seed is not None else seed)
            return [seed]
        else:
            raise ValueError(f"Unsupported rng_type: {self.rng_type}")
