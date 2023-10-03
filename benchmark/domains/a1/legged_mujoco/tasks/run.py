from typing import Dict, Optional, Tuple

import dm_control.utils.transformations as tr
import numpy as np
import copy
from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.composer.observation import observable
from dm_control.utils import rewards

from benchmark.domains.a1.legged_mujoco.tasks.utils import _find_non_contacting_height
from benchmark.domains.a1.legged_mujoco.arenas import RampHField, HField

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.001

class SimpleRun(composer.Task):

    def __init__(self,
                 robot,
                 terminate_pitch_roll: Optional[float] = 60,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 floor_friction: Tuple[float] = (1, 0.005, 0.0001),
                 add_velocity_to_observations: bool = True,
                 target_linear_velocity: float = 1.0):

        self.floor_friction = None
        self._robot = robot
        self._target_linear_velocity = target_linear_velocity

        self._smooth_floor = arenas.Floor(size=(10, 10))
        # if hasattr(self._smooth_floor, '_top_camera'):
        #     self._smooth_floor._top_camera.remove()
        self.setup_floor(self._smooth_floor)
        self._floor = self._smooth_floor

        self._floor.add_free_entity(self._robot)

        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = floor_friction

        observables = (self._robot.observables.proprioception +
                       self._robot.observables.kinematic_sensors)
        for observable in observables:
            observable.enabled = True

        if not add_velocity_to_observations:
            self._robot.observables.sensors_velocimeter.enabled = False

        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1, 0.5],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=70.0)

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._terminate_pitch_roll = terminate_pitch_roll

    def setup_floor(self, floor):
        # set the contact parameters
        # remove cameras/other visual elements
        # add goal sensor
        floor.mjcf_model.size.nconmax = 400
        floor.mjcf_model.size.njmax = 2000
        if hasattr(floor, '_top_camera'):
            floor._top_camera.remove()
        if hasattr(floor.mjcf_model.visual, 'headlight'):
            floor.mjcf_model.visual.headlight.remove()

    def get_reward(self, physics):
        obs = {}
        for k, v in self.observables.items():
            if v.enabled:
                obs[k] = v(physics)

        # Get run reward
        x_velocity = obs['a1/sensors_velocimeter'][0]
        _, pitch, yaw = tr.quat_to_euler(obs['a1/sensors_framequat'])

        reward_v = rewards.tolerance(x_velocity * np.cos(pitch) * np.cos(yaw),
                                     bounds=(self._target_linear_velocity - 0.1,
                                             self._target_linear_velocity + 0.1),
                                     margin=self._target_linear_velocity,
                                     value_at_margin=0,
                                     sigmoid='linear')
        bonus = 10 * reward_v

        self._x_vel = x_velocity
        
        # Get energy penalty
        qvel = physics.bind(self._robot.joints).qvel
        torque = physics.bind(self._robot.actuators).force
        energy = np.sum(qvel * torque)
        energy_penalty = 0.01 * np.sum(qvel * torque)
        
        self._energy = energy

        # Get pitch and roll penalty
        tar_pose = self._robot._INIT_QPOS
        offset = self._robot._QPOS_OFFSET
        bounds = (tar_pose - offset, tar_pose + offset)

        qpos = physics.bind(self._robot.joints).qpos

        up = physics.bind(self._robot.root_body).xmat[-1]
        upright = rewards.tolerance(up,
                                    bounds=(0.9, float('inf')),
                                    sigmoid='quadratic',
                                    margin=0.9,
                                    value_at_margin=0)
        standing_reward = upright
        for i in range(len(qpos)):
            standing = rewards.tolerance(qpos[i],
                                         bounds=(bounds[0][i], bounds[1][i]),
                                         margin=offset[i],
                                         value_at_margin=0.5)
            standing_reward *= standing

        return bonus * standing_reward - energy_penalty

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False

        _find_non_contacting_height(physics,
                                    self._robot,
                                    qpos=self._robot._INIT_QPOS)

    def before_step(self, physics, action, random_state):
        pass

    def before_substep(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec(physics)

    def after_step(self, physics, random_state):
        self._failure_termination = False

        if self._terminate_pitch_roll is not None:
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)

            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor

class RunToward(SimpleRun):

    def __init__(self, robot, **kwargs):
        super().__init__(robot, **kwargs)
        self.floor_type = 'smooth'
        self._target_delta_yaw = 0.0

        self._bumpy_floor = HField(size=(10, 10))

        self._goal_site = self._add_goal_sensor(self._smooth_floor)
        self.setup_floor(self._bumpy_floor)


    @property
    def task_observables(self):
        target_delta_yaw = observable.Generic(lambda _: np.asarray(
            [self._target_delta_yaw], dtype=np.float32))
        target_delta_yaw.enabled = True
        task_observables = super().task_observables
        task_observables['target_delta_yaw'] = target_delta_yaw
        return task_observables

    def get_reward(self, physics):
        obs = {}
        for k, v in self.observables.items():
            if v.enabled:
                obs[k] = v(physics)

        # Get run reward
        x_velocity = obs['a1/sensors_velocimeter'][0]
        _, pitch, _ = tr.quat_to_euler(obs['a1/sensors_framequat'])
        reward_v = rewards.tolerance(x_velocity * np.cos(pitch),
                                     bounds=(self._target_linear_velocity - 0.1,
                                             self._target_linear_velocity + 0.1),
                                     margin=self._target_linear_velocity,
                                     value_at_margin=0,
                                     sigmoid='linear')
        bonus = reward_v

        target_delta_yaw = obs['target_delta_yaw']
        reward_d = rewards.tolerance(target_delta_yaw,
                                     bounds=(-np.pi/12, np.pi/12),
                                     margin=np.pi,
                                     value_at_margin=0,
                                     sigmoid='linear')
        bonus *= reward_d
        bonus *= 10

        self._x_vel = x_velocity
        
        # Get energy penalty
        qvel = physics.bind(self._robot.joints).qvel
        torque = physics.bind(self._robot.actuators).force
        energy = np.sum(qvel * torque)
        energy_penalty = 0.001 * np.sum(qvel * torque)
        
        self._energy = energy

        # Get pitch and roll penalty
        tar_pose = self._robot._INIT_QPOS
        offset = self._robot._QPOS_OFFSET
        bounds = (tar_pose - offset, tar_pose + offset)

        qpos = physics.bind(self._robot.joints).qpos

        up = physics.bind(self._robot.root_body).xmat[-1]
        upright = rewards.tolerance(up,
                                    bounds=(0.9, float('inf')),
                                    sigmoid='quadratic',
                                    margin=0.9,
                                    value_at_margin=0)
        standing_reward = upright
        for i in range(len(qpos)):
            standing = rewards.tolerance(qpos[i],
                                         bounds=(bounds[0][i], bounds[1][i]),
                                         margin=offset[i],
                                         value_at_margin=0.5)
            standing_reward *= standing

        return bonus * standing_reward - energy_penalty

    def _add_goal_sensor(self, floor, pos=[-10.0, 0.0, .125]):
        return floor.mjcf_model.worldbody.add('site', 
                                            name="goal_loc"+str(pos), 
                                            size=[0.1]*3, 
                                            rgba=[0, 1, 0, 1],
                                            group=0,
                                            pos=pos)
    
    def _set_goal_loc(self, goal_loc):
        # self._goal_site.remove()
        # self._goal_site.pos = goal_loc
        self._goal_site = self._add_goal_sensor(self._floor, goal_loc)

    def _get_goal_loc(self):
        return self._goal_site.pos

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        new_floor_type = random_state.choice(["smooth", "bumpy"])
        if new_floor_type != self.floor_type:
            self._robot.detach()
            self._floor = self._bumpy_floor if new_floor_type == "bumpy" else self._smooth_floor
            self._floor.add_free_entity(self._robot)
            self.floor_type = new_floor_type
            
        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)
        
        # randomize friction 
        floor_friction = (random_state.uniform(low=0.75, high=1.25), 
                          0.005, 
                          0.0001) 
        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = floor_friction
        
        # randomize mass
        trunk = self._robot.mjcf_model.find('body', 'trunk')
        trunk.inertial.mass = str(random_state.uniform(low=4.714-.25, high=4.714+.25))

    def sample_goal(self, random_state):
        x_pos = random_state.uniform(low=-10.0, high=10.0)
        y_pos = random_state.uniform(low=-10.0, high=10.0)
        return np.array([x_pos, y_pos, 0.125], dtype=np.float32)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False

        _find_non_contacting_height(physics,
                                    self._robot,
                                    qpos=self._robot._INIT_QPOS)
        
        goal_loc = self.sample_goal(random_state)
        self._set_goal_loc(goal_loc)

    def before_step(self, physics, action, random_state):
        pass

    def before_substep(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec(physics)

    def after_step(self, physics, random_state):
        self._failure_termination = False

        if self._terminate_pitch_roll is not None:
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)

            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True

        pos, quat = self._robot.get_pose(physics)

        # calculate the angle between the robot's heading and the goal
        goal_loc = self._get_goal_loc()
        displacement_vector = goal_loc[:2] - pos[:2]
        while np.linalg.norm(displacement_vector) < 0.1:
            # sample a new goal
            new_goal_loc = self.sample_goal(random_state)
            self._set_goal_loc(new_goal_loc)
            goal_loc = self._get_goal_loc()
            displacement_vector = goal_loc[:2] - pos[:2]
        heading_vector = tr.quat_rotate(quat, [1, 0, 0])[:2]
        self._target_delta_yaw = np.arctan2(displacement_vector[1], displacement_vector[0]) - np.arctan2(heading_vector[1], heading_vector[0])

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor

class Hiking(RunToward):

    def __init__(self, robot, **kwargs):

        super(RunToward, self).__init__(robot, **kwargs)

        self._target_delta_yaw = 0.0

        self._bumpy_floor = HField(size=(10, 10))

        self.setup_floor(self._bumpy_floor)

        self.floor_type = 'bumpy'
        self._ramp_frames = None
        self._last_xy_pos = [-20.0, 0]

        self._ramp_floor = RampHField(size=(10, 10))
        self.setup_floor(self._ramp_floor)

        self._bowl_floor = arenas.Bowl(size=(10, 10))
        self.setup_floor(self._bowl_floor)

        self._floor = self._ramp_floor
        self._robot.detach()
        self._floor.add_free_entity(self._robot)

        self._ramp_frames = [self._floor.attach(ramp) for ramp in self._floor._ramps]

        robot_heading = np.pi/2
        robot_offset = 10
        ramp_height = 0.2

        # first ramp up
        self._ramp_frames[0].pos = (robot_offset + 0, 0, ramp_height)
        self._ramp_frames[0].euler =  (0, 0, robot_heading + 0)

        # second ramp down
        self._ramp_frames[1].pos = (robot_offset + 4.5, 0 + 0.5, ramp_height)
        self._ramp_frames[1].euler =  (0, 0, robot_heading -np.pi)

        # third ramp back up
        self._ramp_frames[2].pos = (robot_offset + 14, 0, ramp_height)
        self._ramp_frames[2].euler =  (0, 0, robot_heading + 0)

        # fourth ramp back down
        self._ramp_frames[3].pos = (robot_offset + 18.5, 0 + 0.5, ramp_height)
        self._ramp_frames[3].euler =  (0, 0, robot_heading -np.pi)

        # schedule goals
        self._goal_xs = [-10.0, 0.0, 20.0, 40.0]
        self._goal_sites = [self._add_goal_sensor(self._floor, pos=[x, 0.0, .125]) for x in self._goal_xs]
        self._curr_goal_idx = 0

    def initialize_episode(self, physics, random_state):
        super(RunToward, self).initialize_episode(physics, random_state)

        _find_non_contacting_height(physics,
                                    self._robot,
                                    x_pos=-20.0,
                                    qpos=self._robot._INIT_QPOS)
        self._curr_goal_idx = 0
        for site in self._goal_sites:
            physics.bind(site).pos[-1] = .125

    def initialize_episode_mjcf(self, random_state):
        super(RunToward, self).initialize_episode_mjcf(random_state)

        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)

    def _get_goal_loc(self):
        return self._goal_sites[self._curr_goal_idx].pos

    def after_step(self, physics, random_state):
        super(RunToward, self).after_step(physics, random_state)
        self._failure_termination = False

        if self._terminate_pitch_roll is not None:
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)
            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True

        pos, quat = self._robot.get_pose(physics)
        self._last_xy_pos = pos[:2]

        if np.abs(self._last_xy_pos[1]) > 3:
            self._failure_termination = True
            
        # calculate the angle between the robot's heading and the goal
        goal_loc = self._get_goal_loc()
        displacement_vector = goal_loc[:2] - pos[:2]
            
        if np.linalg.norm(displacement_vector) < 0.1:
            physics.bind(self._goal_sites[self._curr_goal_idx]).pos[-1] -= 1.0
            self._curr_goal_idx += 1
            
            goal_loc = self._get_goal_loc()
            displacement_vector = goal_loc[:2] - pos[:2]

        heading_vector = tr.quat_rotate(quat, [1, 0, 0])[:2]
        self._target_delta_yaw = np.arctan2(displacement_vector[1], displacement_vector[0]) - np.arctan2(heading_vector[1], heading_vector[0])
