import copy

import numpy as np


class Robot:

    def __init__(self, n_jnt, actuator_specs=None):
        self.n_jnt = n_jnt
        self.load_specs(actuator_specs)

    def load_specs(self, actuator_specs):
        self.robot_pos_bound = actuator_specs[:, [0, 1]]
        self.robot_vel_bound = actuator_specs[:, [2]]
        self.pos_noise_amp = actuator_specs[:, 3]
        self.vel_noise_amp = actuator_specs[:, 4]

    def step(self, env, ctrl, skip, mode='velact'):
        step_duration = skip * env.model.opt.timestep

        # enforce velocity limits
        if mode == 'velact':
            # this converts ctrl in vel to pos
            ctrl_feasible = self.ctrl_velocity_limits_velact(
                ctrl, step_duration)
        elif mode == 'posact':
            ctrl_feasible = self.ctrl_velocity_limits_posact(
                ctrl, step_duration)
        else:
            raise ValueError

        # enforce position limits
        ctrl_feasible = self.enforce_position_limits(ctrl_feasible)

        return env.do_simulation(ctrl_feasible, skip)

    def enforce_position_limits(self, qpos):
        qpos_limited = np.clip(
            qpos,
            self.robot_pos_bound[:self.n_jnt, 0],
            self.robot_pos_bound[:self.n_jnt, 1],
        )
        return qpos_limited

    def enforce_velocity_limits(self, qvel):
        qvel_limited = np.clip(
            qvel,
            -self.robot_vel_bound[:self.n_jnt, 0],
            self.robot_vel_bound[:self.n_jnt, 0],
        )
        return qvel_limited

    # enforce velocity specs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits_posact(self, ctrl_position, step_duration):
        ctrl_desired_vel = (ctrl_position -
                            self.last_qpos[:self.n_jnt]) / step_duration

        ctrl_feasible_vel = np.clip(
            ctrl_desired_vel,
            -self.robot_vel_bound[:self.n_jnt, 0],
            self.robot_vel_bound[:self.n_jnt, 0],
        )
        ctrl_feasible_position = self.last_qpos[:self.
                                                n_jnt] + ctrl_feasible_vel * step_duration
        return ctrl_feasible_position

    # enforce velocity specs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits_velact(self, ctrl_velocity, step_duration):
        ctrl_feasible_vel = np.clip(
            ctrl_velocity,
            -self.robot_vel_bound[:self.n_jnt, 0],
            self.robot_vel_bound[:self.n_jnt, 0],
        )
        ctrl_feasible_position = self.last_qpos[:self.
                                                n_jnt] + ctrl_feasible_vel * step_duration
        return ctrl_feasible_position

    def cache_obs(self, qpos, qvel):
        self.last_qpos = copy.deepcopy(qpos)
        self.last_qvel = copy.deepcopy(qvel)
