import os

import mujoco
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from benchmark.domains.adroit.src.utils.quatmath import euler2quat


class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense", early_termination=False, **kwargs):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0

        self.reward_type = reward_type
        self.early_termination = early_termination

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            curr_dir + "/src/assets/DAPG_pen.xml",
            5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
            ),
        )
        mujoco.mj_forward(self.model, self.data)

        # change actuator sensitivity
        self.model.actuator_gainprm[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_WRJ1"
            ) : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_WRJ0")
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_FFJ3"
            ) : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_THJ0")
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_WRJ1"
            ) : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_WRJ0")
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_FFJ3"
            ) : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_THJ0")
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        self.target_obj_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.S_grasp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp"
        )
        self.obj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Object")
        self.eps_ball_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "eps_ball"
        )
        self.obj_t_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_top"
        )
        self.obj_b_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_bottom"
        )
        self.tar_t_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_top"
        )
        self.tar_b_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_bottom"
        )

        self.pen_length = np.linalg.norm(
            self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]
        )
        self.tar_length = np.linalg.norm(
            self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]
        )

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        self.action_space.low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            starting_up = True
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos = self.data.body(self.obj_bid).xpos.ravel()
        desired_loc = self.data.site(self.eps_ball_sid).xpos.ravel()
        obj_orien = (
            self.data.site(self.obj_t_sid).xpos - self.data.site(self.obj_b_sid).xpos
        ) / self.pen_length
        desired_orien = (
            self.data.site(self.tar_t_sid).xpos - self.data.site(self.tar_b_sid).xpos
        ) / self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos - desired_loc)
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        sparse_reward = 50 * goal_achieved
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            reward += -dist
            reward += orien_similarity
            # bonus for being close to desired orientation
            if orien_similarity > 0.9:
                reward += 10
            if obj_pos[2] < 0.075:
                reward -= 5
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            raise ValueError(f"{self.reward_type} reward type not supported.")

        # penalty for dropping the pen
        done = False
        if self.early_termination:
            done = True if not starting_up else False

        return self.get_obs(), reward, done, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body(self.obj_bid).xpos.ravel()
        desired_pos = self.data.site(self.eps_ball_sid).xpos.ravel()
        obj_orien = (
            self.data.site(self.obj_t_sid).xpos - self.data.site(self.obj_b_sid).xpos
        ) / self.pen_length
        desired_orien = (
            self.data.site(self.tar_t_sid).xpos - self.data.site(self.tar_b_sid).xpos
        ) / self.tar_length
        return np.concatenate(
            [
                qp[:-6],
                obj_pos,
                obj_vel,
                obj_orien,
                desired_orien,
                obj_pos - desired_pos,
                obj_orien - desired_orien,
            ]
        )

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.set_state(qp, qv)
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        desired_orien = state_dict["desired_orien"]
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.set_state(qp, qv)

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.azimuth = -45
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path["env_infos"]["goal_achieved"]) > 20:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
