import os

import mujoco
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

ADD_BONUS_REWARDS = True


class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense", **kwargs):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.reward_type = reward_type

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            curr_dir + "/src/assets/DAPG_relocate.xml",
            5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64
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

        self.target_obj_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.S_grasp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp"
        )
        self.obj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Object")
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        self.action_space.low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body(self.obj_bid).xpos.ravel()
        palm_pos = self.data.site(self.S_grasp_sid).xpos.ravel()
        target_pos = self.data.site(self.target_obj_sid).xpos.ravel()

        goal_achieved = True if np.linalg.norm(obj_pos - target_pos) < 0.1 else False

        sparse_reward = 20 * goal_achieved + 10 * (
            np.linalg.norm(obj_pos - target_pos) < 0.1
        )
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            reward += -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
            if obj_pos[2] > 0.04:  # if object off the table
                reward += 1.0  # bonus for lifting the object
                reward += -0.5 * np.linalg.norm(
                    palm_pos - target_pos
                )  # make hand go to target
                reward += -0.5 * np.linalg.norm(
                    obj_pos - target_pos
                )  # make object go to target
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            raise ValueError(f"{self.reward_type} reward type not supported.")

        return ob, reward, False, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body(self.obj_bid).xpos.ravel()
        palm_pos = self.data.site(self.S_grasp_sid).xpos.ravel()
        target_pos = self.data.site(self.target_obj_sid).xpos.ravel()
        return np.concatenate(
            [qp[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos]
        )

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(
            low=-0.15, high=0.15
        )
        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(
            low=-0.15, high=0.3
        )
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(
            low=-0.2, high=0.2
        )
        self.model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(
            low=-0.2, high=0.2
        )
        self.model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(
            low=0.15, high=0.35
        )
        self.set_state(qp, qv)
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.data.body(self.obj_bid).xpos.ravel()
        palm_pos = self.data.site(self.S_grasp_sid).xpos.ravel()
        target_pos = self.data.site(self.target_obj_sid).xpos.ravel()
        return dict(
            hand_qpos=hand_qpos,
            obj_pos=obj_pos,
            target_pos=target_pos,
            palm_pos=palm_pos,
            qpos=qp,
            qvel=qv,
        )

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        obj_pos = state_dict["obj_pos"]
        target_pos = state_dict["target_pos"]
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.set_state(qp, qv)

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path["env_infos"]["goal_achieved"]) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
