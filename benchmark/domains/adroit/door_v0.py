import os

import mujoco
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

ADD_BONUS_REWARDS = True


class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense", **kwargs):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.reward_type = reward_type

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            curr_dir + "/src/assets/DAPG_door.xml",
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

        utils.EzPickle.__init__(self)
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        self.action_space.low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])
        self.door_hinge_did = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        ]
        self.grasp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp"
        )
        self.handle_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_handle"
        )
        self.door_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "frame")

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        goal_achieved = True if door_pos >= 1.35 else False

        sparse_reward = (
            10 * goal_achieved
            + 8 * (door_pos > 1.0)
            + 2 * (door_pos > 1.2)
            - 0.1 * (door_pos - 1.57) * (door_pos - 1.57)
        )
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            # get to handle
            reward += -0.1 * np.linalg.norm(palm_pos - handle_pos)
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
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        # len(qp[1:-2]) == 27
        # import ipdb; ipdb.set_trace()
        return np.concatenate(
            [
                qp[1:-2],
                [latch_pos],
                door_pos,
                palm_pos,
                handle_pos,
                palm_pos - handle_pos,
                [door_open],
            ]
        )

    def reset_model(self):
        self.model.body_pos[self.door_bid, 0] = self.np_random.uniform(
            low=-0.3, high=-0.2
        )
        self.model.body_pos[self.door_bid, 1] = self.np_random.uniform(
            low=0.25, high=0.35
        )
        self.model.body_pos[self.door_bid, 2] = self.np_random.uniform(
            low=0.252, high=0.35
        )
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        self.model.body_pos[self.door_bid] = state_dict["door_body_pos"]
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        self.set_state(qp, qv)

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path["env_infos"]["goal_achieved"]) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
