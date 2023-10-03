import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS
from roboverse.envs.widow250_binsort import bin_sort_hash

class Stitching:
    def __init__(self, env, random_steps=10, xyz_action_scale=7.0, random_gripper_open=False, distance_thresh=0.03, **kwargs):
        self.env = env
        self.random_steps = random_steps
        self.xyz_action_scale = xyz_action_scale
        self.random_gripper_open = random_gripper_open
        self.xy_diff_thresh = distance_thresh

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        ee_pos, _ = bullet.get_link_state(self.env.robot_id, self.env.end_effector_index)
        random_pos = np.random.uniform(low=self.env.ee_pos_low, high=self.env.ee_pos_high)
        random_pos[-1] = ee_pos[-1] + np.random.normal(scale=0.2)
        self.random_pos = np.clip(random_pos, self.env.ee_pos_low, self.env.ee_pos_high)
        self.num_random_steps = 0
        self.reached_random_pos = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        done = False

        rand_num = np.random.rand()

        if self.reached_random_pos or self.num_random_steps < self.random_steps:
            action_xyz = (np.random.uniform(low=self.env.ee_pos_low, high=self.env.ee_pos_high) - ee_pos) * self.xyz_action_scale
            action_angles = [0.0, 0.0, 0.0]
            action_gripper = [rand_num if self.random_gripper_open else 0.0]
        else:
            action_xyz = (self.random_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0.0, 0.0, 0.0]
            action_gripper = [rand_num if self.random_gripper_open else 0.0]
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            self.reached_random_pos = self.reached_random_pos or (xy_diff < self.xy_diff_thresh)
        
        self.num_random_steps += 1

        agent_info = dict(place_attempted=False, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
