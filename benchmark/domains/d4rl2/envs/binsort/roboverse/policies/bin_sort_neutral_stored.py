import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS, BIN_SORT_OBJECTS2NOISE
from roboverse.envs.widow250_binsort import bin_sort_hash

class BinSortNeutralStored:

    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=7.0,
                 pick_point_noise=0.00, drop_point_noise=0.00, 
                 correct_bin_per=1.0, open_steps=5, use_stored_noise=True):
        self.env = env
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            # + np.random.normal(scale=0.01)
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise
        self.correct_bin_per = correct_bin_per

        self.grasp_distance_thresh = 0.02,
        self.open_steps=open_steps
        self.curr_steps=0
        self.use_stored_noise = use_stored_noise

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.curr_steps=0
        if hasattr(self.env, 'bin_obj') and self.env.bin_obj and len(self.env.object_names) > 1:
            self.object_to_target = np.random.choice(self.env.object_names[:-1]) # last object ignored if one in bin    
        else:
            self.object_to_target = np.random.choice(self.env.object_names)
        self.get_pickpoint()
        self.drop_point = self.env.container_position if bin_sort_hash(self.object_to_target) % 2 == 0 else self.env.container2_position
        alternate_drop_point = self.env.container_position if bin_sort_hash(self.object_to_target) % 2 == 1 else self.env.container2_position
        if self.use_stored_noise:
            self.correct_bin_per = BIN_SORT_OBJECTS2NOISE[self.object_to_target]
        self.drop_point = self.drop_point if np.random.rand() < self.correct_bin_per else alternate_drop_point
        self.drop_point[2] = -0.2
        self.place_attempted = False

        self.reset_pos, _ = bullet.get_link_state(self.env.robot_id, self.env.end_effector_index)

    def get_pickpoint(self):
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point[2] = -0.32

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False

        # print('gripper_pickpoint_dist ', gripper_pickpoint_dist )
        # print('self.env.is_gripper_open ', self.env.is_gripper_open)
        if self.place_attempted:
            # print('Neutral attemtped')
            # Reset after one attempt
            action_xyz = (self.reset_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            self.place_attempted = True
        elif gripper_pickpoint_dist > self.grasp_distance_thresh and self.env.is_gripper_open:
            self.get_pickpoint()
            # print('moving near object ')
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # print('peform grasping, gripper open:', self.env.is_gripper_open)
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # print('lift object')
            # lifting objects above the height threshold for picking
            # action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_xyz = np.array([0., 0., 0.08]) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_droppoint_dist > 0.02:
            # print('move towards container')
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            # print('drop')
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True
            self.curr_steps += 1

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info


class PickPlaceOld:

    def __init__(self, env, pick_height_thresh=-0.31):
        self.env = env
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            + np.random.normal(scale=0.01)
        self.xyz_action_scale = 7.0
        self.reset()

    def reset(self):
        self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.place_attempted = False
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)

        container_pos = self.env.container_position
        target_pos = np.append(container_pos[:2], container_pos[2] + 0.15)
        target_pos = target_pos + np.random.normal(scale=0.01)
        gripper_target_dist = np.linalg.norm(target_pos - ee_pos)
        gripper_target_threshold = 0.03

        done = False

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif object_gripper_dist > self.dist_thresh and self.env.is_gripper_open:
            # move near the object
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_target_dist > gripper_target_threshold:
            # lifted, now need to move towards the container
            action_xyz = (target_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info
