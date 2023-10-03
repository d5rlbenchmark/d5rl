import gym
import numpy as np

from roboverse.bullet.serializable import Serializable
import roboverse.bullet as bullet
from roboverse.envs import objects
from roboverse.bullet import object_utils
# from .multi_object import MultiObjectEnv
from roboverse.envs.multi_object import MultiObjectEnv

END_EFFECTOR_INDEX = 8
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14  # TODO(avi) This is a guess, need to verify what joint this is
JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]
JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)

GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]
GRIPPER_OPEN_STATE = [0.036, -0.036]
GRIPPER_CLOSED_STATE = [0.015, -0.015]

ACTION_DIM = 7


class Widow250Env(gym.Env, Serializable):

    def __init__(self,
                 control_mode='continuous',
                 observation_mode='pixels',
                 observation_img_dim=128,
                 transpose_image=True,

                 object_names=('beer_bottle', 'gatorade'),
                 object_scales=(0.75, 0.75),
                 object_orientations=((0, 0, 1, 0), (0, 0, 1, 0)),
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 target_object='gatorade',
                 load_tray=True,

                 num_sim_steps=10,
                 num_sim_steps_reset=50,
                 num_sim_steps_discrete_action=75,

                 reward_type='grasping',
                 grasp_success_height_threshold=-0.25,
                 grasp_success_object_gripper_threshold=0.1,

                 xyz_action_scale=0.2,
                 abc_action_scale=20.0,
                 gripper_action_scale=20.0,

                 ee_pos_low=(.4, -.2, -.34),
                 ee_pos_high=(0.8, .4, -0.1),
                 camera_target_pos=(0.6, 0.2, -0.28),
                 camera_distance=0.29,
                 camera_roll=0.0,
                 camera_pitch=-40,
                 camera_yaw=180,

                 gui=False,
                 in_vr_replay=False,

                 terminate_on_success=True,
                 max_reward=1,
                 objects_in_container=False,
                 ):
        print(object_names)

        self.render_enabled = True

        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim
        self.transpose_image = transpose_image
        self.max_reward = max_reward
        self.objects_in_container=objects_in_container

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = \
            grasp_success_object_gripper_threshold

        self.gui = gui

        # TODO(avi): Add limits to ee orientation as well
        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)

        # object stuff
        assert target_object in object_names
        assert len(object_names) == len(object_scales)
        self.load_tray = load_tray
        self.num_objects = len(object_names)
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.object_names = object_names
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_name in \
                zip(object_orientations, object_scales, self.object_names):
            self.object_orientations[object_name] = orientation
            self.object_scales[object_name] = object_scale

        self.in_vr_replay = in_vr_replay
        self._load_meshes()

        self.movable_joints = bullet.get_movable_joints(self.robot_id)
        self.end_effector_index = END_EFFECTOR_INDEX
        self.reset_joint_values = RESET_JOINT_VALUES
        self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale
        self.gripper_action_scale = gripper_action_scale

        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_roll = camera_roll
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance,
                                yaw=self.camera_yaw,
                                pitch=self.camera_pitch,
                                roll=self.camera_roll,
                                up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_dim, self.observation_img_dim)

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True  # TODO(avi): Clean this up

        self.terminate_on_success = terminate_on_success

        self.reset()
        self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)

    def disable_render(self):
        self.render_enabled = False

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        if self.in_vr_replay:
            object_positions = self.original_object_positions
        else:
            object_positions = object_utils.generate_object_positions(
                self.object_position_low, self.object_position_high,
                self.num_objects,
            )
            self.original_object_positions = object_positions
        for object_name, object_position in zip(self.object_names, object_positions):
            self.objects[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name])
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True  # TODO(avi): Clean this up

        return self.get_observation()

    def step(self, action):
        # TODO Clean this up
        if np.isnan(np.sum(action)):
            print('action', action)
            assert False

        action = np.clip(action, -1, +1)  # TODO Clean this up

        xyz_action = action[:3]  # ee position actions
        abc_action = action[3:6]  # ee orientation actions
        gripper_action = action[6]

        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        ee_deg = bullet.quat_to_deg(ee_quat)
        target_ee_deg = ee_deg + self.abc_action_scale * abc_action
        target_ee_quat = bullet.deg_to_quat(target_ee_deg)

        if self.control_mode == 'continuous':
            num_sim_steps = self.num_sim_steps
            target_gripper_state = gripper_state + \
                                   [-self.gripper_action_scale * gripper_action,
                                    self.gripper_action_scale * gripper_action]

        elif self.control_mode == 'discrete_gripper':
            if gripper_action > 0.5 and not self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_OPEN_STATE
                self.is_gripper_open = True  # TODO(avi): Clean this up

            elif gripper_action < -0.5 and self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_CLOSED_STATE
                self.is_gripper_open = False  # TODO(avi): Clean this up
            else:
                num_sim_steps = self.num_sim_steps
                if self.is_gripper_open:
                    target_gripper_state = GRIPPER_OPEN_STATE
                else:
                    target_gripper_state = GRIPPER_CLOSED_STATE
                # target_gripper_state = gripper_state
        else:
            raise NotImplementedError

        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_low,
                                self.ee_pos_high)
        target_gripper_state = np.clip(target_gripper_state, GRIPPER_LIMITS_LOW,
                                       GRIPPER_LIMITS_HIGH)

        bullet.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state,
            self.robot_id,
            self.end_effector_index, self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=num_sim_steps)

        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        if self.terminate_on_success:
            if reward >= self.max_reward:
                done = True
            elif self.objects_in_container and 'all_placed' in info and info['all_placed']:
                done = True
        return self.get_observation(), reward, done, info

    def get_observation(self):
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        if self.observation_mode == 'pixels':
            observation = {
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_state, gripper_binary_state)),
            }
            if self.render_enabled:
                image_observation = self.render_obs()
                image_observation = np.float32(image_observation.flatten()) / 255.0
                observation['image'] = image_observation
        else:
            raise NotImplementedError

        return observation

    def get_reward(self, info):
        if self.reward_type == 'grasping':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):

        info = {'grasp_success': False}
        for object_name in self.object_names:
            grasp_success = object_utils.check_grasp(
                object_name, self.objects, self.robot_id,
                self.end_effector_index, self.grasp_success_height_threshold,
                self.grasp_success_object_gripper_threshold)
            if grasp_success:
                info['grasp_success'] = True

        info['grasp_success_target'] = object_utils.check_grasp(
            self.target_object, self.objects, self.robot_id,
            self.end_effector_index, self.grasp_success_height_threshold,
            self.grasp_success_object_gripper_threshold)
        return info

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.observation_img_dim, self.observation_img_dim,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=0)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def _set_action_space(self):
        self.action_dim = ACTION_DIM
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def get_gripper_state(self):
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray(joint_states[-2:])
        return gripper_state

    def close(self):
        bullet.disconnect()


class Widow250MultiObjectEnv(MultiObjectEnv, Widow250Env):
    """Grasping Env but with a random object each time."""


if __name__ == "__main__":
    env = Widow250Env(gui=True)
    import time

    env.reset()
    # import IPython; IPython.embed()

    for i in range(20):
        print(i)
        obs, rew, done, info = env.step(
            np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0.]))
        print("reward", rew, "info", info)
        time.sleep(0.1)

    env.reset()
    time.sleep(1)
    for _ in range(25):
        env.step(np.asarray([0., 0., 0., 0., 0., 0., 0.6]))
        time.sleep(0.1)

    env.reset()
