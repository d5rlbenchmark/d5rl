import numpy as np

import d4rl2.envs.widowx.roboverse.bullet as bullet


class DrawerClose:

    def __init__(self, env, suboptimal=False):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_z = -0.25
        self.top_drawer_offset = np.array([0, 0, 0.02])
        self.suboptimal = suboptimal

        self.reset()

    def reset(self):
        self.drawer_never_opened = True
        offset_coeff = (-1)**(1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])
        self.reached_pushing_region = False

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(self.env.robot_id,
                                          self.env.end_effector_index)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        top_drawer_pos = self.env.get_drawer_pos("drawer_top")
        top_drawer_push_target_pos = (top_drawer_pos +
                                      np.array([0.15, 0, 0.05]))
        is_gripper_ready_to_push = (ee_pos[0] > top_drawer_push_target_pos[0]
                                    and
                                    ee_pos[2] < top_drawer_push_target_pos[2])
        done = False
        neutral_action = [0.0]
        if (not self.env.is_top_drawer_closed()
                and not self.reached_pushing_region
                and not is_gripper_ready_to_push):
            # print("move up and left")
            action_xyz = [0.3, -0.2, -0.15]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_top_drawer_closed():
            # print("close top drawer")
            self.reached_pushing_region = True
            action_xyz = (top_drawer_pos + self.top_drawer_offset -
                          ee_pos) * 7.0
            action_xyz[0] *= 3
            action_xyz[1] *= 0.6
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class MultiDrawerClose:
    """Scripted policy to close one of two drawers."""

    def __init__(self, env, close_drawer_name='main_top', suboptimal=False):
        self.env = env
        self.suboptimal = suboptimal
        self.xyz_action_scale = 7.0
        if close_drawer_name == 'main_top':
            # Do not change this
            self.gripper_dist_thresh = 0.04
            self.gripper_xy_dist_thresh = 0.02
        else:
            self.gripper_dist_thresh = 0.05
            self.gripper_xy_dist_thresh = 0.04

        self.ending_z = -0.25
        self.top_drawer_offset = np.array([0, 0, 0.02])

        self.close_drawer_name = close_drawer_name
        self.reset()

        self.drawer_name_map = dict()
        self.drawer_name_map['main_drawer'] = 'drawer'
        self.drawer_name_map['main_top'] = 'drawer_top'
        self.drawer_name_map['second_drawer'] = 'second_drawer'
        self.drawer_name_map['second_top'] = 'second_drawer_top'

    def reset(self):
        self.drawer_never_opened = True
        offset_coeff = (-1)**(1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.01, 0.0, -0.01])
        self.reached_pushing_region = False

    def get_action(self):
        if 'main' in self.close_drawer_name:
            env_get_drawer_handle_pos = self.env.get_drawer_handle_pos()
            if self.close_drawer_name == 'main_drawer':
                env_is_drawer_closed = self.env.is_drawer_closed()
            else:
                env_is_drawer_closed = self.env.is_top_drawer_closed()
            env_is_left_opening = self.env.left_opening
        elif 'second' in self.close_drawer_name:
            env_get_drawer_handle_pos = self.env.get_second_drawer_handle_pos()
            if self.close_drawer_name == 'second_drawer':
                env_is_drawer_closed = self.env.is_second_drawer_closed()
            else:
                env_is_drawer_closed = self.env.is_second_top_drawer_closed()
            env_is_left_opening = self.env.second_drawer_left_opening

        ee_pos, _ = bullet.get_link_state(self.env.robot_id,
                                          self.env.end_effector_index)
        handle_pos = env_get_drawer_handle_pos + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        top_drawer_pos = self.env.get_drawer_pos(
            self.drawer_name_map[self.close_drawer_name])

        if self.close_drawer_name == 'main_top':
            top_drawer_push_target_pos = (top_drawer_pos +
                                          np.array([0.15, 0, 0.05]))
        elif self.close_drawer_name == 'second_top':
            top_drawer_push_target_pos = (top_drawer_pos +
                                          np.array([-0.12, 0, 0.05]))
        elif self.close_drawer_name == 'main_drawer':
            top_drawer_push_target_pos = (top_drawer_pos +
                                          np.array([0.05, 0, 0.07]))
        elif self.close_drawer_name == 'second_drawer':
            top_drawer_push_target_pos = (top_drawer_pos +
                                          np.array([-0.05, 0, 0.05]))

        if 'main' in self.close_drawer_name:
            is_gripper_ready_to_push = (
                ee_pos[0] > top_drawer_push_target_pos[0]
                and ee_pos[2] < top_drawer_push_target_pos[2])
        else:
            is_gripper_ready_to_push = (
                ee_pos[0] < top_drawer_push_target_pos[0]
                and ee_pos[2] < top_drawer_push_target_pos[2])

        done = False
        neutral_action = [0.0]
        if (not env_is_drawer_closed and not self.reached_pushing_region
                and not is_gripper_ready_to_push):
            print("move up and left", ee_pos, env_get_drawer_handle_pos,
                  ee_pos[0] < top_drawer_push_target_pos[0],
                  ee_pos[2] < top_drawer_push_target_pos[2], top_drawer_pos)
            if self.close_drawer_name == 'main_top':
                action_xyz = [0.3, -0.2, -0.15]
            elif self.close_drawer_name == 'second_top':
                action_xyz = [-0.05, 0.2, 0.05]
            elif self.close_drawer_name == 'main_drawer':
                action_xyz = [0.02, 0.2, -0.05]
            elif self.close_drawer_name == 'second_drawer':
                action_xyz = [-0.04, 0.2, -0.05]

            if self.suboptimal:
                """Occasionally flip the action."""
                if np.random.uniform() > 0.3:
                    action_xyz[0] = action_xyz[0] * float(
                        np.random.uniform() > 0.5)
                    action_xyz[1] = action_xyz[1] * float(
                        np.random.uniform() > 0.5)

            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not env_is_drawer_closed:
            print("close drawer")
            self.reached_pushing_region = True
            action_xyz = (top_drawer_pos + self.top_drawer_offset -
                          ee_pos) * 7.0
            action_xyz[0] *= 3
            action_xyz[1] *= 0.6

            if self.suboptimal:
                """Occasionally flip the action"""
                if np.random.uniform() > 0.3:
                    action_xyz = -1.0 * action_xyz

            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            print('Done with the action')
            if 'main' in self.close_drawer_name:
                action_xyz = [0.05, 0.2, 0.05]
            else:
                action_xyz = [-0.05, 0.2, 0.05]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class MultiDrawerCloseSuboptimal(MultiDrawerClose):

    def __init__(self, env, **kwargs):
        super(MultiDrawerCloseSuboptimal, self).__init__(
            env,
            suboptimal=True,
            **kwargs,
        )
