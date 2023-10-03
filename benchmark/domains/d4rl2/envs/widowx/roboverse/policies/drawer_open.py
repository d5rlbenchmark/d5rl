import numpy as np

import d4rl2.envs.widowx.roboverse.bullet as bullet


class DrawerOpen:

    def __init__(self, env):
        self.env = env
        self.xyz_action_scale = 7.0
        self.gripper_dist_thresh = 0.06
        self.gripper_xy_dist_thresh = 0.04
        self.ending_height_thresh = 0.2
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.drawer_never_opened = True
        offset_coeff = (-1)**(1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.005, 0.0, -0.01])

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(self.env.robot_id,
                                          self.env.end_effector_index)
        handle_pos = self.env.get_drawer_handle_pos() + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        # print("gripper_handle_dist", gripper_handle_dist)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        done = False

        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not self.env.is_drawer_open()):
            print('xy - approaching handle')
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (gripper_handle_dist > self.gripper_dist_thresh
              and not self.env.is_drawer_open()):
            print("moving down toward handle")
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not self.env.is_drawer_open():
            print("opening drawer")
            x_command = (-1)**(1 - self.env.left_opening)
            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (np.abs(ee_pos[2] - self.ending_height_thresh) >
              self.gripper_dist_thresh):
            print("Lift upward")
            self.drawer_never_opened = False
            action_xyz = np.array([0, 0, 0.7])  # force upward action
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            print("done with drawer open")
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        agent_info = dict(done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return action, agent_info


class MultiDrawerOpen:
    """Scripted policy to open one of two drawers."""

    def __init__(self, env, open_drawer_name='main', suboptimal=False):
        self.env = env
        self.xyz_action_scale = 7.0
        self.suboptimal = suboptimal
        if open_drawer_name == 'main':
            # Do not change this
            self.gripper_dist_thresh = 0.04
            self.gripper_xy_dist_thresh = 0.02
            self.ending_height_thresh = 0.2
        else:
            self.gripper_dist_thresh = 0.05
            self.gripper_xy_dist_thresh = 0.04
            self.ending_height_thresh = 0.2

        self.open_drawer_name = open_drawer_name
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.drawer_never_opened = True
        offset_coeff = (-1)**(1 - self.env.left_opening)
        self.handle_offset = np.array([offset_coeff * 0.005, 0.0, -0.01])

    def get_action(self):
        if self.open_drawer_name == 'main':
            env_get_drawer_handle_pos = self.env.get_drawer_handle_pos()
            env_is_drawer_open = self.env.is_drawer_open()
            env_is_left_opening = self.env.left_opening
        else:
            # import pdb; pdb.set_trace()
            env_get_drawer_handle_pos = self.env.get_second_drawer_handle_pos()
            env_is_drawer_open = self.env.is_second_drawer_open()
            env_is_left_opening = self.env.second_drawer_left_opening

        ee_pos, _ = bullet.get_link_state(self.env.robot_id,
                                          self.env.end_effector_index)
        handle_pos = env_get_drawer_handle_pos + self.handle_offset
        gripper_handle_dist = np.linalg.norm(handle_pos - ee_pos)
        # print("gripper_handle_dist", gripper_handle_dist)
        gripper_handle_xy_dist = np.linalg.norm(handle_pos[:2] - ee_pos[:2])
        done = False

        if (gripper_handle_xy_dist > self.gripper_xy_dist_thresh
                and not env_is_drawer_open):
            print('xy - approaching handle', gripper_handle_xy_dist)
            action_xyz = (handle_pos - ee_pos) * 7.0
            action_xyz = list(action_xyz[:2]) + [0.]  # don't droop down.
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (gripper_handle_dist > self.gripper_dist_thresh
              and not env_is_drawer_open):
            print("moving down toward handle", gripper_handle_dist,
                  env_is_drawer_open, self.gripper_dist_thresh)
            action_xyz = (handle_pos - ee_pos) * 7.0
            if self.suboptimal:
                # if the policy is suboptimal, then occasionally flip the
                # actions
                if np.random.uniform() > 0.5:
                    action_xyz = -1 * action_xyz
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif not env_is_drawer_open:
            print("opening drawer")
            if self.open_drawer_name == 'main':
                x_command = (-1)**(1 - self.env.left_opening)
            else:
                x_command = (-1)**(self.env.left_opening)

            if self.suboptimal:
                if np.random.uniform() > 0.5:
                    # close the drawer sometimes randomly
                    # since this policy is suboptimal
                    x_command = -x_command

            action_xyz = np.array([x_command, 0, 0])
            # action = np.asarray([0., 0., 0.7])
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif (np.abs(ee_pos[2] - self.ending_height_thresh) >
              self.gripper_dist_thresh):
            print("Lift upward")
            self.drawer_never_opened = False
            action_xyz = np.array([0, 0, 0.7])  # force upward action
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        else:
            # print("done with drawer open")
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]

        agent_info = dict(done=done)
        # action = np.concatenate((action_xyz, action_angles, action_gripper))
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, [0]))
        return action, agent_info


class MultiDrawerOpenSuboptimal(MultiDrawerOpen):

    def __init__(self, env, **kwargs):
        super(MultiDrawerOpenSuboptimal, self).__init__(
            env,
            suboptimal=True,
            **kwargs,
        )
