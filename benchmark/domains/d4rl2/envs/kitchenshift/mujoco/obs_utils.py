import numpy as np
from dm_control.mujoco import engine

from .rotations import euler2quat, mat2euler, mat2quat, quat2euler, quat_mul


def get_obs_ee(
        sim,
        rot_use_euler=False):  # should in in global coordinates, world frame
    i = sim.model.site_name2id('end_effector')
    pos = sim.data.site_xpos[i, ...].copy()
    xmat = sim.data.site_xmat[i, ...].copy()
    xmat = xmat.reshape(3, 3)
    if rot_use_euler:
        rot = mat2euler(xmat)
    else:
        rot = mat2quat(xmat)

    # i = self.sim.model.body_name2id('panda0_link7')
    # pos = self.sim.data.body_xpos[i, ...]
    # if rot_use_euler:
    #     xmat = self.sim.data.body_xmat[i, ...]
    #     xmat = xmat.reshape(3, 3)
    #     rot = mat2euler(xmat)
    # else:
    #     rot = self.sim.data.xquat[i, ...]
    return np.concatenate([pos, rot])


def get_obs_forces(sim):
    # explicitly compute contacts since mujoco2.0 doesn't call this
    # see https://github.com/openai/gym/issues/1541
    engine.mjlib.mj_rnePostConstraint(sim.model.ptr, sim.data.ptr)
    finger_id_left = sim.model.body_name2id('panda0_leftfinger')
    finger_id_right = sim.model.body_name2id('panda0_rightfinger')

    l = sim.data.cfrc_ext[finger_id_left, ...].copy()
    r = sim.data.cfrc_ext[finger_id_right, ...].copy()
    return np.concatenate([l, r])
