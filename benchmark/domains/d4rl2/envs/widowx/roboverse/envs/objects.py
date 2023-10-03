import os

import pybullet as p
import pybullet_data

import d4rl2.envs.widowx.roboverse.bullet as bullet

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')
"""
NOTE: Use this file only for core objects, add others to bullet/object_utils.py
This file will likely be deprecated in the future.
"""


def table():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    table_id = p.loadURDF('table/table.urdf',
                          basePosition=[.75, -.2, -1],
                          baseOrientation=[0, 0, 0.707107, 0.707107],
                          globalScaling=1.0)
    return table_id


def tray(base_position=(.60, 0.3, -.37), scale=0.5):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    tray_id = p.loadURDF('tray/tray.urdf',
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id


def tray_no_divider(base_position=(.60, 0.25, -.37), scale=0.6):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
    tray_id = p.loadURDF(os.path.join(BASE_ASSET_PATH,
                                      'tray/tray_no_divider.urdf'),
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id


def tray_no_divider_scaled(base_position=(.60, 0.25, -.37), scale=0.6):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
    tray_id = p.loadURDF(os.path.join(BASE_ASSET_PATH,
                                      'tray/tray_no_divider_scaled.urdf'),
                         basePosition=base_position,
                         baseOrientation=[0, 0, 0.707107, 0.707107],
                         globalScaling=scale)
    return tray_id


def widow250(scale=1.0):
    widow250_path = os.path.join(ASSET_PATH,
                                 'interbotix_descriptions/urdf/wx250s.urdf')
    widow250_id = p.loadURDF(widow250_path,
                             basePosition=[0.6, 0, -0.4],
                             baseOrientation=bullet.deg_to_quat([0., 0., 0]),
                             globalScaling=scale)
    return widow250_id
