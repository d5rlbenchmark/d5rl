import copy
import os

import gym
import numpy as np
# import xml.etree.ElementTree as ET
from lxml import etree as ET

from .constants import FRANKA_INIT_QPOS, OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES
from .kitchen_v1 import Kitchen_v1

PARAMS = dict(
    kettle_positions=dict(
        top_right_angled_randomized=[0.23, 0.75, 1.625],
        #bot_right_angled_randomized = [0.23, 0.3504, 1.625],
        bot_left_angled_randomized=[-0.2694, 0.3504, 1.625]),
    cabinet_textures=[
        'wood1', 'wood2', 'metal1', 'metal2', 'marble1', 'tile1'
    ],
    counter_textures=[
        'white_marble_tile2',
        'tile1',
        'wood1',
        'wood2',
    ],
    floor_textures=[
        'white_marble_tile',
        'marble1',
        'tile1',
        'wood1',
        'wood2',
        'checker',
    ])


class Randomized_Kitchen(Kitchen_v1):
    """
    Mujoco does not support runtime modification of the environment, it can only load from a
    static XML. We have to modify the XML and reload it for changes to propogate to the sim.

    see https://github.com/deepmind/dm_control/issues/54
    https://github.com/deepmind/dm_control/issues/125
    http://mujoco.org/forum/index.php?threads/programatically-creating-the-bodies.3805/#post-4940

    worldgen refs
    https://github.com/PSVL/DoorGym/blob/903493a6bd518bed5452c683c3857f1bfc5a209d/world_generator/world_generator.py
    https://github.com/ARISE-Initiative/robosuite/blob/2218845cc83fb43bb96ba274cea8fab45095adce/robosuite/wrappers/domain_randomization_wrapper.py
    https://github.com/matwilso/domrand/blob/c7cc6252656d88836037b86cafd8a88021f64ce4/domrand/sim_manager.py
    https://github.com/openai/orrb
    """
    TASK_ELEMENTS = ['microwave', 'kettle', 'bottomknob', 'switch', 'slide']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.defaults = dict(
            camera_ids=self.camera_ids,
            noise_ratio=self.noise_ratio,
            robot_cache_noise_ratio=self.robot_cache_noise_ratio,
            init_qpos=self.init_qpos,
            #
            init_random_steps_set=self.init_random_steps_set,
            init_perturb_robot_ratio=self.init_perturb_robot_ratio,
            init_perturb_object_ratio=self.init_perturb_object_ratio,
            rng_type=self.rng_type,
        )

        self.parser = ET.XMLParser(remove_blank_text=True,
                                   remove_comments=True)
        self.reset_domain_changes()

    def sample_domain_parameters(self):
        domain_parameters = []
        domain_parameters.append(
            ('change_microwave', self.np_random2.choice(4)))
        domain_parameters.append(('change_kettle', self.np_random2.choice(8)))

        microwave_params = dict(
            pos="{x} {y} 1.6".format(
                x=-0.800 + self.np_random2.normal() * 0.025,
                y=-0.100 + self.np_random2.normal() * 0.025),
            euler="0 0 {e}".format(e=0.1 + self.np_random2.uniform() * 0.4))
        domain_parameters.append(
            ('change_objects_layout', 'microwave', microwave_params))

        ovenhood_params = dict(pos="{x} 0.0 0.0".format(
            x=self.np_random2.normal() * 0.05))
        domain_parameters.append(
            ('change_objects_layout', 'ovenhood', ovenhood_params))

        kettle_position = self.np_random2.choice(
            list(PARAMS['kettle_positions'].keys()))
        kettle_params = dict(kettle_qpos=self.np_random2.normal(
            loc=PARAMS['kettle_positions'][kettle_position],
            scale=[0.025, 0.025, 0.00]),
                             kettle_qrot=self.np_random2.normal(
                                 loc=[-0.99, 0.0, 0.0, -0.5236],
                                 scale=[0.0, 0.0, 0.0, 0.15]))

        domain_parameters.append(
            ('change_objects_layout', 'kettle', kettle_params))

        cabinet_tecture = self.np_random2.choice(PARAMS['cabinet_textures'])
        domain_parameters.append(('change_hinge_texture', cabinet_tecture))
        domain_parameters.append(('change_slide_texture', cabinet_tecture))
        domain_parameters.append(
            ('change_counter_texture',
             self.np_random2.choice(PARAMS['counter_textures'])))
        light = np.zeros(2)
        light[0] = self.np_random2.uniform(-1, 1)
        light[1] = self.np_random2.uniform(0, 1)
        domain_parameters.append(('change_lighting', light))

        reset_qpos = FRANKA_INIT_QPOS.copy()
        init_robot_noise = 0.05 * self.np_random2.uniform(
            low=self.robot.robot_pos_bound[:self.N_DOF_ROBOT, 0],
            high=self.robot.robot_pos_bound[:self.N_DOF_ROBOT, 1],
        )
        reset_qpos[:self.N_DOF_ROBOT] += init_robot_noise
        reset_qpos[:self.N_DOF_ROBOT] = self.robot.enforce_position_limits(
            reset_qpos[:self.N_DOF_ROBOT])
        domain_parameters.append(
            ('change_robot_init_qpos', reset_qpos[:self.N_DOF_ROBOT]))

        return domain_parameters

    def change_microwave(self, microwave_id):
        n = self.worldbody.find('.//body[@name="microwave"]/include')
        n.attrib['file'] = n.attrib['file'].replace(
            'microwave_chain0.xml', f'microwave_chain{microwave_id}.xml')

    def change_kettle(self, kettle_id):
        n = self.worldbody.find('.//body[@name="kettle"]/include')
        n.attrib['file'] = n.attrib['file'].replace(
            'kettle_chain0.xml', f'kettle_chain{kettle_id}.xml')

    def change_hinge_texture(self, cabinet_texture):
        return self._change_cabinet_texture('hinge', cabinet_texture)

    def change_slide_texture(self, cabinet_texture):
        return self._change_cabinet_texture('slide', cabinet_texture)

    def _change_cabinet_texture(self, cabinet, cabinet_texture):
        if cabinet_texture not in [
                'wood1', 'wood2', 'metal1', 'metal2', 'marble1', 'tile1'
        ]:
            raise ValueError(f'Unsupported cabinet texture {cabinet_texture}.')

        fn = f'objects/{cabinet}cabinet_asset.xml'
        p = os.path.join(os.path.dirname(__file__), 'assets', fn)
        n = self.domain_model_xml_tree.find(f'include[@file="{fn}"]')
        # read in file, modify, then insert xml
        include = ET.fromstring(open(p, 'r').read(), parser=self.parser)
        i = self.domain_model_xml_tree.getchildren().index(n)
        self.domain_model_xml_tree.remove(n)

        t = include.find('asset/texture[@file="textures/wood1.png"]')
        m = include.find(
            f'default/default[@class="{cabinet}cabinet"]/geom[@material="M_{cabinet}_blue"]'
        )

        if cabinet_texture == 'metal1':  # texture already used for something else, so can just swap
            m.attrib['material'] = f'M_{cabinet}_metal'
        else:
            # wood1.png texture is defined as a placeholder and not used for anything else
            t.attrib['file'] = f'textures/{cabinet_texture}.png'
            m.attrib['material'] = f'M_{cabinet}_wood'

        for j, child in enumerate(include):
            self.domain_model_xml_tree.insert(i + j, child)

    def change_lighting(self, light):
        n = self.worldbody.findall('light')
        n[0].attrib['dir'] = '1 {y} -1'.format(y=light[0])
        n[1].attrib['dir'] = '-1 {y} -1'.format(y=light[1])

    def change_objects_layout(self, object, layout):

        if object == 'kettle':
            # kettle position is a task goal param, so we change it by modifying init_qpos
            kettle_qpos = layout['kettle_qpos']
            kettle_qrot = layout['kettle_qrot']
            if kettle_qpos is not None:
                self.changed['init_qpos'][
                    OBS_ELEMENT_INDICES['kettle']] = kettle_qpos
            if kettle_qrot is not None:
                self.changed['init_qpos'][
                    OBS_ELEMENT_INDICES['kettle_rot']] = kettle_qrot
        else:
            o = object
            if object == 'hinge' or object == 'slide':
                o += 'cabinet'

            n = self.worldbody.find(f'.//body[@name="{o}"]')
            for k, v in layout.items():
                n.attrib[k] = v

    def change_counter_texture(self, counter_texture):
        if counter_texture not in [
                'white_marble_tile2',
                'tile1',
                'wood1',
                'wood2',
        ]:
            raise ValueError(f'Unsupported counter texture {counter_texture}.')

        fn = 'objects/counters_asset.xml'
        p = os.path.join(os.path.dirname(__file__), 'assets', fn)
        n = self.domain_model_xml_tree.find(f'include[@file="{fn}"]')
        # read in file, modify, then insert xml
        include = ET.fromstring(open(p, 'r').read(), parser=self.parser)
        i = self.domain_model_xml_tree.getchildren().index(n)
        self.domain_model_xml_tree.remove(n)

        t = include.find('asset/texture[@name="T_counter_marble"]')
        t.attrib['file'] = f'textures/{counter_texture}.png'

        for j, child in enumerate(include):
            self.domain_model_xml_tree.insert(i + j, child)

    def change_floor_texture(self, floor_texture):
        if floor_texture not in [
                'white_marble_tile',
                'marble1',
                'tile1',
                'wood1',
                'wood2',
                'checker',
        ]:
            raise ValueError(f'Unsupported floor texture {floor_texture}.')

        n = self.domain_model_xml_tree.find('asset/texture[@name="texplane"]')
        if floor_texture == 'checker':
            pass
        else:
            n.attrib['file'] = f'textures/{floor_texture}.png'

    def spawn_object(self, object):
        if object == 'bin':
            pass
        else:
            raise ValueError

    def change_camera_params(self, POSES=None):
        if POSES is None:
            POSES = dict()
            for camera_id in self.cameras.keys():
                if "gripper" in camera_id:
                    continue
                else:
                    POSE = self.cameras[camera_id].get_pose()._asdict().copy()
                    POSE['distance'] += self.np_random2.normal(0.0, 0.05)
                    POSE['lookat'] += self.np_random2.normal(0.0, 0.05, size=3)
                    POSE['azimuth'] += self.np_random2.normal(0.0, 1.0)
                    POSE['elevation'] += self.np_random2.normal(0.0, 1.0)
                    self.cameras[camera_id].set_pose(**POSE)
                    POSES[camera_id] = POSE.copy()
        else:
            for camera_id in POSES:
                self.cameras[camera_id].set_pose(**POSES[camera_id])

        return POSES

    def change_noise_ratio(self, noise_ratio):
        self.changed['noise_ratio'] = noise_ratio

    def change_init_noise_params(
        self,
        init_random_steps_set,
        init_perturb_robot_ratio,
        init_perturb_object_ratio,
        rng_type,
    ):
        self.init_random_steps_set = init_random_steps_set
        self.init_perturb_robot_ratio = init_perturb_robot_ratio
        self.init_perturb_object_ratio = init_perturb_object_ratio
        self.rng_type = rng_type

    def change_robot_init_qpos(self, init_qpos):
        self.changed['init_qpos'][:self.N_DOF_ROBOT] = np.array(init_qpos)

    def change_robot(self):
        pass

    def change_objects_done(self, objects):
        self.objects_done_set += objects

    def change_object_done(self, object):
        self.change_objects_done([object])

    def change_force_sensor_noise(self, noise_type):
        if noise_type == 'zeroed':
            pass

        elif noise_type == 'gaussian0.1':
            pass

        else:
            raise ValueError(f'Unsupported force sensor noise {noise_type}.')

    def reset_domain_changes(self):
        # returns Element rather than ElementTree like ET.parse, so don't need to getroot()
        self.domain_model_xml_tree = ET.fromstring(self.model_xml,
                                                   parser=self.parser)
        self.worldbody = self.domain_model_xml_tree.find('worldbody')

        self.changed = copy.deepcopy(self.defaults)

        # NOTE: this will override init kettle position
        self.set_init_qpos(FRANKA_INIT_QPOS.copy())
        self.objects_done_set = []

    def domain_model_xml(self):
        domain_model_xml = ET.tostring(
            self.domain_model_xml_tree,
            encoding='utf8',
            method='xml',
            pretty_print=True,
        ).decode('utf8')
        return domain_model_xml

    def reset(self, randomize_model=True, domain=None, **kwargs):
        if domain is not None:
            self.reset_domain_changes()
            domain_parameters = domain['environment']
            for p in domain_parameters:
                fn = getattr(self, p[0])
                fn(*p[1:])
            domain_model_xml = self.domain_model_xml()
            self.set_init_qpos(self.changed['init_qpos'])
            self.load_sim(domain_model_xml)
            _ = self.change_camera_params(domain['cameras'])
            self.domain = domain

        elif randomize_model:
            self.reset_domain_changes()
            domain_parameters = self.sample_domain_parameters()
            for p in domain_parameters:
                fn = getattr(self, p[0])
                fn(*p[1:])

            domain_model_xml = self.domain_model_xml()
            self.set_init_qpos(self.changed['init_qpos'])
            self.load_sim(domain_model_xml)
            POSES = self.change_camera_params()
            self.domain = dict()
            self.domain['environment'] = domain_parameters
            self.domain['cameras'] = POSES

        objects_done_set = self.objects_done_set if len(
            self.objects_done_set) > 0 else None
        return super().reset(objects_done_set=objects_done_set, **kwargs)
