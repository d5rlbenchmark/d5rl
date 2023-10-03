import numpy as np

CAMERAS = {
    0: dict(distance=2.1, lookat=[-0.4, 0.5, 2.0], azimuth=70,
            elevation=-37.5),
    1: dict(distance=2.2,
            lookat=[-0.2, 0.75, 2.0],
            azimuth=150,
            elevation=-30.0),
    2: dict(distance=4.5, azimuth=-66, elevation=-65),
    3: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            ),  # original, as in https://relay-policy-learning.github.io/
    4: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70,
            elevation=-50),  # angled up to get a more top-down view
    5: dict(distance=2.65, lookat=[0, 0, 2.0], azimuth=90, elevation=-60
            ),  # similar to appendix D of https://arxiv.org/pdf/1910.11956.pdf
    6: dict(distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-60
            ),  # 3-6 are first person views at different angles and distances
    7: dict(
        distance=2.5, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-45
    ),  # problem w/ POV is that the knobs can be hidden by the hinge drawer and arm
    8: dict(distance=2.9, lookat=[-0.05, 0.5, 2.0], azimuth=90, elevation=-50),
    9: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90,
            elevation=-50),  # move back so less of cabinets
    10: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-35),
    11: dict(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=90, elevation=-10)
}

# from https://github.com/rail-berkeley/d4rl/blob/9b1eeab56d2adc5afa0541b6dd95399de9635390/d4rl/kitchen/kitchen_envs.py#L9
ORIG_OBS_ELEMENT_INDICES = {
    # oven_chain_orig
    'bottomknob': np.array([11, 12]),  # bottom burner
    'topknob': np.array([15, 16]),  # top burner
    #
    'bottomknobr': np.array([9, 10]),  # bottom burner, right
    'topknobr': np.array([13, 14]),  # top burner, right
    #
    'switch': np.array([17, 18]),  # light switch
    'slide': np.array([19]),  # slide cabinet
    'hinge': np.array([20, 21]),  # hinge cabinet
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25]),  # kettle, position on stove only
    # 'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_INDICES = {
    # order matters, flip b/c original had knob body before burner body
    'bottomknob': np.array([14, 10]),  # bottom burner, left
    'topknob': np.array([16, 12]),  # top burner, right
    #
    'bottomknobr': np.array([13, 9]),  # bottom burner, right
    'topknobr': np.array([15, 11]),  # top burner, right
    #
    'switch': np.array([17, 18]),  # light switch
    'slide': np.array([19]),  # slide cabinet
    'hinge': np.array([20, 21]),  # hinge cabinet
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25]),  # kettle, position on stove only
    'kettle_rot': np.array([26, 27, 28, 29]),
    # 'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    'bottomknob': np.array([-0.88, -0.01]),
    'topknob': np.array([-0.92, -0.01]),
    'switch': np.array([-0.69, -0.05]),
    'slide': np.array([0.37]),
    'hinge': np.array([0.0, 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62]),
    # 'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    'bottomknobr': np.array([-0.88, -0.01]),
    'topknobr': np.array([-0.92, -0.01]),
}

# fmt: off
ORIG_FRANKA_INIT_QPOS = np.array([
    1.48388023e-01, -1.76848573e+00, 1.84390296e+00, -2.47685760e+00,
    2.60252026e-01, 7.12533105e-01, 1.59515394e+00, 4.79267505e-02,
    3.71350919e-02, -2.66279850e-04, -5.18043486e-05, 3.12877220e-05,
    -4.51199853e-05, -3.90842156e-06, -4.22629655e-05, 6.28065475e-05,
    4.04984708e-05, 4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
    -6.44129196e-03, -1.77048263e-03, 1.08009684e-03, -2.69397440e-01,
    3.50383255e-01, 1.61944683e+00, 1.00618764e+00, 4.06395120e-03,
    -6.62095997e-03, -2.68278933e-04
])
FRANKA_INIT_QPOS = np.array([
    1.48388023e-01, -1.76848573e+00, 1.84390296e+00, -2.47685760e+00,
    2.60252026e-01, 7.12533105e-01, 1.59515394e+00, 4.79267505e-02,
    3.71350919e-02, -4.51199853e-05, -4.51199853e-05, 3.12877220e-05,
    6.28065475e-05, -4.51199853e-05, 3.12877220e-05, -4.51199853e-05,
    6.28065475e-05, 4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
    -6.44129196e-03, -1.77048263e-03, 1.08009684e-03, -2.69397440e-01,
    3.50383255e-01, 1.61944683e+00, 1.00618764e+00, 4.06395120e-03,
    -6.62095997e-03, -2.68278933e-04
])
# fmt: on

INIT_QPOS = {
    'franka':
    np.array([
        1.48388023e-01,
        -1.76848573e00,
        1.84390296e00,
        -2.47685760e00,
        2.60252026e-01,
        7.12533105e-01,
        1.59515394e00,
        4.79267505e-02,
        3.71350919e-02,
    ]),
    'bottomknob':
    np.array([3.12877220e-05, -4.51199853e-05]),
    'topknob':
    np.array([6.28065475e-05, 4.04984708e-05]),
    'bottomknobr':
    np.array([-2.66279850e-04, -5.18043486e-05]),
    'topknobr':
    np.array([-3.90842156e-06, -4.22629655e-05]),
    'switch':
    np.array([0.00046273, -0.00022691]),
    'slide':
    np.array([-0.0004655]),
    'hinge':
    np.array([-0.00644129, -0.00177048]),
    'microwave':
    np.array([0.0010801]),
    'kettle':
    np.array([
        -2.69397440e-01,
        3.50383255e-01,
        1.61944683e00,
        1.00618764e00,
        4.06395120e-03,
        -6.62095997e-03,
        -2.68278933e-04,
    ]),
}

# see Appendix B of RPL
#BONUS_THRESH = 0.3

BONUS_THRESH = {
    'bottomknob': 0.3,
    'topknob': 0.3,
    'bottomknobr': 0.3,
    'topknobr': 0.3,
    'switch': 0.3,
    'slide': 0.1,
    'hinge': 0.3,
    'microwave': 0.2,
    'kettle': 0.1
}

DEMO_OBJ_ORDER = [
    'microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge'
]
