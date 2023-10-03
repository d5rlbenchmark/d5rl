# 16 pairs -- can make 32 tasks out of it
PICK_PLACE_TRAIN_TASK_OBJECTS = [
    ['conic_cup', 'fountain_vase'],
    ['circular_table', 'hex_deep_bowl'],
    ['smushed_dumbbell', 'square_prism_bin'],
    ['narrow_tray', 'colunnade_top'],
    ['stalagcite_chunk', 'bongo_drum_bowl'],
    ['pacifier_vase', 'beehive_funnel'],
    ['crooked_lid_trash_can', 'toilet_bowl'],
    ['pepsi_bottle', 'tongue_chair'],
    ['modern_canoe', 'pear_ringed_vase'],
    ['short_handle_cup', 'bullet_vase'],
    ['glass_half_gallon', 'flat_bottom_sack_vase'],
    ['trapezoidal_bin', 'vintage_canoe'],
    ['bathtub', 'flowery_half_donut'],
    ['t_cup', 'cookie_circular_lidless_tin'],
    ['box_sofa', 'two_layered_lampshade'],
    ['conic_bin', 'jar'],
    #   'aero_cylinder',
]

# 8 objects, just doubled it to match length of object pairs
PICK_PLACE_TRAIN_TASK_CONTAINERS = [
    'plate',
    'cube_concave',
    'table_top',
    'bowl_small',
    'tray',
    'long_open_box',
    'cube',
    'torus',
    'plate',
    'cube_concave',
    'table_top',
    'bowl_small',
    'tray',
    'long_open_box',
    'cube',
    'torus',
]

# 4 pairs -- can make 8 pick + place tasks out of it
PICK_PLACE_TEST_TASK_OBJECTS = [
    ['square_rod_embellishment', 'grill_trash_can'],
    ['shed', 'sack_vase'],
    ['two_handled_vase', 'thick_wood_chair'],
    ['curved_handle_cup', 'baseball_cap'],
    # 'elliptical_capsule',
]

PICK_PLACE_TEST_TASK_CONTAINERS = [
    'pan_tefal',
    'marble_cube',
    'basket',
    'checkerboard_table',
]

# Repeating this to match length 8
PICK_PLACE_TEST_TASK_OBJECTS_REPEATED = [
    ['square_rod_embellishment', 'grill_trash_can'],
    ['shed', 'sack_vase'],
    ['two_handled_vase', 'thick_wood_chair'],
    ['curved_handle_cup', 'baseball_cap'],
    # REPEAT
    ['square_rod_embellishment', 'grill_trash_can'],
    ['shed', 'sack_vase'],
    ['two_handled_vase', 'thick_wood_chair'],
    ['curved_handle_cup', 'baseball_cap'],
]

PICK_PLACE_TEST_TASK_CONTAINERS_REPEATED = [
    'pan_tefal',
    'marble_cube',
    'basket',
    'checkerboard_table',
    # REPEAT
    'pan_tefal',
    'marble_cube',
    'basket',
    'checkerboard_table',
]
