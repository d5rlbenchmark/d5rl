#blender --background --python myscript.py
#blender -b --python-console
#blender --background --python convert.py -- test.wrl test.obj

import bpy
import sys

#obj_in = '/home/jonathan/Desktop/Projects/objects/6a0c8c43b13693fa46eb89c2e933753d.obj'
#stl_out = '/home/jonathan/Desktop/Projects/objects/out.stl'

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]
obj_out = argv[1]

objs = bpy.data.objects
objs.remove(objs["Cube"], True)
bpy.ops.import_scene.x3d(filepath=stl_in, axis_forward='-Z', axis_up='Y')
bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y', use_selection=True)
