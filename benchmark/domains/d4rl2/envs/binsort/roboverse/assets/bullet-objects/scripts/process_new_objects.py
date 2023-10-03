import sys
import os
import os.path as osp
import importlib.util

SHAPENET_ASSET_PATH = "../ShapeNetCore/"
SHAPENET_IN_DIR = "/home/albert/mice-bucket/albert/ShapeNetCore_v2/ShapeNetCore.v2"
SHAPENET_OUT_DIR = "/home/albert/dev/ShapeNet_novel/ShapeNetCore.v2"
VHACD_OUT_DIR = "/home/albert/dev/ShapeNet_novel/ShapeNetCore_vhacd" # the one corresponding to SHAPENET_OUT_DIR
# SHAPENET_GOOD_OUT_DIR = "/home/albert/dev/bullet-objects/ShapeNetCore/ShapeNetCore.v2"
# VHACD_GOOD_OUT_DIR = "/home/albert/dev/bullet-objects/ShapeNetCore/ShapeNetCore_vhacd"

def import_shapenet_metadata():
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", osp.join(SHAPENET_ASSET_PATH, "metadata.py"))
    shapenet_metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(shapenet_metadata)
    return shapenet_metadata.obj_path_map, shapenet_metadata.path_scaling_map

def output_novel_obj_ids(all_obj_ids_f_in, novel_obj_ids_path, curr_obj_ids):
    print("len(curr_object_ids)", len(curr_obj_ids))
    with open(all_obj_ids_f_in, "r") as f_in, \
          open(novel_obj_ids_path, "w") as f_out:
        for line in f_in:
            obj_id = line.strip()
            if obj_id not in curr_obj_ids:
                f_out.write(obj_id + "\n")

def copy_novel_objs_to_folder(novel_obj_ids_path):
    with open(novel_obj_ids_path, "r") as f:
        for line in f:
            if len(line) <= 1:
                continue
            print("line", line)
            class_id, obj_id = line.strip().split("/")
            print("line", line)
            os.system("mkdir {}/{}".format(SHAPENET_OUT_DIR, class_id))

            src_dir = "{}/{}/{}".format(SHAPENET_IN_DIR, class_id, obj_id)
            dest_dir = "{}/{}".format(SHAPENET_OUT_DIR, class_id)
            copy_command = "cp -r {} {}".format(src_dir, dest_dir)
            print(copy_command)
            os.system(copy_command)

def copy_good_objs_and_vhacd(annotated_novel_obj_ids_path, good_obj_ids_path):
    with open(annotated_novel_obj_ids_path, "r") as f_in,\
        open(good_obj_ids_path, "w") as f_out:
        for line in f_in:
            if len(line) <= 1 or not ("good" in line):
                continue
            print("line", line)
            class_obj_id, annotation = line.strip().split(",")
            class_id, obj_id = class_obj_id.split("/")

            # remove [good] tags from annotation. Write line to fout.
            annotation = annotation[:annotation.index("[")].strip()
            f_out.write(class_obj_id + "," + annotation + "\n")

            os.system("mkdir {}/{}".format(SHAPENET_GOOD_OUT_DIR, class_id))
            src_shapenet_dir = "{}/{}/{}".format(
                SHAPENET_OUT_DIR, class_id, obj_id)
            dest_shapenet_dir = "{}/{}".format(
                SHAPENET_GOOD_OUT_DIR, class_id)
            copy_shapenet_command = "cp -r {} {}".format(
                src_shapenet_dir, dest_shapenet_dir)
            print(copy_shapenet_command)

            os.system("mkdir {}/{}".format(VHACD_GOOD_OUT_DIR, class_id))
            src_vhacd_dir = "{}/{}/{}".format(VHACD_OUT_DIR, class_id, obj_id)
            dest_vhacd_dir = "{}/{}".format(VHACD_GOOD_OUT_DIR, class_id)
            copy_vhacd_command = "cp -r {} {}".format(src_vhacd_dir, dest_vhacd_dir)
            print(copy_vhacd_command)

            os.system(copy_shapenet_command)
            os.system(copy_vhacd_command)

if __name__ == "__main__":
    obj_ids_path = "obj_candidate_ids.txt"
    novel_obj_ids_path = "novel_obj_ids.txt"
    annotated_novel_obj_ids_path = "novel_obj_ids_annotated.txt"
    good_obj_ids_path = "good_obj_ids.txt"
    # _, path_scaling_map = import_shapenet_metadata()
    # output_novel_obj_ids(obj_ids_path, novel_obj_ids_path, list(path_scaling_map.keys()))
    # copy_novel_objs_to_folder(novel_obj_ids_path)
    copy_good_objs_and_vhacd(annotated_novel_obj_ids_path, good_obj_ids_path)

