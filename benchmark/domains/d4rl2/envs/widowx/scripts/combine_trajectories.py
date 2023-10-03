import argparse
import os
import os.path as osp

import numpy as np

NFS_PATH = '/nfs/kun2/users/aviralkumar/d4rl2_widowx_data_final/'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data-save-path", type=str)
args = parser.parse_args()

if osp.exists(NFS_PATH):
    parent_dir = osp.join(NFS_PATH, args.data_save_path)
else:
    parent_dir = osp.join(__file__, "../..", "data", args.data_save_path)

parent_dir = osp.abspath(parent_dir)

all_files = []
for root, dirs, files in os.walk(parent_dir):
    for f in files:
        f_path = os.path.join(root, f)
        print(f_path)
        data = np.load(f_path, allow_pickle=True)
        all_files.append(data)

all_data = np.concatenate(all_files, axis=0)

save_all_path = os.path.join(
    parent_dir, "{}_{}.npy".format(args.data_save_path, len(all_data)))
print(save_all_path)
np.save(save_all_path, all_data)
