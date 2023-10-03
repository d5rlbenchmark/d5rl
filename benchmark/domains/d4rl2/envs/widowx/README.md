# Manipulation Stitching Environments

## Scripted data collection

see `scripts/generate_widowx_datasets.py` for all commands to run scripted collection. After saving all data, run the following script to get the datasets.

`python scripts/generate_widowx_datasets.py --dataset_save_path=<where you want to save data> --dataset_type=<dataset_type>`

where `dataset_type` takes one of four values: `adversarial_stitch+expert`, `adversarial_stitch`, `stitch`, `stitch+expert`. More datasets would be added.


The four environments are mentioned in the `__init__.py` file.


## Pulling all submodules (like bullet-objects)

First time: `git submodule update --init --recursive`

Subsequent updates: `git submodule update --recursive --remote`
