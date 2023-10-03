# Start here

`python roboverse/envs/widow250.py`

## Scripted data collection

`python scripts/scripted_collect_parallel.py -n 5000 -p 20 -t 25 -e Widow250PickPlace-v0 -pl grasp -a grasp_success -d testing`

`python scripts/scripted_collect.py -n 100 -t 25 -e Widow250PickPlace-v0 -pl grasp -a grasp_success --gui`

## Pulling all submodules (like bullet-objects)

First time: `git submodule update --init --recursive`

Subsequent updates: `git submodule update --recursive --remote`