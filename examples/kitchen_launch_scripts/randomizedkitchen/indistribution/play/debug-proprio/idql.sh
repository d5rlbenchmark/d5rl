cd ./examples
conda activate d5rl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export KITCHEN_DATASETS=/PATH/TO/datasets/randomized_kitchen
export STANDARD_KITCHEN_DATASETS=/PATH/TO/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz
export RELAY_POLICY_REPO="./benchmark/domains/relay-policy-learning/adept_envs"


XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_randomizedkitchen.py \
--task "randomizedkitchen_indistribution-play_data" \
--tqdm=true \
--project bench_randomizedkitchen_debug3 \
--algorithm idql \
--proprio=true \
--description proprio \
--eval_episodes 100 \
--eval_interval 50000 \
--online_eval_interval 50000 \
--log_interval 1000 \
--max_gradient_steps 500_000 \
--max_online_gradient_steps 500_000 \
--replay_buffer_size 400_000 \
--batch_size 256 \
--im_size 64 \
--use_wrist_cam=false \
--camera_ids "12" \
--seed 0 