import argparse
import sys
import imp
from jaxrl2.utils.general_utils import AttrDict

from examples.train_offline_pixels_widowx import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,
                        help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=20000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--offline_finetuning_start', default=-1, help='Number of training steps after which to start training online.', type=int)
    parser.add_argument('--online_start', default=int(2e5), help='Number of training steps after which to start training online.', type=int)
    parser.add_argument('--max_steps', default=int(1e9), help='Number of training steps.', type=int)
    parser.add_argument('--tqdm', default=1, help='Use tqdm progress bar.', type=int)
    parser.add_argument('--save_video', action='store_true', help='Save videos during evaluation.')
    parser.add_argument('--use_negatives', action='store_true', help='Use negative_data')
    parser.add_argument('--reward_scale', default=11.0, help='Scale for the reward', type=float)
    parser.add_argument('--reward_shift', default=-1, help='Shift for the reward', type=float)
    parser.add_argument('--reward_type' ,default='final_one', help='reward type')
    parser.add_argument('--cql_alpha_offline_finetuning', default=-1.0, help='alpha for finetuning', type=float)
    parser.add_argument('--cql_alpha_online_finetuning', default=-1.0, help='alpha for finetuning', type=float)

    parser.add_argument('--obs_latency', default=0, help='Number of timesteps of observation latency', type=int)
    parser.add_argument('--frame_stack', default=1, help='Number of frames stacked', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--add_prev_actions', action='store_true', help='whether to add low-dim previous actions to the obervations')

    parser.add_argument('--dataset', default='ball_in_bowl', help='name of dataset')
    parser.add_argument("--multi_viewpoint", default=1, help="whether to use multiple camreas", type=int)

    parser.add_argument('--online_mixing_ratio', default=0.5,
                        help='fraction of batch composed of old data to be used, the remainder part is newly collected data', type=float)

    parser.add_argument('--target_mixing_ratio', default=0.9,
                        help='fraction of batch composed of bridge data, the remainder is target data',
                        type=float)
    parser.add_argument('--num_bridge_traj', default=-1,
                        help='num trajectories used for the target task',
                        type=int)
    parser.add_argument('--num_target_traj', default=-1,
                        help='num trajectories used for the target task',
                        type=int)
    parser.add_argument('--trajwise_alternating', default=1,
                        help='alternate between training and data collection after each trajectory', type=int)
    parser.add_argument('--restore_path',
                        default='',
                        help='folder inside $EXP where weights are stored')
    parser.add_argument('--only_add_success', action='store_true', help='only add successful traj to buffer')

    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')

    parser.add_argument('--from_states', action='store_true', help='only use states, no images')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--online_from_scratch', action='store_true', help='train online from scratch.')
    parser.add_argument("--use_terminals", default=1, help="whether to use terminals", type=int)

    parser.add_argument('--stochastic_data_collect', default=1, help='sample from stochastic policy for data collection.', type=int)

    parser.add_argument('--algorithm', default='iql', help='type of algorithm')

    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--config', default='examples/configs/offline_pixels_default_real.py', help='File path to the training hyperparameter configuration.')
    parser.add_argument("--azure", action='store_true', help="run on azure")
    parser.add_argument("--offline_only", default=1, help="whether to only perform offline training", type=int)
    parser.add_argument("--eval_only", action='store_true', help="perform evals only")
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step', type=int)

    #environment
    parser.add_argument('--episode_timelimit', default=40, help='prefix to use', type=int)
    parser.add_argument('--save_replay_buffer', action='store_true', help='whether to save the repaly buffer')
    parser.add_argument('--reward_func_type', type=int, default=0, help='type of reward function')
    parser.add_argument('--rew_func_for_target_only', type=int, default=0, help='type of reward function')
    parser.add_argument('--bridge_clip_traj_length', type=int, default=-1, help='clip traj length')
    
    parser.add_argument('--num_demo', type=int, default=5, help='read num demo for prepared numpy files like out5.npy')
    parser.add_argument('--tpu_port', type=int, default=8476, help='define a unique port num for each tpu process to run multipule jobs in single instance')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--online_pos_neg_ratio', type=float, default=-1, help='online_pos_neg_ratio')
    parser.add_argument('--offline_only_restore_onlinedata', type=str, default=None, help='offline_only_restore_onlinedata')
    parser.add_argument('--num_final_reward_steps', default=1, help='number of final reward timesteps', type=int)
    parser.add_argument('--reset_critic', default=None, help='whole / decoder', type=str, choices=[None, 'whole', 'decoder'])

    parser.add_argument('--online_cql_alpha_sep_offline', default=-1.0, help='alpha for offline data during online finetuning', type=float)
    parser.add_argument('--online_cql_alpha_sep_online', default=-1.0, help='alpha for online data during online finetuning', type=float)
    parser.add_argument('--alpha_schedule_interval', default=-1.0, help='alpha_schedule_interval', type=int)
    parser.add_argument('--alpha_schedule_ratio', default=1.0, help='alpha_schedule_ratio', type=float)
    parser.add_argument('--online_cql_alpha_sep_offline_bridge', default=-1.0, help='alpha for offline bridge data during online finetuning', type=float)
    parser.add_argument('--online_cql_alpha_sep_offline_target', default=-1.0, help='alpha for offline target data during online finetuning', type=float)
    parser.add_argument('--posneg_schedule_interval', default=-1.0, help='online pos neg ratio schedule _interval', type=int)
    parser.add_argument('--posneg_schedule_ratio', default=1.0, help='online pos neg ratio schedule _ratio', type=float)
    parser.add_argument('--freeze_encoders_actor', action='store_true', help='freeze encoder or not')
    parser.add_argument('--freeze_encoders_critic', action='store_true', help='freeze encoder or not')
    parser.add_argument('--pretrained_encoder', default=None, help='Path to pretrained encoder to initialize', type=str)
    parser.add_argument('--encoder_key', default='agent/value/params/encoder', help='Key for encoder params in pretrained encder', type=str)
    parser.add_argument('--utd_schedule_interval', default=-1.0, help='alpha_schedule_interval', type=int)
    parser.add_argument('--utd_schedule_ratio', default=1.0, help='alpha_schedule_ratio', type=float)

    parser.add_argument('--override_critic_lr', default=-1.0, help='online pos neg ratio schedule _ratio', type=float)
    parser.add_argument('--rescale_critic_last_layer_ratio', default=-1, help='rescale_critic_last_layer_ratio', type=float)
    parser.add_argument('--wait_actor_update', default=-1, help='wait actor update', type=int)
    parser.add_argument('--bound_q_with_mc', default=0, help='wait actor update', type=int)
    parser.add_argument('--num_online_gradsteps_batch', default=-1, help='take a certain gradstep in batch setting', type=int)
    parser.add_argument('--keep_mc_bound_online', action='store_true', help='keep mc bound in online phase or not')
    parser.add_argument('--online_bound_nstep_return', default=-1, help='use Nstep return instead of MC return to go for lowerbound during online finetuning', type=int)


    # algorithm args:
    train_args_dict = dict(
        latent_dim = 50,
    )
    
    variant, args = parse_training_args(train_args_dict, parser)

    main(variant)