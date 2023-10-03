import os

import copy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import jax.numpy as jnp
import wandb
from jaxrl2.evaluation import evaluate
import collections
from jaxrl2.utils.visualization_utils import visualize_image_actions
from jaxrl2.utils.visualization_utils import visualize_states_rewards, visualize_image_rewards
from jaxrl2.utils.visualization_utils import sigmoid
from jaxrl2.data.dataset import PropertyReplayBuffer, MixingReplayBuffer
import gc;

def offline_training_loop(variant, agent, eval_env, replay_buffer, eval_replay_buffer=None, wandb_logger=None, perform_control_evals=True, task_id_mapping=None):
    if eval_replay_buffer is None:
        eval_replay_buffer = replay_buffer
    
    if variant.offline_finetuning_start != -1:
        changed_buffer_to_finetuning = False
        if isinstance(replay_buffer, MixingReplayBuffer):
            replay_buffer.set_mixing_ratio(1) #offline only    
            print("set target ratio to 1 for offline pretraining")
    
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if eval_replay_buffer is not None:
        eval_replay_buffer_iterator = eval_replay_buffer.get_iterator(variant.batch_size)

    # perform eval for initial checkpoint
    if hasattr(agent, 'unreplicate'):
        agent.unreplicate()
    print("performing evaluation for initial checkpoint")
    # perform_control_eval(agent, eval_env, 0, variant, wandb_logger)        
    # agent.perform_eval(variant, 0, wandb_logger, eval_replay_buffer, eval_replay_buffer_iterator, eval_env)
    if hasattr(agent, 'replicate'):
        agent.replicate()

    for i in tqdm(range(1, variant.online_start + 1), smoothing=0.1,):
        
        t0 = time.time()
        batch = next(replay_buffer_iterator)
        tget_data = time.time() - t0
        t1 = time.time()

        out = agent.update(batch)
        
        if isinstance(out, tuple):
            agent, update_info = out
        else:
            update_info = out

        tupdate = time.time() - t1

        if variant.offline_finetuning_start != -1:
            if not changed_buffer_to_finetuning and i >= variant.offline_finetuning_start and isinstance(replay_buffer, MixingReplayBuffer):
                replay_buffer.set_mixing_ratio(variant.target_mixing_ratio)
                print(f"set target ratio to {variant.target_mixing_ratio} for offline finetuning")
                del replay_buffer_iterator
                replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
                changed_buffer_to_finetuning = True
                
                if hasattr(agent, '_cql_alpha') and hasattr(variant, 'cql_alpha_offline_finetuning') and variant.cql_alpha_offline_finetuning > 0:
                    agent._cql_alpha = variant.cql_alpha_offline_finetuning

        if i % variant.eval_interval == 0:
            if hasattr(agent, 'unreplicate'):
                agent.unreplicate()
            wandb_logger.log({'t_get_data': tget_data}, step=i)
            wandb_logger.log({'t_update': tupdate}, step=i)
            if 'pixels' in update_info and i % (variant.eval_interval*10) == 0:
                if variant.algorithm == 'reward_classifier':
                    image = visualize_image_rewards(update_info.pop('pixels'), batch['rewards'], update_info.pop('rewards_mean'), batch['observations'], task_id_mapping=task_id_mapping)
                    wandb_logger.log({'training/image_rewards': wandb.Image(image)}, step=i)
                else:
                    image = visualize_image_actions(update_info.pop('pixels'), batch['actions'], update_info.pop('pred_actions_mean'))
                    wandb_logger.log({'training/image_actions': wandb.Image(image)}, step=i)
            if perform_control_evals:
                perform_control_eval(agent, eval_env, i, variant, wandb_logger)

            # agent.perform_eval(variant, i, wandb_logger, eval_replay_buffer, eval_replay_buffer_iterator, eval_env)
            if hasattr(agent, 'replicate'):
                agent.replicate()

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                
                if v.ndim == 0:
                    wandb_logger.log({f'training/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'training/{k}', v, i)

        if variant.checkpoint_interval != -1:
            if i % variant.checkpoint_interval == 0:
                if hasattr(agent, 'unreplicate'):
                    agent.unreplicate()
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                if hasattr(agent, 'replicate'):
                    agent.replicate()

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, real_env=True, saver=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)

    traj_collect_func = collect_traj

    traj_id = 0
    
    if variant.get('alpha_schedule_interval', False) and variant.alpha_schedule_interval > 0:
        if isinstance(online_replay_buffer, PropertyReplayBuffer):
            # note that in this case we only schedule the online alpha
            wandb_logger.log({f'online_cql_alpha': online_replay_buffer.property_dict["cql_alpha"]}, step=0)
        elif isinstance(online_replay_buffer, MixingReplayBuffer):
            # in this case we change all
            wandb_logger.log({f'cql_alpha': agent._cql_alpha}, step=0)

    if variant.get('posneg_schedule_interval', False) and variant.posneg_schedule_interval > 0:
        wandb_logger.log({f'online_pos_neg_ratio': variant.online_pos_neg_ratio}, step=0)

    if variant.get('utd_schedule_interval', False) and variant.utd_schedule_interval > 0:
        wandb_logger.log({f'utd': variant.multi_grad_step}, step=0)

    i = variant.online_start + 1
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)

    with tqdm(total=variant.max_steps + 1, initial=variant.online_start + 1) as pbar:
        while i < variant.max_steps + 1:
            traj = traj_collect_func(variant, agent, env, not variant.stochastic_data_collect, traj_id=traj_id)
            traj_id += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if saver is not None:
                saver.save(traj)
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj)*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    # if i == variant.online_start + 1:
                    #     agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch, i)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger)
                        
                        try:
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)
                        except:
                            pass

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                            if hasattr(variant, 'save_replay_buffer') and variant.save_replay_buffer:
                                print('saving replay buffer to ', variant.outputdir + '/replaybuffer.npy')
                                online_replay_buffer.save(variant.outputdir + '/replaybuffer.npy')

                    if variant.get('alpha_schedule_interval', False) and variant.alpha_schedule_interval > 0 and i % variant.alpha_schedule_interval == 0:
                        if isinstance(online_replay_buffer, PropertyReplayBuffer):
                             # note that in this case we only schedule the online alpha
                            online_replay_buffer.property_dict["cql_alpha"] /= variant.alpha_schedule_ratio
                            wandb_logger.log({f'online_cql_alpha': online_replay_buffer.property_dict["cql_alpha"]}, step=i)
                            del replay_buffer_iterator
                            replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
                        elif isinstance(online_replay_buffer, MixingReplayBuffer):
                            # in this case we change all
                            agent._cql_alpha /= variant.alpha_schedule_ratio
                            wandb_logger.log({f'cql_alpha': agent._cql_alpha}, step=i)
                        else:
                            raise NotImplementedError()

                    if variant.get('posneg_schedule_interval', False) and variant.posneg_schedule_interval > 0 and i % variant.posneg_schedule_interval == 0:
                        if isinstance(online_replay_buffer, PropertyReplayBuffer):
                            pos_neg_ratio = max(online_replay_buffer.replay_buffer.mixing_ratio / variant.posneg_schedule_ratio, 0.3)
                            online_replay_buffer.replay_buffer.set_mixing_ratio(pos_neg_ratio)
                        else:
                            pos_neg_ratio = max(online_replay_buffer.mixing_ratio / variant.posneg_schedule_ratio, 0.3)
                            online_replay_buffer.set_mixing_ratio(pos_neg_ratio)
                        wandb_logger.log({f'online_pos_neg_ratio': pos_neg_ratio}, step=i)
                        print("setting online pos neg ratio to", pos_neg_ratio)
                        del replay_buffer_iterator
                        replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)

                    if variant.get('utd_schedule_interval', False) and variant.utd_schedule_interval > 0 and i % variant.utd_schedule_interval == 0:
                        current_utd = variant.multi_grad_step 
                        utd = max(current_utd // variant.utd_schedule_ratio, 1)
                        variant.multi_grad_step = int(utd)
                        wandb_logger.log({f'utd': variant.multi_grad_step}, step=i)
                        print("setting utd to", variant.multi_grad_step)


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    if variant.only_add_success:
        if traj[-1]['reward'] < 1e-3:
            print('trajecotry discarded because unsuccessful')
            return
    
    if isinstance(online_replay_buffer, PropertyReplayBuffer):
        online_replay_buffer = online_replay_buffer.replay_buffer

    if variant.get('online_pos_neg_ratio', False) >= 0:
        is_negative = traj[-1]['reward'] < 1e-3
        for t, step in enumerate(traj):
            obs = step['observation']
            next_obs = step['next_observation']
            if not variant.add_states and 'state' in obs:
                obs.pop('state')
            if not variant.add_states and 'state' in next_obs:
                next_obs.pop('state')
                
            if variant.dataset in ['binsort', 'multi_object_in_bowl', 'multi_object_in_bowl_interfering']:
                reward = variant.reward_func(step['reward'])
                mask = 1 - int(reward == 10)
            else:
                reward = variant.reward_func(step['reward'])
    
            insert_dict = dict(
                observations=obs,
                actions=step['action'],
                next_actions=traj[t + 1]['action'] if t < len(traj) - 1 else step['action'],
                rewards=reward,
                masks=mask,
                dones=step['done'],
                next_observations=next_obs,
                trajectory_id=online_replay_buffer.replay_buffers[int(is_negative)]._traj_counter
            )            
            online_replay_buffer.replay_buffers[int(is_negative)].insert(insert_dict)
        online_replay_buffer.replay_buffers[int(is_negative)].increment_traj_counter()
    else:
        masks = []
        rewards = []
        for t, step in enumerate(traj):
            if variant.reward_type == 'dense':
                reward = step['reward']
                # TODO: fix this mask
                mask = 1
            elif variant.dataset in ['binsort', 'multi_object_in_bowl', 'multi_object_in_bowl_interfering']:
                reward = variant.reward_func(step['reward'])
                mask = 1 - int(reward == 10)
            else:
                reward = variant.reward_func(step['reward'])
            masks.append(mask)
            rewards.append(reward)
        
        monte_carlo_return = calc_return_to_go(rewards, masks, variant.discount)
        if variant.get("online_bound_nstep_return", -1) > 0:
            nstep_return = calc_nstep_return(variant.online_bound_nstep_return, rewards, masks, variant.discount)

        if variant.reward_type == 'dense':
            is_positive=True
            traj_insert = traj
        else:
            is_positive = rewards[-1] > 1e-3
            # if it is negative sample without terminal, we should remove last n steps
            if variant.get("online_bound_nstep_return", -1) > 0 and masks[-1] == 1:
                traj_insert = traj[:-variant.online_bound_nstep_return]
            else:
                traj_insert = traj
    
        for t, step in enumerate(traj_insert):
            obs = step['observation']
            next_obs = step['next_observation']
            if not variant.add_states and 'state' in obs:
                obs.pop('state')
            if not variant.add_states and 'state' in next_obs:
                next_obs.pop('state')
            if variant.get("online_bound_nstep_return", -1)  > 1:
                nstep_obs = traj[t + variant.online_bound_nstep_return]['observation'] if t < len(traj)-variant.online_bound_nstep_return else traj[-1]['observation']
                if not variant.add_states and 'state' in nstep_obs:
                    nstep_obs.pop('state')
                nstep_actions = traj[t + variant.online_bound_nstep_return]['action'] if t < len(traj)-variant.online_bound_nstep_return else traj[-1]['action']
                nstep_mask = masks[t + variant.online_bound_nstep_return-1] if t < len(traj)-variant.online_bound_nstep_return else masks[-1]
            
            
            if variant.get("online_bound_nstep_return", -1) > 1:
                insert_dict = dict(
                    observations=obs,
                    actions=step['action'],
                    next_actions=traj[t + 1]['action'] if t < len(traj) - 1 else step['action'],
                    rewards=rewards[t],
                    masks=masks[t],
                    dones=step['done'],
                    next_observations=next_obs,
                    trajectory_id=online_replay_buffer._traj_counter,
                    mc_returns=monte_carlo_return[t] if is_positive else -25.0,
                    nstep_returns=nstep_return[t],
                    nstep_observations=nstep_obs,
                    nstep_actions=nstep_actions,
                    nstep_masks = nstep_mask,
                    is_offline = 0

                )
            else:
                insert_dict = dict(
                observations=obs,
                actions=step['action'],
                next_actions=traj[t + 1]['action'] if t < len(traj) - 1 else step['action'],
                rewards=rewards[t],
                masks=masks[t],
                dones=step['done'],
                next_observations=next_obs,
                trajectory_id=online_replay_buffer._traj_counter,
                mc_returns=monte_carlo_return[t] if is_positive else -25.0,
                is_offline = 0
            )
            online_replay_buffer.insert(insert_dict)
        online_replay_buffer.increment_traj_counter()

def run_multiple_trajs(variant, agent, env, num_trajs, deterministic=True):
    returns = []
    mult_stage_returns = []
    lengths = []
    obs = []

    traj_collect_func = collect_traj

    for i in range(num_trajs):
        print('##############################################')
        print('traj', i)
        traj = traj_collect_func(variant, agent, env, deterministic, traj_id=i)
        returns.append(np.sum([step['reward'] for step in traj]))
        mult_stage_returns.append(np.sum([int(step['reward']==2) for step in traj]))
        lengths.append(len(traj))
        obs.append([step['observation'] for step in traj])

    return {
        'return': np.mean(returns),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'return_max': np.max(returns),
        'return_min': np.min(returns),
        'mult_stage_return': np.mean(mult_stage_returns),
        'mult_stage_return_mean': np.mean(mult_stage_returns),
        'mult_stage_return_min': np.std(mult_stage_returns),
        'mult_stage_return_max': np.max(mult_stage_returns),
        'mult_stage_return_std': np.min(mult_stage_returns),
        'length': np.mean(lengths),
        'obs': obs[-1],
        'rewards': np.array([step['reward'] for step in traj])
    }

def collect_traj(variant, agent, env, deterministic, traj_id=None, max_len_eval=500):
    obs, done = env.reset(), False
    traj = []

    print('collect traj deterministc', deterministic)
    num_steps = 0
    with tqdm(total=max_len_eval) as pbar:
        while not done and num_steps < max_len_eval:
            if hasattr(variant, 'eval_task_id'):
                if variant.eval_task_id != -1:
                    obs['task_id'] = np.zeros(variant.num_tasks, np.float32)[None]
                    obs['task_id'][:, variant.eval_task_id] = 1.
            if variant.from_states:
                obs_filtered = copy.deepcopy(obs)
                if 'pixels' in obs_filtered:
                    obs_filtered.pop('pixels')
            else:
                obs_filtered = obs

            if deterministic:
                out = agent.eval_actions(obs_filtered)
                if isinstance(out, tuple):
                    action, agent = out
                else:
                    action = out
                
                action = action.squeeze()
            else:

                out = agent.sample_actions(obs_filtered)
                if isinstance(out, tuple):
                    action, agent = out
                else:
                    action = out
                
                action = action.squeeze()

            next_obs, reward, done, info = env.step(action)

            if hasattr(variant, 'eval_task_id'):
                if variant.eval_task_id != -1:
                    next_obs['task_id'] = obs['task_id']

            traj.append({
                'observation': obs,
                'action': action,
                'reward': reward,
                'next_observation': next_obs,
                'done': done,
                'info': info
            })
            obs = next_obs
            
            pbar.update(1)
            num_steps += 1
    return traj


def stepwise_alternating_training_loop(variant, batch_size, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger):
    replay_buffer_iterator = replay_buffer.get_iterator(batch_size)
    observation, done = env.reset(), False
    print('stepwise alternating loop')
    for i in tqdm(range(variant.online_start + 1, variant.max_steps + 1),
                       smoothing=0.1,
                       ):

        if len(replay_buffer.replay_buffers[1]) > variant.start_online_updates:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)
            print('gradient update')

        if done:
            observation, done = env.reset(), False
            online_replay_buffer.increment_traj_counter()

        action = agent.eval_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        reward = reward * variant.reward_scale + variant.reward_shift

        online_replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation,
                 trajectory_id=online_replay_buffer._traj_counter
                 ))
        observation = next_observation

        if i % variant.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb_logger.log({f'training/{k}': v}, step=i)
                elif v.ndim <= 2:
                    wandb_logger.log_histogram(f'training/{k}', v, i)

        # if i % variant.eval_interval == 0:
        #     agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

        if variant.checkpoint_interval != -1:
            if i % variant.checkpoint_interval == 0:
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

def perform_control_eval(agent, eval_env, i, variant, wandb_logger):
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.enable_render()
    eval_info = run_multiple_trajs(variant, agent,
                                   eval_env,
                                   num_trajs=variant.eval_episodes, deterministic=not variant.stochastic_evals)
    print('eval runs done.')
    if variant.from_states:
        if hasattr(eval_env, 'enable_render'):
            eval_env.disable_render()
    obs = eval_info.pop('obs')
    if 'pixels' in obs[0]:
        video = np.stack([ts['pixels'] for ts in obs]).squeeze()
        if len(video.shape) == 5:
            video = video[..., -1] # visualizing only the last frame of the stack when using framestacking
        video = video.transpose(0, 3, 1, 2)
        wandb_logger.log({'eval_video': wandb.Video(video[:, -3:], fps=8)}, step=i)
    
    if 'state' in obs[0] and variant.reward_type == 'dense':
        states = np.stack([ts['state'] for ts in obs])
        states_image = visualize_states_rewards(states, eval_info['rewards'], eval_env.target)
        wandb_logger.log({'state_traj_image': wandb.Image(states_image)}, step=i)

    for k, v in eval_info.items():
        if v.ndim == 0:
            wandb_logger.log({f'evaluation/{k}': v}, step=i)

    print('finished evals avg. return ', eval_info['return'])
    print('finished evals avg. length ', eval_info['length'])


def run_evals_only(variant, agent, eval_env, wandb_logger):
    i = 0
    while True:
        perform_control_eval(agent, eval_env, i, variant, wandb_logger)
        i += 1

def is_positive_sample(traj, i, variant, task_name):
    return i >= len(traj['observations']) - variant.num_final_reward_steps

def is_positive_traj(traj):
    return traj['rewards'][-1, 0] >= 1


def load_buffer(dataset_file, variant, task_aliasing_dict=None, multi_viewpoint=False, data_count_dict=None, split_pos_neg=False, num_traj_cutoff=None, traj_len_cutoff=None,  split_by_traj=False):
    print('loading buffer data from ', dataset_file)
    if variant.cond_interfing:
        if isinstance(variant.ti, tuple):
            splt = str.split(dataset_file, '/')
            task_name = ''.join([splt[x] for x in variant.ti])
        elif isinstance(variant.ti, int):
            task_name = str.split(dataset_file, '/')[variant.ti]
        else:
            raise NotImplementedError()
        env_name = str.split(dataset_file, '/')[-3]
    else:
        task_name = str.split(dataset_file, '/')[-3]
        env_name = str.split(dataset_file, '/')[-4]
    if task_aliasing_dict and task_name in task_aliasing_dict:
        task_name = task_aliasing_dict[task_name]
    trajs = np.load(os.environ['DATA'] + dataset_file, allow_pickle=True)
    if data_count_dict is not None:
        if env_name not in data_count_dict:
            data_count_dict[env_name] = {}
        if task_name in data_count_dict[env_name]:
            data_count_dict[env_name][task_name] += len(trajs)
        else:
            data_count_dict[env_name][task_name] = len(trajs)
    if len(trajs) == 0:
        return 0, trajs

    pos_num_transitions = 0
    neg_num_transitions = 0
    num_transitions = 0

    # Count number of viewpoints
    if multi_viewpoint:
        viewpoints = trajs[0]['observations'][0].keys()
        viewpoints = [viewpoint for viewpoint in viewpoints if viewpoint.startswith('images')]
        num_viewpoints = len(viewpoints)
        print('num viewpoints', num_viewpoints)
    else:
        num_viewpoints = 1

    if num_traj_cutoff is not None and num_traj_cutoff != -1:
        np.random.shuffle(trajs)
        trajs = trajs[:num_traj_cutoff]
        print('traj cutoff', num_traj_cutoff)
        gc.collect()
    
    if traj_len_cutoff is not None:
        for traj in trajs:
            for key in traj:
                traj[key] = traj[key][:traj_len_cutoff]
        print('traj len cutoff', traj_len_cutoff)
        gc.collect()

    for traj in trajs:
        if "rewards" in traj.keys() and isinstance(traj["rewards"], list):
            traj["rewards"] = np.array(traj["rewards"]).reshape(-1, 1)
        for i in range(len(traj['observations'])):

            if split_by_traj:
                if is_positive_traj(traj):
                    pos_num_transitions += num_viewpoints
                else:
                    neg_num_transitions += num_viewpoints
            elif split_pos_neg:
                if is_positive_sample(traj, i, variant, task_name):
                    pos_num_transitions += num_viewpoints
                else:
                    neg_num_transitions += num_viewpoints
            else:
                num_transitions += num_viewpoints
        # num_transitions += 1  # needed because of memory efficient replay buffer
        # pos_num_transitions += 1  # needed because of memory efficient replay buffer
        # neg_num_transitions += 1  # needed because of memory efficient replay buffer
        traj['task_description'] = task_name
    if split_pos_neg:
        return (pos_num_transitions, neg_num_transitions), trajs
    return num_transitions, trajs


def _reshape_image(obs):
    if len(obs.shape) == 1:
        obs = np.reshape(obs, (3, 128, 128))
        return np.transpose(obs, (1, 2, 0))
    elif len(obs.shape) == 3:
        return obs
    else:
        raise ValueError

RETURN_TO_GO_DICT = dict()

def calc_return_to_go(rewards, masks, gamma):
    global RETURN_TO_GO_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in RETURN_TO_GO_DICT.keys():
        reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0]*len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            reward_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            prev_return = reward_to_go[-i-1]
        RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go

NSTEP_RETURN_DICT = dict()
def calc_nstep_return(n, rewards, masks, gamma):
    global NSTEP_RETURN_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in NSTEP_RETURN_DICT.keys():
        nstep_return = NSTEP_RETURN_DICT[rewards_str]
    else:
        nstep_return = [0]*len(rewards)
        prev_return = 0
        terminal_counts=1
        for i in range(len(rewards)):
            if i < n + terminal_counts - 1:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            else:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1] - (gamma**n) * rewards[-i-1+n] * masks[-i-1]
            prev_return = nstep_return[-i-1]
            if i!= 0 and masks[-i-1] == 0: # deal with the negative traj wich does not have terminal
                terminal_counts+=1
        NSTEP_RETURN_DICT[rewards_str] = nstep_return
    return nstep_return