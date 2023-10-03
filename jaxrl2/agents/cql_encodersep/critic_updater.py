from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

from flax.core import frozen_dict
import copy


NUM_CQL_REPEAT = 4
CLIP_MIN=0
CLIP_MAX=20


def extend_and_repeat(tensor, axis, repeat):
    if isinstance(tensor, frozen_dict.FrozenDict):
        new_tensor = {}
        for key in tensor:
            new_tensor[key] = jnp.repeat(jnp.expand_dims(tensor[key], axis), repeat, axis=axis)
        new_tensor = tensor.copy(add_or_replace=new_tensor)
        return new_tensor
    else:
        return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def reshape_for_cql_computation(tensor, num_cql_repeat):
    if isinstance(tensor, frozen_dict.FrozenDict):
        new_tensor = {}
        for key in tensor:
            new_tensor[key] = jnp.reshape(tensor[key],
                    [tensor[key].shape[0] * num_cql_repeat, *tensor[key].shape[2:]])
        new_tensor = tensor.copy(add_or_replace=new_tensor)
        return new_tensor
    else:
        return jnp.reshape(tensor,
                [tensor.shape[0] * num_cql_repeat, *tensor.shape[2:]])

def update_critic(
        key: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
        target_critic_encoder: TrainState, target_critic_decoder: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool, critic_reduction: str, cql_alpha: float, max_q_backup: bool, dr3_coefficient: float,tr_penalty_coefficient:float, mc_penalty_coefficient:float, pretrained_critic_encoder: TrainState,
        method:bool=False, method_const:float=1.0, method_type:int=0, cross_norm:bool=False, bound_q_with_mc:bool=False, online_bound_nstep_return:int=-1 
    ) -> Tuple[TrainState, Dict[str, float]]:

    key, key_pi, key_random, key_temp, key_nstep = jax.random.split(key, num=5)
    
    enc_rng_keys_for_nets = ('noise', 'drop_path')
    len_key = len(enc_rng_keys_for_nets)
    
    which_call = 0
    num_calls = 4
    rng_keys_per_call = jax.random.split(key, num= len_key * num_calls + 1)
    key, rng_keys_per_call = rng_keys_per_call[0], rng_keys_per_call[1:]
    
    
    def enc_rng_key():
        nonlocal which_call
        assert 0 <= which_call < num_calls, f"{which_call} invalid"
        keys = dict(zip(enc_rng_keys_for_nets, rng_keys_per_call[len_key*which_call : len_key * (which_call + 1)]))
        which_call += 1 
        return keys
    
    if hasattr(target_critic_encoder, 'batch_stats') and target_critic_encoder.batch_stats is not None:
        embed_next_obs, _ = target_critic_encoder.apply_fn({'params': target_critic_encoder.params, 'batch_stats': target_critic_encoder.batch_stats}, batch['next_observations'], training=False, mutable=['batch_stats'], rngs=enc_rng_key())
    else:
        embed_next_obs = target_critic_encoder.apply_fn({'params': target_critic_encoder.params}, batch['next_observations'], rngs=enc_rng_key())
    
    if max_q_backup:
        # needed for actor
        next_observations_tiled = extend_and_repeat(
            batch['next_observations'], axis=1, repeat=NUM_CQL_REPEAT
        )
        next_observations_tiled = reshape_for_cql_computation(
            next_observations_tiled, num_cql_repeat=NUM_CQL_REPEAT)
        
        #embedding tiled
        embed_next_obs = extend_and_repeat(
            embed_next_obs, axis=1, repeat=NUM_CQL_REPEAT
        )
        embed_next_obs = reshape_for_cql_computation(
            embed_next_obs, num_cql_repeat=NUM_CQL_REPEAT)
    else:
        next_observations_tiled = batch['next_observations']

    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        dist, _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats}, next_observations_tiled, mutable=['batch_stats'], rngs=enc_rng_key())
    else:
        dist = actor.apply_fn({'params': actor.params}, next_observations_tiled, rngs=enc_rng_key())

    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    if hasattr(target_critic_decoder, 'batch_stats') and target_critic_decoder.batch_stats is not None:
        next_qs, _ = target_critic_decoder.apply_fn({'params': target_critic_decoder.params, 'batch_stats': target_critic_decoder.batch_stats},
                                    embed_next_obs, next_actions, mutable=['batch_stats'], training=False) # make sure to use the target critic with val mode
    else:
        next_qs = target_critic_decoder.apply_fn({'params': target_critic_decoder.params}, embed_next_obs, next_actions)
    
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    if max_q_backup:
        """Now reduce next q over the actions"""
        next_q_reshape = jnp.reshape(next_q, (batch['actions'].shape[0], NUM_CQL_REPEAT))
        next_q = jnp.max(next_q_reshape, axis=-1)

    target_q = batch['rewards'] + discount * batch['masks'] * next_q

    if backup_entropy:
        target_q -= discount * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs

    # if bound_q_with_mc:
    #     target_q = jnp.maximum(target_q, batch['mc_returns'])

    # CQL sample actions
    observations_tiled = extend_and_repeat(batch['observations'], axis=1, repeat=NUM_CQL_REPEAT)
    observations_tiled = reshape_for_cql_computation(
        observations_tiled, num_cql_repeat=NUM_CQL_REPEAT)

    next_observations_tiled_temp = extend_and_repeat(batch['next_observations'], axis=1, repeat=NUM_CQL_REPEAT)
    next_observations_tiled_temp = reshape_for_cql_computation(
        next_observations_tiled_temp, num_cql_repeat=NUM_CQL_REPEAT)
    
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        policy_dist, _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats}, observations_tiled, mutable=['batch_stats'], rngs=enc_rng_key())
    else:
        policy_dist = actor.apply_fn({'params': actor.params}, observations_tiled, rngs=enc_rng_key())
    
    policy_actions, policy_log_probs = policy_dist.sample_and_log_prob(seed=key_pi)
    
    if isinstance(batch['observations'], frozen_dict.FrozenDict):
        n = batch['observations']['pixels'].shape[0]
    else:
        n = batch['observations'].shape[0]

    random_actions = jax.random.uniform(
        key_random, shape=(n * NUM_CQL_REPEAT, policy_actions.shape[-1]),
        minval=-1.0, maxval=1.0
    )
    random_pi = (1.0/2.0) ** policy_actions.shape[-1]

    if pretrained_critic_encoder is not None:
        global pretrained_embed_obs
        pretrained_embed_obs = pretrained_critic_encoder.apply_fn({'params': pretrained_critic_encoder.params}, batch['observations'])
        pretrained_embed_obs = jax.lax.stop_gradient(pretrained_embed_obs)
    global  bound_q_with_mc_global, online_bound_nstep_return_global
    bound_q_with_mc_global, online_bound_nstep_return_global = None, None
    bound_q_with_mc_global = bound_q_with_mc
    online_bound_nstep_return_global = online_bound_nstep_return

    def critic_loss_fn(critic_encoder_params: Params, critic_decoder_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            embed_curr_obs, new_model_state_encoder = critic_encoder.apply_fn({'params': critic_encoder_params, 'batch_stats': critic_encoder.batch_stats}, 
                                                                              batch['observations'], mutable=['batch_stats'], rngs=enc_rng_key())
        else:
            embed_curr_obs = critic_encoder.apply_fn({'params': critic_encoder_params}, batch['observations'], rngs=enc_rng_key())
            new_model_state_encoder = {}

        embed_curr_obs_tiled = extend_and_repeat(embed_curr_obs, axis=1, repeat=NUM_CQL_REPEAT)
        embed_curr_obs_tiled = reshape_for_cql_computation(embed_curr_obs_tiled, num_cql_repeat=NUM_CQL_REPEAT)


        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            qs, new_model_state_decoder = critic_decoder.apply_fn({'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                                embed_curr_obs, batch['actions'], mutable=['batch_stats'])
        else:
            qs = critic_decoder.apply_fn({'params': critic_decoder_params}, embed_curr_obs, batch['actions'])
            new_model_state_decoder = {}
            
            
        new_model_state = (new_model_state_encoder, new_model_state_decoder)

        critic_loss = ((qs - target_q)**2).mean()
        bellman_loss = critic_loss

        qs_to_log = copy.deepcopy(qs) # copy to avoid modifying the original qs
        
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            q_pi, _ = critic_decoder.apply_fn({'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                                    embed_curr_obs_tiled, policy_actions, mutable=['batch_stats'])
        else:
            q_pi = critic_decoder.apply_fn({'params': critic_decoder_params}, embed_curr_obs_tiled, policy_actions)
        
        if bound_q_with_mc_global:
            if online_bound_nstep_return_global > 0:
                # use MC return for offline and Nstep return for online as lowerbound
                # calculate nstep returns
                embed_nstep_obs = target_critic_encoder.apply_fn({'params': target_critic_encoder.params}, batch['nstep_observations'])
                dist = actor.apply_fn({'params': actor.params}, batch['nstep_observations'])
                nstep_actions, _ = dist.sample_and_log_prob(seed=key_nstep)
                nstep_q = target_critic_decoder.apply_fn({'params': target_critic_decoder.params}, embed_nstep_obs, nstep_actions)
                nstep_q = nstep_q.mean(axis=0)
                nstep_q = batch['nstep_returns'] + (discount**online_bound_nstep_return) * batch['nstep_masks'] * nstep_q
                nstep_q = jnp.nan_to_num(nstep_q, nan=-100000.0, posinf=-100000.0, neginf=-100000.0)
                nstep_returns_tiled = jnp.reshape(jnp.repeat(nstep_q, NUM_CQL_REPEAT), (1, -1))
                nstep_q = jnp.repeat(nstep_returns_tiled, q_pi.shape[0], axis=0)

                # calculate mc return for offline
                mc_returns_tiled = jnp.reshape(jnp.repeat(batch['mc_returns'], NUM_CQL_REPEAT), (1, -1))
                mc_returns = jnp.repeat(mc_returns_tiled, q_pi.shape[0], axis=0)

                # process offline/online/masks
                is_offline_masks = jnp.repeat(jnp.reshape(jnp.repeat(batch['is_offline'], NUM_CQL_REPEAT), (1, -1)), q_pi.shape[0], axis=0)
                lower_bounds = mc_returns * is_offline_masks + nstep_q * (1 - is_offline_masks)
                q_pi = jnp.maximum(q_pi, lower_bounds)

                q_pi_offline = q_pi * is_offline_masks
                q_pi_online = q_pi * (1 - is_offline_masks)

                q_pi_bounded_rate_offline = jnp.sum(q_pi_offline==lower_bounds) / jnp.sum(is_offline_masks)
                q_pi_bounded_rate_online = jnp.sum(q_pi_online==lower_bounds) / jnp.sum(1 - is_offline_masks)
                qpi_bounded_rate = jnp.sum(q_pi==lower_bounds) / (jnp.sum(lower_bounds==lower_bounds))
            else:
                # use MC return as lowerboud for both online and offline samples
                mc_returns_tiled = jnp.reshape(jnp.repeat(batch['mc_returns'], NUM_CQL_REPEAT), (1, -1))
                lower_bounds = jnp.repeat(mc_returns_tiled, q_pi.shape[0], axis=0)
                q_pi = jnp.maximum(q_pi, lower_bounds)
                qpi_bounded_rate = jnp.sum(q_pi==lower_bounds) / (jnp.sum(lower_bounds==lower_bounds) - jnp.sum(lower_bounds==-25.0))


        q_pi_for_is = (q_pi[0] - policy_log_probs, q_pi[1] - policy_log_probs)
        q_pi_for_is = (
            jnp.reshape(q_pi_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_pi_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_pi_for_is = jnp.stack(q_pi_for_is, axis=0)

        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            q_random, _ = critic_decoder.apply_fn({'params': critic_decoder_params, 'batch_stats': critic_decoder.batch_stats},
                                    embed_curr_obs_tiled, random_actions, mutable=['batch_stats'])
        else:
            q_random = critic_decoder.apply_fn({'params': critic_decoder_params}, embed_curr_obs_tiled, random_actions)

        q_random_for_is = (q_random[0] - np.log(random_pi), q_random[1] - np.log(random_pi))
        q_random_for_is = (
            jnp.reshape(q_random_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_random_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )    
        q_random_for_is = jnp.stack(q_random_for_is, axis=0)

        # if bound_q_with_mc_global:
        #     # reshape to (num_critic, batch, NUM_CQL_REPEAT) as q_random_for_is and q_pi_for_is
        #     mc_returns = jnp.repeat(jnp.repeat(batch['mc_returns'].reshape(1, -1, 1), q_pi_for_is.shape[0], axis=0), q_pi_for_is.shape[-1], axis=-1)
        #     q_pi_for_is = jnp.maximum(q_pi_for_is, mc_returns)
        #     # q_random_for_is = jnp.maximum(q_random_for_is, mc_returns)
        #     mc_bounded_rate_qpi = jnp.sum(q_pi_for_is==mc_returns) / jnp.sum(mc_returns==mc_returns)
            # mc_bounded_rate_qrandom = jnp.sum(q_random_for_is==mc_returns) / jnp.sum(mc_returns==mc_returns)
        
        cat_q = jnp.concatenate([q_pi_for_is, q_random_for_is], axis=-1)
        lse_q = jax.scipy.special.logsumexp(cat_q, axis=-1)

        ### max(q_pi, MC return)
        # if bound_q_with_mc_global:
        #     mc_returns = jnp.repeat(batch['mc_returns'].reshape(1, -1), lse_q.shape[0], axis=0)
        #     lse_q = jnp.maximum(lse_q, mc_returns)
        #     mc_bounded_rate = jnp.sum(lse_q==mc_returns) / jnp.sum(mc_returns==mc_returns)

        cql_loss_per_element = lse_q - qs

        if "cql_alpha" in batch.keys():
            # if usint property_buffer to use diffrent cal_alpha among smaples
            cql_loss_per_element = cql_loss_per_element.mean(axis=0) * batch["cql_alpha"].squeeze()
            cql_loss = cql_loss_per_element.mean()
        else:
            cql_loss = cql_alpha * cql_loss_per_element.mean()

        critic_loss = critic_loss + cql_loss

        # Trust Region Penalty between pretrained encoders and current encoders
        if tr_penalty_coefficient != 0:
            tr_penalty = ((embed_curr_obs["pixels"] - pretrained_embed_obs["pixels"])**2).mean()
            critic_loss = critic_loss + tr_penalty_coefficient * tr_penalty

        if mc_penalty_coefficient > 0:
            # mc_penalty = ((qs.mean(axis=0) - batch["mc_returns"])**2).mean()
            # mc_penalty = batch["mc_returns"].mean() - q_pi.mean()
            q_pi_mean = q_pi.mean(axis=0).reshape(-1,  NUM_CQL_REPEAT).mean(axis=1) # prcess to shape (batchsize, )
            mc_penalty = jnp.clip(batch["mc_returns"] - q_pi_mean, 0).mean()
            critic_loss = critic_loss + mc_penalty_coefficient * mc_penalty

        ## Logging only
        diff_rand_data = q_random.mean() - qs_to_log.mean()
        diff_pi_data = q_pi.mean() - qs_to_log.mean()

        things_to_log = {   
            'critic_loss': critic_loss,
            'bellman_loss': bellman_loss,
            'cql_loss_mean': cql_loss,
            'cql_loss_max': jnp.max(cql_loss_per_element),
            'cql_loss_min': jnp.min(cql_loss_per_element),
            'cql_loss_std': jnp.std(cql_loss_per_element),
            'lse_q': lse_q.mean(),
            'q_pi_avg': q_pi.mean(),
            'q_random': q_random.mean(),
            'q_data_avg': qs_to_log.mean(),
            'q_data_max': qs_to_log.max(),
            'q_data_min': qs_to_log.min(),
            'q_data_std': qs_to_log.std(),
            'weighted_q_data_avg': qs.mean(),
            'weighted_q_data_max': qs.max(),
            'weighted_q_data_min': qs.min(),
            'weighted_q_data_std': qs.std(),
            'q_pi_max': q_pi.max(),
            'q_pi_min': q_pi.min(),
            'diff_pi_data_mean': diff_pi_data,
            'diff_rand_data_mean': diff_rand_data, 
            'target_actor_entropy': -next_log_probs.mean(),
            'rewards_mean': batch['rewards'].mean(),
            'actions_mean': batch['actions'].mean(),
            'actions_max': batch['actions'].max(),
            'actions_min': batch['actions'].min(),
            'terminals_mean': batch['masks'].mean(),
            'log_pis_mean': policy_log_probs.mean(),
            'log_pis_max': policy_log_probs.max(),
            'log_pis_min': policy_log_probs.min(),
            'target_q_avg': target_q.mean(),
            'target_q_max': target_q.max(),
            'target_q_min': target_q.min(),
            'target_q_std': target_q.std(),
        }

        if tr_penalty_coefficient != 0:
            things_to_log['trust_region_penalty'] = tr_penalty
        if mc_penalty_coefficient != 0:
            things_to_log['monte_carlo_penalty'] = mc_penalty
        if bound_q_with_mc_global:
            things_to_log['qpi_bounded_rate'] = qpi_bounded_rate
            things_to_log['qpi_lower_bounds_avg'] = lower_bounds.mean()
            if online_bound_nstep_return_global > 0:
                things_to_log['q_pi_bounded_rate_offline'] = q_pi_bounded_rate_offline
                things_to_log['q_pi_bounded_rate_online'] = q_pi_bounded_rate_online        
        return critic_loss, (things_to_log, new_model_state)


    (grads_encoder,grads_decoder), (info,new_model_state) = jax.grad(critic_loss_fn, has_aux=True, argnums=(0,1))(critic_encoder.params, critic_decoder.params)

    if 'batch_stats' in new_model_state[0]:
        new_critic_encoder = critic_encoder.apply_gradients(grads=grads_encoder, batch_stats=new_model_state[0]['batch_stats'])
    else:
        new_critic_encoder = critic_encoder.apply_gradients(grads=grads_encoder)
        
    if 'batch_stats' in new_model_state[1]:
        new_critic_decoder = critic_decoder.apply_gradients(grads=grads_decoder, batch_stats=new_model_state[1]['batch_stats'])
    else:
        new_critic_decoder = critic_decoder.apply_gradients(grads=grads_decoder)
    new_critic = (new_critic_encoder, new_critic_decoder)
    return new_critic, info