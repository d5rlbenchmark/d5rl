from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.kitchen_data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False) -> Tuple[TrainState, Dict[str, float]]:
    
    key, key_act = jax.random.split(key, num=2)
    
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

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'], rngs=enc_rng_key())
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'], rngs=enc_rng_key())
            new_model_state = {}
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        if hasattr(critic_encoder, 'batch_stats') and critic_encoder.batch_stats is not None:
            embed_curr_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, batch['observations'], mutable=['batch_stats'], rngs=enc_rng_key())
        else:
            embed_curr_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, batch['observations'], rngs=enc_rng_key())
            
        if hasattr(critic_decoder, 'batch_stats') and critic_decoder.batch_stats is not None:
            qs, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, embed_curr_obs, actions, mutable=['batch_stats'])
        else:
            qs = critic_decoder.apply_fn({'params': critic_decoder.params}, embed_curr_obs, actions)
        
        q = qs.min(axis=0)
        
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
            'pred_actions_mean': dist.distribution._loc,
            'dataset_actions': batch['actions']
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info