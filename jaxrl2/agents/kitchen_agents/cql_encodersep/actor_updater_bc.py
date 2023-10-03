from statistics import mean
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update_bc(
        rng: PRNGKey, actor: TrainState, critic_encoder: TrainState, critic_decoder: TrainState,
        temperature:int, batch: FrozenDict,  dueling_num_a=None, dueling_avg_dist=None,):

    rng, key, key_encoder, key_decoder1, key_decoder2 , key_sample, key_act = jax.random.split(rng, 7)
    key, key_random, key_seed = jax.random.split(key, num=3)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                              batch['observations'], training=True, rngs={'dropout': key}, mutable=['batch_stats'])
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'], training=True, rngs={'dropout': key})
            new_model_state = {}
            
        log_probs = dist.log_prob(batch['actions'])
        log_prob_loss = - (log_probs).mean()

        mse = (dist.mode() - batch['actions']) ** 2
        mse = mse.mean(axis=-1) # mean over action dimension
        mse_loss = (mse).mean()
        
        action_pi, log_pi = dist.sample_and_log_prob(seed=key)
                
        actor_loss = log_prob_loss

        # sample log pis for entropy calculation
        _, entropy = dist.sample_and_log_prob(seed=key_act)
    
        if hasattr(dist, 'distribution') and hasattr(dist.distribution, '_loc'):
            mean_dist = dist.distribution._loc
            std_diag_dist = dist.distribution._scale_diag
        else:
            mean_dist = dist._loc
            std_diag_dist = dist._scale_diag
            
        things_to_log={        
            'log_prob_loss': log_prob_loss,
            'mse_loss': mse_loss, 

            'dataset_actions': batch['actions'],
            'pred_actions_mean': mean_dist, 
            'action_std': std_diag_dist,
            'loss': actor_loss,

            'entropy': -entropy.mean(),
        }

        things_to_log = {'bc_' + k: v for k, v in things_to_log.items()}

        return actor_loss, (new_model_state, things_to_log)

    grads, (new_model_state, info) = jax.grad(loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)
        
    return new_actor, info