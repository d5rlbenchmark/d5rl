from typing import Dict, Tuple
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey

def get_stats(name, vector):
    return {name + 'mean': vector.mean(),
            name + 'min': vector.min(),
            name + 'max': vector.max(),
            }


def update_critic(key: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState, batch: FrozenDict, discount: float) -> Tuple[TrainState, Dict[str, float]]:
    key, key_pi, key_noise = jax.random.split(key,3)
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key_pi)
    policy_noise = 0.1 
    lim = 1 - 1e-5
    noise = jnp.clip(jax.random.normal(key_noise, next_actions.shape) * policy_noise, -0.5, 0.5)
    next_actions = jnp.clip(next_actions + noise, -lim, lim)

    next_qs = target_critic.apply_fn({'params': target_critic.params}, batch['next_observations'], next_actions)
    next_q = next_qs.min(axis=0)
    target_q = batch['rewards'] + discount * batch['masks'] * next_q
    
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        if critic.batch_stats is not None:
            qs, new_model_state = critic.apply_fn({'params': critic_params, 'batch_stats': critic.batch_stats}, batch['observations'], batch['actions'],
                                 training=True, mutable=['batch_stats'])
        else:
            qs = critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
            new_model_state = {}
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, (new_model_state, {'critic_loss': critic_loss, 'q_data_avg': qs.mean(), 'q_data_min': qs.min(), 'q_data_max': qs.max(), "target_q_avg":target_q.mean(), 'target_q_max': target_q.max(), 'target_q_min': target_q.min(), 'rewards': batch['rewards'], **get_stats('rewards', batch['rewards'])})

    grads, (new_model_state, info) = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    if 'batch_stats' in new_model_state:
        new_critic = critic.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info