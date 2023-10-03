from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey

def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                batch: FrozenDict, alpha: float
                 ) -> Tuple[TrainState, Dict[str, float]]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                                  batch['observations'],
                                  training=True,
                                  mutable=['batch_stats'],
                                  rngs={'dropout': key})
        else:
            dist = actor.apply_fn({'params': actor_params},
                                  batch['observations'],
                                  training=True,
                                  rngs={'dropout': key})
            new_model_state = {}

        lim = 1 - 1e-5
        actions = jnp.clip(dist.mode(), -lim, lim)
        mse_loss = ((actions - batch['actions']) ** 2).mean()

        qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        abs_q = jnp.abs(qs.mean(axis=0))
        lmda = jax.lax.stop_gradient(alpha / abs_q.mean())

        actor_loss = -lmda * qs.mean(axis=0).mean() + mse_loss

        infos = {'actor_loss': actor_loss, 'lmda': lmda, 'mse_loss': mse_loss, 'actor_qpi_avg': qs.mean(axis=0).mean()}
        return actor_loss, (new_model_state, infos)

    grads, (new_model_state, info) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info