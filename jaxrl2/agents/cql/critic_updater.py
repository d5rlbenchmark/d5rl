from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

NUM_CQL_REPEAT = 4


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def extend_and_repeat_dict(obs_dict, axis, repeat):
    return {key:extend_and_repeat(val, axis, repeat) for key, val in obs_dict.items()}

def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool,
        critic_reduction: str, cql_alpha: float,
        max_q_backup: bool, dr3_coefficient: float,
        use_sarsa_backups: bool, bound_q_with_mc: bool
) -> Tuple[TrainState, Dict[str, float]]:

    key, key_pi, key_random = jax.random.split(key, num=3)
    if max_q_backup:
        next_observations_tiled = extend_and_repeat(
            batch['next_observations'], axis=1, repeat=NUM_CQL_REPEAT
        )
        next_observations_tiled = jnp.reshape(
            next_observations_tiled, [batch['next_observations'].shape[0] * NUM_CQL_REPEAT,
                                      batch['next_observations'].shape[1],
                                      batch['next_observations'].shape[2],
                                      batch['next_observations'].shape[3], batch['next_observations'].shape[4]])
    else:
        next_observations_tiled = batch['next_observations']

    # print("next_observations_tiled.keys():", next_observations_tiled.keys())
    dist = actor.apply_fn({'params': actor.params}, next_observations_tiled)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    # print("next_observations_tiled.keys():", next_observations_tiled.keys())
    next_qs = target_critic.apply_fn({'params': target_critic.params}, next_observations_tiled, next_actions)
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

    if use_sarsa_backups:
        """When SARSA needs to be done."""
        next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     next_observations_tiled, batch['next_actions'])
        next_q = next_qs.mean(axis=0)

    target_q = batch['rewards'] + discount * batch['masks'] * next_q

    if backup_entropy:
        target_q -= discount * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs


    # DR3 logging and loss computation
    def dr3_loss_fn(critic_params: Params):
        import pdb; pdb.set_trace()
        _, next_feat_ff, next_feat_conv = critic.apply_fn({'params': critic.params},
                                                      next_observations_tiled, next_actions, True)
        next_feat_ff = jnp.reshape(next_feat_ff, (2, batch['actions'].shape[0], NUM_CQL_REPEAT, -1))
        next_feat_conv = jnp.reshape(next_feat_conv,
                                     (2, batch['actions'].shape[0], NUM_CQL_REPEAT, -1))
        avg_feat_ff = jnp.mean(next_feat_ff, axis=2)
        avg_feat_conv = jnp.mean(next_feat_conv, axis=2)

        _, feat_ff, feat_conv = critic.apply_fn({'params': critic.params}, batch['observations'],
                                                batch['actions'], True)

        dr3_dot_produce_ff = jnp.mean(jnp.sum(avg_feat_ff * feat_ff, axis=-1))
        dr3_dot_product_conv = jnp.mean(jnp.sum(avg_feat_conv * feat_conv, axis=-1))
        return {
            'dr3_dot_prodct_ff': dr3_dot_product_ff,
            'dr3_dot_product_conv': dr3_dot_product_conv
        }


    # CQL sample actions
    new_observations_tiled = {}
    observations_tiled = extend_and_repeat_dict(batch['observations'], axis=1, repeat=NUM_CQL_REPEAT)
    for key, val in observations_tiled.items():
        new_shape = (batch['observations'][key].shape[0] * NUM_CQL_REPEAT,) + batch['observations'][key].shape[1:]
        new_observations_tiled[key] = jnp.reshape(val, new_shape)
    observations_tiled = new_observations_tiled

    # observations_tiled_ = extend_and_repeat(batch['observations']['pixels'], axis=1, repeat=NUM_CQL_REPEAT)
    # observations_tiled_ = {"pixels":jnp.reshape(
    #     observations_tiled_, [batch['observations']['pixels'].shape[0] * NUM_CQL_REPEAT,
    #                          batch['observations']['pixels'].shape[1],
    #                          batch['observations']['pixels'].shape[2],
    #                          batch['observations']['pixels'].shape[3], batch['observations']['pixels'].shape[4]])
    #                          }
    #
    # print("jnp.array_equal(observations_tiled_[\"pixels\"], observations_tiled[\"pixels\"]):", jnp.array_equal(observations_tiled_["pixels"], observations_tiled["pixels"]))
    # # assert jnp.array_equal(observations_tiled_["pixels"], observations_tiled["pixels"])
    policy_dist = actor.apply_fn({'params': actor.params}, observations_tiled)
    policy_actions, policy_log_probs = policy_dist.sample_and_log_prob(seed=key_pi)

    N = batch['observations']['state'].shape[0] if "state" in batch['observations'] else batch['observations']['pixels'].shape[0]
    random_actions = jax.random.uniform(
        key_random, shape=(N * NUM_CQL_REPEAT, policy_actions.shape[-1]),
        minval=-1.0, maxval=1.0

    )
    random_pi = (1.0/2.0) ** policy_actions.shape[-1]
    
    if bound_q_with_mc is not None:
        global  bound_q_with_mc_global
        bound_q_with_mc_global = bound_q_with_mc

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        bellman_loss = critic_loss

        # CQL loss
        q_pi = critic.apply_fn({'params': critic_params}, observations_tiled,
                                policy_actions)
        if bound_q_with_mc_global:
            mc_returns_tiled = jnp.reshape(jnp.repeat(batch['mc_returns'], NUM_CQL_REPEAT), (1, -1))
            mc_returns = jnp.repeat(mc_returns_tiled, q_pi.shape[0], axis=0)
            q_pi = jnp.maximum(q_pi, mc_returns)
            mc_bounded_rate = jnp.sum(q_pi==mc_returns) / jnp.sum(mc_returns==mc_returns)
            
        q_pi_for_is = (q_pi[0] - policy_log_probs, q_pi[1] - policy_log_probs)
        q_pi_for_is = (
            jnp.reshape(q_pi_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_pi_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_pi_for_is = jnp.stack(q_pi_for_is, axis=0)

        q_random = critic.apply_fn({'params': critic_params}, observations_tiled,
                                    random_actions)
        q_random_for_is = (q_random[0] - np.log(random_pi), q_random[1] - np.log(random_pi))
        q_random_for_is = (
            jnp.reshape(q_random_for_is[0], (batch['actions'].shape[0], NUM_CQL_REPEAT)),
            jnp.reshape(q_random_for_is[1], (batch['actions'].shape[0], NUM_CQL_REPEAT))
        )
        q_random_for_is = jnp.stack(q_random_for_is, axis=0)

        cat_q = jnp.concatenate([q_pi_for_is, q_random_for_is], axis=-1)
        lse_q = jax.scipy.special.logsumexp(cat_q, axis=-1)
        cql_loss_per_element = lse_q - qs
        cql_loss = cql_loss_per_element.mean()

        critic_loss = critic_loss + cql_alpha * cql_loss

        ## Logging only
        diff_rand_data = q_random.mean() - qs.mean()
        diff_pi_data = q_pi.mean() - qs.mean()
        
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
            'q_data_avg': qs.mean(),
            'q_data_max': qs.max(),
            'q_data_min': qs.min(),
            'q_pi_max': q_pi.max(),
            'q_pi_min': q_pi.min(),
            'diff_pi_data_mean': diff_pi_data,
            'diff_rand_data_mean': diff_rand_data,
            'target_actor_entropy': -next_log_probs.mean(),
            'rewards_mean': batch['rewards'].mean(),
            'actions_mean': batch['actions'].mean(),
            'actions_max': batch['actions'].max(),
            'actions_min': batch['actions'].min(),
        }
        
        if bound_q_with_mc_global:
            things_to_log['mc_bounded_rate'] = mc_bounded_rate

        return critic_loss, things_to_log

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    # dr3_loss_dict = dr3_loss_fn(critic.params)
    # info.extend(dr3_loss_dict)

    return new_critic, info
