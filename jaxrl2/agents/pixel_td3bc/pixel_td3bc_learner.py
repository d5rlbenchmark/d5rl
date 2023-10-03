"""Implementations of algorithms for continuous control."""
from jaxrl2.networks.learned_std_normal_policy import LearnedStdNormalPolicy
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_td3bc.actor_updater import update_actor
from jaxrl2.agents.pixel_td3bc.critic_updater import update_critic
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.normal_policy import NormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('color_jitter', 'share_encoders', 'aug_next'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, target_actor_params: Params, critic: TrainState,
    target_critic_params: Params, batch: TrainState,
    discount: float, tau: float, alpha: float, color_jitter: bool, share_encoders: bool, aug_next: bool
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:

    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        # randmo crop
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.) * 255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    target_actor = actor.replace(params=target_actor_params)
    target_critic = critic.replace(params=target_critic_params)
    key, rng = jax.random.split(rng)

    new_critic, critic_info = update_critic(key, target_actor, critic, target_critic, batch, discount)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, critic, batch, alpha)

    new_target_critic_params = soft_target_update(new_critic.params,
                                                  target_critic_params, tau)
    new_target_actor_params = soft_target_update(new_actor.params, target_actor_params, tau)
    return rng, new_actor, new_target_actor_params, new_critic, new_target_critic_params, {
        **critic_info,
        **actor_info,
        'pixels': aug_pixels
    }


class PixelTD3BCLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 2.0,
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='batch',
                 policy_type='unit_std_normal',
                 policy_std=1.,
                 color_jitter = True,
                 share_encoders = False,
                 mlp_init_scale=1.,
                 mlp_output_scale=1.,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=False,
                 use_bottleneck=True
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.share_encoders = share_encoders

        action_dim = actions.shape[-1]

        self.tau = tau
        self.discount = discount
        self.alpha = alpha

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if policy_type == 'unit_std_normal':
            policy_def = NormalPolicy(hidden_dims,
                                      action_dim,
                                      dropout_rate=dropout_rate,
                                      std=policy_std,
                                      init_scale=mlp_init_scale,
                                      output_scale=mlp_output_scale,
                                      )
        elif policy_type == 'learned_std_normal':
            policy_def = LearnedStdNormalPolicy(hidden_dims,
                                            action_dim,
                                            dropout_rate=dropout_rate)
        else:
            raise ValueError('policy type not found!')


        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     stop_gradient=share_encoders,
                                     use_bottleneck=use_bottleneck
                                     )
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        critic_def_init = critic_def.init(critic_key, observations,
                                        actions)
        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)
        target_actor_params = copy.deepcopy(actor_params)

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._target_actor_params = target_actor_params

    def update(self, batch: FrozenDict, i=-1) -> Dict[str, float]:
        new_rng, new_actor, new_target_actor, new_critic, new_target_critic,info = _update_jit(
            self._rng, self._actor, self._target_actor_params, self._critic, self._target_critic_params,
            batch, self.discount, self.tau, self.alpha,
            self.color_jitter, self.share_encoders, self.aug_next
            )

        self._rng = new_rng
        self._actor = new_actor
        self._target_actor_params = new_target_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic

        return info

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            "target_actor_params": self._target_actor_params,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._target_actor_params = output_dict['target_actor_params']
        print('restored from ', dir)