"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.drq.drq_learner import _share_encoder, _unpack
from jaxrl2.agents.kitchen_agents.iql.actor_updater import update_actor
from jaxrl2.agents.kitchen_agents.iql.critic_updater import update_q, update_v
from jaxrl2.data.kitchen_data.dataset import DatasetDict
from jaxrl2.networks.kitchen_networks.encoders import ResNetV2Encoder, ImpalaEncoder
from jaxrl2.networks.kitchen_networks.encoders.resnet_encoderv1 import GroupConvWrapper, ResNet18, ResNet34, ResNetSmall, ResNet50
from jaxrl2.networks.kitchen_networks.encoders import D4PGEncoder, D4PGEncoderGroups ###===### ###---###
from jaxrl2.networks.kitchen_networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.kitchen_networks.pixel_multiplexer import PixelMultiplexer, PixelMultiplexerMultiple
from jaxrl2.networks.kitchen_networks.values import StateActionEnsemble, StateValue
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update

import os ###===###
from flax.training import checkpoints
from glob import glob ###---###


@functools.partial(jax.jit, static_argnames=("critic_reduction", "share_encoder"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    value: TrainState,
    batch: TrainState,
    discount: float,
    tau: float,
    expectile: float,
    A_scaling: float,
    critic_reduction: str,
    share_encoder: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    batch = _unpack(batch)

    if share_encoder:
        actor = _share_encoder(source=critic, target=actor)
        value = _share_encoder(source=critic, target=value)

    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, key = jax.random.split(rng)
    aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
    next_observations = batch["next_observations"].copy(
        add_or_replace={"pixels": aug_next_pixels}
    )
    batch = batch.copy(add_or_replace={"next_observations": next_observations})

    target_critic = critic.replace(params=target_critic_params)
    new_value, value_info = update_v(
        target_critic, value, batch, expectile, critic_reduction
    )
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, target_critic, new_value, batch, A_scaling, critic_reduction
    )

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_value,
        {**critic_info, **value_info, **actor_info},
    )


class PixelIQLLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        cnn_groups: int = 1, ###===### ###---###
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.9,
        A_scaling: float = 10.0,
        critic_reduction: str = "min",
        dropout_rate: Optional[float] = None,
        share_encoder: bool = False,
        encoder: str = "d4pg",
        encoder_norm: str = 'batch',
        use_spatial_softmax=False,
        softmax_temperature=-1,
        use_multiplicative_cond=False,
        use_spatial_learned_embeddings=False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        # assert observations["pixels"].shape[-2] / cnn_groups == 3, f"observations['pixels'].shape: {observations['pixels'].shape}, cnn_groups: {cnn_groups}"


        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.A_scaling = A_scaling
        self.share_encoder = share_encoder

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        # encoder_defs = []
        # for i in range(cnn_groups):
        if encoder == "d4pg":
            encoder_def = D4PGEncoder(cnn_features, cnn_filters, cnn_strides, cnn_padding)
            # encoder_def = D4PGEncoderGroups(cnn_features, cnn_filters, cnn_strides, cnn_padding, cnn_groups) ###===### ###---###
        elif encoder == "impala":
            encoder_def = ImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
        elif encoder == "resnet":
            encoder_def = ResNetV2Encoder((2, 2, 2, 2))
        elif encoder == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                   num_spatial_blocks=8)
        elif encoder == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                   num_spatial_blocks=8,)
            # encoder_defs.append(encoder_def)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = UnitStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate, apply_tanh=False
        )
        # actor_def = PixelMultiplexerMultiple(
        #     encoders=encoder_defs,
        #     network=policy_def,
        #     latent_dim=latent_dim,
        #     stop_gradient=share_encoder,
        # )
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=share_encoder,
        )

        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(hidden_dims, num_qs=2)
        # critic_def = PixelMultiplexerMultiple(
        #     encoders=encoder_defs, network=critic_def, latent_dim=latent_dim
        # )
        critic_def = PixelMultiplexer(
            encoder=encoder_def, network=critic_def, latent_dim=latent_dim
        )

        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        # value_def = PixelMultiplexerMultiple(
        #     encoders=encoder_defs,
        #     network=value_def,
        #     latent_dim=latent_dim,
        #     stop_gradient=share_encoder,
        # )
        value_def = PixelMultiplexer(
            encoder=encoder_def,
            network=value_def,
            latent_dim=latent_dim,
            stop_gradient=share_encoder,
        )
        value_params = value_def.init(value_key, observations)["params"]
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=value_lr),
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._value = value

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_value,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._value,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.A_scaling,
            self.critic_reduction,
            self.share_encoder,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        return info

    ###===###
    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            "value":self._value,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        if os.path.isfile(dir):
            checkpoint_file = dir
        else:
            def sort_key_fn(checkpoint_file):
                chkpt_name = checkpoint_file.split("/")[-1]
                return int(chkpt_name[len("checkpoint"):])

            checkpoint_files = glob(os.path.join(dir, "checkpoint*"))
            checkpoint_files = sorted(checkpoint_files, key=sort_key_fn)
            checkpoint_file = checkpoint_files[-1]

        output_dict = checkpoints.restore_checkpoint(checkpoint_file, self._save_dict)
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._actor = output_dict['actor']
        self._value = output_dict["value"]
    ###---###
