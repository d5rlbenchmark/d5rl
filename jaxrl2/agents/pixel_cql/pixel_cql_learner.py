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
# from jaxrl2.agents.iql.actor_updater import update_actor
# from jaxrl2.agents.iql.critic_updater import update_q, update_v

from jaxrl2.agents.cql.actor_updater import update_actor
from jaxrl2.agents.cql.critic_updater import update_critic
from jaxrl2.agents.cql.temperature_updater import update_temperature
from jaxrl2.agents.cql.temperature import Temperature
from jaxrl2.networks.normal_tanh_policy import NormalTanhPolicy

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import ResNetV2Encoder, ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import GroupConvWrapper, ResNet18, ResNet34, ResNetSmall, ResNet50
from jaxrl2.networks.encoders import D4PGEncoder, D4PGEncoderGroups ###===### ###---###
# from jaxrl2.networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer, PixelMultiplexerMultiple
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update

import os ###===###
from flax.training import checkpoints
from glob import glob ###---###

@functools.partial(jax.jit, static_argnames=("critic_reduction", "share_encoder", 'backup_entropy', 'max_q_backup', 'use_sarsa_backups', 'bound_q_with_mc'))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: TrainState,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
    cql_alpha: float,
    max_q_backup: bool,
    dr3_coefficient: float,
    use_sarsa_backups: bool,
    bound_q_with_mc: bool,
    share_encoder: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    batch = _unpack(batch)

    if share_encoder:
        actor = _share_encoder(source=critic, target=actor)

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

    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy,
                                            critic_reduction=critic_reduction,
                                            cql_alpha=cql_alpha,
                                            max_q_backup=max_q_backup,
                                            dr3_coefficient=dr3_coefficient,
                                            use_sarsa_backups=use_sarsa_backups,
                                            bound_q_with_mc=bound_q_with_mc)
    new_target_critic_params = soft_target_update(new_critic.params,
                                                  target_critic_params, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'],
                                              target_entropy)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class PixelCQLLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        cnn_groups: int = 1, ###===### ###---###
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.0,
        cql_alpha: float = 0.0,
        backup_entropy: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        max_q_backup: bool = False,
        dr3_coefficient: float = 0.0,
        use_sarsa_backups: bool = False,
        bound_q_with_mc: bool = False,
        critic_reduction: str = 'min',
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

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.max_q_backup = max_q_backup
        self.dr3_coefficient = dr3_coefficient
        self.use_sarsa_backups = use_sarsa_backups
        self.bound_q_with_mc = bound_q_with_mc
        self.share_encoder = share_encoder

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

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
        # policy_def = UnitStdNormalPolicy(hidden_dims, action_dim, dropout_rate=dropout_rate, apply_tanh=False)
        policy_def = NormalTanhPolicy(hidden_dims, action_dim)
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
        critic_def = PixelMultiplexerMultiple(
            encoders=encoder_defs, network=critic_def, latent_dim=latent_dim
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._temp = temp
        self._target_critic_params = target_critic_params
        self._cql_alpha = cql_alpha

        print ('Discount: ', self.discount)
        print ('CQL Alpha: ', self._cql_alpha)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
            self._cql_alpha,
            self.max_q_backup,
            self.dr3_coefficient,
            self.use_sarsa_backups,
            self.bound_q_with_mc,
            self.share_encoder,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp

        return info

    ###===###
    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            "temp":self._temp,
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
        self._temp = output_dict["temp"]

@functools.partial(jax.jit)
def get_q_value(critic, obs, act):
    return critic.apply_fn({'params': critic.params}, obs, act)

# @functools.partial(jax.jit)
# def get_q_value(actions, observations, critic):
#     # q_pred = critic.apply_fn({'params': critic.params}, {"pixels":images[..., None]}, actions)
#     # q_pred = critic.apply_fn({'params': critic.params}, {"pixels":images[..., :3, :-1]}, actions)
#     # q_pred = critic.apply_fn({'params': critic.params}, {"pixels":images[..., :-1]}, actions)
#     obs = observations.copy()
#     obs["pixels"] = observations["pixels"][..., :-1]
#     q_pred = critic.apply_fn({'params': critic.params}, obs, actions)
#     return q_pred

    ###---###
