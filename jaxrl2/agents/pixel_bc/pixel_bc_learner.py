"""Implementations of algorithms for continuous control."""

from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
# from flax.training import train_state

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.bc.actor_updater import log_prob_update
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.drq.drq_learner import _unpack
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders import ResNetV2Encoder, ImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import GroupConvWrapper, ResNet18, ResNet34, ResNetSmall, ResNet50
from jaxrl2.networks.encoders import D4PGEncoder, D4PGEncoderGroups ###===### ###---###
from jaxrl2.networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer, PixelMultiplexerMultiple
from jaxrl2.types import Params, PRNGKey


import os ###===###
from flax.training import checkpoints
from glob import glob ###---###

from functools import partial
from typing import Any


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    # batch = _unpack(batch)
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']

    if batch['observations']['pixels'].squeeze().ndim != 2:
        # random crop
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        rng, key = jax.random.split(rng)
        aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    rng, new_actor, actor_info = log_prob_update(rng, actor, batch)

    return rng, new_actor, actor_info


class PixelBCLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        cnn_groups: int = 1,  ###===### ###---###
        latent_dim: int = 50,
        dropout_rate: Optional[float] = None,
        encoder: str = "d4pg",
        encoder_norm: str = 'batch',
        use_spatial_softmax=False,
        softmax_temperature=-1,
        use_multiplicative_cond=False,
        use_spatial_learned_embeddings=False,
        **kwargs,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        # assert observations["pixels"].shape[-2] / cnn_groups == 3, f"observations['pixels'].shape: {observations['pixels'].shape}, cnn_groups: {cnn_groups}"

        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

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

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = UnitStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def, network=policy_def, latent_dim=latent_dim
        )


        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        self._rng = rng
        self._actor = actor

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, info = _update_jit(self._rng, self._actor, batch)

        self._rng = new_rng
        self._actor = new_actor

        return info

    ###===###
    @property
    def _save_dict(self):
        save_dict = {
            'actor': self._actor,
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
        self._actor = output_dict['actor']
    ###---###
