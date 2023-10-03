from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


class PixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        assert (
            len(observations.keys()) <= 2
        ), "Can include only pixels and states fields."

        x = self.encoder(observations["pixels"])

        """
        resnet_34_v1
        x.shape (512,) for (128, 128, 9, 3)
        x.shape (512,) for (64, 64, 9, 3)

        resnetv2
        x.shape (131_072,) 128x128
        x.shape (32_768,) 64x64


        resnet_34_v1, groups=3
        input  (128, 128, 27)
        post conv1 (64, 64, 192)
        post maxpool1 (32, 32, 192)
        post block layer  (32, 32, 192)
        post block layer  (32, 32, 192)
        post block layer  (32, 32, 192)
        post block  (32, 32, 192)
        post block layer  (16, 16, 384)
        post block layer  (16, 16, 384)
        post block layer  (16, 16, 384)
        post block layer  (16, 16, 384)
        post block  (16, 16, 384)
        post block layer  (16, 16, 768)
        post block layer  (16, 16, 768)
        post block layer  (16, 16, 768)
        post block layer  (16, 16, 768)
        post block layer  (16, 16, 768)
        post block layer  (16, 16, 768)
        post block  (16, 16, 768)
        post block layer  (16, 16, 1536)
        post block layer  (16, 16, 1536)
        post block layer  (16, 16, 1536)
        post block  (16, 16, 1536)
        post flatten (1536,)
        mlp post flatten (100,)
        """

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)


        # xs = []
        # for i in range(3):
        #     x_ = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        #     x_ = nn.LayerNorm()(x_)
        #     x_ = nn.tanh(x_)
        #     xs.append(x_)
        #     print(f"[{i}] x_.shape:", x_.shape)
        #
        # x = jnp.concatenate(xs, axis=-1)
        # print(f"x.shape:", x.shape)



        if "states" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["states"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)



class PixelMultiplexerMultiple(nn.Module):
    encoders: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        assert (
            len(observations.keys()) <= 2
        ), "Can include only pixels and states fields."

        # # x = self.encoder(observations["pixels"])
        # xs = []
        # for i, encoder in enumerate(self.encoders):
        #     xs.append(encoder(observations["pixels"][..., i:i+3, :]))
        #
        # for i in range(len(xs)):
        #     print(f"xs[{i}].shape:", xs[i].shape)
        #
        # x = jnp.concatenate(xs, axis=-1)
        #
        #
        # print("x.shape:", x.shape)
        #
        # if self.stop_gradient:
        #     # We do not update conv layers with policy gradients.
        #     x = jax.lax.stop_gradient(x)
        #
        # x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        # x = nn.LayerNorm()(x)
        # x = nn.tanh(x)

        xs = []
        for i, encoder in enumerate(self.encoders):
            x_i = encoder(observations["pixels"][..., i:i+3, :])
            print(f"(post encoder) x_{i}.shape:", x_i.shape)

            if self.stop_gradient:
                # We do not update conv layers with policy gradients.
                x_i = jax.lax.stop_gradient(x_i)

            x_i = nn.Dense(self.latent_dim, kernel_init=default_init())(x_i)
            x_i = nn.LayerNorm()(x_i)
            x_i = nn.tanh(x_i)
            xs.append(x_i)

        for i in range(len(xs)):
            print(f"(post dense) xs[{i}].shape:", xs[i].shape)
        x = jnp.concatenate(xs, axis=-1)
        print("x.shape:", x.shape)

        if "states" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["states"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
