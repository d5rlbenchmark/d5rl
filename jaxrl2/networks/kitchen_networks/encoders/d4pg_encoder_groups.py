from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

import jax
from jaxrl2.networks.kitchen_networks.constants import default_init



class D4PGEncoderGroups(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"
    groups: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        ### FOR DEBUGGING ###
        # x[..., :3] = 0
        # x[..., 3:6] = 1
        # x[..., 6:] = 10

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            # print("\n[Before] x.shape:", x.shape)
            x = nn.Conv(
                features * self.groups,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=default_init(),
                # kernel_init=jax.nn.initializers.constant(1), ### FOR DEBUGGING ###
                padding=self.padding,
                feature_group_count=self.groups,
            )(x)
            # print("[After] x.shape:", x.shape)
            x = nn.relu(x)

        ### FOR DEBUGGING ###
        # print("x[..., 0:256]:", x[..., 0:256])
        # print("x[..., 256:512]:", x[..., 256:512])
        # print("x[..., 512:]:", x[..., 512:])
        # print("x[0, 0]:", x[0, 0])

        return x.reshape((*x.shape[:-3], -1))
