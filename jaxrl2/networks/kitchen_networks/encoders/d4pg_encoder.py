from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.kitchen_networks.constants import default_init


class D4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train=True) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
