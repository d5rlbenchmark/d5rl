import jax
from jax.lib import xla_bridge
import jax.numpy as jnp

print("\n\n" + "=" * 30 + " GPU TEST " + "=" * 30)

print("jax.__version__:", jax.__version__)
print("xla_bridge.get_backend().platform:", xla_bridge.get_backend().platform)
print("jax.devices():", jax.devices())
print("jax.default_backend():", jax.default_backend())

key = jax.random.PRNGKey(42)
x = jax.random.uniform(key, (5,))
print("x:", x)
y = jnp.ones_like(x)
print("y:", y)
print("x + y:", x + y)
print("=" * 70 + "\n")

import tensorflow as tf
print("tf.__version__:", tf.__version__)
print("tf.test.is_gpu_available():", tf.test.is_gpu_available(), "\n")
print("tf.test.is_gpu_available(cuda_only=True):", tf.test.is_gpu_available(cuda_only=True), "\n")
print("tf.config.list_physical_devices('\GPU\'):", tf.config.list_physical_devices('GPU'), "\n")
with tf.device('gpu:0'):
    x = tf.random.uniform((5, 1))
    print("x:", x)
    y = tf.ones_like(x)
    print("y:", y)
    print("x + y:", x + y)

print("=" * 70 + "\n")
print("Hello")
