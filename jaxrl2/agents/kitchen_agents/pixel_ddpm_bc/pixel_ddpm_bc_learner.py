from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple, Union, Dict

import gym
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState
import jax.numpy as jnp
import flax.linen as nn
from jaxrl2.agents.drq.augmentations import batched_random_crop
from jaxrl2.data.kitchen_data.dataset import DatasetDict
from jaxrl2.networks.jaxrl5_networks import (MLP, Ensemble, StateActionValue, StateValue,
                                          DDPM, FourierFeatures, cosine_beta_schedule,
                                          ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule, PixelMultiplexer)
from jaxrl2.networks.jaxrl5_networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.types import Params, PRNGKey
import numpy as np

from jaxrl2.networks.kitchen_networks.encoders import ImpalaEncoder
# from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer as PixelMultiplexerJaxRL2

from flax.training import checkpoints ###===### ###---###

def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][..., :-1]
            next_obs_pixels = batch["observations"][pixel_key][..., 1:]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=self.rng)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)

    ###===###
    @property
    def _save_dict(self):
        raise NotImplementedError

    def save_checkpoint(self, dir, step, keep_every_n_steps):
        checkpoints.save_checkpoint(dir, self._save_dict, step, prefix='checkpoint', overwrite=False, keep_every_n_steps=keep_every_n_steps)

    def restore_checkpoint(self, dir):
        raise NotImplementedError
    ###---###

class PixelDDPMBCLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    actor_tau: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        use_layer_norm: bool = True,
        dropout_rate: Optional[float] = 0.1,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        time_dim: int = 64,
        actor_architecture: str = 'ln_resnet',
        actor_num_blocks: int = 3,
        beta_schedule: str = 'cosine',
        T: int = 20,
        clip_sampler: bool = True,
        ddpm_temperature: float = 1.0,
        actor_tau: float = 0.001,
        decay_steps: Optional[int] = None,
        use_multiplicative_cond=False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = actions.shape[0]

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(2*time_dim, time_dim),
                                activations=mish,
                                activate_final=False)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=use_layer_norm,
                                    activate_final=False)

            actor_cls = partial(DDPM, time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=use_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)

            actor_cls = partial(DDPM, time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')

        time = jnp.zeros((1,))

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )

            actor_cls = PixelMultiplexer(
                encoder_cls=encoder_cls,
                network_cls=actor_cls,
                latent_dim=latent_dim,
                pixel_keys=pixel_keys,
                depth_keys=depth_keys,
            )

        elif encoder == "impala":
            # encoder_cls = ImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
            encoder_cls = partial(ImpalaEncoder, use_multiplicative_cond=use_multiplicative_cond)
            # actor_cls = PixelMultiplexerJaxRL2(
            #     encoder=encoder_cls,
            #     network=actor_cls,
            #     latent_dim=latent_dim,
            # )

            print("encoder_cls:", encoder_cls)

            actor_cls = PixelMultiplexer(
                encoder_cls=encoder_cls,
                network_cls=actor_cls,
                latent_dim=latent_dim,
                pixel_keys=pixel_keys,
                depth_keys=depth_keys,
                skip_normalization=True,
            )
            # actor_cls = PixelMultiplexer(encoder_cls=encoder_cls, network_cls=actor_cls, latent_dim=latent_dim, pixel_keys=pixel_keys, depth_keys=depth_keys, skip_normalization=True)
        elif encoder == "resnet":
            # encoder_cls = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
            encoder_cls = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2))

            actor_cls = PixelMultiplexer(
                encoder_cls=encoder_cls,
                network_cls=actor_cls,
                latent_dim=latent_dim,
                pixel_keys=pixel_keys,
                depth_keys=depth_keys,
            )

            actor_cls = PixelMultiplexer(
                encoder_cls=encoder_cls,
                network_cls=actor_cls,
                latent_dim=latent_dim,
                pixel_keys=pixel_keys,
                depth_keys=depth_keys,
            )
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        # actor_cls = PixelMultiplexer(
        #     encoder_cls=encoder_cls,
        #     network_cls=actor_cls,
        #     latent_dim=latent_dim,
        #     pixel_keys=pixel_keys,
        #     depth_keys=depth_keys,
        # )
        #observations = jnp.expand_dims(observations, axis=0)
        #actions = jnp.expand_dims(actions, axis=0)
        #import pdb; pdb.set_trace()
        actor_params = actor_cls.init(actor_key, observations, actions, time)["params"]
        actor = TrainState.create(
            apply_fn=actor_cls.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        target_actor = TrainState.create(
                        apply_fn=actor_cls.apply,
                        params=actor_params,
                        tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            rng=rng,
            actor=None,
            score_model=actor,
            target_score_model=target_actor,
            N=1,
            M=0,
            T=T,
            data_augmentation_fn=data_augmentation_fn,
            betas=betas,
            alpha_hats=alpha_hat,
            alphas=alphas,
            act_dim=action_dim,
            clip_sampler=clip_sampler,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
        )

    @jax.jit
    def update(self, batch: DatasetDict):
        agent = self

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, key = jax.random.split(agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(add_or_replace={"observations": observations})

        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))

        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()

            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)

        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    # @jax.jit
    # def eval_actions(self, observations: jnp.ndarray):
    #     rng = self.rng
    #
    #     score_params = self.target_score_model.params
    #     actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
    #
    #     return jnp.array(actions.squeeze()), self.replace(rng=rng)
    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        # observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0).repeat(self.N, axis = 0), observations) #Add dim
        observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0).repeat(1, axis = 0), observations) #Add dim
        score_params = self.target_score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)

        return jnp.array(actions.squeeze()), self.replace(rng=rng)

    ###===###
    @property
    def _save_dict(self):
        save_dict = {
            'score_model': self.score_model,
            'target_score_model': self.target_score_model,
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
        # self._actor = output_dict['actor']
        self.score_model = output_dict["score_model"]
        self.target_score_model = output_dict["target_score_model"]
    ###---###
