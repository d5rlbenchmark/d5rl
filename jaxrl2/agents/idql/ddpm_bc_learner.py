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
                                          DDPM, FourierFeatures, cosine_beta_schedule, PixelMultiplexer,
                                          state_ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule)
from jaxrl2.networks.jaxrl5_networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.kitchen_networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.types import Params, PRNGKey
import numpy as np

from flax.training import checkpoints ###===### ###---###

# from jaxrl2.agents.idql.ddpm_iql_learner import compute_q
@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

# Helps to minimize CPU to GPU transfer.

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


class DDPMBCLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    score_model: TrainState
    target_score_model: TrainState
    # data_augmentation_fn: Callable = struct.field(pytree_node=False)
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
    tau: float
    discount: float
    expectile: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
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
        N: int = 1,
        M: int = 0,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        clip_sampler: bool = True,
        ddpm_temperature: float = 1.0,
        actor_tau: float = 0.001,
        tau: float = 0.005,
        discount: float = 0.99,
        expectile: float = 0.7,
        num_qs: int = 2,
        decay_steps: Optional[int] = None,
        use_multiplicative_cond=False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

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

            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=use_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)

            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')

        time = jnp.zeros((1, ))

        

        
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        target_actor = TrainState.create(
                        apply_fn=actor_def.apply,
                        params=actor_params,
                        tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        

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

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)

        
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_def = StateValue(base_cls=critic_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=value_lr),
        )

        return cls(
            rng=rng,
            actor=None,
            critic=critic,
            target_critic=target_critic,
            value=value,
            score_model=actor,
            target_score_model=target_actor,
            N=N,
            M=M,
            T=T,
            # data_augmentation_fn=data_augmentation_fn,
            betas=betas,
            alpha_hats=alpha_hat,
            alphas=alphas,
            act_dim=action_dim,
            clip_sampler=clip_sampler,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            tau=tau,
            discount=discount,
            expectile=expectile,
        )

    def update_actor(agent, batch: DatasetDict):
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

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            value_loss = expectile_loss(q - v, agent.expectile).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)

        agent = agent.replace(value=value)

        return agent, info

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    @jax.jit
    def update(self, batch: DatasetDict):
        agent = self

        # if "pixels" not in batch["next_observations"]:
        #     batch = _unpack(batch)

        # value = _share_encoder(source=agent.critic, target=agent.value)
        # agent = agent.replace(value=value)

        # Not sure if actor should share encoder with critic, but it's fully disconnected so it probably doesn't matter...?

        # rng, key = jax.random.split(agent.rng)
        # observations = self.data_augmentation_fn(key, batch["observations"])
        # rng, key = jax.random.split(rng)
        # next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        # batch = batch.copy(
        #     add_or_replace={
        #         "observations": observations,
        #         "next_observations": next_observations,
        #     }
        # )

        # batch_size = batch['observations']['pixels'].shape[0]
        batch_size = batch['observations'].shape[0]

        def first_half(x):
            return x[:batch_size//2]

        def second_half(x):
            return x[batch_size//2:]

        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        agent, _ = agent.update_actor(first_batch)
        agent, actor_info = agent.update_actor(second_batch) #Take two steps on actor for every step on critic

        critic_batch_size = min(256, batch_size)

        def slice(x):
            return x[:critic_batch_size]

        mini_batch = jax.tree_util.tree_map(slice, batch)

        agent, v_info = agent.update_v(mini_batch)
        agent, q_info = agent.update_q(mini_batch)

        info = {**actor_info, **v_info, **q_info}

        return agent, info

    @jax.jit
    def update_online(self, batch: DatasetDict):
        #Don't update actor during online finetuning
        agent = self

        # if "pixels" not in batch["next_observations"]:
        #     batch = _unpack(batch)

        # value = _share_encoder(source=agent.critic, target=agent.value)
        # new_agent = agent.replace(value=value)

        # rng, key = jax.random.split(agent.rng)
        # observations = self.data_augmentation_fn(key, batch["observations"])
        # rng, key = jax.random.split(rng)
        # next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        # batch = batch.copy(
        #     add_or_replace={
        #         "observations": observations,
        #         "next_observations": next_observations,
        #     }
        # )

        def slice(x):
            return x[:256]

        batch = jax.tree_util.tree_map(slice, batch)

        agent, v_info = agent.update_v(batch)
        agent, q_info = agent.update_q(batch)

        info = {**v_info, **q_info}

        return agent, info

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)

        # observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)
        # observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0).repeat(self.N, axis = 0), observations)
        observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0).repeat(1, axis = 0), observations)

        score_params = self.target_score_model.params
        actions, rng = state_ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        idx = jnp.argmax(qs)
        action = actions[idx]
        new_rng = rng

        return action.squeeze(), self.replace(rng=new_rng)

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        return self.eval_actions(observations) #Just take argmax for online finetuning

    ###===###
    @property
    def _save_dict(self):
        save_dict = {
            'score_model': self.score_model,
            'target_score_model': self.target_score_model,
            "value":self.value,
            "critic":self.critic,
            "target_critic":self.target_critic,
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
        self.value = output_dict["value"]
        self.critic = output_dict["critic"]
        self.target_critic = output_dict["target_critic"]
    ###---###