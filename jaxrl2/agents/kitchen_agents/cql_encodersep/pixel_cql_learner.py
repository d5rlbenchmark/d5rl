"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flax.training import checkpoints
import pathlib

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple
import pickle

import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze, unfreeze
from flax.training import train_state

from jaxrl2.agents.agent import Agent

from jaxrl2.agents.kitchen_agents.cql_encodersep.actor_updater import update_actor
from jaxrl2.agents.kitchen_agents.cql_encodersep.critic_updater import update_critic
from jaxrl2.agents.kitchen_agents.cql_encodersep.temperature_updater import update_temperature
from jaxrl2.agents.kitchen_agents.cql_encodersep.temperature import Temperature

from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.types import Params, PRNGKey

from jaxrl2.agents.agent import Agent
from jaxrl2.networks.kitchen_networks.learned_std_normal_policy import LearnedStdNormalPolicy, LearnedStdTanhNormalPolicy
from jaxrl2.agents.drq.augmentations import batched_random_crop, color_transform
# from jaxrl2.agents.common import _unpack
from jaxrl2.agents.drq.drq_learner import _unpack
from jaxrl2.agents.drq.drq_learner import _share_encoder
# from jaxrl2.networks.kitchen_networks.encoders.networks import Encoder, PixelMultiplexer, PixelMultiplexerEncoder, PixelMultiplexerDecoder
from jaxrl2.networks.kitchen_networks.encoders.networks import Encoder, PixelMultiplexerEncoder, PixelMultiplexerDecoder
from jaxrl2.networks.kitchen_networks.pixel_multiplexer import PixelMultiplexer

from jaxrl2.networks.kitchen_networks.encoders.impala_encoder import ImpalaEncoder
from jaxrl2.networks.kitchen_networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.kitchen_networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.kitchen_networks.encoders import D4PGEncoder
from jaxrl2.data.kitchen_data.dataset import DatasetDict
from jaxrl2.networks.kitchen_networks.normal_policy import NormalPolicy
from jaxrl2.networks.kitchen_networks.values import StateActionEnsemble, StateValue
from jaxrl2.networks.kitchen_networks.values.state_action_value import StateActionValue
from jaxrl2.networks.kitchen_networks.values.state_value import StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update
# from jaxrl_m.vision import bigvision_resnetv2 as resnet
# from jaxrl_m.vision import encoders as encoders
from icecream import ic
import wandb

import numpy as np
from typing import Any

class TrainState(train_state.TrainState):
    batch_stats: Any = None

@functools.partial(jax.jit, static_argnames=['critic_reduction', 'backup_entropy', 'max_q_backup', 'method', 'method_type', 'cross_norm', 'color_jitter', 'tr_penalty_coefficient', 'mc_penalty_coefficient', 'bound_q_with_mc', 'online_bound_nstep_return'])
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic_encoder: TrainState,
    critic_decoder: TrainState, target_critic_encoder_params: Params,
    target_critic_decoder_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float, backup_entropy: bool,
    critic_reduction: str, cql_alpha: float, max_q_backup: bool, dr3_coefficient: float,tr_penalty_coefficient:float, mc_penalty_coefficient:float, pretrained_critic_encoder: TrainState,
    method:bool = False, method_const:float = 0.0, method_type:int=0,
    cross_norm:bool = False, color_jitter:bool = False, bound_q_with_mc:bool = False, online_bound_nstep_return:int=-1
    ) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:

    # Comment out when using the naive replay buffer
    batch = _unpack(batch)

    # print("batch[\'observations\'][\'pixels\'].shape:", batch['observations']['pixels'].shape)

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

    # print("batch[\'observations\'][\'pixels\'].shape:", batch['observations']['pixels'].shape)

    key, rng = jax.random.split(rng)
    if True:
        target_critic_encoder = critic_encoder.replace(params=target_critic_encoder_params)
        target_critic_decoder = critic_decoder.replace(params=target_critic_decoder_params)

        (new_critic_encoder, new_critic_decoder), critic_info = update_critic(
            key,
            actor,
            critic_encoder,
            critic_decoder,
            target_critic_encoder,
            target_critic_decoder,
            temp,
            batch,
            discount,
            backup_entropy=backup_entropy,
            critic_reduction=critic_reduction,
            cql_alpha=cql_alpha,
            max_q_backup=max_q_backup,
            dr3_coefficient=dr3_coefficient,
            tr_penalty_coefficient=tr_penalty_coefficient,
            mc_penalty_coefficient=mc_penalty_coefficient,
            pretrained_critic_encoder=pretrained_critic_encoder,
            method=method,
            method_const=method_const,
            method_type=method_type,
            cross_norm=cross_norm,
            bound_q_with_mc=bound_q_with_mc,
            online_bound_nstep_return=online_bound_nstep_return
        )
        new_target_critic_encoder_params = soft_target_update(new_critic_encoder.params, target_critic_encoder_params, tau)
        new_target_critic_decoder_params = soft_target_update(new_critic_decoder.params, target_critic_decoder_params, tau)
    else:
        new_critic_encoder, new_critic_decoder, critic_info, new_target_critic_encoder_params, new_target_critic_decoder_params = \
            critic_encoder, critic_decoder, {}, target_critic_encoder_params, target_critic_decoder_params

    if True:
        rng, key = jax.random.split(rng)
        new_actor, actor_info = update_actor(key, actor, new_critic_encoder, new_critic_decoder, temp, batch, cross_norm=cross_norm)
        new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    else:
        new_actor, actor_info, new_temp, alpha_info = actor, {}, temp, {}

    return rng, new_actor, (new_critic_encoder, new_critic_decoder), (new_target_critic_encoder_params, new_target_critic_decoder_params), new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
        # 'pixels': aug_pixels  # only use for debugging, since it may cause slowdown!
    }


class PixelCQLLearnerEncoderSep(Agent):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_filters: Sequence[int] = (3, 3, 3, 3),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 cql_alpha: float = 0.0,
                 tau: float = 0.0,
                 backup_entropy: bool = False,
                 target_entropy: Optional[float] = None,
                 critic_reduction: str = 'min',
                 dropout_rate: Optional[float] = None,
                 init_temperature: float = 1.0,
                 max_q_backup: bool = False,
                 policy_encoder_type: str = 'resnet_small',
                 encoder_type: str ='resnet_small',
                 encoder_resize_dim: int = 128,
                 encoder_norm: str = 'batch',
                 dr3_coefficient: float = 0.0,
                 tr_penalty_coefficient:float = 0.0,
                 mc_penalty_coefficient:float = 0.0,
                 method:bool = False,
                 method_const:float = 0.0,
                 method_type:int=0,
                 cross_norm:bool = False,
                 use_spatial_softmax=True,
                 use_multiplicative_cond=False,
                 softmax_temperature=-1,
                 share_encoders=False,
                 color_jitter=True,
                 use_bottleneck=True,
                 adam_weight_decay=None,
                 freeze_encoders_actor=False,
                 freeze_encoders_critic=False,
                 wait_actor_update=-1,
                 bound_q_with_mc=False,
                 online_bound_nstep_return=-1,
                 mae_type='vc1',
                 **kwargs,
        ):
        print('Unused', kwargs)

        self.color_jitter=color_jitter

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.max_q_backup = max_q_backup
        self.dr3_coefficient = dr3_coefficient

        self.method = method
        self.method_const = method_const
        self.method_type = method_type
        self.cross_norm = cross_norm
        self.tr_penalty_coefficient = tr_penalty_coefficient
        self.mc_penalty_coefficient = mc_penalty_coefficient
        self._pretrained_critic_encoder = None
        self.wait_actor_update = wait_actor_update
        self.bound_q_with_mc = bound_q_with_mc
        self.online_bound_nstep_return = online_bound_nstep_return

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        rng, noise1_key, noise2_key, noise3_key, drop1_key, drop2_key, drop3_key = jax.random.split(rng, 7)


        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
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
        elif encoder_type == 'resnetv2-50-1':
            encoder_def = encoders[encoder_type]()
        elif encoder_type == 'mae':
            from jaxrl2.networks import mae, loading_utils
            encoder_def = mae.MAEEncoder()
        elif encoder_type == "d4pg":
            encoder_def = D4PGEncoder(cnn_features, cnn_filters, cnn_strides, cnn_padding)
        else:
            raise ValueError('encoder type not found!')

        if policy_encoder_type == 'small':
            policy_encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif policy_encoder_type == 'impala':
            print('using impala')
            policy_encoder_def = ImpalaEncoder(use_multiplicative_cond=use_multiplicative_cond)
        elif policy_encoder_type == 'resnet_small':
            policy_encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_18_v1':
            policy_encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_34_v1':
            policy_encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif policy_encoder_type == 'resnet_small_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_18_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif policy_encoder_type == 'resnet_34_v2':
            policy_encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        elif policy_encoder_type == 'resnetv2-50-1':
            policy_encoder_def = encoders[policy_encoder_type]()
        elif policy_encoder_type == 'same':
            policy_encoder_def = encoder_def
        elif policy_encoder_type == 'mae':
            from jaxrl2.networks import mae, loading_utils
            policy_encoder_def = mae.MAEEncoder()
        elif policy_encoder_type == "d4pg":
            policy_encoder_def = D4PGEncoder(cnn_features, cnn_filters, cnn_strides, cnn_padding)
        else:
            raise ValueError('encoder type not found!')

        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, action_dim, dropout_rate=dropout_rate)

        # actor_def = PixelMultiplexer(encoder=policy_encoder_def,
        #                              network=policy_def,
        #                              latent_dim=latent_dim,
        #                              stop_gradient=share_encoders or freeze_encoders_actor,
        #                              use_bottleneck=use_bottleneck)
        actor_def = PixelMultiplexer(encoder=policy_encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim)


        actor_def_init = actor_def.init({'params': actor_key, 'noise':noise1_key, 'drop_path':drop1_key}, observations)
        actor_params = actor_def_init['params']
        if policy_encoder_type == 'mae':
            actor_params = loading_utils.load_pytorch_weights(mae_type, actor_params)

        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  batch_stats=actor_batch_stats,
                                  tx=optax.adam(learning_rate=actor_lr))

        network_def = StateActionEnsemble(hidden_dims, num_qs=2)
        critic_def_encoder = PixelMultiplexerEncoder(encoder=encoder_def,latent_dim=latent_dim, use_bottleneck=use_bottleneck, stop_gradient=freeze_encoders_critic)
        critic_def_decoder = PixelMultiplexerDecoder(network=network_def)

        critic_key_encoder, critic_key_decoder = jax.random.split(critic_key, 2)

        critic_def_encoder_init = critic_def_encoder.init({'params': critic_key_encoder, 'noise':noise2_key, 'drop_path':drop2_key}, observations)
        critic_encoder_params = critic_def_encoder_init['params']
        if encoder_type == 'mae':
            critic_encoder_params = loading_utils.load_pytorch_weights(mae_type, critic_encoder_params)

        critic_encoder_batch_stats = critic_def_encoder_init['batch_stats'] if 'batch_stats' in critic_def_encoder_init else None

        if 'batch_stats' in critic_def_encoder_init:
            embed_obs, _ = critic_def_encoder.apply({'params': critic_encoder_params, 'batch_stats': critic_def_encoder_init['batch_stats']}, observations, mutable=['batch_stats'], rngs={'noise':noise3_key, 'drop_path':drop3_key})
        else:
            embed_obs = critic_def_encoder.apply({'params': critic_encoder_params}, observations, rngs={'noise':noise3_key, 'drop_path':drop3_key})

        critic_def_decoder_init = critic_def_decoder.init(critic_key_decoder, embed_obs, actions)
        critic_decoder_params = critic_def_decoder_init['params']
        critic_decoder_batch_stats = critic_def_decoder_init['batch_stats'] if 'batch_stats' in critic_def_decoder_init else None

        critic_encoder = TrainState.create(apply_fn=critic_def_encoder.apply,
                                params=critic_encoder_params,
                                batch_stats=critic_encoder_batch_stats,
                                tx=optax.adam(learning_rate=critic_lr))

        critic_decoder = TrainState.create(apply_fn=critic_def_decoder.apply,
                                params=critic_decoder_params,
                                batch_stats=critic_decoder_batch_stats,
                                tx=optax.adam(learning_rate=critic_lr))
        if adam_weight_decay > 0:
            # no decay list for adamw
            self.encoder_no_decay_list = critic_def_encoder.no_decay_list()
            self.decoder_no_decay_list = critic_def_decoder.no_decay_list()
            self.adam_weight_decay = adam_weight_decay
            self.critic_lr = critic_lr

        target_critic_encoder_params = copy.deepcopy(critic_encoder.params)
        target_critic_decoder_params = copy.deepcopy(critic_decoder.params)


        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))

        self._rng = rng
        self._actor = actor
        self._critic_encoder = critic_encoder
        self._critic_decoder = critic_decoder
        self._critic = (critic_encoder, critic_decoder)

        self._temp = temp
        self._target_critic_encoder_params = target_critic_encoder_params
        self._target_critic_decoder_params = target_critic_decoder_params
        self._target_critic_params = (target_critic_encoder_params, target_critic_decoder_params)

        self._cql_alpha = cql_alpha
        print ('Discount: ', self.discount)
        print ('CQL Alpha: ', self._cql_alpha)
        print('Method: ', self.method, 'Const: ', self.method_const)
        print('Actor Params', jax.tree_map(lambda x: x.shape, actor.params))
        print('Critic Encoder Params', jax.tree_map(lambda x: x.shape, critic_encoder.params))


    def update(self, batch: FrozenDict, i=-1) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic_params, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic_encoder, self._critic_decoder,
            self._target_critic_encoder_params, self._target_critic_decoder_params,
            self._temp, batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.critic_reduction, self._cql_alpha, self.max_q_backup,
            self.dr3_coefficient, tr_penalty_coefficient=self.tr_penalty_coefficient, mc_penalty_coefficient=self.mc_penalty_coefficient, pretrained_critic_encoder=self._pretrained_critic_encoder,color_jitter=self.color_jitter,
            method=self.method, method_const=self.method_const,
            method_type=self.method_type, cross_norm=self.cross_norm, bound_q_with_mc=self.bound_q_with_mc, online_bound_nstep_return=self.online_bound_nstep_return)

        new_critic_encoder, new_critic_decoder = new_critic
        new_target_critic_encoder_params, new_target_critic_decoder_params = new_target_critic_params

        self._rng = new_rng
        if self.wait_actor_update > 0 and i >= 0 and i <= self.wait_actor_update:
            pass
        else:
            self._actor = new_actor
        self._critic_encoder = new_critic_encoder
        self._critic_decoder = new_critic_decoder
        self._critic = (new_critic_encoder, new_critic_decoder)

        self._target_critic_encoder_params = new_target_critic_encoder_params
        self._target_critic_decoder_params = new_target_critic_decoder_params
        self._target_critic_params = (new_target_critic_encoder_params, new_target_critic_decoder_params)

        self._temp = new_temp

        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        # try:
        from examples.train_utils import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)
        # except Exception as e:
        #     print(e)
        #     print('Could not visualize')

    def make_value_reward_visulization(self, variant, trajs, **kwargs):
        # try:
        num_traj = len(trajs['rewards'])
        traj_images = []
        num_stack = variant.frame_stack

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            target_critic_encoder = self._critic_encoder.replace(params=self._target_critic_encoder_params)
            target_critic_decoder = self._critic_decoder.replace(params=self._target_critic_decoder_params)

            q_pred = []
            target_q_pred = []
            bellman_loss = []
            task_ids = []

            # Do the frame stacking thing for observations
            # images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'), num_stack + 1, axis=0)

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]

                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_q_value(action, obs_dict, self._critic_encoder, self._critic_decoder)
                next_action = get_action(next_obs_dict, self._actor)
                target_q_value = get_q_value(next_action, next_obs_dict, target_critic_encoder, target_critic_decoder)
                target_q_value = rewards[t] + target_q_value.min() * self.discount * masks[t]
                q_pred.append(q_value)
                target_q_pred.append(target_q_value.item())
                bellman_loss.append(((q_value-target_q_value)**2).mean().item())
                if 'task_id' in observations.keys():
                    task_ids.append(np.argmax(observations['task_id']))

            # print ('lengths for verification: ', len(task_ids), len(q_pred), len(masks), len(bellman_loss))

            traj_images.append(make_visual(q_pred, rewards, observations['pixels'], masks, target_q_pred, bellman_loss, task_ids))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

        # except Exception as e:
        #     print(e)
        #     return np.zeros((num_traj, 128, 128, 3))

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir, reset_critic=None, rescale_critic_last_layer_ratio=-1):
        assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._actor = output_dict['actor']
        self._temp = output_dict['temp']

        if reset_critic is None and rescale_critic_last_layer_ratio <= 0:
            self._critic_encoder, self._critic_decoder = output_dict['critic']
            self._critic=(self._critic_encoder, self._critic_decoder)
            self._target_critic_encoder_params, self._target_critic_decoder_params = output_dict['target_critic_params']
            self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        elif rescale_critic_last_layer_ratio > 0:
            self._critic_encoder = output_dict['critic'][0]
            critic_decoder = output_dict['critic'][1]
            decoder_params = flax.core.frozen_dict.unfreeze(critic_decoder.params)
            decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['bias'] = decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['bias'] * rescale_critic_last_layer_ratio
            decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['kernel'] = decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['kernel'] * rescale_critic_last_layer_ratio
            self._critic_decoder = TrainState.create(apply_fn=critic_decoder.apply_fn,
                    batch_stats=critic_decoder.batch_stats,
                    tx=critic_decoder.tx,
                    params=flax.core.frozen_dict.freeze(decoder_params),
                    )
            self._target_critic_encoder_params= output_dict['target_critic_params'][0]
            target_decoder_params = flax.core.frozen_dict.unfreeze(output_dict['target_critic_params'][1])
            target_decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['bias'] = target_decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['bias'] * rescale_critic_last_layer_ratio
            target_decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['kernel'] = target_decoder_params['network']['VmapStateActionValue_0']['MLP_0']['Dense_2']['kernel'] * rescale_critic_last_layer_ratio
            self._target_critic_decoder_params = flax.core.frozen_dict.freeze(target_decoder_params)
            self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
            print("rescaled critic last layer:", rescale_critic_last_layer_ratio)


        elif reset_critic == 'decoder':
            print("initializing critic decoder")
            self._critic_encoder, _ = output_dict['critic']
            self._critic=(self._critic_encoder, self._critic_decoder)
            self._target_critic_encoder_params, _ = output_dict['target_critic_params']
            self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        elif reset_critic == 'whole':
            print("initializing whole critic")
            pass



        print('restored from ', dir)

        if self.tr_penalty_coefficient != 0:
            print("restored pretrained critic encoder for calculating trust region penalty")
            self._pretrained_critic_encoder, _ = copy.deepcopy(output_dict['critic'])

    def load_encoder(self, pretrained_file, encoder_key):
        if pretrained_file.endswith('.pkl'):
            with open(pretrained_file, 'rb') as f:
                pretrained_params = pickle.load(f)
        elif pretrained_file.endswith('.npz'):
            assert encoder_key == 'encoder/encoder'
            init_params = flax.core.frozen_dict.unfreeze(self._critic_encoder.params)
            encoder_params = resnet.load(init_params['encoders_actor']['encoder'], pretrained_file, None, dont_load=('head/bias', 'head/kernel'))
            init_params['encoders_actor']['encoder'] = encoder_params
            pretrained_params = flax.core.frozen_dict.freeze(init_params)
        else:
            pretrained_params = checkpoints.restore_checkpoint(pretrained_file, None)
        for k in encoder_key.split('/'):
            pretrained_params = pretrained_params[k]
        assert set(self._critic_encoder.params['encoder']['encoder'].keys()) == set(pretrained_params.keys()), f"{set(self._critic_encoder.params['encoder']['encoder'].keys())} != {set(pretrained_params.keys())}"
        def replace_encoder_params(params_dict):
            params_dict = unfreeze(params_dict)
            params_dict['encoder']['encoder'] = pretrained_params
            return freeze(params_dict)
        params_to_edit = (self._actor.params, self._critic_encoder.params, self._target_critic_encoder_params)
        actor_params, critic_encoder_params, target_critic_encoder_params = map(replace_encoder_params, params_to_edit)
        self._target_critic_encoder_params = target_critic_encoder_params
        self._target_critic_params = (target_critic_encoder_params, self._target_critic_params[1])
        self._actor = self._actor.replace(params=actor_params)
        self._critic_encoder = self._critic_encoder.replace(params=critic_encoder_params)

        def trees_equal(tree1, tree2):
            return all(jax.tree_leaves(jax.tree_map(np.array_equal, tree1, tree2)))
        assert(trees_equal(self._actor.params['encoder']['encoder'], freeze(pretrained_params))), 'Parameters not updated'
        assert(trees_equal(self._critic_encoder.params['encoder']['encoder'], freeze(pretrained_params))), 'Parameters not updated'

    def apply_adamw(self):
        encoder_decay_mask = get_weight_decay_mask(params=self._critic_encoder.params, no_decay_list=self.encoder_no_decay_list)
        self._critic_encoder = TrainState.create(apply_fn=self._critic_encoder.apply_fn,
                    params=flax.core.frozen_dict.unfreeze(self._critic_encoder.params),
                    batch_stats=self._critic_encoder.batch_stats,
                    tx=optax.adamw(learning_rate=self.critic_lr,
                    weight_decay=self.adam_weight_decay,
                    b1=0.9, b2=0.999,
                    mask=encoder_decay_mask
                    ))

        decoder_decay_mask = get_weight_decay_mask(params=self._critic_decoder.params, no_decay_list=self.decoder_no_decay_list)
        self._critic_decoder = TrainState.create(apply_fn=self._critic_decoder.apply_fn,
                                params=flax.core.frozen_dict.unfreeze(self._critic_decoder.params),
                                batch_stats=self._critic_decoder.batch_stats,
                                tx=optax.adamw(learning_rate=self.critic_lr,
                                weight_decay=self.adam_weight_decay,
                                b1=0.9, b2=0.999,
                                mask=decoder_decay_mask
                                ))

        self._critic=(self._critic_encoder, self._critic_decoder)


        self._target_critic_encoder_params = flax.core.frozen_dict.unfreeze(self._target_critic_encoder_params)
        self._target_critic_decoder_params = flax.core.frozen_dict.unfreeze(self._target_critic_decoder_params)


        print("==============encoder_decay_mask:=====================")
        print(encoder_decay_mask)
        print("==============decoder_decay_mask:=====================")
        print(decoder_decay_mask)
        print("======================================================")


@functools.partial(jax.jit)
def get_action(obs_dict, actor):
    # print(f'{images.shape=}')
    # print(f'{images[..., None]=}')
    key_dropout, key_pi = jax.random.split(jax.random.PRNGKey(0))
    dist = actor.apply_fn({'params': actor.params}, obs_dict, rngs={'dropout': key_dropout})
    actions, policy_log_probs = dist.sample_and_log_prob(seed=key_pi)
    return actions

@functools.partial(jax.jit)
def get_q_value(actions, obs_dict, critic_encoder, critic_decoder):
    if critic_encoder.batch_stats is not None:
        embed_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, obs_dict, mutable=['batch_stats'])
    else:
        embed_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, obs_dict)

    if critic_decoder.batch_stats is not None:
        q_pred, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, embed_obs, actions, mutable=['batch_stats'])
    else:
        q_pred = critic_decoder.apply_fn({'params': critic_decoder.params}, embed_obs, actions)

    return q_pred

def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr


def make_visual(q_estimates, rewards, images, masks, target_q_pred, bellman_loss, task_ids):
    q_estimates_np = np.stack(q_estimates, 0)

    fig, axs = plt.subplots(7, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, C, H, W shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)

    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)

    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')

    axs[2].plot(target_q_pred, linestyle='--', marker='o')
    axs[2].set_ylabel('target_q_pred')
    axs[2].set_xlim([0, len(target_q_pred)])

    axs[3].plot(bellman_loss, linestyle='--', marker='o')
    axs[3].set_ylabel('bellman_loss')
    axs[3].set_xlim([0, len(bellman_loss)])

    axs[4].plot(rewards, linestyle='--', marker='o')
    axs[4].set_ylabel('rewards')
    axs[4].set_xlim([0, len(rewards)])

    axs[5].plot(masks, linestyle='--', marker='o')
    axs[5].set_ylabel('masks')
    axs[5].set_xlim([0, len(masks)])

    axs[6].plot(task_ids, linestyle='--', marker='o')
    axs[6].set_ylabel('task_ids')
    axs[6].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return out_image


def get_weight_decay_mask(no_decay_list, params):
    param_info = {key: (id(value), jnp.ndim(value)) for key, value in flax.traverse_util.flatten_dict(flax.core.frozen_dict.unfreeze(params)).items()}
    for key, (i, d) in param_info.items():
        if all([k not in no_decay_list for k in key]) and d != 1:
            param_info[key] = True
        else:
            param_info[key] = False
    mask = flax.traverse_util.unflatten_dict(param_info)
    return mask
