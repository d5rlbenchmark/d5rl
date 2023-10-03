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
from jaxrl2.agents.drq.augmentations import batched_random_crop
from jaxrl2.agents.drq.drq_learner import _unpack
from jaxrl2.data.kitchen_data.dataset import DatasetDict
from jaxrl2.networks.kitchen_networks.encoders import ResNetV2Encoder, ImpalaEncoder
from jaxrl2.networks.kitchen_networks.encoders.resnet_encoderv1 import GroupConvWrapper, ResNet18, ResNet34, ResNetSmall, ResNet50
from jaxrl2.networks.kitchen_networks.encoders import D4PGEncoder, D4PGEncoderGroups ###===### ###---###
from jaxrl2.networks.kitchen_networks.normal_policy import UnitStdNormalPolicy
from jaxrl2.networks.kitchen_networks.pixel_multiplexer import PixelMultiplexer, PixelMultiplexerMultiple
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
    batch = _unpack(batch)

    rng, key = jax.random.split(rng)
    aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, new_actor, actor_info = log_prob_update(rng, actor, batch)

    return rng, new_actor, actor_info


# import torch
# from torchvision.io import read_image

# from voltron import instantiate_extractor, load

# import numpy as np 
# import cv2 
# from copy import deepcopy 

# class PretrainedEncoderWrapper:
    # def __init__(self, encoder_name):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     vcond, preprocess = load(encoder_name, device=device, freeze=True)

    #     img = preprocess(read_image("peel-carrot-initial.png"))[None, ...].to(device)
    #     print("img.shape:", img.shape)
        

    #     with torch.no_grad():
    #         if "v-cond" in encoder_name:
    #             visual_features = vcond(img, mode="visual")  # Vision-only features (no language)
    #         else:
    #             visual_features = vcond(img)  # Vision-only features (no language)

    #     vector_extractor = instantiate_extractor(vcond, n_latents=1)().to(device)
    #     print("vector_extractor(visual_features).shape:", vector_extractor(visual_features).shape)

    #     self._vcond = vcond
    #     self._preprocess = preprocess
    #     self._imsize = 224
    #     self._encoder_name = encoder_name
    #     self._vector_extractor = vector_extractor
    #     self._device = device

    # def __call__(self, pixels):
    #     pixels = deepcopy(pixels)

    #     # if len(pixels.shape) == 4:
    #     #     pixels_batched = np.zeros((9, pixels.shape[0], pixels.shape[1], 3))

    #     #     for i in range(3):
    #     #         for j in range(3):
    #     #             pixels_batched[i*3 + j] = pixels[..., i*3:i*3+3, j]

    #     #     # arranges them in [cam0-frame0, cam0-frame1, cam0-frame2, cam1-frame0, cam1-frame1, cam1-frame2, cam2-frame0, cam2-frame1, cam2-frame2]
    #     #     assert np.array_equal(pixels_batched[0], pixels[..., :3, 0])
    #     #     assert np.array_equal(pixels_batched[-1], pixels[..., 6:, -1])
    #     #     assert np.array_equal(pixels_batched[2], pixels[..., :3, -1])
    #     #     assert np.array_equal(pixels_batched[5], pixels[..., 3:6, -1])

    #     #     pixels_batched = pixels_batched.transpose((0, 3, 1, 2))
    #     #     pixels_batched = torch.tensor(pixels_batched)
    #     #     img = self._preprocess(pixels_batched)[None, ...].to(device)

    #     #     with torch.no_grad():
    #     #         if "v-cond" in self._encoder_name:
    #     #             visual_features = self._vcond(img, mode="visual")  # Vision-only features (no language)
    #     #         else:
    #     #             visual_features = self._vcond(img)  # Vision-only features (no language)
            
    #     #     features = self._vector_extractor(visual_features) # (batch, 384)
    #     #     features = features.view(-1)

    #     # else:
    #     #     import pdb; pdb.set_trace()

    #     no_batch_dim = False
    #     if len(pixels.shape) == 4:
    #         pixels = pixels[None, ...]
    #         no_batch_dim = True

    #     embed_dim = 384

    #     features_allcams = torch.zeros((pixels.shape[0], embed_dim*3), device=self._device)

    #     for cam_no in range(3):
    #         pixels_cam = deepcopy(pixels[..., cam_no*3:cam_no*3+3, 0]

    #         # pixels_cam = pixels_cam.transpose((0, -1, -3, -2)) 
    #         pixels_cam = pixels_cam.transpose((0, 3, 1, 2)) 
            
    #         pixels_cam = torch.tensor(pixels_cam)
    #         img = self._preprocess(pixels_cam).to(self._device)

    #         with torch.no_grad():
    #             if "v-cond" in self._encoder_name:
    #                 visual_features = self._vcond(img, mode="visual")  # Vision-only features (no language)
    #             else:
    #                 visual_features = self._vcond(img)  # Vision-only features (no language)
            
    #         features = self._vector_extractor(visual_features) # (batch, 384)
            
    #         features_allcams[:, cam_no*embed_dim:cam_no*embed_dim+embed_dim] = features

    #     features_allcams = features_allcams.detach().cpu().numpy()

    #     if no_batch_dim:
    #         assert features_allcams.shape[0] == 1
    #         features_allcams = np.squeeze(features_allcams)

    #     return features_allcams
import voltron

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
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        # assert observations["pixels"].shape[-2] / cnn_groups == 3, f"observations['pixels'].shape: {observations['pixels'].shape}, cnn_groups: {cnn_groups}"

        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        use_pretrained_representations = False

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
        elif encoder in voltron.available_models():
            encoder_def = None
            use_pretrained_representations = True
            # encoder_defs.append(encoder_def)
        # encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
        #                        use_multiplicative_cond=use_multiplicative_cond,
        #                        use_spatial_learned_embeddings=use_spatial_learned_embeddings,
        #                        num_spatial_blocks=8,
        #                        conv=partial(GroupConvWrapper, groups=cnn_groups))
        # encoder_def = D4PGEncoderGroups(cnn_features, cnn_filters, cnn_strides, cnn_padding, cnn_groups)


        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        policy_def = UnitStdNormalPolicy(
            hidden_dims, action_dim, dropout_rate=dropout_rate
        )
        # actor_def = PixelMultiplexerMultiple(
        #     encoders=encoder_defs, network=policy_def, latent_dim=latent_dim
        # )
        actor_def = PixelMultiplexer(encoder=encoder_def, network=policy_def, latent_dim=latent_dim, use_pretrained_representations=use_pretrained_representations)


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
