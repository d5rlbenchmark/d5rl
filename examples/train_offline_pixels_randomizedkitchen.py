import torch

import os
import pickle
import numpy as np

import sys; sys.path.append("..")

import gym
from benchmark.domains import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags



from jaxrl2.evaluation import evaluate_kitchen

from jaxrl2.agents.kitchen_agents.pixel_cql import PixelCQLLearner
from jaxrl2.agents.kitchen_agents.pixel_iql import PixelIQLLearner
from jaxrl2.agents.kitchen_agents.pixel_bc import PixelBCLearner
from jaxrl2.agents.kitchen_agents import PixelIDQLLearner, PixelDDPMBCLearner
from jaxrl2.agents.kitchen_agents import PixelCQLLearnerEncoderSepParallel
from jaxrl2.agents.kitchen_agents import PixelCQLLearnerEncoderSep
from jaxrl2.agents.kitchen_agents import PixelTD3BCLearner
from jaxrl2.agents.kitchen_agents import DrQLearner

import jaxrl2.wrappers.combo_wrappers as wrappers
from jaxrl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data.kitchen_data import MemoryEfficientReplayBuffer

from glob import glob

from flax.core import frozen_dict

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

# flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './results', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "vd5rl", 'WandB project.')
flags.DEFINE_string('algorithm', "cql", 'Which offline RL algorithm to use.')
flags.DEFINE_string('description', "default", 'WandB project.')
flags.DEFINE_string('task', "microwave", 'Task for the kitchen env.')
# flags.DEFINE_string('camera_angle', "camera2", 'Camera angle.')
flags.DEFINE_string('datadir', "microwave", 'Directory with dataset files.')
flags.DEFINE_integer('ep_length', 280, 'Episode length.')
flags.DEFINE_integer('action_repeat', 1, 'Random seed.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Number of transitions the (offline) replay buffer can hold.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 250,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('online_eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_gradient_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('max_online_gradient_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_boolean('finetune_online', True, 'Save videos during evaluation.')
flags.DEFINE_boolean('proprio', False, 'Save videos during evaluation.')
flags.DEFINE_float("take_top", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('debug', False, 'Set to debug params (shorter).')
flags.DEFINE_float("discount", 0.99, "Take top N% trajectories.")
flags.DEFINE_integer("im_size", 128, "Image size.")
flags.DEFINE_boolean("use_wrist_cam", True, "Use the wrist cam?")
flags.DEFINE_string('camera_ids', "12", 'Eg: 0,1')
flags.DEFINE_integer("framestack", 3, "Image size.")

# flags.DEFINE_string('pretrained_encoder', "none", 'Eg: 0,1')


import torch
from torchvision.io import read_image

import voltron
from voltron import instantiate_extractor, load

import numpy as np 
import cv2 

class PretrainedEncoder:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vcond, preprocess = load(model_name, device=device, freeze=True)

        img = preprocess(read_image("peel-carrot-initial.png"))[None, ...].to(device)
        print("img.shape:", img.shape)
        

        with torch.no_grad():
            if "v-cond" in model_name:
                visual_features = vcond(img, mode="visual")  # Vision-only features (no language)
            else:
                visual_features = vcond(img)  # Vision-only features (no language)

        vector_extractor = instantiate_extractor(vcond, n_latents=1)().to(device)
        print("vector_extractor(visual_features).shape:", vector_extractor(visual_features).shape)

        self._vcond = vcond
        self._preprocess = preprocess
        self._imsize = 224
        self._model_name = model_name
        self._vector_extractor = vector_extractor
        self._device = device
        self._embed_dim = vector_extractor(visual_features).squeeze().shape[0] * 3


    def __call__(self, pixels):
        pixels = pixels.copy()

        pixels = np.stack([pixels[..., :3], pixels[..., 3:6], pixels[..., 6:]])

        assert len(pixels.shape) == 4, f"pixels.shape: {pixels.shape}"

        pixels = pixels.transpose((0, 3, 1, 2))
        pixels = torch.tensor(pixels, device=self._device)
        
        img = self._preprocess(pixels).to(self._device)

        with torch.no_grad():
            if "v-cond" in self._model_name:
                visual_features = self._vcond(img, mode="visual")  # Vision-only features (no language)
            else:
                visual_features = self._vcond(img)  # Vision-only features (no language)
        
        features = self._vector_extractor(visual_features) # (batch, 384)
        features = features.view(-1)
        return features.detach().cpu().numpy()


import sys
algname = sys.argv[sys.argv.index("--algorithm") + 1]
print("algname:", algname)
assert algname in ["bc", "iql", "cql_slow", "cql", "calql", "td3bc", "ddpm_bc", "idql", "drq"], f"algname: {algname}"

config_flags.DEFINE_config_file(
    'config',
    f'./configs/offline_pixels_randomizedkitchen_debug_config.py:{algname}',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    if FLAGS.debug:
        FLAGS.project = "trash_results"
        # FLAGS.batch_size = 16
        FLAGS.max_gradient_steps = 500
        FLAGS.eval_interval = 400
        FLAGS.online_eval_interval = 400
        FLAGS.eval_episodes = 2
        FLAGS.log_interval = 300

        if FLAGS.max_online_gradient_steps > 0:
            FLAGS.max_online_gradient_steps = 1000

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.project, FLAGS.task, FLAGS.algorithm, FLAGS.description, f"seed_{FLAGS.seed}")
    os.makedirs(os.path.join(save_dir, "wandb"), exist_ok=True)
    group_name = f"{FLAGS.task}_{FLAGS.algorithm}_{FLAGS.description}"
    name = f"seed_{FLAGS.seed}"


    wandb.init(project=FLAGS.project,
              dir=os.path.join(save_dir, "wandb"),
              id=group_name + "-" + name,
              group=group_name,
              save_code=True,
              name=name,
              resume=None,
              entity="iris_intel")

    wandb.config.update(FLAGS)


    if FLAGS.config.model_config.encoder in voltron.available_models():
        pretrained_encoder = PretrainedEncoder(FLAGS.config.model_config.encoder)
    else:
        pretrained_encoder = None
    


    env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, im_size=FLAGS.im_size, camera_ids=FLAGS.camera_ids, use_wrist_cam=FLAGS.use_wrist_cam, framestack=FLAGS.framestack, pretrained_encoder=pretrained_encoder)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, im_size=FLAGS.im_size, camera_ids=FLAGS.camera_ids, use_wrist_cam=FLAGS.use_wrist_cam, framestack=FLAGS.framestack, pretrained_encoder=pretrained_encoder)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        # kwargs['decay_steps'] = FLAGS.max_gradient_steps
        kwargs['decay_steps'] = FLAGS.max_gradient_steps + FLAGS.max_online_gradient_steps

    # assert kwargs["cnn_groups"] == 1
    print(globals()[FLAGS.config.model_constructor])
    if FLAGS.algorithm in ('idql', 'ddpm_bc'):
        agent = globals()[FLAGS.config.model_constructor].create(
        FLAGS.seed, env.observation_space, env.action_space,
        **kwargs)
    else:
        agent = globals()[FLAGS.config.model_constructor](
            FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
            **kwargs)
        # obs = env.observation_space.sample()
        # obs["pixels"] = obs["pixels"][None]
        # obs["states"] = obs["states"][None]
        #
        # agent = globals()[FLAGS.config.model_constructor](
        #     FLAGS.seed, obs, env.action_space.sample()[None],
        #     **kwargs)
    print('Agent created')

    print("Loading replay buffer")
    # replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    # replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False})
    # load_data(replay_buffer, FLAGS.datadir, FLAGS.task, FLAGS.ep_length, 3, FLAGS.proprio, debug=FLAGS.debug)
    replay_buffer = env.q_learning_dataset(include_pixels=False, size=FLAGS.replay_buffer_size, discount=FLAGS.discount, debug=FLAGS.debug)

    if FLAGS.take_top is not None or FLAGS.filter_threshold is not None:
        ds.filter(take_top=FLAGS.take_top, threshold=FLAGS.filter_threshold)


    print('Replay buffer loaded')

    print('Start offline training')
    tbar = tqdm.tqdm(range(1, FLAGS.max_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    for i in tbar:
        tbar.set_description(f"[{FLAGS.algorithm} {FLAGS.seed}]  (offline)")
        # batch = next(replay_buffer_iterator)
        batch = replay_buffer.sample(FLAGS.batch_size)

        # new_batch = {}
        # new_batch["actions"] = batch["actions"][None]
        # new_batch["dones"] = batch["dones"][None]
        # new_batch["masks"] = batch["masks"][None]
        # new_batch["rewards"] = batch["rewards"][None]
        # new_batch["observations"] = {"pixels":None, "states":None}
        # new_batch["observations"]["pixels"] = batch["observations"]["pixels"][None]
        # new_batch["observations"]["states"] = batch["observations"]["states"][None]
        # new_batch["next_observations"] = {"states":None}
        # new_batch["next_observations"]["states"] = batch["next_observations"]["states"][None]
        # new_batch = frozen_dict.freeze(new_batch)
        # out = agent.update(new_batch)

        out = agent.update(batch)


        if isinstance(out, tuple):
            agent, update_info = out
        else:
            update_info = out

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    # print(k, v)

        if i % FLAGS.eval_interval == 0 or i == 1000:
            eval_info = evaluate_kitchen(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False)
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

            wandb.log({f"replay_buffer/capacity": replay_buffer._capacity}, step=i)
            wandb.log({f"replay_buffer/size": replay_buffer._size}, step=i)
            wandb.log({f"replay_buffer/fullness": replay_buffer._size / replay_buffer._capacity}, step=i)

    eval_info = evaluate_kitchen(agent,
                         eval_env,
                         num_episodes=2 if FLAGS.debug else 100,
                         progress_bar=False)
    for k, v in eval_info.items():
        wandb.log({f'evaluation/{k}': v}, step=i)

    agent.save_checkpoint(os.path.join(save_dir, "offline_checkpoints"), i, -1)

    if FLAGS.finetune_online and FLAGS.max_online_gradient_steps > 0:
        print('Start online training')
        observation, done = env.reset(), False
        transitions = []
        tbar = tqdm.tqdm(range(1, FLAGS.max_online_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
        for i in tbar:
            tbar.set_description(f"[{FLAGS.algorithm} {FLAGS.seed} (online)]")

            out = agent.sample_actions(observation)
            if isinstance(out, tuple):
                action, agent = out
            else:
                action = out

            env_step = env.step(action)
            if len(env_step) == 4:
                next_observation, reward, done, info = env_step
            elif len(env_step) == 5:
                next_observation, reward, done, _, info = env_step
            else:
                raise ValueError(f"env_step should be length 4 or 5 but is length {len(env_step)}")

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            if pretrained_encoder is not None:
                import pdb; pdb.set_trace()
                # Need to add pretrained representations to the online data

            transitions.append(dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            ))
            # replay_buffer.insert(
            #     dict(
            #         observations=observation,
            #         actions=action,
            #         rewards=reward,
            #         masks=mask,
            #         dones=done,
            #         next_observations=next_observation,
            #     )
            # )

            observation = next_observation

            if done:
                for k, v in info["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.max_gradient_steps)
                observation, done = env.reset(), False

                reward_to_go = [0]*len(transitions)
                prev_return = transitions[-1]["rewards"]/(1- FLAGS.discount)
                for i in range(len(transitions)):
                    reward_to_go[-i-1] = transitions[-i-1]["rewards"] + FLAGS.discount * prev_return
                    prev_return = reward_to_go[-i-1]

                for i, transition in enumerate(transitions):
                    transition["mc_returns"] = reward_to_go[i]
                    replay_buffer.insert(transition)
                transitions = []

            batch = replay_buffer.sample(FLAGS.batch_size, reinsert_offline=False)
            if FLAGS.algorithm == "idql":
                out = agent.update_online(batch)
            else:
                out = agent.update(batch)

            if isinstance(out, tuple):
                agent, update_info = out
            else:
                update_info = out

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        wandb.log({f'training/{k}': v}, step=i + FLAGS.max_gradient_steps)
                        # print(k, v)

            if i % FLAGS.online_eval_interval == 0 or i == 1000:
                eval_info = evaluate_kitchen(agent,
                                     eval_env,
                                     num_episodes=FLAGS.eval_episodes,
                                     progress_bar=False)

                for k, v in eval_info.items():
                    if FLAGS.debug:
                        v += 1000

                    wandb.log({f'evaluation/{k}': v}, step=i + FLAGS.max_gradient_steps)

                wandb.log({f"replay_buffer/capacity": replay_buffer._capacity}, step=i + FLAGS.max_gradient_steps)
                wandb.log({f"replay_buffer/size": replay_buffer._size}, step=i + FLAGS.max_gradient_steps)
                wandb.log({f"replay_buffer/fullness": replay_buffer._size / replay_buffer._capacity}, step=i + FLAGS.max_gradient_steps)

        agent.save_checkpoint(os.path.join(save_dir, "online_checkpoints"), i + FLAGS.max_gradient_steps, -1)


def make_env(task, ep_length, action_repeat, proprio, im_size=128, camera_ids="0,1", use_wrist_cam=True, framestack=3, pretrained_encoder=None):
    suite, task = task.split('_', 1)

    if "randomizedkitchen" in suite:

        """
        Check what the episode length is for standard kitchen
        """

        task_set, datasets = task.split("-")
        datasets = datasets.split("+")

        if task_set == "indistribution":
            tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide']
        elif task_set == "outofdistribution":
            tasks_to_complete = ['microwave', 'kettle', "bottomknob", 'switch']
        else:
            raise ValueError(f"Unsupported tasks set: \"{task_set}\".")

        # env = gym.make(task)
        env = gym.make("random_kitchen-v1",
                       tasks_to_complete=tasks_to_complete,
                       datasets=datasets,
                       framestack=framestack,
                       pretrained_encoder=pretrained_encoder)

        print("\nenv:", env)
        print("\nenv._max_episode_steps:", env._max_episode_steps)
        print("\nenv.env.env.tasks_to_complete:", env.env.env.tasks_to_complete)
        print("\nenv.env.env._datasets_urls:", env.env.env._datasets_urls)
        print("\nenv.env.env.env._frames:", env.env.env.env._frames)
        print("\nenv.cameras:", env.cameras)
        print("\nenv.observation_space:", env.observation_space)
        # dataset = env.q_learning_dataset()
        return env
        # assert proprio
    else:
        raise ValueError(f"Unsupported environment suite: \"{suite}\".")
    return env




if __name__ == '__main__':
    app.run(main)

