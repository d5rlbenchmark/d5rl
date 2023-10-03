import torch

import os
import pickle
import numpy as np

import gym
# import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

import sys; sys.path.append("..")


from jaxrl2.evaluation import evaluate_kitchen

from jaxrl2.agents.kitchen_agents.pixel_cql import PixelCQLLearner
from jaxrl2.agents.kitchen_agents.pixel_iql import PixelIQLLearner
from jaxrl2.agents.kitchen_agents.pixel_bc import PixelBCLearner
from jaxrl2.agents.kitchen_agents.pixel_ddpm_bc import PixelDDPMBCLearner
from jaxrl2.agents.kitchen_agents import PixelCQLLearnerEncoderSepParallel
from jaxrl2.agents.kitchen_agents import PixelCQLLearnerEncoderSep
from jaxrl2.agents.kitchen_agents import PixelTD3BCLearner
from jaxrl2.agents.kitchen_agents import PixelIDQLLearner
from jaxrl2.agents.kitchen_agents import DrQLearner

import jaxrl2.wrappers.combo_wrappers as wrappers
from jaxrl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data.kitchen_data import MemoryEfficientReplayBuffer

from glob import glob

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS



# flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './results', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "vd5rl", 'WandB project.')
flags.DEFINE_string('algorithm', "cql", 'Which offline RL algorithm to use.')
flags.DEFINE_string('description', "default", 'WandB project.')
flags.DEFINE_string('task', "diversekitchen_indistribution-expert_demos", 'Task for the kitchen env.')
# flags.DEFINE_string('camera_angle', "camera2", 'Camera angle.')
# flags.DEFINE_string('datadir', "microwave", 'Directory with dataset files.')
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

#config_flags.DEFINE_config_file(
#     'config',
#     './configs/offline_pixels_config.py:cql',
#     'File path to the training hyperparameter configuration.',
#     lock_config=False)

import sys
algname = sys.argv[sys.argv.index("--algorithm") + 1]
print("algname:", algname)
assert algname in ["bc", "iql", "cql_slow", "cql", "calql", "td3bc", "ddpm_bc", "idql", "drq"], f"algname: {algname}"

config_flags.DEFINE_config_file(
    'config',
    f'./configs/offline_pixels_standardkitchen_debug_config.py:{algname}',
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
        FLAGS.eval_episodes = 2
        FLAGS.log_interval = 200

        if FLAGS.max_online_gradient_steps > 0:
            FLAGS.max_online_gradient_steps = 500

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

    env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, im_size=FLAGS.im_size, camera_ids=FLAGS.camera_ids, use_wrist_cam=FLAGS.use_wrist_cam)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, im_size=FLAGS.im_size, camera_ids=FLAGS.camera_ids, use_wrist_cam=FLAGS.use_wrist_cam)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        # kwargs['decay_steps'] = FLAGS.max_gradient_steps
        kwargs['decay_steps'] = FLAGS.max_gradient_steps + FLAGS.max_online_gradient_steps

    # assert kwargs["cnn_groups"] == 1
    print(globals()[FLAGS.config.model_constructor])
    # obs = env.observation_space.sample()
    # obs["pixels"] = obs["pixels"][None]
    # obs["states"] = obs["states"][None]
    # agent = globals()[FLAGS.config.model_constructor](FLAGS.seed, obs, env.action_space.sample()[None], **kwargs)
    if FLAGS.algorithm in ('idql', 'ddpm_bc'):
        agent = globals()[FLAGS.config.model_constructor].create(
        FLAGS.seed, env.observation_space, env.action_space,
        **kwargs)
    else:
        agent = globals()[FLAGS.config.model_constructor](
            FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
            **kwargs)
    print('Agent created')

    print("Loading replay buffer")
    replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size})

    DATADIR = os.environ.get('STANDARD_KITCHEN_DATASETS', None)
    print("DATADIR:", DATADIR)
    load_data(replay_buffer, env, DATADIR, FLAGS.task, FLAGS.ep_length, 3, FLAGS.proprio, FLAGS.discount, debug=FLAGS.debug)

    if FLAGS.take_top is not None or FLAGS.filter_threshold is not None:
        ds.filter(take_top=FLAGS.take_top, threshold=FLAGS.filter_threshold)

    print('Replay buffer loaded')

    print('Start offline training')
    tbar = tqdm.tqdm(range(1, FLAGS.max_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    for i in tbar:
        tbar.set_description(f"[{FLAGS.algorithm} {FLAGS.seed}] (offline)")
        batch = next(replay_buffer_iterator)
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



            batch = next(replay_buffer_iterator)
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


def get_task_list(task):
    if task == "indistribution":
        tasks_list = ['microwave', 'kettle', 'light switch', 'slide cabinet']
    elif task == "outofdistribution":
        tasks_list = ['microwave', 'kettle', "bottom burner", 'light switch']
    else:
        raise ValueError(f"Unsupported task: \"{task}\".")

    return tasks_list

def make_env(task, ep_length, action_repeat, proprio, im_size=128, camera_ids="0,1", use_wrist_cam=True):
    suite, task = task.split('_', 1)

    tasks_list = get_task_list(task)

    if "singleviewkitchen" in suite:
        assert not proprio
        assert action_repeat == 1
        # tasks_list = task.split("+")

        env = wrappers.Kitchen(task=tasks_list, size=(im_size, im_size), proprio=proprio, log_only_target_tasks=True)
        env = wrappers.ActionRepeat(env, action_repeat)
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, ep_length)
        env = FrameStack(env, num_stack=3)
    elif "standardkitchen" in suite:
        camera_ids = camera_ids.split(",")
        camera_ids = [int(camera_id) for camera_id in camera_ids]
        # assert proprio
        assert action_repeat == 1
        # tasks_list = task.split("+")
        # env = wrappers.KitchenMultipleViews(task=tasks_list, size=(128, 128), camera_ids=[0, 1], proprio=proprio, log_only_target_tasks=True)
        env = wrappers.KitchenMultipleViews(task=tasks_list, size=(im_size, im_size), camera_ids=camera_ids, use_wrist_cam=use_wrist_cam, proprio=proprio, log_only_target_tasks=True)
        env = wrappers.ActionRepeat(env, action_repeat)
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, ep_length)
        env = FrameStack(env, num_stack=3)
    else:
        raise ValueError(f"Unsupported environment suite: \"{suite}\".")
    return env

def load_episode(env, episode_file, suite, tasks_list, discount):
    with open(episode_file, 'rb') as f:
        episode = np.load(f, allow_pickle=True)

        if tasks_list is None:
            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object}
        else:
            if "reward" in episode:
                rewards = episode["reward"]
            else:
                rewards = sum([episode[f"reward {obj}"] for obj in tasks_list])

            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object and "init_q" not in k and "observation" not in k and "terminal" not in k and "goal" not in k}
            episode["reward"] = rewards

        reward_to_go = np.zeros_like(episode["reward"])
        prev_return = episode["reward"][-1]/(1- discount)
        # prev_return = 0
        for i in range(episode["reward"].shape[0]):
            reward_to_go[-i-1] = episode["reward"][-i-1] + discount * prev_return
            prev_return = reward_to_go[-i-1]

        episode["mc_returns"] = reward_to_go


        # extra_image_camera_0_rgb
        # extra_image_camera_1_rgb
        # extra_image_camera_gripper_rgb

        if "standardkitchen" in suite:
            # keys = ["extra_image_camera_0_rgb", "extra_image_camera_1_rgb", "extra_image_camera_gripper_rgb"]
            imgs = {}
            for camera_id, camera in env.cameras.items():
                # imgs[camera_id + "_rgb"] = episode[camera_id + "_rgb"]
                imgs[camera_id + "_rgb"] = episode[f"extra_image_{camera_id}_rgb"]



            img = np.concatenate([imgs[key] for key in sorted(list(imgs.keys()))], axis=-1)
            episode["image"] = img

    return episode

def load_data(replay_buffer, env, offline_dataset_path, task, ep_length, num_stack, proprio, discount, debug=False):
    suite, task = task.split('_', 1)
    tasks_list = get_task_list(task)

    episode_files = glob(os.path.join(offline_dataset_path, '**', '*.npz'), recursive=True)
    total_transitions = 0

    for episode_file in tqdm.tqdm(episode_files, total=len(episode_files), desc="Loading offline data"):
        episode = load_episode(env, episode_file, suite, tasks_list, discount)

        # observation, done = env.reset(), False
        frames = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            frames.append(episode["image"][0])

        observation = dict(pixels=np.stack(frames, axis=-1))
        if proprio:
            observation["states"] = episode["proprio"][0]
        done = False

        i = 1
        while not done:
            # action = agent.sample_actions(observation)
            action = episode["action"][i]
            mc_returns = episode["mc_returns"][i]

            # next_observation, reward, done, info = env.step(action)
            frames.append(episode["image"][i])
            next_observation = dict(pixels=np.stack(frames, axis=-1))
            if proprio:
                next_observation["states"] = episode["proprio"][i]
            reward = episode["reward"][i]
            done = i >= episode["image"].shape[0] - 1
            # print(f"i: {i}, done: {done}")
            info = {}

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0
            replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                    mc_returns=mc_returns,
                )
            )
            observation = next_observation
            total_transitions += 1
            i += 1

            if debug and total_transitions > 5000:
                return

    print(f"Loaded {len(episode_files)} episodes and {total_transitions} total transitions.")
    print(f"replay_buffer capacity {replay_buffer._capacity}, replay_buffer size {replay_buffer._size}.")
    assert replay_buffer._capacity >= total_transitions


if __name__ == '__main__':
    app.run(main)

