import os
import random

import numpy as np
import tqdm
from absl import app, flags
from jaxrl.agents import SACLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.wrappers import VideoRecorder
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from d4rl2.envs.a1.env_utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', int(1e4), 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e7), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_policy', False,
                     'Save the policy during evaluation.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(__file__), 'configs', 'sac_default.py'),
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    env = make_env("walk", FLAGS.seed, None)
    eval_env = make_env("walk", FLAGS.seed + 42, None)

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        env = VideoRecorder(env, video_train_folder, 480, 640)

        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
        eval_env = VideoRecorder(eval_env, video_eval_folder, 480, 640)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    replay_buffer_size = kwargs.pop('replay_buffer_size')

    observation = env.observation_space.sample()[np.newaxis]
    agent = SACLearner(FLAGS.seed, observation,
                       env.action_space.sample()[np.newaxis], **kwargs)
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        summary_writer.add_scalar(
            'motion/x_velocity', env.unwrapped._env.physics.named.data.
            sensordata['unitree_a1/velocimeter'][0],
            info['total']['timesteps'])
        summary_writer.add_scalar(
            'motion/y_velocity', env.unwrapped._env.physics.named.data.
            sensordata['unitree_a1/velocimeter'][1],
            info['total']['timesteps'])
        summary_writer.add_scalar(
            'motion/z_ang_velocity', env.unwrapped._env.physics.named.data.
            sensordata['unitree_a1/sensor_gyro'][2],
            info['total']['timesteps'])

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            if FLAGS.save_policy:
                agent.actor.save(
                    os.path.join(FLAGS.save_dir, 'models',
                                 f'{FLAGS.seed}_{i}'))


if __name__ == '__main__':
    app.run(main)
