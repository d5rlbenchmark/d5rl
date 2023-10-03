#! /usr/bin/env python
import os
import pickle

import gym
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from d4rl2.envs.a1.collect.saving import save_data

tf.config.experimental.set_visible_devices([], "GPU")

from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import VideoRecorder, wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 'sac_default.py'),
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', 'original'))
    env = gym.make('a1-walk-v0')
    env = wrap_gym(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    relabel_summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', 'relabel'))
    relabel_env = gym.make('a1-walk_stable-v0')
    relabel_env = wrap_gym(relabel_env)
    relabel_env = gym.wrappers.RecordEpisodeStatistics(relabel_env,
                                                       deque_size=1)
    relabel_env.seed(FLAGS.seed)

    eval_env = gym.make('a1-walk-v0')
    eval_env = wrap_gym(eval_env)
    eval_env = VideoRecorder(eval_env, 'a1_walk_videos', height=256, width=256)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space,
                       **kwargs)

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    relabel_replay_buffer = ReplayBuffer(env.observation_space,
                                         env.action_space, FLAGS.max_steps)
    relabel_replay_buffer.seed(FLAGS.seed)

    returns = []
    observation, done = env.reset(), False
    r_observation, r_done = relabel_env.reset(), False
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

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))

        r_next_observation, r_reward, r_done, r_info = relabel_env.step(action)

        if not r_done or 'TimeLimit.truncated' in r_info:
            r_mask = 1.0
        else:
            r_mask = 0.0

        relabel_replay_buffer.insert(
            dict(observations=r_observation,
                 actions=action,
                 rewards=r_reward,
                 masks=r_mask,
                 dones=r_done,
                 next_observations=r_next_observation))

        assert np.allclose(r_observation, observation)
        assert np.allclose(r_next_observation, next_observation)
        assert r_done == done
        assert r_mask == mask

        observation = next_observation
        r_observation = r_next_observation

        if done:
            observation, done = env.reset(), False
            r_observation, r_done = relabel_env.reset(), False

            decode = {'r': 'return', 'l': 'length', 't': 'time'}
            for k, v in info['episode'].items():
                summary_writer.scalar(f'training/{decode[k]}', v, i)

            for k, v in r_info['episode'].items():
                relabel_summary_writer.scalar(f'training/{decode[k]}', v, i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                summary_writer.scalar(f'evaluation/{k}', v, i)
            returns.append((i, eval_info['return']))
            savefile = os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt')
            np.savetxt(savefile, returns, fmt=['%d', '%.1f'])

    os.makedirs('datasets', exist_ok=True)
    h5path = os.path.join('datasets', 'a1-walk.hdf5')
    save_data(replay_buffer, h5path)


if __name__ == '__main__':
    app.run(main)
