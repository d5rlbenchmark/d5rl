

import os
import pickle
import numpy as np

import sys; sys.path.append("..")
import gym
from benchmark.domains.a1 import legged_mujoco




# import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags






from jaxrl2.evaluation import evaluate

from jaxrl2.agents import DDPMIQLLearner, DDPMBCLearner


import jaxrl2.wrappers.combo_wrappers as wrappers
from jaxrl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data.kitchen_data import ReplayBuffer

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

flags.DEFINE_boolean('normalize', False, 'Set to debug params (shorter).')

import sys
algname = sys.argv[sys.argv.index("--algorithm") + 1]
print("algname:", algname)
assert algname in ["bc", "iql", "cql_slow", "cql", "calql", "td3bc", "ddpm_bc", "idql", "drq"], f"algname: {algname}"

config_flags.DEFINE_config_file(
    'config',
    f'./configs/offline_a1_config.py:{algname}',
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

    env = make_env(FLAGS.task)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = make_env(FLAGS.task)

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

    # print("Loading replay buffer")
    # replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    # replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False})

    # DATADIR = os.environ.get('STANDARD_KITCHEN_DATASETS', None)
    # print("DATADIR:", DATADIR)
    # load_data(replay_buffer, env, DATADIR, FLAGS.task, FLAGS.ep_length, 3, FLAGS.proprio, FLAGS.discount, debug=FLAGS.debug)

    # rootroot = "/home/laurasmith/finetuning_benchmark/code_testing/"
    _, task = FLAGS.task.split('_', 1)

    rootroot = os.environ.get('A1_DATASETS', None)
    if task == "a1-interpolate-v0":
        load_buffer_dir_root = "a1_interpolate"
    elif task == "a1-extrapolate-above-v0":
        load_buffer_dir_root = "a1_extrapolate_above"
    elif task == "a1-hiking-v0":
        load_buffer_dir_root = "a1_hiking"
    else:
        raise NotImplementedError

    import pickle

    load_buffer_dir = os.path.join(os.path.join(rootroot, load_buffer_dir_root), "buffers")  
    with open(os.path.join(load_buffer_dir, 'buffer.pkl'), 'rb') as f:
        dataset = pickle.load(f).dataset_dict
        print("Loaded replay buffer from", load_buffer_dir)
    
    my_dataset = get_laura_dataset(dataset, clip_action=0.99999)
    offline_dataset_size = my_dataset['rewards'].shape[0]

    # observation_space: gym.Space,
    #     action_space: gym.Space,
    #     capacity: int,
    #     next_observation_space: Optional[gym.Space] = None,
        
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=offline_dataset_size+int(FLAGS.max_online_gradient_steps))
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size})

    
    convert_D4RL(replay_buffer, my_dataset)
    if FLAGS.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1


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

        if i % FLAGS.eval_interval == 0: # or i == 1000:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False)
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

            wandb.log({f"replay_buffer/capacity": replay_buffer._capacity}, step=i)
            wandb.log({f"replay_buffer/size": replay_buffer._size}, step=i)
            wandb.log({f"replay_buffer/fullness": replay_buffer._size / replay_buffer._capacity}, step=i)

    eval_info = evaluate(agent,
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

            if i % FLAGS.online_eval_interval == 0: # or i == 1000:
                eval_info = evaluate(agent,
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






def make_env(task):
    suite, task = task.split('_', 1)

    from jaxrl2.wrappers.jaxrl5_wrappers import wrap_gym

    env = gym.make(task)
    env = wrap_gym(env, rescale_actions=True)
    return env


def get_laura_dataset(dataset, clip_action):
    terminate_on_end = False
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    terminals = dataset['masks'] == 0.0
    timeouts = dataset['dones']

    dataset['terminals'] = terminals
    dataset['timeouts'] = timeouts

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
    # first process by traj
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['dones'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep or i == N-1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in ['actions', 'next_observations', 'observations', 'rewards', 'terminals', 'timeouts']:
                    data_[k].append(dataset[k][i])
            if 'next_observations' not in dataset.keys():
                data_['next_observations'].append(dataset['observations'][i+1])
            episode_step += 1

        if (done_bool or final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = episode_data["rewards"]
            episode_data['actions'] = np.clip(episode_data['actions'], -clip_action, clip_action)
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)
    dataset = concatenate_batches(episodes_dict_list)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
        terminals=dataset['terminals'].astype(np.float32),
        timeouts=dataset['timeouts'].astype(np.float32),
    )
    
def concatenate_batches(batches):
	concatenated = {}
	for key in batches[0].keys():
		concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
	return concatenated


def convert_D4RL(replay_buffer, dataset):
		# replay_buffer.state[0:dataset['observations'].shape[0]] = dataset['observations']
		# replay_buffer.action[0:dataset['actions'].shape[0]] = dataset['actions']
		# replay_buffer.next_state[0:dataset['next_observations'].shape[0]] = dataset['next_observations']
		# reward = dataset['rewards'].reshape(-1,1)
		# replay_buffer.reward[0:reward.shape[0]] = reward
		# not_done = 1. - dataset['terminals'].reshape(-1,1)
		# replay_buffer.not_done[0:not_done.shape[0]] = not_done
		# replay_buffer.size = replay_buffer.state.shape[0]
		# replay_buffer.ptr = replay_buffer.size
		# print("Converted D4RL dataset to replay buffer with size:, ", replay_buffer.size)

        assert dataset['dones'].sum() == 0
        assert dataset['terminals'].sum() == 0

        for i in tqdm.trange(dataset['observations'].shape[0]):
            replay_buffer.insert(
                dict(
                    observations=dataset['observations'][i],
                    actions=dataset['actions'][i],
                    rewards=dataset['rewards'][i],
                    masks=np.logical_not(dataset['dones'][i]),
                    mc_returns=-42, ###CHANGE ME###
                    dones=dataset['dones'][i],
                    next_observations=dataset['next_observations'][i],
                )
            )

if __name__ == '__main__':
    app.run(main)



"""
export A1_DATASETS=
"""