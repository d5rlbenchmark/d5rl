import numpy as np
import time
import os
import os.path as osp
import roboverse
from roboverse.policies import policies
import argparse
import pickle
from tqdm import tqdm

from roboverse.utils import get_timestamp
# import railrl.torch.pytorch_util as ptu
EPSILON = 0.1

# TODO(avi): Clean this up
# NFS_PATH = '/nfs/kun1/users/avi/imitation_datasets/'
import collections
import gym



class ObsLatency(gym.Wrapper):
    def __init__(self, env, latency: int):
        super().__init__(env)
        self._latency = latency

        self._image = collections.deque(maxlen=latency + 1)

        self._reward_frames = collections.deque(maxlen=latency + 1)
        self._terminal_frames = collections.deque(maxlen=latency + 1)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._latency):
            self._image.append(obs['image'])
            self._reward_frames.append(0.0)
            self._terminal_frames.append(False)
        obs['image'] = self._image[0]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._image.append(obs['image'])
        self._reward_frames.append(reward)
        self._terminal_frames.append(done)

        # For debugging purposes only, one can actually print or log these
        # this info is returned to the final algorithm main runner loop
        info['orig_obs'] = obs
        info['orig_reward'] = reward
        info['orig_terminal'] = done

        obs['image'] = self._image[0]
        return obs, self._reward_frames[0], self._terminal_frames[0], info


def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim, transpose_image):
    def reshape_image(obs, img_dim, transpose_image):
        if transpose_image:
            obs["image"] = np.reshape(obs["image"], (3, img_dim, img_dim))
            obs["image"] = np.transpose(obs["image"], [1, 2, 0])
            obs["image"] = np.uint8(obs["image"] * 255.)
        else:
            obs["image"] = np.reshape(np.uint8(obs["image"] * 255.),
                                              (img_dim, img_dim, 3))
        return obs

    reshape_image(observation, img_dim, transpose_image)
    traj["observations"].append(observation)
    reshape_image(next_observation, img_dim, transpose_image)
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def collect_one_traj(env, policy, num_timesteps, noise,
                     accept_trajectory_key, transpose_image,
                     use_pretrained_policy=False, ignore_done=False):
    num_steps = -1
    rewards = []
    success = False
    img_dim = env.observation_img_dim
    observation = env.reset()
    policy.reset()
    time.sleep(1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )
    for j in range(num_timesteps):

        observation = env.get_observation()

        if use_pretrained_policy:
            action, agent_info = policy.get_action(observation)
        else:
            action, agent_info = policy.get_action()

        # In case we need to pad actions by 1 for easier realNVP modelling
        env_action_dim = env.action_space.shape[0]
        if 'modify_noise' in agent_info and agent_info['modify_noise']:
            action += np.random.normal(scale=noise*3, size=(env_action_dim,))
        else:
            action += np.random.normal(scale=noise, size=(env_action_dim,))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        next_observation, reward, done, info = env.step(action)
        
        if False:
            print(reward, done, info)
        
        add_transition(traj, observation,  action, reward, info, agent_info,
                       done, next_observation, img_dim, transpose_image)

        if info[accept_trajectory_key] and num_steps < 0:
            num_steps = j

        rewards.append(reward)
        if not use_pretrained_policy and not ignore_done and (done or agent_info['done']):
            break

    if info[accept_trajectory_key]:
        success = True
    return traj, success, num_steps


def main(args):

    timestamp = get_timestamp()
    data_save_path = args.save_directory
    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)
        
    extra_kwargs = {}
    if args.specific_task_id:
        extra_kwargs['specific_task_id'] = args.specific_task_id
        extra_kwargs['desired_task_id'] = args.desired_task_id

    data = []
    use_pretrained_policy = False

    if args.pretrained_policy:
        with open(args.pretrained_policy, 'rb') as f:
            params = pickle.load(f)
        # Assumes realNVP policy
        policy = params['trainer/bijector']
        use_pretrained_policy = True
        transpose_image=True
        ptu.set_gpu_mode(True)
        env = roboverse.make(args.env_name, gui=args.gui, transpose_image=transpose_image, **extra_kwargs)
        print("reward_type", env.reward_type)
    else:
        assert args.policy_name in policies.keys(), f"The policy name must be one of: {policies.keys()}"
        policy_class = policies[args.policy_name]
        transpose_image = False
        env = roboverse.make(args.env_name, gui=args.gui, transpose_image=transpose_image, terminate_on_success=not args.run_until_end, **extra_kwargs)
        policy = policy_class(env)

        if args.delay > 0:
            env = ObsLatency(env, latency=args.delay)
        
        binsort_classes = [
            policies['binsort'], 
            policies['binsortneutral'], 
            policies['binsortneutralstored'], 
            policies['binsortmult'],
            policies['binsortneutralmult'],
            policies['binsortneutralmultstored'],
            policies['binsortmultstored'],
        ]

        if args.p_place_correct > 0 and policy_class in binsort_classes:
            policy = policy_class(env,correct_bin_per=args.p_place_correct)
        elif args.p_place_correct > 0:
            policy = policy_class(env,p_place_correct=args.p_place_correct)
        else:    
            policy = policy_class(env)
            
        assert args.accept_trajectory_key in env.get_info().keys(), \
            f"""The accept trajectory key must be one of: {env.get_info().keys()}"""

    # TODO: clean up
    if args.target_object:
        env.target_object = args.target_object

    if args.reward_type:
        env.reward_type = args.reward_type

    num_success = 0
    num_saved = 0
    num_attempts = 0
    accept_trajectory_key = args.accept_trajectory_key

    progress_bar = tqdm(total=args.num_trajectories)

    while num_saved < args.num_trajectories:
        num_attempts += 1
        print('num attemtps', num_attempts)
        traj, success, num_steps = collect_one_traj(
            env, policy, args.num_timesteps, args.noise,
            accept_trajectory_key, transpose_image,
            use_pretrained_policy, ignore_done=args.ignore_done)

        print('success', success)
        if args.save_failonly:
            print('use failonly')
            if not success:
                data.append(traj)
                num_saved += 1
                progress_bar.update(1)
            else:
                num_success += 1
        else:
            if success:
                if args.gui:
                    print("num_timesteps of success: ", num_steps)
                    print('num tsteps', len(traj['actions']))
                data.append(traj)
                num_success += 1
                num_saved += 1
                progress_bar.update(1)
            elif args.save_all:
                data.append(traj)
                num_saved += 1
                progress_bar.update(1)

        if args.gui:
            print("success rate: {}".format(num_success/(num_attempts)))

    progress_bar.close()
    print("success rate: {}".format(num_success / (num_attempts)))

    pivot = int(len(data)*0.9)

    if args.use_timestamp:
        save_path = data_save_path + '/{}_{}/train'.format(args.env_name, timestamp)
    else:
        save_path = data_save_path + '/{}/train'.format(args.env_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    np.save(save_path + '/out.npy', data[:pivot])

    if args.use_timestamp:
        save_path = data_save_path + '/{}_{}/val'.format(args.env_name, timestamp)
    else:
        save_path = data_save_path + '/{}/val'.format(args.env_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    np.save(save_path + '/out.npy', data[pivot:])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=False, default=None)
    parser.add_argument("-pp", "--pretrained-policy", type=str, required=False, default=None)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-o", "--target-object", type=str)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--reward-type", type=str, default=None)
    parser.add_argument("--save_failonly", action='store_true', default=False)
    parser.add_argument("--run_until_end", action='store_true', default=False)
    parser.add_argument('--specific_task_id', type=int, default=0)
    parser.add_argument('--desired_task_id', type=int, nargs="+", default=(0,0))
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument('--p_place_correct', type=float, default=0.0)
    parser.add_argument('--trunc', type=int, default=0)
    parser.add_argument('--ignore_done', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')

    args = parser.parse_args()

    main(args)
