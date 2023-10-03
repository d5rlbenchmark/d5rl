
import os

import gym
import imageio
import numpy as np

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning



class KitchenVideoRecorder(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 save_folder: str = '',
                 fps: int = 25):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.fps = fps
        self.frames = []

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def step(self, action: np.ndarray):
                
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation['pixels'][:, :, :3, 0])

        if done:
            save_file = os.path.join(self.save_folder,
                                         f'{self.current_episode}.gif')
            imageio.mimwrite(save_file, self.frames, fps=self.fps)
            self.frames = []
            self.current_episode += 1

        return observation, reward, done, info
    
    
    
#class KitchenVideoRecorder(gym.Wrapper):
#
#    def __init__(self,
#                 env: gym.Env,
#                 save_folder: str = '',
#                 fps: int = 25):
#        super().__init__(env)
#
#        self.current_episode = 0
#        self.save_folder = save_folder
#        self.fps = fps
#        self.frames = dict()
#
#        try:
#            os.makedirs(save_folder, exist_ok=True)
#        except:
#            pass
#
#    def step(self, action: np.ndarray):
#
#        frames = self.env.render(mode='rgb_array')
#        for key in frames.keys():
#            if key in self.frames.keys():
#                self.frames[key].append(frames[key].copy())
#            else:
#                self.frames[key] = []
#                self.frames[key].append(frames[key].copy())
#                
#        observation, reward, done, info = self.env.step(action)
#
#        if done:
#            for key in self.frames.keys():
#                save_file = os.path.join(self.save_folder,
#                                         f'{self.current_episode}_{key}.gif')
#                imageio.mimwrite(save_file, self.frames[key], fps=self.fps)
#                self.frames[key] = []
#            self.current_episode += 1
#
#        return observation, reward, done, info