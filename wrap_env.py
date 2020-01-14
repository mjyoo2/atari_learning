from skimage.transform import resize

import gym
import numpy as np
from skimage.color import rgb2gray

class AtariWrapEnv(gym.Env):
    def __init__(self, game_id, render_mode=True):
        self.game_id = game_id
        self.atari = gym.make(game_id)
        self.resize_shape = (84, 84, 1)
        self.real_state = [np.zeros(self.resize_shape), np.zeros(self.resize_shape), np.zeros(self.resize_shape), np.zeros(self.resize_shape)]
        self.render_mode = render_mode
        self.action_space = self.atari.action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 4))
        self.num_timesteps = 0
        self.sum_reward = 0

    def step(self, action):
        self.num_timesteps += 1
        state, reward, done, info = self.atari.step(action)
        if self.render_mode:
            self.render()
        state = resize(rgb2gray(state), (84, 84)).reshape((84, 84, 1))
        self.real_state.append(state)
        del self.real_state[0]
        output = (np.concatenate([self.real_state[0], self.real_state[1], self.real_state[2], self.real_state[3]], axis=2))
        self.sum_reward += reward
        if done:
            info = {'episode': {'r': self.sum_reward, 'l': self.num_timesteps}, 'game_reward': reward}
        else:
            info = {'episode': None, 'game_reward': reward}
        return output, reward, done, info


    def reset(self):
        state = self.atari.reset()
        state = resize(rgb2gray(state), (84, 84)).reshape(self.resize_shape)
        self.sum_reward = 0
        self.num_timesteps = 0
        self.real_state = [np.zeros(self.resize_shape), np.zeros(self.resize_shape), np.zeros(self.resize_shape), state]
        output = (np.concatenate([self.real_state[0], self.real_state[1], self.real_state[2], self.real_state[3]], axis=2))
        return output


    def render(self):
        self.atari.render()