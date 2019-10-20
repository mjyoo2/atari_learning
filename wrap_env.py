import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import block_reduce

class AtariWrapEnv(gym.Env):
    def __init__(self, game_id, render_mode=True):
        self.game_id = game_id
        self.atari = gym.make(game_id, obs_type='image', frameskip=3)
        self.real_state = [np.zeros([105, 80, 1]), np.zeros([105, 80, 1]), np.zeros([105, 80, 1]), np.zeros([105, 80, 1])]
        self.render_mode = render_mode
        self.action_space = self.atari.action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(105, 80, 4))
        self.num_timesteps = 0
        self.sum_reward = 0

    def step(self, action):
        self.num_timesteps += 1
        state, reward, done, info = self.atari.step(action)
        if self.render_mode:
            self.render()
        state = rgb2gray(state)
        state = block_reduce(state, block_size=(2, 2), func=np.max)
        self.real_state.append(state.reshape([105, 80, 1]))
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
        state = rgb2gray(state)
        state = block_reduce(state, block_size=(2, 2), func=np.max)
        self.sum_reward = 0
        self.num_timesteps = 0
        self.real_state = [np.zeros([105, 80, 1]), np.zeros([105, 80, 1]), np.zeros([105, 80, 1]), state.reshape([105, 80, 1])]
        output = (np.concatenate([self.real_state[0], self.real_state[1], self.real_state[2], self.real_state[3]], axis=2))
        return output


    def render(self):
        self.atari.render()