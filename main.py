import gym
import os
from wrap_env import AtariWrapEnv
from stable_baselines import ACKTR
from save_PPO2 import savePPO2
from DQN import AtariDQN, nature_cnn_ex, nature_cnn_ex2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy
from ACKTR import checkpoint_ACKTR

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    game_id = "Boxing-v0"

    n_cpu = 6
    env = SubprocVecEnv([lambda: AtariWrapEnv(game_id, render_mode=False) for i in range(n_cpu)])

    model = savePPO2(CnnPolicy, env, verbose=1, policy_kwargs={'cnn_extractor': nature_cnn_ex2, 'layers': [512]})

    # model = ACKTR.load('/model/Breakout_16000_ACKTR.pkl', env)
    # model.full_tensorboard_log = True
    # model.tensorboard_log='/model'
    model.learn(total_timesteps=50000000, game_id = game_id, log_interval= 30, reset_num_timesteps=False)
    model.save('/model/Atari_game/{}/{}_ACKTR'.format(game_id, game_id))