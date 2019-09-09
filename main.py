import gym
import os
from wrap_env import AtariWrapEnv
from stable_baselines import ACKTR
from DQN import AtariDQN, nature_cnn_ex, nature_cnn_ex2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy
from ACKTR import checkpoint_ACKTR

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    game_id = "BeamRider-v0"

    n_cpu = 16
    env = SubprocVecEnv([lambda: AtariWrapEnv(game_id, render_mode=False) for i in range(n_cpu)])

    model = checkpoint_ACKTR(CnnPolicy, env, verbose=1, policy_kwargs={'cnn_extractor': nature_cnn_ex2, 'layers': []}, full_tensorboard_log=True,
                tensorboard_log='/model/Atari_game/{}/'.format(game_id), n_steps=64)

    # model = ACKTR.load('/model/Breakout_16000_ACKTR.pkl', env)
    # model.full_tensorboard_log = True
    # model.tensorboard_log='/model'
    model.learn(total_timesteps=50000000, log_interval= 30, reset_num_timesteps=False)