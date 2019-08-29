import gym
import os
from wrap_env import AtariWrapEnv
from stable_baselines import ACKTR
from DQN import AtariDQN, nature_cnn_ex
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import LnCnnPolicy
from stable_baselines.common.policies import CnnPolicy
from ACKTR import checkpoint_ACKTR

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    game_id = "Breakout-v0"

    n_cpu = 12
    env = SubprocVecEnv([lambda: AtariWrapEnv(game_id, render_mode=False) for i in range(n_cpu)])

    model = checkpoint_ACKTR(CnnPolicy, env, verbose=1, policy_kwargs={'cnn_extractor': nature_cnn_ex, 'layers': []}, full_tensorboard_log=True,
                tensorboard_log='/model')

    # model = ACKTR.load('/model/Breakout_16000_ACKTR.pkl', env)
    # model.full_tensorboard_log = True
    # model.tensorboard_log='/model'
    model.learn(total_timesteps=50000000, log_interval= 30, reset_num_timesteps=False)