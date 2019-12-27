from wrap_env import AtariWrapEnv
from network import nature_cnn_ex2
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy

if __name__ == '__main__':
    game_id = "Breakout-v0"

    n_cpu = 12
    env = SubprocVecEnv([lambda: AtariWrapEnv(game_id, render_mode=False) for i in range(n_cpu)])

    model = PPO2(CnnPolicy, env, verbose=1, policy_kwargs={'cnn_extractor': nature_cnn_ex2, 'layers': [256]},
                     tensorboard_log='./result/{}/'.format(game_id))

    model.learn(total_timesteps=10000000, log_interval= 30, reset_num_timesteps=False)
    model.save('./result/{}/model.pkl'.format(game_id))