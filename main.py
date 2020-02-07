from network import nature_cnn_ex2, local_nature_cnn_ex2
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy
from wrap_env import AtariWrapEnv

if __name__ == '__main__':
    game_id = "BreakoutDeterministic-v4"

    n_cpu = 12
    env = SubprocVecEnv([lambda: AtariWrapEnv(game_id, render_mode=False) for i in range(n_cpu)])

    model = PPO2(CnnPolicy, env, verbose=1,  n_steps=32, tensorboard_log='./result/{}/'.format(game_id),
                 policy_kwargs={'layers': [256], 'cnn_extractor': local_nature_cnn_ex2})

    model.learn(total_timesteps=50000000, log_interval= 30, reset_num_timesteps=False)
    model.save('./result/{}/local_model'.format(game_id))