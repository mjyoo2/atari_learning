from old_env import AtariWrapEnv
from stable_baselines import PPO2

if __name__ =='__main__':
    env = AtariWrapEnv('Breakout-v0')
    model = PPO2.load('C:\model\Atari_game\Breakout-v0/Breakout-v0_PPO2.pkl')

    for _ in range(10000):
        done = False
        state = env.reset()
        while not done:
            action = model.predict(state)[0]
            state, reward, done, info = env.step(action)