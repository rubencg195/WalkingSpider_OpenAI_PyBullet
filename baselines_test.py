import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('Ant-v2') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000000)

# model.save("ppo2_ant_v2")
# del model # remove to demonstrate saving and loading
model = PPO2.load("ppo2_ant_v2")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()