import gym
import walking_spider
import pybullet as p
import pybullet_data
import os
from gym.utils import seeding

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from gym import spaces
import numpy as np

# n_cpu = 4
# total_timesteps = 200000000
# # total_timesteps = 200000
# env = SubprocVecEnv([lambda: gym.make('WalkingSpider-v0') for i in range(n_cpu)])
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=total_timesteps)
# model.save("experience_learned/ppo2_WalkingSpider_v0_testing")
# del model # remove to demonstrate saving and loading

# # # Enjoy trained agent
model = PPO2.load("experience_learned/ppo2_WalkingSpider_v0_testing_3")
print("Enjoy trained agent")
env = DummyVecEnv([lambda: gym.make('WalkingSpider-v0')])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Random Environment
# env = gym.make('WalkingSpider-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action


    # print("Obs Shape ", observation, " Action Shape ", env.action_space.sample().shape)
