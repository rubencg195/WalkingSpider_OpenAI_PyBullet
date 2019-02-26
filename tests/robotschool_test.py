import gym
import roboschool

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('RoboschoolAnt-v1') for i in range(n_cpu)])
# env = gym.make('RoboschoolAnt-v1')

model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000000)

# model.save("ppo2_robotschool_ant_v2")
# del model # remove to demonstrate saving and loading
model = PPO2.load("ppo2_robotschool_ant_v2")
# model = PPO2.load("ppo2_robotschool_ant_v2_2")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(obs.shape)
    env.render()


    

# env.reset()
# while True:
#     env.step(env.action_space.sample())
#     env.render()