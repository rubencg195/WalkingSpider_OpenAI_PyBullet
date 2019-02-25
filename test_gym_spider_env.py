import gym
import walking_spider


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('WalkingSpider-v0') for i in range(n_cpu)])
# env = gym.make('RoboschoolAnt-v1')
env = DummyVecEnv([lambda: gym.make('WalkingSpider-v0')])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000000)

# model.save("ppo2_WalkingSpider_v0")
# del model # remove to demonstrate saving and loading
# # model = PPO2.load("ppo2_WalkingSpider_v0")
# model = PPO2.load("ppo2_WalkingSpider_v0")

# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# env = gym.make('Ant-v2')
# env = gym.make('WalkingSpider-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

#     print("Obs Shape ", env.action_space.sample().shape)
