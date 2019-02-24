import gym
import walking_spider

env = gym.make('WalkingSpider-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action