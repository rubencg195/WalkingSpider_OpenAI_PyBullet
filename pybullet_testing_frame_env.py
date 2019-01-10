import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

import time
import math
import numpy as np


class WalkingSpider(gym.Env):
  def __init__(self, render=True):
    self._observation      = []
    self.action_space      = spaces.Box(low=-1, high=1, shape=(10,))
    self.observation_space = spaces.Box(low=-1, high=1, shape=(8,)) 
    if (render):
        self.physicsClient = p.connect(p.GUI)
    else:
        self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
    p.resetDebugVisualizerCamera( cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])
    self._seed()

    p.resetSimulation()
    p.setGravity(0,0,-10) # m/s^2
    p.setTimeStep(0.01)   # sec
    self.plane               = p.loadURDF("plane.urdf")

    self.cubeStartPos         = [0,0,0.06]
    self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    self.robotId              = p.loadURDF(
      "src/spider.xml",
      self.cubeStartPos, 
      self.cubeStartOrientation
    )
    self.movingJoints = [0, 2, 3, 5, 6, 8, 9, 11]
    #paramId = p.addUserDebugParameter("My Param", 0, 100, 50)
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    self.assign_throttle(action)
    p.stepSimulation()
    self.observation    = self.compute_observation()
    reward              = self.compute_reward()
    done                = self.compute_done()
    self.envStepCounter += 1
    return np.array(self.observation), reward, done, {}
  
  def reset(self):
    self.vt              = [0,0,0,0,0,0,0,0]
    self.vd              = 0
    self.maxV            = 8.72  #0.12sec/60 deg = 500 deg/s = 8.72 rad/s
    self.envStepCounter  = 0
    

    p.resetBasePositionAndOrientation(
      self.robotId,
      posObj=self.cubeStartPos,
      ornObj=self.cubeStartOrientation
    )
    
    self.observation     = self.compute_observation()
    return np.array(self._observation)
    
  def moveLeg(self, robot, id, target ):
    if(robot is None):
        return;
    p.setJointMotorControl2(
        bodyUniqueId  = robot,
        jointIndex    = id,
        # controlMode   = p.VELOCITY_CONTROL,        #p.POSITION_CONTROL,
        # targetVelocity= target                     #targetPosition=position,
        controlMode   = p.POSITION_CONTROL,          #p.POSITION_CONTROL,
        targetVelocity= target  
    )
  def assign_throttle(self, action):
    deltav = 0.05
    for i, key in enumerate(self.movingJoints) :
      self.vt[i] = self.clamp(self.vt[i] + action[i], -1, 1)
      self.moveLeg( robot=self.robotId, id=key,  target= self.vt[i] ) 
    #print(self.vt)

  def clamp(self, n, minn, maxn):
    return max(min(maxn, n), minn)

  def compute_observation(self):
    p.addUserDebugLine(lineFromXYZ=(0,0,0),lineToXYZ=(0.3,0,0), lineWidth=5, lineColorRGB=[0,255,0] ,parentObjectUniqueId=self.robotId )
    p.addUserDebugText("Rewards {}".format(0.0), [0,0,0.3], textSize=2.5, parentObjectUniqueId=self.robotId)

    baseOri       = np.array( p.getBasePositionAndOrientation(self.robotId) )
    JointStates   = p.getJointStates(self.robotId, self.movingJoints ) 
    BaseAngVel    = p.getBaseVelocity(self.robotId)
    ContactPoints = p.getContactPoints(self.robotId, self.plane)
    print("\nBase Orientation \nPos( x= {} , y = {} , z = {} )\nRot Quaternion( x = {} , y = {} , z = {}, w = {} )\n\n".format(
      baseOri[0][0], baseOri[0][1], baseOri[0][2], 
      baseOri[1][0], baseOri[1][1], baseOri[1][2], baseOri[1][3]
    ))

    # print("\nJointStates: (Pos,Vel,6 Forces [Fx, Fy, Fz, Mx, My, Mz], appliedJointMotorTorque)\n\n".format(

    # ))
    # print("\nBase Angular Velocity (Linear Vel(xyz) Algular Vel(wx,wy,wz)) \n\n".format(

    # ))
    # print("\ngetContactPoints (contactFlag, idA, idB, linkidA, linkidB, posA, posB, contactnormalonb, contactdistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2)\n\n".format(

    # ))
    return []

  def compute_reward(self):
    forward_reward = 0.0
    contact_cost   = 0.0
    ctrl_cost      = 0.0
    survive_reward = 1.0
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    return reward

  def compute_done(self):
    return False

  def render(self, mode='human', close=False):
    pass

if __name__ == "__main__":
  #Hyper-parameters
  max_steps    = 10000
  max_episodes = 100


  env = WalkingSpider(render=True)
  # print("Env Action Sample", env.action_space.sample())
  # print("Env Action Sample", env.observation_space.sample())
  for i in range (max_episodes):
    env.reset()
    for i in range (max_steps):
      env.render()
      env.step(env.action_space.sample())
      break
    break


"""
Limitations
left_front_joint      => lower="-0.4" upper="2.5" id=0
left_front_leg_joint  => lower="-0.6" upper="0.7" id=2

right_front_joint     => lower="-2.5" upper="0.4" id=3
right_front_leg_joint => lower="-0.6" upper="0.7" id=5

left_back_joint       => lower="-2.5" upper="0.4" id=6
left_back_leg_joint   => lower="-0.6" upper="0.7" id=8

right_back_joint      => lower="-0.4" upper="2.5" id=9
right_back_leg_joint  => lower="-0.6" upper="0.7" id=11
"""
