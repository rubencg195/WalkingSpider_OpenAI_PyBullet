import gym
from gym import spaces
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data

import time
import math
import numpy as np

# Import GIF recorder for visual debugging
try:
    from .gif_recorder import GifRecorder
    GIF_RECORDER_AVAILABLE = True
except ImportError:
    GIF_RECORDER_AVAILABLE = False
    print("⚠️  GIF recorder not available")


class WalkingSpiderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self, render=True, enable_gif_recording=False):
        super(WalkingSpiderEnv, self).__init__()
        # FIXED: Action space should be (8,) not (10,) - we have 8 moving joints
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(75,), dtype=np.float32)

        self._observation = []
        self.render_mode = render
        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
        
        # Initialize GIF recorder for visual debugging
        if GIF_RECORDER_AVAILABLE and enable_gif_recording:
            self.gif_recorder = GifRecorder(
                save_dir='videos', 
                duration_seconds=10, 
                fps=30,
                enabled=True
            )
            self.gif_recorder.start_recording()
        else:
            self.gif_recorder = None
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        self._seed()

        p.resetSimulation()
        p.setGravity(0, 0, -10)  # m/s^2
        # p.setTimeStep(1./60.)   # sec
        p.setTimeStep(0.01)   # sec
        self.plane = p.loadURDF("plane.urdf")

        self.cubeStartPos = [0, 0, 0.06]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.robotId = p.loadURDF(
            os.path.join(path, "spider.xml"),
            self.cubeStartPos,
            self.cubeStartOrientation
        )
        self.movingJoints = [0, 2, 3, 5, 6, 8, 9, 11]
        
        # IMPROVEMENT: Set friction parameters to prevent slipping
        # Set friction for the plane (ground)
        p.changeDynamics(
            self.plane, -1,
            lateralFriction=1.5,      # Increased from default ~0.5
            spinningFriction=0.1,
            rollingFriction=0.01,
            restitution=0.0           # No bouncing
        )
        
        # Set friction for robot leg tips (link indices for leg endpoints)
        leg_link_indices = [2, 5, 8, 11]  # Front-Left, Front-Right, Back-Left, Back-Right
        for link in leg_link_indices:
            p.changeDynamics(
                self.robotId, link,
                lateralFriction=1.5,
                spinningFriction=0.1,
                rollingFriction=0.01,
                contactStiffness=30000,   # Stiffer contact
                contactDamping=1000       # Damped contact
            )
        
        # Set friction for base body to prevent sliding
        p.changeDynamics(
            self.robotId, -1,
            lateralFriction=0.8,
            mass=0.2
        )

    def reset(self):
        self.vt = [0, 0, 0, 0, 0, 0, 0, 0]
        self.vd = 0
        self.maxV = 8.72  # 0.12sec/60 deg = 500 deg/s = 8.72 rad/s
        self.envStepCounter = 0
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=self.cubeStartPos,
            ornObj=self.cubeStartOrientation
        )
        observation = self.compute_observation()
        return observation

    def step(self, action):
        self.assign_throttle(action)
        observation = self.compute_observation()
        reward = self.compute_reward()
        done = self.compute_done()
        self.envStepCounter += 1
        
        # Capture frame for GIF recording (every other frame for 30fps)
        if self.gif_recorder and self.envStepCounter % 2 == 0:
            self.gif_recorder.capture_frame(self.physicsClient)
        
        return observation, reward, done, {}

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
    def moveLeg(self, robot, id, target):
      if(robot is None):
          return
      # IMPROVEMENT: Added PD control parameters and force limits for better motor control
      p.setJointMotorControl2(
          bodyUniqueId=robot,
          jointIndex=id,
          controlMode=p.POSITION_CONTROL,
          targetPosition=target,
          force=10.0,              # Maximum force (Newton-meters)
          positionGain=0.5,        # P gain for position control
          velocityGain=0.1,        # D gain for damping
          maxVelocity=3.0          # Limit maximum velocity (rad/s)
      )    

    def assign_throttle(self, action):
      for i, key in enumerate(self.movingJoints):
        self.vt[i] = self.clamp(self.vt[i] + action[i], -2, 2)
        self.moveLeg(robot=self.robotId, id=key,  target=self.vt[i])


    def compute_observation(self):
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      JointStates = p.getJointStates(self.robotId, self.movingJoints)
      BaseAngVel = p.getBaseVelocity(self.robotId)
      ContactPoints = p.getContactPoints(self.robotId, self.plane)

      debug = False
      if (debug):
        print("\nBase Orientation \nPos( x= {} , y = {} , z = {} )\nRot Quaternion( x = {} , y = {} , z = {}, w = {} )\n\n".format(
            baseOri[0][0], baseOri[0][1], baseOri[0][2],
            baseOri[1][0], baseOri[1][1], baseOri[1][2], baseOri[1][3]
        ))
        print(
            "\nJointStates: (Pos,Vel,6 Forces [Fx, Fy, Fz, Mx, My, Mz], appliedJointMotorTorque)\n")
        for i, joint in enumerate(JointStates):
            print("Joint #{} State: Pos {}, Vel {} Fx {} Fy {} Fz {} Mx {} My {} Mz {}, ApliedJointTorque {} ".format(
                i, joint[0], joint[1], joint[2][0], joint[2][1], joint[2][2], joint[2][3], joint[2][4], joint[2][5], joint[3]))
        print("\nBase Angular Velocity (Linear Vel( x= {} , y= {} , z=  {} ) Algular Vel(wx= {} ,wy= {} ,wz= {} ) ".format(
            BaseAngVel[0][0], BaseAngVel[0][1], BaseAngVel[0][2], BaseAngVel[1][0], BaseAngVel[1][1], BaseAngVel[1][2]))
      #print( "\n\nContact Points" ,ContactPoints if len(ContactPoints) > 0 else None )
      # print("\nContactPoints (contactFlag, idA, idB, linkidA, linkidB, posA, posB, contactnormalonb, contactdistance, normalForce, lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2)\n\n".format(
      # ))
      # print("Contact Point ", len(ContactPoints), ContactPoints[0] if len(ContactPoints) > 0 else None )

      obs = np.array([
          baseOri[0][2],  # z (height) of the Torso -> 1
          # orientation (quarternion x,y,z,w) of the Torso -> 4
          baseOri[1][0],
          baseOri[1][1],
          baseOri[1][2],
          baseOri[1][3],
          JointStates[0][0],  # Joint angles(Pos) -> 8
          JointStates[1][0],
          JointStates[2][0],
          JointStates[3][0],
          JointStates[4][0],
          JointStates[5][0],
          JointStates[6][0],
          JointStates[7][0],
          # 3-dim directional velocity and 3-dim angular velocity -> 3+3=6
          BaseAngVel[0][0],
          BaseAngVel[0][1],
          BaseAngVel[0][2],
          BaseAngVel[1][0],
          BaseAngVel[1][1],
          BaseAngVel[1][2],
          JointStates[0][1],  # Joint Velocities -> 8
          JointStates[1][1],
          JointStates[2][1],
          JointStates[3][1],
          JointStates[4][1],
          JointStates[5][1],
          JointStates[6][1],
          JointStates[7][1]
      ])
      # External forces (force x,y,z + torque x,y,z) applied to the CoM of each link (Ant has 14 links: ground+torso+12(3links for 4legs) for legs -> (3+3)*(14)=84
      external_forces = np.array([np.array(joint[2])
                                  for joint in JointStates])
      # print("Joint State Shape" , external_forces.shape, external_forces.flatten().shape )
      obs = np.append(obs, external_forces.ravel())
      # print("Obs: ", obs.shape, obs)
      return obs.tolist()

    def compute_reward(self):
      """
      IMPROVEMENT: Multi-objective reward function for better walking behavior.
      Rewards forward motion, stability, and energy efficiency while penalizing bad contacts.
      """
      # Get current state
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      BaseAngVel = p.getBaseVelocity(self.robotId)
      JointStates = p.getJointStates(self.robotId, self.movingJoints)
      ContactPoints = p.getContactPoints(self.robotId, self.plane)
      
      p.stepSimulation()
      
      # Get state after simulation step
      baseOri_after = np.array(p.getBasePositionAndOrientation(self.robotId))
      BaseAngVel_after = p.getBaseVelocity(self.robotId)
      
      # 1. FORWARD VELOCITY REWARD (main objective)
      forward_velocity = BaseAngVel_after[0][0]
      forward_reward = 10.0 * forward_velocity
      
      # 2. STABILITY REWARD (keep upright at target height)
      z_height = baseOri_after[0][2]
      target_height = 0.06
      height_reward = -5.0 * abs(z_height - target_height)
      
      # 3. ORIENTATION PENALTY (prevent flipping)
      orientation = baseOri_after[1]  # quaternion
      roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
      orientation_penalty = -2.0 * (abs(roll) + abs(pitch))
      
      # 4. ENERGY EFFICIENCY (minimize torque usage)
      torques = np.array([joint[3] for joint in JointStates])
      energy_cost = -0.1 * np.square(torques).sum()
      
      # 5. SMOOTH MOTION (minimize jerky movements)
      joint_vels = np.array([joint[1] for joint in JointStates])
      smoothness_penalty = -0.01 * np.square(joint_vels).sum()
      
      # 6. CONTACT PENALTY (only penalize body/base contacts, not foot contacts)
      leg_links = {2, 5, 8, 11}  # Foot link indices
      bad_contacts = [c for c in ContactPoints if c[4] not in leg_links]
      contact_penalty = -5.0 * len(bad_contacts)
      
      # 7. SURVIVAL BONUS
      alive_bonus = 1.0
      
      # TOTAL REWARD
      total_reward = (
          forward_reward + 
          height_reward + 
          orientation_penalty + 
          energy_cost + 
          smoothness_penalty + 
          contact_penalty + 
          alive_bonus
      )
      
      # Clip to non-negative (optional - can allow negative for learning)
      reward = max(0, total_reward)
      
      # Visual debug info
      p.addUserDebugLine(
          lineFromXYZ=(0, 0, 0), lineToXYZ=(0.3, 0, 0), 
          lineWidth=5, lineColorRGB=[0, 255, 0], 
          parentObjectUniqueId=self.robotId
      )
      p.addUserDebugText(
          f"R:{reward:.1f} V:{forward_velocity:.2f}", 
          [0, 0, 0.3], lifeTime=0.25, textSize=2.5, 
          parentObjectUniqueId=self.robotId
      )
      
      return reward 

    def compute_done(self):
      """
      IMPROVEMENT: Proper termination conditions.
      Episode ends if robot falls, flips over, or reaches max steps.
      """
      baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
      z_height = baseOri[0][2]
      
      # Terminate if fallen (too low)
      if z_height < 0.03:
          return True
      
      # Terminate if flipped over (extreme orientation)
      orientation = baseOri[1]
      roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
      if abs(roll) > 1.5 or abs(pitch) > 1.5:  # ~85 degrees
          return True
      
      # Terminate after maximum steps (prevent infinite episodes)
      if self.envStepCounter > 1000:
          return True
      
      return False
    
    def render(self, mode='human', close=False):
      pass