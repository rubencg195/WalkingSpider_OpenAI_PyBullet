import pybullet as p
import time
import pybullet_data

def moveLeg( robot=None, id=0, position=0, force=0.5  ):
    if(robot is None):
        return;
    p.setJointMotorControl2(
        robot,
        id,
        p.POSITION_CONTROL,
        targetPosition=position,
        force=force,
        #maxVelocity=5
    )

pixelWidth = 1000
pixelHeight = 1000
camTargetPos = [0,0,0]
camDistance = 0.5
pitch = -10.0
roll=0
upAxisIndex = 2
yaw = 0

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
p.resetDebugVisualizerCamera( cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])

viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0.05]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
boxId = p.loadURDF("src/spider.xml",cubeStartPos, cubeStartOrientation)
# boxId = p.loadURDF("spider_simple.urdf",cubeStartPos, cubeStartOrientation)



toggle = 1



p.setRealTimeSimulation(1)

for i in range (10000):
    #p.stepSimulation()
    

    moveLeg( robot=boxId, id=0,  position= toggle * -2 ) #LEFT_FRONT
    moveLeg( robot=boxId, id=2,  position= toggle * -2 ) #LEFT_FRONT

    moveLeg( robot=boxId, id=3,  position= toggle * -2 ) #RIGHT_FRONT
    #moveLeg( robot=boxId, id=5,  position= toggle *  2 ) #RIGHT_FRONT

    moveLeg( robot=boxId, id=6,  position= toggle *  2 ) #LEFT_BACK
    #moveLeg( robot=boxId, id=8,  position= toggle * -2 ) #LEFT_BACK

    moveLeg( robot=boxId, id=9,  position= toggle *  2 ) #RIGHT_BACK
    #moveLeg( robot=boxId, id=11, position= toggle *  2 ) #RIGHT_BACK
    #time.sleep(1./140.)g
    #time.sleep(0.01)
    time.sleep(1)
    
    toggle = toggle * -1

    #viewMatrix        = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
    #projectionMatrix  = [1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
    #img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, shadow=1,lightDirection=[1,1,1])

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()


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


"""
 def init_custom_camera(self):
    self.pixelWidth = 1000
    self.pixelHeight = 1000
    self.camTargetPos = [0,0,0]
    self.camDistance = 0.5
    self.pitch = -10.0
    self.roll=0
    self.upAxisIndex = 2
    self.yaw = 0
  
  def render_custom_camera(self):
    viewMatrix        = p.computeViewMatrixFromYawPitchRoll(
      self.camTargetPos, 
      self.camDistance, 
      self.yaw, 
      self.pitch, 
      self.roll, 
      self.upAxisIndex
    )
    projectionMatrix  = [1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
    img_arr           = p.getCameraImage(
      self.pixelWidth, 
      self.pixelHeight, 
      viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, shadow=1,lightDirection=[1,1,1])
    print(len(img_arr))
"""