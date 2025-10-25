# WalkingSpider - AI-Powered Quadruped Robot Simulation & Training

A machine learning project that trains a simulated 8-legged quadruped spider robot to walk using reinforcement learning. The project combines PyBullet physics simulation with OpenAI Gym and Stable Baselines PPO2 algorithm for autonomous locomotion learning.

## Overview

This project demonstrates training a physics-simulated spider robot to develop walking behaviors through reinforcement learning. The spider has 8 controllable joints (2 legs × 4 vertical/horizontal servo pairs) and learns to coordinate these joints to achieve forward locomotion.

**Key Features:**
- **PyBullet Physics Simulation**: Realistic physics-based robot simulation with gravity, collisions, and joint constraints
- **OpenAI Gym Integration**: Standard RL environment interface for training algorithms
- **PPO2 Reinforcement Learning**: Proximal Policy Optimization algorithm from Stable Baselines for efficient training
- **8-Legged Quadruped Design**: Biologically-inspired spider robot with 4 legs (front-left, front-right, back-left, back-right)
- **Detailed Robot Model**: URDF/XML definitions with accurate joint limits, masses, and inertia parameters
- **Multi-CPU Training**: Support for parallel environment instances to accelerate learning

## Project Structure

```
WalkingSpider_OpenAI_PyBullet/
├── environment/
│   └── walking-spider/                    # Custom Gym environment package
│       ├── walking_spider/
│       │   ├── __init__.py               # Gym environment registration
│       │   └── envs/
│       │       ├── walking_spider_env.py # Main environment implementation
│       │       ├── spider.xml            # Robot URDF definition (8 joints)
│       │       ├── spider_simple.urdf    # Alternative simplified model
│       │       └── meshes/               # 3D models for robot visualization
│       └── setup.py                       # Environment package installation
├── src/
│   ├── spider.xml                        # Robot model (duplicate for reference)
│   └── meshes/                           # Asset files for simulation
├── experience_learned/                   # Trained model weights
│   └── ppo2_WalkingSpider_v0_*.pkl      # Pre-trained PPO2 models
├── tests/                                # Testing and comparison scripts
│   ├── ant.py                           # Ant robot baseline
│   ├── mujoco_test.py                   # MuJoCo comparisons
│   ├── robotschool_test.py              # RobotSchool environment tests
│   └── pybullet_testing_env.py          # PyBullet environment validation
├── docs/                                # Reference documentation
│   ├── Balancing bot building using OpenAI's Gym and pyBullet.pdf
│   └── PyBullet Quickstart Guide.pdf
├── files/                               # 3D printable STL files for physical robot
│   ├── Base_MG90.stl
│   ├── Servo_leg_MG90.stl
│   ├── Top_cover.stl
│   ├── Battery_cover.stl
│   └── U_servo_MG90.stl
├── images/                              # Screenshots and visualizations
│   ├── URDF.png                        # Robot structure diagram
│   ├── spider.gif                      # Walking animation
│   ├── PyBullet.png                    # Simulation screenshot
│   ├── CAD/                            # CAD design images
│   └── RENDERS/                        # 3D render previews
├── logs/                               # Training logs and performance data
├── videos/                             # Recorded simulation videos
│   ├── AntV2_RobotSchool_WalkingSpider_V1.mp4
│   └── AntV2_RobotSchool_WalkingSpider_V2.mp4
├── Walking_Spider_Training.ipynb       # Jupyter notebook for training
├── test_gym_spider_env.py              # Quick environment test script
└── walking_spider.yml                  # Conda environment specification
```

## Robot Design

The simulated spider robot has the following characteristics:

**Structure:**
- 4 legs (Front-Left, Front-Right, Back-Left, Back-Right)
- Each leg has 2 joints: vertical (yaw, Z-axis rotation) and horizontal (pitch, Y-axis rotation)
- Total of 8 controllable joints
- Central base body: 125×125×35mm with 0.2kg mass

**Joint Configuration:**
- **Front-Left**: Joints 0 (vertical), 1 (horizontal)
- **Front-Right**: Joints 2 (vertical), 3 (horizontal)
- **Back-Left**: Joints 4 (vertical), 5 (horizontal)
- **Back-Right**: Joints 6 (vertical), 7 (horizontal)

**Action Space:**
- 8-dimensional continuous control: [-1, 1] for each joint
- Actions are velocity targets converted to position commands

**Observation Space:**
- 75-dimensional state vector including:
  - Base position and orientation (quaternion)
  - Base linear and angular velocities
  - Joint positions and velocities (all 8 joints)
  - Contact forces and sensor data

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- CUDA/GPU support (optional, for faster training)
- Git Bash or similar terminal (Windows)

### 1. Clone the Repository

```bash
cd "C:\Users\ruben\Documents\Projects"
git clone <repository-url>
cd WalkingSpider_OpenAI_PyBullet
```

### 2. Create Conda Environment

**Option A: Using the provided environment file**

```bash
conda env create -f walking_spider.yml
conda activate walking_spider
```

**Option B: Create a fresh environment**

```bash
conda create -n walking_spider python=3.6
conda activate walking_spider
```

### 3. Install the Custom Gym Environment

```bash
cd environment/walking-spider
pip install -e .
```

This will install:
- `gym==0.10.9`
- `pybullet==2.4.1`
- `stable-baselines==0.1.5` (for PPO2)

### 4. Install Additional Dependencies (if needed)

For the full feature set with TensorFlow logging and other tools:

```bash
pip install tensorflow==1.12.0 tensorboard==1.12.1
```

**All dependencies from `walking_spider.yml`:**
- Core: gym, pybullet, numpy, scipy
- Training: stable-baselines, tensorflow, keras, torch, torchvision
- Utilities: opencv-python, matplotlib, pillow, h5py
- Environment: mujoco-py, atari-py, box2d-py

## Usage

### Quick Test: Run a Pre-trained Agent

```bash
# Run the pre-trained spider model
python test_gym_spider_env.py
```

This loads the best trained model (`ppo2_WalkingSpider_v0_testing_3.pkl`) and demonstrates the learned walking behavior in the PyBullet GUI.

### Train a New Model

#### Option 1: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook Walking_Spider_Training.ipynb
```

The notebook includes:
- Environment initialization
- PPO2 model training configuration
- Training loop with progress tracking
- Model saving and loading
- Visualization of learned behaviors

#### Option 2: Using Python Script

Create a training script (e.g., `train_spider.py`):

```python
import gym
import walking_spider
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# Parallel training on 4 CPUs
n_cpu = 4
total_timesteps = 200000000

env = SubprocVecEnv([lambda: gym.make('WalkingSpider-v0') for i in range(n_cpu)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=total_timesteps)
model.save("experience_learned/ppo2_WalkingSpider_v0_training")
```

Run it:

```bash
python train_spider.py
```

### Test the Environment

```bash
python test_gym_spider_env.py
```

Or create a simple test script:

```python
import gym
import walking_spider

env = gym.make('WalkingSpider-v0')
env.reset()

for episode in range(5):
    obs = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
        if done:
            break
    
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
```

### Run Comparison Tests

Compare the spider with other environments:

```bash
python tests/ant.py              # Test against OpenAI's Ant robot
python tests/mujoco_test.py      # MuJoCo physics comparison
python tests/robotschool_test.py # RobotSchool environment tests
```

## Training Details

### PPO2 Algorithm Configuration

The project uses PPO2 (Proximal Policy Optimization v2) from Stable Baselines:

**Typical Training Parameters:**
- **Algorithm**: PPO2
- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Total Timesteps**: 200M+ (can vary)
- **Parallel Environments**: 4-8 CPUs
- **Network Architecture**: 2 hidden layers (64 units each)

**Reward Function:**
The environment calculates rewards based on:
- Forward velocity of the robot (positive reward for moving forward)
- Energy efficiency (penalty for excessive joint movements)
- Stability (penalty for falling or extreme tilting)
- Joint position smoothness

### Training Progress

Training timeline for best results:
- **0-10M steps**: Initial exploration, basic walking emerges
- **10-50M steps**: Improved coordination, stable gait development
- **50-100M steps**: Fine-tuning, speed optimization
- **100M+ steps**: Refinement, robust behavior

Pre-trained models are provided in `experience_learned/`:
- `ppo2_WalkingSpider_v0.pkl` - Initial training
- `ppo2_WalkingSpider_v0_testing.pkl` - Testing phase v1
- `ppo2_WalkingSpider_v0_testing_3.pkl` - Best version (recommended)

## Rendering Modes

### PyBullet GUI Visualization

When `render=True` in the environment:
- **Interactive 3D camera**: Click and drag to rotate
- **Real-time physics**: Watch joint movements and body dynamics
- **Collision visualization**: See ground contact points

### Video Recording

The environment supports rendering to video files:

```python
env = gym.make('WalkingSpider-v0')
env = gym.wrappers.Monitor(env, 'videos/', force=True)
```

Sample videos are in the `videos/` directory showing different training stages.

## File Formats

### Robot Definition Files

- **spider.xml** (URDF format): Complete robot model with joints and links
  - 8 controllable revolute joints
  - 4 fixed horizontal servo mounts
  - Base body link with collision geometry
  
- **spider_simple.urdf**: Simplified version for faster simulation

- **Mesh files**:
  - `base.obj/mtl`: 3D model of robot base body
  - `leg.stl`, `servo_joint.stl`: STL files for 3D reference

### Model Checkpoints

- **ppo2_*.pkl**: Pickled Python files containing trained neural network weights
  - Can be loaded with `PPO2.load("path/to/model")`
  - Compatible with Stable Baselines

## Physical Robot Implementation

This project includes CAD designs and 3D-printable STL files for building the physical spider robot:

**3D Printable Parts** (in `files/` directory):
- Base servo mount (MG90 compatible)
- Servo leg attachments
- Top/bottom covers
- Battery holder

**Components Needed for Physical Build:**
- 4× MG90 servo motors
- Microcontroller (Raspberry Pi, Arduino, or similar)
- Power supply for servos
- Structural materials (3D printed parts or aluminum)
- Wiring and connectors

**Physical Integration:**
The real robot would use the same servo coordinate system as the simulation, allowing direct transfer of trained policies.

## Performance Metrics

### Learning Curves

Monitor training progress:
- **Reward per episode**: Tracked in TensorBoard logs
- **Average episode length**: Should increase with better walking
- **Success rate**: Episodes where robot moves forward > minimum threshold

### Comparison Benchmarks

The project includes comparisons with:
- **OpenAI Ant**: 8-legged agent from OpenAI baseline
- **MuJoCo physics**: Alternative physics engine comparison
- **RobotSchool environments**: General robotics benchmarks

## Troubleshooting

### Common Issues

**Issue**: ImportError for walking_spider
- **Solution**: Ensure you've run `pip install -e .` in the `environment/walking-spider/` directory

**Issue**: PyBullet window doesn't open (GUI mode)
- **Solution**: Ensure you have display server available. Use DIRECT mode for headless training.

**Issue**: Out of memory during training
- **Solution**: Reduce `n_cpu` from 4 to 2 or 1, or reduce total timesteps

**Issue**: Slow training
- **Solution**: Use GPU acceleration, reduce environment complexity, or use DIRECT mode instead of GUI

### Debugging

Enable verbose output:

```python
model = PPO2(MlpPolicy, env, verbose=2)  # More detailed logging
model.learn(total_timesteps=100000, log_interval=10)
```

## Advanced Topics

### Hyperparameter Tuning

Modify PPO2 parameters for better performance:

```python
model = PPO2(
    MlpPolicy, env,
    learning_rate=3e-4,
    n_steps=512,
    nminibatches=8,
    noptepochs=4,
    ent_coef=0.01,
    verbose=1
)
```

### Custom Reward Functions

Modify `compute_reward()` in `environment/walking-spider/walking_spider/envs/walking_spider_env.py` to shape the reward signal differently.

### Environment Variations

Create variants by modifying:
- Joint limits and velocities
- Gravity and friction coefficients
- Initial robot position/orientation
- Observation/action spaces

## References

### Key Papers & Resources

- **PPO Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **PyBullet Documentation**: https://pybullet.org
- **OpenAI Gym**: https://gym.openai.com
- **Stable Baselines**: https://stable-baselines.readthedocs.io

### Related Projects

- OpenAI's Ant environment
- DeepMind's locomotion suite
- MIT's Cheetah robot control
- Boston Dynamics-inspired research

### Included Documentation

- `docs/PyBullet Quickstart Guide.pdf` - PyBullet API basics
- `docs/Balancing bot building using OpenAI's Gym and pyBullet.pdf` - Design methodology

## Video Demo

See the trained spider robot in action:

**YouTube Demo**: https://youtu.be/j9sysG-EIkQ

[![Watch the video](/images/youtube.png)](https://youtu.be/j9sysG-EIkQ)

## Project Showcase

### Design & CAD

![CAD Design](/images/CAD/1aab238b7800d3096ec45e017554c280_preview_featured.jpg)

### Simulation Screenshot

![PyBullet Simulation](/images/PyBullet.png)

### Robot URDF Structure

![URDF Structure](/images/URDF.png)

### Walking Animation

![Spider Walking](/images/spider.gif)

### Physical Prototype

![Physical Robot Construction](/images/spider(2).jpeg)
![Physical Robot Assembly](/images/spider(4).jpeg)
![Detailed View](/images/spider(8).jpeg)
![Robot Setup](/images/spider(1).jpeg)

## TODO: Planned Improvements

The following improvements have been identified to enhance the robot's walking behavior, physics realism, and training effectiveness:

### Improvement Tracking Table

| Priority | Category | Issue | Status | Estimated Time | Impact |
|----------|----------|-------|--------|----------------|--------|
| 🔴 Critical | Physics | **Fix Reward Function Bug** - Line 177 rewards slowing down instead of speeding up (`xvelbefore - xvelafter` should be reversed) | 🔄 Pending Testing | 5 min | High |
| 🔴 Critical | Physics | **Add Friction Parameters** - No friction coefficients set causing slippery legs and floor. Need lateralFriction, spinningFriction, rollingFriction for plane and legs | 🔄 Pending Testing | 10 min | High |
| 🔴 Critical | Code Quality | **Fix Action Space Mismatch** - Action space is `(10,)` but only 8 joints exist. Should be `shape=(8,)` | 🔄 Pending Testing | 2 min | Medium |
| 🟡 High | Control | **Improve Motor Control Parameters** - Add force limits, positionGain, velocityGain, and maxVelocity to `setJointMotorControl2()` | 🔄 Pending Testing | 10 min | High |
| 🟡 High | Reward | **Better Reward Shaping** - Replace simple reward with multi-objective: forward velocity + stability + orientation + energy efficiency + smoothness (includes Fix Contact Cost Logic) | 🔄 Pending Testing | 20 min | High |
| 🟡 High | Termination | **Add Proper Done Conditions** - Currently never terminates. Add height check, flip detection, and max steps | 🔄 Pending Testing | 10 min | Medium |
| 🟢 Medium | Physics | **Add Joint Damping to URDF** - Add `<dynamics damping="0.5" friction="0.1"/>` to all revolute joints in `spider.xml` | 🔄 Pending Testing | 15 min | Medium |
| 🟢 Medium | Debugging | **Add GIF Snapshot System** - Automatically capture random 10-second GIF snapshots during training with timestamp filenames saved to videos/ folder for visual debugging | 🔄 Pending Testing | 25 min | Medium |
| 🟢 Medium | Observation | **Add Foot Contact Sensors** - Include binary foot contact state (4 values) in observation space for better gait learning | ⏳ Pending | 15 min | Medium |
| 🟢 Medium | Physics | **Improve Contact Parameters** - Add contactStiffness and contactDamping to leg links for more realistic ground interaction | 🔄 Pending Testing | 10 min | Medium |
| 🔵 Low | Training | **Add Training Curriculum** - Implement difficulty levels (easy: high friction → hard: slippery floor) for progressive learning | ⏳ Pending | 30 min | Low |
| 🔵 Low | Visualization | **Debug Visualization** - Add visual indicators for contact forces, friction vectors, and reward components | ⏳ Pending | 20 min | Low |
| 🔵 Low | Reward | **Add Gait Quality Metrics** - Reward coordinated leg movement patterns and penalize chaotic motions | ⏳ Pending | 30 min | Low |

**Status Legend:**
- ⏳ Pending - Not started
- 🔄 Pending Testing - Implemented but not tested
- ✅ Complete - Implemented and tested
- ❌ Blocked - Waiting on dependencies

### Detailed Improvement Specifications

#### 1. Fix Reward Function Bug (CRITICAL)

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py:177`

**Current Code:**
```python
forward_reward = 20 * (xvelbefore - xvelafter)  # WRONG: Rewards slowing down!
```

**Fixed Code:**
```python
forward_reward = 10 * xvelafter  # Directly reward forward velocity
```

#### 2. Add Friction Parameters (CRITICAL)

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py` in `__init__()` method

**Add after robot loading:**
```python
# Set friction for the plane
p.changeDynamics(
    self.plane, -1,
    lateralFriction=1.5,
    spinningFriction=0.1,
    rollingFriction=0.01,
    restitution=0.0
)

# Set friction for robot legs (link indices for leg tips)
leg_link_indices = [2, 5, 8, 11]
for link in leg_link_indices:
    p.changeDynamics(
        self.robotId, link,
        lateralFriction=1.5,
        spinningFriction=0.1,
        rollingFriction=0.01,
        contactStiffness=30000,
        contactDamping=1000
    )
```

#### 3. Fix Action Space Mismatch (CRITICAL)

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py:26-27`

**Change from:**
```python
self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
```

**To:**
```python
self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
```

#### 4. Improve Motor Control Parameters

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py` in `moveLeg()` method

**Enhanced motor control:**
```python
def moveLeg(self, robot, id, target):
    if robot is None:
        return
    p.setJointMotorControl2(
        bodyUniqueId=robot,
        jointIndex=id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target,
        force=10.0,              # Max force (Newton-meters)
        positionGain=0.5,        # P gain for position control
        velocityGain=0.1,        # D gain for damping
        maxVelocity=3.0          # Limit maximum velocity
    )
```

#### 5. Better Reward Shaping

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py` in `compute_reward()` method

**Comprehensive multi-objective reward:**
```python
def compute_reward(self):
    baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
    BaseAngVel = p.getBaseVelocity(self.robotId)
    JointStates = p.getJointStates(self.robotId, self.movingJoints)
    
    p.stepSimulation()
    
    baseOri_after = np.array(p.getBasePositionAndOrientation(self.robotId))
    BaseAngVel_after = p.getBaseVelocity(self.robotId)
    
    # 1. Forward velocity reward (main objective)
    forward_velocity = BaseAngVel_after[0][0]
    forward_reward = 10.0 * forward_velocity
    
    # 2. Stability reward (keep upright)
    z_height = baseOri_after[0][2]
    target_height = 0.06
    height_reward = -5.0 * abs(z_height - target_height)
    
    # 3. Orientation penalty (prevent flipping)
    orientation = baseOri_after[1]
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    orientation_penalty = -2.0 * (abs(roll) + abs(pitch))
    
    # 4. Energy efficiency
    torques = np.array([joint[3] for joint in JointStates])
    energy_cost = -0.1 * np.square(torques).sum()
    
    # 5. Smooth motion
    joint_vels = np.array([joint[1] for joint in JointStates])
    smoothness_penalty = -0.01 * np.square(joint_vels).sum()
    
    # 6. Bad contact penalty (body touching ground)
    ContactPoints = p.getContactPoints(self.robotId, self.plane)
    leg_links = {2, 5, 8, 11}
    bad_contacts = [c for c in ContactPoints if c[4] not in leg_links]
    contact_penalty = -5.0 * len(bad_contacts)
    
    # 7. Survival bonus
    alive_bonus = 1.0
    
    total_reward = (
        forward_reward + height_reward + orientation_penalty + 
        energy_cost + smoothness_penalty + contact_penalty + alive_bonus
    )
    
    return max(0, total_reward)
```

#### 6. Add Proper Termination Conditions

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py` in `compute_done()` method

**Replace:**
```python
def compute_done(self):
    return False
```

**With:**
```python
def compute_done(self):
    baseOri = np.array(p.getBasePositionAndOrientation(self.robotId))
    z_height = baseOri[0][2]
    
    # Terminate if fallen
    if z_height < 0.03:
        return True
    
    # Terminate if flipped over
    orientation = baseOri[1]
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    if abs(roll) > 1.5 or abs(pitch) > 1.5:  # ~85 degrees
        return True
    
    # Terminate after max steps
    if self.envStepCounter > 1000:
        return True
    
    return False
```

#### 7. Add Joint Damping to URDF

**File:** `environment/walking-spider/walking_spider/envs/spider.xml`

**Add to each revolute joint:**
```xml
<dynamics damping="0.5" friction="0.1"/>
```

**Example:**
```xml
<joint name="left_front_joint" type="revolute">
    <parent link="base_link" />
    <child link="left_v_front_link" />
    <origin xyz="0.04 0.05 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
    <limit lower="-0.4" upper="2.5" effort="10" velocity="3" />
    <dynamics damping="0.5" friction="0.1"/>
</joint>
```

#### 8. Add GIF Snapshot System

**New File:** `environment/walking-spider/walking_spider/envs/gif_recorder.py`

**Create a GIF recording utility:**
```python
import os
import imageio
import numpy as np
from datetime import datetime
import pybullet as p

class GifRecorder:
    def __init__(self, save_dir='videos', duration_seconds=10, fps=30):
        self.save_dir = save_dir
        self.duration_seconds = duration_seconds
        self.fps = fps
        self.max_frames = duration_seconds * fps
        self.frames = []
        self.recording = False
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start_recording(self):
        self.frames = []
        self.recording = True
    
    def capture_frame(self):
        if not self.recording:
            return
        
        # Capture frame from PyBullet
        width, height = 640, 480
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=0.8,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/height,
            nearVal=0.1, farVal=100.0
        )
        
        (_, _, px, _, _) = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
        self.frames.append(rgb_array)
        
        # Auto-save when duration reached
        if len(self.frames) >= self.max_frames:
            self.save_gif()
            self.start_recording()  # Start new recording
    
    def save_gif(self):
        if not self.frames:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f'spider_snapshot_{timestamp}.gif')
        
        imageio.mimsave(filename, self.frames, fps=self.fps)
        print(f"✅ GIF saved: {filename} ({len(self.frames)} frames)")
        
        self.frames = []
        self.recording = False
```

**Usage in environment:**
```python
# In walking_spider_env.py __init__:
from .gif_recorder import GifRecorder

self.gif_recorder = GifRecorder(save_dir='videos', duration_seconds=10, fps=30)
self.gif_recorder.start_recording()

# In step() method:
if self.envStepCounter % 2 == 0:  # Capture every other frame
    self.gif_recorder.capture_frame()
```

#### 9. Add Foot Contact Sensors (PENDING)

**File:** `environment/walking-spider/walking_spider/envs/walking_spider_env.py` in `compute_observation()` method

**Add before return statement:**
```python
# Add foot contact sensors (4 binary values)
foot_contacts = [0, 0, 0, 0]  # [FL, FR, BL, BR]
ContactPoints = p.getContactPoints(self.robotId, self.plane)
for contact in ContactPoints:
    link_id = contact[4]
    if link_id == 2: foot_contacts[0] = 1    # Front Left
    elif link_id == 5: foot_contacts[1] = 1  # Front Right
    elif link_id == 8: foot_contacts[2] = 1  # Back Left
    elif link_id == 11: foot_contacts[3] = 1 # Back Right

obs = np.append(obs, foot_contacts)
```

**Also update observation space size from 75 to 79.**

**Note:** This improvement is pending implementation.

#### 10. Improve Contact Parameters (ALREADY IMPLEMENTED)

**Status:** This was already implemented in step #2 (Add Friction Parameters) where we added:
- `contactStiffness=30000` 
- `contactDamping=1000`

These parameters were applied to leg links to improve ground interaction. No additional work needed.

### Quick Wins (Start Here!)

The following changes provide maximum impact with minimal effort:

1. **Fix reward function** (5 min) - Line 177 single line change
2. **Add friction** (10 min) - Copy-paste friction code block
3. **Fix action space** (2 min) - Change 10 to 8

These three fixes will immediately address the "slippery robot" issue visible in the training videos.

## Contributing

To contribute improvements:

1. Create a feature branch for each improvement (e.g., `feature/fix-reward-bug`)
2. Make your changes in that branch
3. Commit early and often - one commit per logical change
4. Test with `python test_gym_spider_env.py`
5. Push your branch and create a Pull Request
6. Update the TODO table status (⏳ → 🔄 → ✅)
7. Merge after review and testing

**Recent Commits (October 2025):**
- `6691d77` - docs: Update TODO table - fix ordering, change status to Pending Testing
- `2d03e40` - feat: Add joint damping to URDF (#7)
- `13c3ccf` - feat: Add GIF snapshot recording system (#8)
- `ef847f5` - feat: Add comprehensive debug logging system
- `5e2f0a7` - feat: Critical environment improvements (#1-6, #8 integration)
- `11cf0d3` - chore: Update package metadata from environment changes

**Note:** The environment improvements (#1-6) were committed together as they were intertwined in a single file. In future, commit each improvement in a separate branch/PR for better tracking.

## License

[Specify your license here]

## Contact & Support

For questions or issues:
- Check the included PDF documentation
- Review training notebooks for examples
- Examine test files for usage patterns

## Changelog

### Version 1.0
- Initial project setup with 8-joint spider robot
- PPO2 training integration
- Successful walking behavior demonstrated
- Pre-trained models included
- CAD files for physical construction
- Jupyter notebook for training

---

**Last Updated**: October 2025

**Project Status**: Active - Continuously improving robot locomotion behaviors through reinforcement learning


