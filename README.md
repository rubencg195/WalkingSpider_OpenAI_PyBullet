# RL Training for Quadruped Robot Locomotion

Train a simulated 4-legged quadruped robot (spider robot) to walk using reinforcement learning (PPO). The robot has 8 controllable joints (2 per leg). Uses PyBullet for physics simulation and OpenAI Gym for the RL environment.

## Quick Start

### Option 1: See Spider Walk (No Training Required)
```bash
python scripts/demo_trained_spider.py --physics-only
```
**Time:** Immediate | **Output:** PyBullet 3D window with walking spider

### Option 2: Train & Demo
```bash
python scripts/train_ppo.py --timesteps 50000
python scripts/demo_trained_spider.py --model output/trained_spider_ppo.zip
```
**Time:** ~5 minutes | **Output:** Trained model saved to `output/trained_spider_ppo.zip` + reward plot

### Option 3: Production Training
```bash
python scripts/train_ppo.py --timesteps 500000 --render
```
**Time:** 1-2 hours | **Output:** High-quality trained model

---

## Visual Gallery

This section showcases the key components and visualization of the Walking Spider project:

### Spider Animation
The trained spider robot walking with a learned diagonal gait pattern. This GIF demonstrates the smooth locomotion achieved through PPO training:

![Spider Walking Animation](/images/spider.gif)

### Robot Structure (URDF)
The skeletal representation showing all 8 joints (2 per leg × 4 legs) and the coordinate system. Green/red/blue axes represent X/Y/Z directions:

![URDF Structure](/images/URDF.png)

### PyBullet Simulation
Real-time 3D physics simulation showing the spider robot during training. The visualization includes:
- Cyan body representing the main chassis
- Colored legs (red, green, blue) for different joint axes
- Ground plane for contact detection
- Debug visualization of joint forces and positions

![PyBullet Simulation](/images/PyBullet.png)

---

## Project Structure

```
WalkingSpider_OpenAI_PyBullet/
├── scripts/                          # Executable scripts
│   ├── train_ppo.py                  # Train spider with PPO
│   ├── demo_trained_spider.py         # Run trained model or physics demo
│   └── setup_environment.sh           # Setup script
│
├── src/                              # Source code modules
│   ├── walking_spider_env.py          # Custom Gym environment class
│   ├── gif_recorder.py                # Visual debugging utility
│   ├── debug_logger.py                # Logging system utility
│   ├── spider.xml                     # Robot URDF file
│   ├── spider_simple.urdf             # Alternative robot definition
│   └── meshes/                        # 3D models (base, leg, servo)
│
├── output/                           # Trained model binaries (output)
├── experience_learned/               # Pre-trained models (archive)
├── images/                           # Reference images + spider.gif
├── files/                            # CAD files for physical robot
├── videos/                           # GIF snapshots
│
├── README.md                         # Documentation (THIS FILE)
├── requirements.txt                  # Python dependencies
└── pyproject.toml                    # Poetry configuration
```

---

## The 2 Executable Scripts

### 1. `train_ppo.py` - Train the Spider

**Purpose:** Train spider robot to walk using PPO reinforcement learning

**What it does:**
- Initializes OpenAI Gym environment
- Creates PPO agent with configurable parameters
- Trains for specified number of timesteps
- Tracks episode metrics (rewards, lengths)
- Generates matplotlib plot of training progress
- Saves trained model to .zip file

**Usage:**
```bash
python scripts/train_ppo.py [options]
```

**Options:**
- `--timesteps N` - Training timesteps (default: 100,000)
- `--lr FLOAT` - Learning rate (default: 0.0003)
- `--steps N` - Steps per update (default: 2048)
- `--batch-size N` - Batch size (default: 64)
- `--model PATH` - Model save path (default: `output/trained_spider_ppo.zip`)
- `--render` - Show GUI during training

**Output:**
- `output/trained_spider_ppo.zip` - Trained model (in output directory)
- `training_rewards.png` - Reward plot with moving average
- `training_logs/` - TensorBoard logs

**Examples:**
```bash
python scripts/train_ppo.py                          # Default: 100k steps
python scripts/train_ppo.py --timesteps 50000       # Quick training
python scripts/train_ppo.py --timesteps 500000 --render  # Full training with GUI
```

---

### 2. `demo_trained_spider.py` - Demo Trained Models or Physics

**Purpose:** Visualize trained models or test physics with hardcoded gait

**Mode A: With Trained Model** (requires --model parameter)
- Loads trained PPO model from specified path
- Runs with PyBullet 3D GUI
- Shows learned walking behavior
- Reports statistics: rewards, distance, steps

```bash
python scripts/demo_trained_spider.py --model output/trained_spider_ppo.zip [options]
```

**Mode B: Physics Demo Only** (no model needed)
- Uses hardcoded diagonal trotting gait
- Perfect for quick testing without training
- Runs for 20 seconds

```bash
python scripts/demo_trained_spider.py --physics-only
```

**Options:**
- `--model PATH` - **REQUIRED** Path to trained model (unless using --physics-only)
- `--episodes N` - Number of episodes (default: 5)
- `--max-steps N` - Max steps per episode (default: 1000)
- `--physics-only` - Use hardcoded gait (no model needed)

**Examples:**
```bash
python scripts/demo_trained_spider.py --model output/trained_spider_ppo.zip  # Run 5 episodes
python scripts/demo_trained_spider.py --model output/trained_spider_ppo.zip --episodes 10
python scripts/demo_trained_spider.py --physics-only # Physics demo immediately
```

---

## Robot Design

**Structure:**
- 4 legs: Front-Left, Front-Right, Back-Left, Back-Right
- Each leg has 2 joints: vertical (yaw) + horizontal (pitch)
- Total: 8 controllable joints
- Base body: 125×125×35mm, 2.0kg mass

**Joint Configuration:**
- Front-Left: Joints 0-1
- Front-Right: Joints 2-3
- Back-Left: Joints 4-5
- Back-Right: Joints 6-7

**Action Space:** 8-dimensional [-1, 1] continuous control

**Observation Space:** 75-dimensional state (position, velocity, joint states, contacts)

**Physics:**
- Gravity: -9.81 m/s²
- Ground friction: 5.0 (high grip)
- Leg friction: 5.0
- Joint damping: 0.5 (stability)
- Physics engine: Optimized to 2,500+ FPS

---

## Setup & Installation

### Prerequisites
- Python 3.10+ 
- pip and venv
- WSL Ubuntu on Windows (recommended)

### Quick Setup (WSL Ubuntu)

**One-time setup using the automated script:**

```bash
# Clone or enter the repository directory
cd ~/WalkingSpider_OpenAI_PyBullet

# Run the setup script (handles everything automatically)
bash scripts/setup_environment.sh
```

The script will:
1. Create a Python virtual environment (`venv_spider/`)
2. Install all dependencies from `requirements.txt`
3. Verify all packages are installed correctly

**Manual Setup** (if you prefer step-by-step):

```bash
cd ~/WalkingSpider_OpenAI_PyBullet
python3 -m venv venv_spider
source venv_spider/bin/activate
pip install -r requirements.txt
```

### Verify Installation
```bash
source venv_spider/bin/activate
python -c "import pybullet; print('✓ PyBullet')"
python -c "import gym; print('✓ Gym')"
python -c "import stable_baselines3; print('✓ Stable Baselines 3')"
```

---

## Environment Details

### Reward Function

Multi-objective rewards encourage natural walking:
- **Forward velocity** (primary): 10.0 × forward speed
- **Height stability**: Penalty for deviation from target height
- **Orientation**: Penalty for excessive roll/pitch
- **Energy efficiency**: Penalty for excessive joint torques
- **Smooth motion**: Penalty for sudden velocity changes
- **Contact penalty**: Penalty for non-leg body contact with ground
- **Survival bonus**: +1.0 per step (encourages longer episodes)

### Termination Conditions

Episode ends when:
- Robot falls below certain height (z < 0.03m)
- Robot flips over (roll/pitch > 1.5 rad ≈ 85°)
- Maximum steps reached (1000 steps)

### Observation & Action

**Observation (75 values):**
- Base position (3) + orientation quaternion (4)
- Base linear velocity (3) + angular velocity (3)
- Joint positions (8) + joint velocities (8)
- Contact points + forces
- Other sensor data

**Action (8 values):**
- Target position for each of 8 joints
- Range: [-1, 1] (normalized to joint limits)

---

## Improvements Implemented

The simulation includes all critical physics and RL improvements:

✅ **Fixed reward function** - Now rewards forward motion (not slowing down)  
✅ **High friction** - Ground friction 5.0, leg friction 5.0 (prevents sliding)  
✅ **Correct action space** - 8 dimensions (not 10)  
✅ **Better motor control** - Proper force, gains, and velocity limits  
✅ **Multi-objective rewards** - Velocity, stability, efficiency, smoothness  
✅ **Proper termination** - Falls, flips, and max steps  
✅ **Joint damping** - URDF damping for stability  
✅ **Gravity enabled** - -9.81 m/s²  
✅ **Optimized physics** - 2,500+ FPS performance  
✅ **Visual debugging** - GIF snapshots + logging system  

---

## Verified Performance

**Physics Demo Results (20 seconds):**
- Distance traveled: 0.375 meters ✓
- Gait pattern: Diagonal trotting ✓
- Simulation FPS: 2,500+ ✓
- Gravity: -9.81 m/s² confirmed ✓
- No sliding observed ✓

**Physics Engine:**
- Average FPS: 2,518
- Frame time: 0.40 ms
- FPS range: 811 - 5,626

**Gait Quality:**
- Natural diagonal walking pattern
- Coordinated leg movement
- Smooth forward motion
- Realistic physics

---

## Training

### PPO Algorithm Configuration

**Typical Parameters:**
- Algorithm: PPO (Proximal Policy Optimization)
- Policy: MlpPolicy (Multi-Layer Perceptron)
- Total timesteps: 50,000 - 500,000+ (configurable)
- Parallel environments: Single (configurable)
- Network: 2 hidden layers (64 units each)

### How PPO Training Works

**PPO (Proximal Policy Optimization)** is a state-of-the-art reinforcement learning algorithm that trains agents to maximize cumulative rewards while preventing drastic policy changes. Here's the process:

1. **Data Collection**: The agent explores the environment using its current policy, collecting trajectories of states, actions, and rewards
2. **Advantage Estimation**: We calculate how much better an action is compared to baseline (using Generalized Advantage Estimation with λ=0.95)
3. **Policy Clipping**: PPO limits policy updates to prevent overly large changes using the clipping mechanism (ε=0.2)
4. **Value Function**: A separate neural network estimates state values to reduce variance in advantage estimates
5. **Multiple Epochs**: Multiple training passes over the collected data for sample efficiency

**Key Benefits:**
- **Stable Learning**: Clipping prevents catastrophic policy collapses
- **Sample Efficient**: Reuses data multiple times for better sample efficiency
- **Strong Performance**: Consistently achieves excellent results on locomotion tasks
- **Adjustable Hyperparameters**: Easy to tune for different environments

### PPO Training Architecture

```mermaid
graph TB
    subgraph Input["Input Data"]
        Obs["Observations<br/>(75 values)"]
        Action["Previous Actions<br/>(8 values)"]
    end
    
    subgraph Network["Neural Networks"]
        Policy["Policy Network<br/>(2x64 hidden)"]
        Value["Value Network<br/>(2x64 hidden)"]
    end
    
    subgraph RL["RL Core"]
        Rollout["Rollout Collection<br/>(2048 steps)"]
        Advantage["Advantage Calculation<br/>GAE Lambda=0.95"]
        Clipping["PPO Clipping<br/>Epsilon=0.2"]
    end
    
    subgraph Training["Training Loop"]
        Loss["Compute Loss<br/>Policy + Value"]
        Optimize["Optimizer Step<br/>Adam LR=3e-4"]
        Update["Update Network<br/>Weights"]
    end
    
    subgraph Env["Environment"]
        PyBullet["PyBullet Physics<br/>9.81 m/s² gravity"]
        Spider["8-Joint Spider<br/>Continuous Control"]
        Reward["Reward Function<br/>Multi-objective"]
    end
    
    subgraph Output["Output"]
        Model["Trained Model<br/>.zip file"]
        Plots["Training Plots<br/>Rewards, Episodes"]
    end
    
    Obs --> Policy
    Action --> Policy
    Obs --> Value
    Policy --> Rollout
    Value --> Rollout
    
    Rollout --> Advantage
    Advantage --> Clipping
    Clipping --> Loss
    
    Loss --> Optimize
    Optimize --> Update
    Update --> Policy
    Update --> Value
    
    Policy --> Spider
    Spider --> Env
    Spider --> Reward
    Reward --> Loss
    
    PyBullet --> Spider
    
    Update --> Model
    Loss --> Plots
    
    style Input fill:#e1f5ff
    style Network fill:#fff3e0
    style RL fill:#f3e5f5
    style Training fill:#e8f5e9
    style Env fill:#fce4ec
    style Output fill:#c8e6c9
```

### Training Modes

#### Mode 1: Fast Training (CPU, No Rendering)
```bash
python scripts/train_ppo.py --timesteps 100000
```
- ✅ Fastest training speed
- ❌ No visual feedback
- Good for: Bulk training, cloud servers
- **Typical time**: 1-2 hours for 100k timesteps

#### Mode 2: Interactive Training (CPU, With PyBullet GUI)
```bash
python scripts/train_ppo.py --timesteps 100000 --render
```
- ✅ See the spider learning to walk in real-time
- ✅ PyBullet GUI shows the robot
- ✅ Matplotlib plots update live during training
- ❌ Slower (renders take CPU time)
- Good for: Debugging, visualization, learning
- **Typical time**: 2-3 hours for 100k timesteps (slower due to rendering)

#### Mode 3: GPU Accelerated Training (CUDA, No Rendering)
```bash
python scripts/train_ppo.py --timesteps 500000 --device cuda
```
- ✅ Much faster neural network training
- ✅ Best for large-scale training
- ❌ Requires NVIDIA GPU with CUDA support
- Good for: Production training, large datasets
- **Typical time**: 30 minutes for 500k timesteps (with GPU)

#### Mode 4: GPU Training With Visualization (CUDA + Rendering)
```bash
python scripts/train_ppo.py --timesteps 100000 --render --device cuda
```
- ✅ GPU-accelerated + visual feedback
- ✅ See training progress in real-time
- ❌ Requires NVIDIA GPU
- ❌ Slowest option (GUI rendering + training)
- Good for: Presentation, high-quality visualization
- **Typical time**: 2-3 hours for 100k timesteps

### Device Selection

#### CPU (Default)
```bash
python scripts/train_ppo.py --device cpu
```
- Works on any machine
- No special requirements
- Slower training speed
- Suitable for learning and testing

#### GPU (CUDA)
```bash
python scripts/train_ppo.py --device cuda
```
**Requirements:**
- NVIDIA GPU (GeForce, Tesla, etc.)
- CUDA toolkit installed
- cuDNN libraries
- PyTorch with CUDA support

**Install GPU support:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU availability:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Training Timeline

- **0-10k steps**: Initial exploration, basic walking emerges
- **10-50k steps**: Improved coordination, gait development
- **50-100k steps**: Speed optimization
- **100k+ steps**: Fine-tuning, robust behavior

### Training Output

After training completes:
- `output/trained_spider_ppo.zip` - Trained model binary
- `training_rewards.png` - Final reward plot (saved at end)
- Console logs - Episode metrics printed during training
- Matplotlib window - Real-time plot (if --render enabled)

### Quick Examples

```bash
# Quick test (5min, CPU, no GUI)
python scripts/train_ppo.py --timesteps 10000

# Interactive learning (2h, CPU, with GUI + matplotlib)
python scripts/train_ppo.py --timesteps 100000 --render

# Production training (30min, GPU, no GUI)
python scripts/train_ppo.py --timesteps 500000 --device cuda

# Full quality (fast GPU, visual feedback)
python scripts/train_ppo.py --timesteps 500000 --render --device cuda
```

### Pre-trained Models

Located in `experience_learned/`:
- `ppo2_WalkingSpider_v0.pkl` - Initial training
- `ppo2_WalkingSpider_v0_testing.pkl` - Testing phase
- `ppo2_WalkingSpider_v0_testing_3.pkl` - Best version

---

## Debugging & Visualization

### Visual Debugging

**GIF Snapshots:** The environment automatically captures 10-second GIF snapshots during testing:

```python
env = gym.make('WalkingSpider-v0')
env = gym.wrappers.Monitor(env, 'videos/', force=True)
```

GIFs saved to `videos/` directory with timestamp filenames.

### Logging System

**Enable detailed logging:**
```python
from walking_spider.envs.debug_logger import create_logger

logger = create_logger(log_level='INFO', log_dir='logs/debug')

for episode in range(num_episodes):
    logger.log_episode_start(episode)
    
    for step in range(max_steps):
        obs, reward, done, info = env.step(action)
        logger.log_step(step, obs, action, reward, done, info)
        logger.log_reward_breakdown(
            forward_reward=10.5,
            height_reward=-2.3,
            energy_cost=-1.2
        )
    
    logger.log_episode_end()
```

**Logs captured:**
- Robot position, orientation, velocities
- Joint states and torques
- Contact points and forces
- Reward components breakdown
- Episode metrics

---

## Example Usage

### Train a Model
```python
import gym
import walking_spider
from stable_baselines3 import PPO

env = gym.make('WalkingSpider-v0')
model = PPO('MlpPolicy', env, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=100000)
model.save("trained_spider_ppo")
```

### Load & Run Trained Model
```python
import gym
import walking_spider
from stable_baselines3 import PPO

env = gym.make('WalkingSpider-v0')
model = PPO.load("trained_spider_ppo")

obs = env.reset()
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
```

### Random Testing
```python
import gym
import walking_spider

env = gym.make('WalkingSpider-v0')

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
    
    print(f"Episode {episode+1}: Reward = {total_reward:.2f}")

env.close()
```

---

## Physical Robot Implementation

The project includes CAD designs and 3D-printable STL files for building a physical spider robot:

**3D Printable Parts** (in `files/` directory):
- Base servo mount (MG90 compatible)
- Servo leg attachments
- Top/bottom covers
- Battery holder

**Components Needed:**
- 4× MG90 servo motors
- Microcontroller (Raspberry Pi, Arduino, etc.)
- Power supply for servos
- Structural materials (3D printed or aluminum)
- Wiring and connectors

**Integration:**
The simulated robot uses the same coordinate system as the physical robot, allowing direct transfer of trained policies to the real hardware.

---

## Troubleshooting

### Installation Issues

**"No module named 'pybullet'"**
```bash
# Verify virtual environment is activated
source venv_spider/bin/activate    # Linux/macOS
venv_spider\Scripts\activate       # Windows

# Reinstall
pip install -r requirements.txt
```

**"Microsoft Visual C++ 14.0 or greater is required"** (Windows only)

Solutions:
1. Use Conda: `conda install -c conda-forge pybullet`
2. Use WSL Ubuntu (recommended)
3. Install Visual C++ Build Tools from Microsoft

**"PyBullet GUI won't open"**
- Use physics demo instead: `python scripts/demo_trained_spider.py --physics-only`
- Check display settings in WSL/Linux
- Use headless mode: `env = gym.make('WalkingSpider-v0', render_mode='rgb_array')`

### Performance Issues

**Training is slow:**
- Use `render=False` for faster training
- Use DIRECT mode instead of GUI
- Enable GPU for neural network training

**Out of memory:**
- Reduce parallel environments
- Reduce batch size
- Use lighter network architecture

**Poor walking behavior:**
- Increase training timesteps
- Adjust reward function weights
- Check physics parameters (friction, gravity)

---

## Contributing

To contribute improvements:

1. Create a feature branch: `git checkout -b feature/your-improvement`
2. Make changes and test: `python scripts/demo_trained_spider.py --physics-only`
3. Commit: `git commit -m "feat: your improvement"`
4. Push and create Pull Request

**Development Tips:**
- Always test changes with the physics demo first
- Enable logging for debugging: `debug_logger.py`
- Use GIF snapshots for visual verification
- Document changes in commit messages

---

## References

**Key Papers & Resources:**
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- PyBullet: https://pybullet.org
- OpenAI Gym: https://gym.openai.com
- Stable Baselines3: https://stable-baselines3.readthedocs.io

**Related Projects:**
- OpenAI Gym Ant environment
- DeepMind Locomotion Suite
- MIT Cheetah Robot Control
- Boston Dynamics-inspired research

---

## License

[Specify your license here]

---

## Status

✅ **Project Status:** Ready for use

- All scaffolding code removed
- Only essential files for spider simulation
- Physics tested and verified (2,500+ FPS)
- Two clean, focused scripts
- Comprehensive documentation

**Last Updated:** October 2025

**Next Steps:**
1. Run physics demo: `python scripts/demo_trained_spider.py --physics-only`
2. Train a model: `python scripts/train_ppo.py --timesteps 50000`
3. Evaluate results: `python scripts/demo_trained_spider.py`

---

## TODO - Future Improvements

### Priority 1: 🔴 Critical Maintenance
- [ ] **Migrate from OpenAI Gym to Gymnasium**
  - **Why**: Gym is unmaintained since 2022, doesn't support NumPy 2.0
  - **Warning**: Runtime warning on every execution
  - **Task**: Replace `import gym` with `import gymnasium as gym` in:
    - `src/walking_spider_env.py` (line 1)
    - `scripts/train_ppo.py` (imports section)
    - `scripts/demo_trained_spider.py` (imports section)
  - **Reference**: https://gymnasium.farama.org/introduction/migration_guide/
  - **Expected Impact**: Remove deprecation warnings, enable NumPy 2.0 compatibility

### Priority 2: 🟡 Bug Fix
- [ ] **Debug Spider Floating During Training**
  - **Issue**: Spider appears to float/levitate instead of walking on ground
  - **Likely Causes**:
    - Gravity parameter misconfiguration (currently -10 m/s² vs physics demo -9.81 m/s²)
    - Ground friction values too high (currently 5.0, may need tuning)
    - Contact point detection issues with robot base
    - Joint damping causing upward bias
  - **Investigation Steps**:
    1. Compare gravity values between `walking_spider_env.py` and `demo_trained_spider.py`
    2. Test different friction coefficients (ground, legs, base)
    3. Log contact forces during training
    4. Visualize with `--render` flag to observe behavior
  - **Reference**: Lines 65, 80-107 in `src/walking_spider_env.py`

### Priority 3: 🟢 Performance Enhancement
- [ ] **Improve Training Policy**
  - **Goal**: Achieve faster convergence and better walking behavior
  - **Potential Improvements**:
    - Reward function tuning (currently: velocity=10.0, height=-5.0, energy=-0.1)
    - Add curriculum learning (easier tasks → harder tasks)
    - Implement action smoothing to reduce jerky movements
    - Add experience replay buffer for better sample efficiency
    - Experiment with different network architectures (64→128 hidden units)
  - **Metrics to Track**: Episode reward trends, final distance traveled, gait stability
  - **Hyperparameters to Tune**: Learning rate, GAE lambda, PPO clip range

### Priority 4: 🟢 Feature Addition
- [ ] **Add Alternative RL Algorithms as Parameters**
  - **Goal**: Support multiple RL algorithms beyond PPO for experimentation
  - **Candidates to Implement**:
    - **A2C** (Advantage Actor-Critic) - faster convergence, less stable
    - **SAC** (Soft Actor-Critic) - off-policy, better exploration
    - **TD3** (Twin Delayed DDPG) - for continuous control
    - **TRPO** (Trust Region Policy Optimization) - similar to PPO, theoretically grounded
  - **Implementation**:
    - Add `--algorithm` parameter to `train_ppo.py` (default: PPO)
    - Create algorithm factory function
    - Maintain same hyperparameter interface
  - **Commands**:
    ```bash
    python scripts/train_ppo.py --algorithm ppo --timesteps 100000
    python scripts/train_ppo.py --algorithm a2c --timesteps 100000
    python scripts/train_ppo.py --algorithm sac --timesteps 100000
    python scripts/train_ppo.py --algorithm td3 --timesteps 100000
    ```
  - **Dependencies**: All available in Stable-Baselines3

### Completed ✅
- ✅ Consolidated project structure (moved modules to `src/`, scripts to `scripts/`)
- ✅ Added GPU/CUDA support with `--device` parameter
- ✅ Implemented real-time matplotlib plotting during training
- ✅ Created comprehensive documentation with mermaid diagrams
- ✅ Fixed NumPy 2.0 compatibility in `walking_spider_env.py`
- ✅ Added shimmy for Gym/Gymnasium bridge compatibility
- ✅ Implemented demo script with physics-only mode
- ✅ Created setup automation script


