#!/usr/bin/env python3
"""
Demo Script for Trained Walking Spider
Loads a trained PPO model and runs it with PyBullet GUI visualization
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import time

# Add src/ to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from local modules
from walking_spider_env import WalkingSpiderEnv

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable_baselines3 not installed. Running physics demo instead.")


def demo_trained_model(model_path=None, episodes=5, max_steps=1000):
    """
    Run a trained model in the PyBullet GUI
    
    Args:
        model_path: Path to trained model (required if not using --physics-only)
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    
    if not HAS_SB3:
        print("stable_baselines3 not available. Cannot load trained model.")
        demo_physics_only()
        return
    
    if model_path is None:
        print("❌ Error: --model parameter is required to run a trained model")
        print()
        print("Usage examples:")
        print("  python scripts/demo_trained_spider.py --model output/trained_spider_ppo.zip")
        print("  python scripts/demo_trained_spider.py --physics-only")
        print()
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Run train_ppo.py to train a model first:")
        print("  python scripts/train_ppo.py --timesteps 50000")
        demo_physics_only()
        return
    
    print("=" * 80)
    print("  Demo: Trained Walking Spider")
    print("=" * 80)
    print()
    print(f"Loading model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = WalkingSpiderEnv(render=True, enable_gif_recording=True)
    
    print(f"✓ Model loaded successfully")
    print(f"✓ PyBullet GUI environment created")
    print()
    
    total_reward = 0
    total_distance = 0
    
    for ep in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        start_pos, _ = p.getBasePositionAndOrientation(env.robotId)
        
        print(f"Episode {ep+1}/{episodes}:")
        
        for step in range(max_steps):
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        end_pos, _ = p.getBasePositionAndOrientation(env.robotId)
        distance = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
        
        print(f"  Steps: {episode_steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Distance Traveled: {distance:.3f}m")
        print()
        
        total_reward += episode_reward
        total_distance += distance
    
    avg_reward = total_reward / episodes
    avg_distance = total_distance / episodes
    
    print("=" * 80)
    print("  Results Summary")
    print("=" * 80)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Distance: {avg_distance:.3f}m")
    print()
    
    env.close()


def demo_physics_only():
    """
    Run a physics demonstration without a trained model
    Shows the spider walking with a hardcoded diagonal gait
    """
    
    print("=" * 80)
    print("  Physics Demo: Walking Spider (Hardcoded Gait)")
    print("=" * 80)
    print()
    
    try:
        client = p.connect(p.GUI)
        print("✓ PyBullet GUI connected!")
    except:
        client = p.connect(p.DIRECT)
        print("⚠️  GUI not available, running headless")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Optimize physics for speed
    p.setPhysicsEngineParameter(
        numSubSteps=1,
        fixedTimeStep=0.001,
        numSolverIterations=4
    )
    
    # Load ground
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    p.changeDynamics(plane_id, -1, lateralFriction=5.0, spinningFriction=5.0,
                     rollingFriction=0.0, restitution=0.0)
    
    # Load spider
    spider_pos = [0, 0, 0.15]
    spider_orn = p.getQuaternionFromEuler([0, 0, 0])
    spider_id = p.loadURDF("src/spider_simple.urdf", spider_pos, spider_orn)
    
    p.changeDynamics(spider_id, -1, mass=2.0, linearDamping=0.01, angularDamping=0.01)
    for i in range(p.getNumJoints(spider_id)):
        p.changeDynamics(spider_id, i, lateralFriction=5.0, spinningFriction=5.0)
    
    # Camera
    p.resetDebugVisualizerCamera(cameraDistance=0.35, cameraYaw=45,
                                  cameraPitch=-30, cameraTargetPosition=[0, 0, 0.1])
    
    print("Running hardcoded diagonal walking gait...")
    print()
    
    start_time = time.time()
    start_pos, _ = p.getBasePositionAndOrientation(spider_id)
    
    while time.time() - start_time < 20:
        t = time.time() - start_time
        
        phase1 = np.sin(t * 2)
        phase2 = np.sin(t * 2 + np.pi)
        
        # Diagonal gait: left-front & right-back move together, then right-front & left-back
        action = np.array([
            2.0 * phase1,                              # left_front
            1.5 * np.cos(t * 2),                       # left_front_leg
            2.0 * phase2,                              # right_front
            1.5 * np.cos(t * 2 + np.pi),              # right_front_leg
            2.0 * phase2,                              # left_back
            1.5 * np.cos(t * 2 + np.pi),              # left_back_leg
            2.0 * phase1,                              # right_back
            1.5 * np.cos(t * 2),                       # right_back_leg
        ])
        
        for joint_id in range(8):
            try:
                p.setJointMotorControl2(
                    spider_id, joint_id, p.POSITION_CONTROL,
                    targetPosition=action[joint_id],
                    force=100,
                    positionGain=1.2,
                    velocityGain=0.8,
                    maxVelocity=20.0
                )
            except:
                pass
        
        p.stepSimulation()
        
        if int(t) % 5 == 0 and t > int(t) - 0.01:  # Print every 5 seconds
            pos, _ = p.getBasePositionAndOrientation(spider_id)
            dist = np.sqrt((pos[0]-start_pos[0])**2 + (pos[1]-start_pos[1])**2)
            print(f"  {t:5.1f}s: Position=({pos[0]:+.3f}, {pos[1]:+.3f}), Distance={dist:.3f}m")
        
        time.sleep(0.001)
    
    end_pos, _ = p.getBasePositionAndOrientation(spider_id)
    distance = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
    
    print()
    print("=" * 80)
    print("  Results")
    print("=" * 80)
    print(f"Final Position: ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    print(f"Distance Traveled: {distance:.3f}m")
    print()
    
    p.disconnect()
    print("✓ Demo complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Walking Spider')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (required unless using --physics-only)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--physics-only', action='store_true', help='Run physics demo without model')
    
    args = parser.parse_args()
    
    if args.physics_only:
        demo_physics_only()
    else:
        demo_trained_model(
            model_path=args.model,
            episodes=args.episodes,
            max_steps=args.max_steps
        )

