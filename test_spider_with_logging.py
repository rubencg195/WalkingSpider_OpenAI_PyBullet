"""
Test Walking Spider Environment with Comprehensive Debug Logging

This script demonstrates how to use the debug logger to track robot behavior,
physics parameters, and training progress for iterative debugging.

Usage:
    # Run with full debug logging
    python test_spider_with_logging.py --log-level DEBUG
    
    # Run with minimal logging
    python test_spider_with_logging.py --log-level INFO
    
    # Run without logging
    python test_spider_with_logging.py --no-log
"""

import sys
import argparse
import gym
import numpy as np

# Add environment to path
sys.path.insert(0, 'environment/walking-spider')
import walking_spider

# Import debug logger
from walking_spider.envs.debug_logger import create_logger


def test_random_actions(num_episodes=5, max_steps=500, logger=None):
    """Test environment with random actions and logging."""
    
    env = gym.make('WalkingSpider-v0')
    
    if logger:
        logger.log("="*80, 'INFO')
        logger.log("RANDOM ACTION TEST", 'INFO')
        logger.log(f"Episodes: {num_episodes}, Max Steps: {max_steps}", 'INFO')
        logger.log("="*80, 'INFO')
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        if logger:
            logger.log_episode_start(episode + 1)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Log step information
            if logger:
                logger.log_step(step, obs, action, reward, done, info)
                
                # Log additional physics information every 50 steps
                if step % 50 == 0:
                    physics_info = {
                        'contacts': [],  # Would extract from PyBullet
                        'joint_states': []  # Would extract from observation
                    }
                    logger.log_physics_state(physics_info)
            
            obs = next_obs
            
            if done:
                break
        
        if logger:
            logger.log_episode_end()
        
        print(f"Episode {episode + 1}: Steps={step+1}, Total Reward={episode_reward:.2f}")
    
    env.close()


def test_trained_model(model_path, num_episodes=3, logger=None):
    """Test a trained model with logging."""
    
    try:
        from stable_baselines import PPO2
        from stable_baselines.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable_baselines not installed. Skipping trained model test.")
        return
    
    if logger:
        logger.log("="*80, 'INFO')
        logger.log("TRAINED MODEL TEST", 'INFO')
        logger.log(f"Model: {model_path}", 'INFO')
        logger.log(f"Episodes: {num_episodes}", 'INFO')
        logger.log("="*80, 'INFO')
    
    # Load model
    try:
        model = PPO2.load(model_path)
        if logger:
            logger.log(f"✅ Model loaded successfully", 'INFO')
    except Exception as e:
        if logger:
            logger.log_error(f"Failed to load model: {model_path}", e)
        print(f"❌ Could not load model: {e}")
        return
    
    # Create environment
    env = DummyVecEnv([lambda: gym.make('WalkingSpider-v0')])
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        if logger:
            logger.log_episode_start(episode + 1)
        
        episode_reward = 0
        step = 0
        
        while step < 1000:
            # Predict action using trained model
            action, _states = model.predict(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Log step
            if logger:
                logger.log_step(step, obs[0], action[0], reward[0], done[0], info)
            
            obs = next_obs
            step += 1
            
            if done[0]:
                break
        
        if logger:
            logger.log_episode_end()
        
        print(f"Episode {episode + 1}: Steps={step}, Total Reward={episode_reward:.2f}")
    
    env.close()


def analyze_logs(log_dir='logs/debug'):
    """Quick analysis of the most recent log file."""
    import os
    import json
    
    # Find most recent JSON log
    json_files = [f for f in os.listdir(log_dir) if f.startswith('spider_data_') and f.endswith('.json')]
    
    if not json_files:
        print(f"❌ No log files found in {log_dir}")
        return
    
    # Get most recent
    latest_file = sorted(json_files)[-1]
    filepath = os.path.join(log_dir, latest_file)
    
    print(f"\n📊 Analyzing: {filepath}")
    print("="*80)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    episodes = data['episodes']
    stats = data['statistics']
    
    print(f"Total Episodes: {metadata['total_episodes']}")
    print(f"Total Steps: {metadata['total_steps']}")
    
    if episodes:
        rewards = [ep['total_reward'] for ep in episodes]
        steps = [len(ep['steps']) for ep in episodes]
        
        print(f"\nReward Stats:")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Min:  {np.min(rewards):.4f}")
        print(f"  Max:  {np.max(rewards):.4f}")
        
        print(f"\nEpisode Length Stats:")
        print(f"  Mean: {np.mean(steps):.2f}")
        print(f"  Min:  {np.min(steps)}")
        print(f"  Max:  {np.max(steps)}")
        
    if stats['velocities']:
        print(f"\nVelocity Stats:")
        print(f"  Mean: {np.mean(stats['velocities']):.4f} m/s")
        print(f"  Max:  {np.max(stats['velocities']):.4f} m/s")
        
    print("="*80)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Test Walking Spider with Debug Logging')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--no-log', action='store_true',
                       help='Disable logging')
    parser.add_argument('--log-dir', type=str, default='logs/debug',
                       help='Directory for log files')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'trained', 'both'],
                       help='Test mode')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--model', type=str, 
                       default='experience_learned/ppo2_WalkingSpider_v0_testing_3.pkl',
                       help='Path to trained model')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing logs instead of running')
    
    args = parser.parse_args()
    
    # Analyze logs if requested
    if args.analyze:
        analyze_logs(args.log_dir)
        return
    
    # Create logger
    logger = None
    if not args.no_log:
        logger = create_logger(
            enabled=True,
            log_level=args.log_level,
            log_dir=args.log_dir
        )
        print(f"✅ Logging enabled: Level={args.log_level}, Dir={args.log_dir}")
    else:
        print("⚠️  Logging disabled")
    
    try:
        # Run tests
        if args.mode in ['random', 'both']:
            print(f"\n🎲 Running random action test...")
            test_random_actions(
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                logger=logger
            )
        
        if args.mode in ['trained', 'both']:
            print(f"\n🤖 Running trained model test...")
            test_trained_model(
                model_path=args.model,
                num_episodes=args.episodes,
                logger=logger
            )
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if logger:
            logger.log_error("Test failed", e)
    
    finally:
        # Close logger
        if logger:
            logger.close()
            print("\n✅ Logging complete")


if __name__ == '__main__':
    main()

