"""
Test Walking Spider with GIF Recording

This script tests the improved walking spider environment with automatic
GIF snapshot recording for visual debugging.

Usage:
    # Test with GIF recording enabled (saves 10-second clips)
    python test_with_gif_recording.py
    
    # Test with trained model
    python test_with_gif_recording.py --model experience_learned/ppo2_WalkingSpider_v0_testing_3.pkl
    
    # Customize GIF settings
    python test_with_gif_recording.py --duration 15 --episodes 3
"""

import sys
import argparse
import gym
import numpy as np

# Add environment to path
sys.path.insert(0, 'environment/walking-spider')
import walking_spider


def test_random_actions(num_episodes=3, max_steps=1000, gif_duration=10):
    """Test environment with random actions and GIF recording."""
    
    print(f"\n{'='*80}")
    print(f"Testing Walking Spider with GIF Recording")
    print(f"{'='*80}")
    print(f"Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"GIF Duration: {gif_duration} seconds")
    print(f"{'='*80}\n")
    
    # Create environment with GIF recording enabled
    env = gym.make('WalkingSpider-v0')
    
    # Manually enable GIF recording (since gym.make doesn't pass kwargs)
    # You may need to modify this based on your environment setup
    try:
        from walking_spider.envs.gif_recorder import GifRecorder
        env.unwrapped.gif_recorder = GifRecorder(
            save_dir='videos', 
            duration_seconds=gif_duration, 
            fps=30,
            enabled=True
        )
        env.unwrapped.gif_recorder.start_recording()
        print("✅ GIF recording enabled\n")
    except Exception as e:
        print(f"⚠️  Could not enable GIF recording: {e}\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                if hasattr(env.unwrapped, 'gif_recorder') and env.unwrapped.gif_recorder:
                    progress = env.unwrapped.gif_recorder.get_progress()
                    print(f"  Step {step + 1:4d} | Reward: {episode_reward:7.2f} | "
                          f"GIF: {progress*100:5.1f}% recorded")
                else:
                    print(f"  Step {step + 1:4d} | Reward: {episode_reward:7.2f}")
            
            obs = next_obs
            
            if done:
                print(f"  Episode ended at step {step + 1}")
                break
        
        print(f"  Total Reward: {episode_reward:.2f}\n")
    
    env.close()
    print("\n✅ Testing complete!")


def test_trained_model(model_path, num_episodes=3, gif_duration=10):
    """Test a trained model with GIF recording."""
    
    try:
        from stable_baselines import PPO2
        from stable_baselines.common.vec_env import DummyVecEnv
    except ImportError:
        print("❌ stable_baselines not installed. Cannot test trained model.")
        print("   Install with: pip install stable-baselines[mpi]")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing Trained Model with GIF Recording")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"GIF Duration: {gif_duration} seconds")
    print(f"{'='*80}\n")
    
    # Load model
    try:
        model = PPO2.load(model_path)
        print(f"✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Create environment
    env = DummyVecEnv([lambda: gym.make('WalkingSpider-v0')])
    
    # Enable GIF recording
    try:
        from walking_spider.envs.gif_recorder import GifRecorder
        env.envs[0].unwrapped.gif_recorder = GifRecorder(
            save_dir='videos', 
            duration_seconds=gif_duration, 
            fps=30,
            enabled=True
        )
        env.envs[0].unwrapped.gif_recorder.start_recording()
        print("✅ GIF recording enabled\n")
    except Exception as e:
        print(f"⚠️  Could not enable GIF recording: {e}\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        episode_reward = 0
        step = 0
        
        while step < 1000:
            # Predict action using trained model
            action, _states = model.predict(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Print progress
            if (step + 1) % 100 == 0:
                if hasattr(env.envs[0].unwrapped, 'gif_recorder') and env.envs[0].unwrapped.gif_recorder:
                    progress = env.envs[0].unwrapped.gif_recorder.get_progress()
                    print(f"  Step {step + 1:4d} | Reward: {episode_reward:7.2f} | "
                          f"GIF: {progress*100:5.1f}% recorded")
                else:
                    print(f"  Step {step + 1:4d} | Reward: {episode_reward:7.2f}")
            
            obs = next_obs
            step += 1
            
            if done[0]:
                print(f"  Episode ended at step {step}")
                break
        
        print(f"  Total Reward: {episode_reward:.2f}\n")
    
    env.close()
    print("\n✅ Testing complete!")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Test Walking Spider with GIF Recording')
    parser.add_argument('--model', type=str, 
                       default=None,
                       help='Path to trained model (optional)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--duration', type=int, default=10,
                       help='GIF duration in seconds')
    
    args = parser.parse_args()
    
    try:
        if args.model:
            test_trained_model(
                model_path=args.model,
                num_episodes=args.episodes,
                gif_duration=args.duration
            )
        else:
            test_random_actions(
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                gif_duration=args.duration
            )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

