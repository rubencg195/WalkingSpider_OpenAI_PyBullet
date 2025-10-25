"""
Comprehensive Test Script for Walking Spider with Full Debugging

This script tests the improved walking spider environment with:
- GUI visualization (for user to see)
- Debug logging (for AI to analyze)
- GIF recording (visual snapshots)
- All improvements active

Usage:
    python run_test_with_full_debug.py
"""

import sys
import gym
import numpy as np
from datetime import datetime

# Add environment to path
sys.path.insert(0, 'environment/walking-spider')
import walking_spider

# Import debug tools
from walking_spider.envs.debug_logger import create_logger
from walking_spider.envs.gif_recorder import GifRecorder


def run_comprehensive_test():
    """Run test with logging, GIF recording, and GUI visualization."""
    
    print("\n" + "="*80)
    print("🤖 WALKING SPIDER COMPREHENSIVE TEST")
    print("="*80)
    print("\n✅ Features Enabled:")
    print("   1. GUI Visualization (you can see the robot)")
    print("   2. Debug Logging (I can analyze behavior)")
    print("   3. GIF Recording (automatic 10-second snapshots)")
    print("   4. All Environment Improvements Active")
    print("\n" + "="*80 + "\n")
    
    # Create logger for debugging
    logger = create_logger(
        enabled=True,
        log_level='INFO',
        log_dir='logs/debug'
    )
    
    logger.log("="*80, 'INFO')
    logger.log("WALKING SPIDER COMPREHENSIVE TEST START", 'INFO')
    logger.log("="*80, 'INFO')
    logger.log("Features: GUI + Logging + GIF Recording", 'INFO')
    logger.log("All improvements active (#1-#8)", 'INFO')
    
    try:
        # Create environment with GUI rendering
        print("🔄 Creating environment with GUI rendering...\n")
        env = gym.make('WalkingSpider-v0')
        
        logger.log("Environment created successfully", 'INFO')
        
        # Enable GIF recording
        print("🔄 Enabling GIF recording...\n")
        try:
            env.unwrapped.gif_recorder = GifRecorder(
                save_dir='videos',
                duration_seconds=10,
                fps=30,
                enabled=True
            )
            env.unwrapped.gif_recorder.start_recording()
            logger.log("GIF recording enabled", 'INFO')
            print("✅ GIF recording enabled - saving to videos/ folder\n")
        except Exception as e:
            logger.log(f"Could not enable GIF recording: {e}", 'WARNING')
            print(f"⚠️  GIF recording unavailable: {e}\n")
        
        # Run test episodes
        num_episodes = 3
        max_steps = 500
        
        for episode in range(num_episodes):
            print(f"\n{'='*80}")
            print(f"📊 EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*80}\n")
            
            logger.log_episode_start(episode + 1)
            
            obs = env.reset()
            episode_reward = 0
            episode_height = []
            episode_velocity = []
            
            print(f"Starting episode with observation shape: {np.array(obs).shape}")
            
            for step in range(max_steps):
                # Random action (not trained model)
                action = env.action_space.sample()
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Extract info for logging
                obs_array = np.array(obs)
                height = obs_array[0]  # First element is height
                vel_x = obs_array[13]  # X velocity
                
                episode_height.append(height)
                episode_velocity.append(vel_x)
                
                # Log step
                logger.log_step(step, obs, action, reward, done, info)
                
                # Log reward components
                if step % 50 == 0:
                    logger.log_reward_breakdown(
                        reward=reward,
                        height=height,
                        velocity=vel_x
                    )
                
                # Print progress
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1:4d} | Reward: {episode_reward:7.2f} | "
                          f"Height: {height:.4f} m | Velocity: {vel_x:.3f} m/s")
                    
                    # Show GIF recording progress
                    if hasattr(env.unwrapped, 'gif_recorder') and env.unwrapped.gif_recorder:
                        progress = env.unwrapped.gif_recorder.get_progress()
                        print(f"           GIF Recording: {progress*100:5.1f}% complete")
                
                obs = next_obs
                
                if done:
                    print(f"\n  ⚠️  Episode ended at step {step + 1} (condition triggered)")
                    logger.log(f"Episode terminated at step {step + 1}", 'INFO')
                    break
            
            # Episode summary
            avg_height = np.mean(episode_height) if episode_height else 0
            max_height = np.max(episode_height) if episode_height else 0
            min_height = np.min(episode_height) if episode_height else 0
            avg_velocity = np.mean(episode_velocity) if episode_velocity else 0
            max_velocity = np.max(episode_velocity) if episode_velocity else 0
            
            print(f"\n📈 Episode Summary:")
            print(f"   Total Steps: {step + 1}")
            print(f"   Total Reward: {episode_reward:.2f}")
            print(f"   Avg Height: {avg_height:.4f} m (ideal: 0.06)")
            print(f"   Height Range: {min_height:.4f} - {max_height:.4f} m")
            print(f"   Avg Forward Velocity: {avg_velocity:.3f} m/s")
            print(f"   Max Forward Velocity: {max_velocity:.3f} m/s")
            
            logger.log(f"Episode {episode + 1} Summary:", 'INFO')
            logger.log(f"  Total Steps: {step + 1}", 'INFO')
            logger.log(f"  Total Reward: {episode_reward:.2f}", 'INFO')
            logger.log(f"  Avg Height: {avg_height:.4f} m", 'INFO')
            logger.log(f"  Avg Velocity: {avg_velocity:.3f} m/s", 'INFO')
            
            logger.log_episode_end()
        
        env.close()
        
        print(f"\n{'='*80}")
        print("✅ TEST COMPLETE")
        print(f"{'='*80}\n")
        
        logger.log("\n" + "="*80, 'INFO')
        logger.log("TEST COMPLETE", 'INFO')
        logger.log("="*80, 'INFO')
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        logger.log_error("Test interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        logger.log_error(f"Error during test: {e}", e)
        import traceback
        traceback.print_exc()
    
    finally:
        # Close logger and save data
        logger.close()
        
        print("\n📊 Debug Information Saved:")
        print("   Logs: logs/debug/spider_debug_*.log")
        print("   Data: logs/debug/spider_data_*.json")
        print("   Summary: logs/debug/spider_summary_*.txt")
        print("\n📹 GIF Snapshots Saved:")
        print("   Videos: videos/spider_snapshot_*.gif")
        print("\n💡 To analyze logs:")
        print("   python analyze_spider_logs.py --summary --problems")
        print("   python analyze_spider_logs.py --episode 1 --verbose")


if __name__ == '__main__':
    run_comprehensive_test()

