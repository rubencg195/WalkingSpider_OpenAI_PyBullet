#!/usr/bin/env python3
"""
PPO Training Script for Walking Spider
Trains the spider robot to walk using Proximal Policy Optimization (PPO2)
Includes real-time reward visualization using matplotlib
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import from local modules in src/
from walking_spider_env import WalkingSpiderEnv


class RewardPlotCallback(BaseCallback):
    """Callback to plot rewards during training"""
    
    def __init__(self, verbose=0, update_freq=10):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.update_freq = update_freq
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        
        # Enable interactive mode for real-time plotting
        plt.ion()
    
    def _on_step(self) -> bool:
        """Called after each step"""
        # Get current episode info
        if 'episode' in self.locals:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.episode_count += 1
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.episodes.append(self.episode_count)
                
                if self.episode_count % self.update_freq == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"Episode {self.episode_count}: Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}")
                    
                    # Update plot in real-time
                    self._update_plot_realtime()
        return True
    
    def _update_plot_realtime(self):
        """Update matplotlib plot in real-time during training"""
        if not self.episodes:
            return
        
        # Create figure on first call
        if self.fig is None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle('Training Progress - Walking Spider PPO')
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot rewards
        self.ax1.plot(self.episodes, self.episode_rewards, alpha=0.6, label='Episode Reward', color='blue')
        if len(self.episodes) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            self.ax1.plot(range(10, len(self.episodes)+1), moving_avg, 'r-', linewidth=2, label='10-Episode Moving Avg')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Total Reward')
        self.ax1.set_title('Training Rewards Over Time')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        self.ax2.plot(self.episodes, self.episode_lengths, alpha=0.6, label='Episode Length', color='orange')
        if len(self.episodes) > 10:
            moving_avg_len = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
            self.ax2.plot(range(10, len(self.episodes)+1), moving_avg_len, 'r-', linewidth=2, label='10-Episode Moving Avg')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Steps per Episode')
        self.ax2.set_title('Episode Lengths Over Time')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update display
    
    def plot_rewards(self, filename='training_rewards.png'):
        """Plot and save reward history"""
        if not self.episodes:
            print("No episodes recorded yet")
            return
        
        # Close real-time plot if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create final plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        ax1.plot(self.episodes, self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episodes) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            ax1.plot(range(10, len(self.episodes)+1), moving_avg, 'r-', linewidth=2, label='10-Episode Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax2.plot(self.episodes, self.episode_lengths, alpha=0.6, label='Episode Length', color='orange')
        if len(self.episodes) > 10:
            moving_avg_len = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
            ax2.plot(range(10, len(self.episodes)+1), moving_avg_len, 'r-', linewidth=2, label='10-Episode Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps per Episode')
        ax2.set_title('Episode Lengths Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"✓ Training plot saved to {filename}")
        plt.close()


def train_spider(
    total_timesteps=100000,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    model_path='output/trained_spider_ppo.zip',
    render=False,
    device='cpu'
):
    """
    Train the walking spider using PPO
    
    Args:
        total_timesteps: Total training steps
        learning_rate: PPO learning rate
        n_steps: Number of steps to collect per update
        batch_size: Batch size for PPO updates
        model_path: Path to save the trained model (default: output/trained_spider_ppo.zip)
        render: Whether to render during training
        device: Device to use for training (cpu or cuda/gpu)
    """
    
    print("=" * 80)
    print("  PPO Training: Walking Spider Robot")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Steps per Update: {n_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Model Output: {model_path}")
    print(f"  Render: {render}")
    print(f"  Device: {device}")
    print()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create environment
    env = WalkingSpiderEnv(render=render, enable_gif_recording=render)
    
    # Wrap in VecEnv for better compatibility
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,  # Disabled by default (install tensorboard to enable)
        device=device
    )
    
    print("Training started...")
    print()
    
    # Create callback for plotting
    callback = RewardPlotCallback()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Save model
    model.save(model_path)
    print()
    print(f"✓ Model saved to {model_path}")
    
    # Plot training results
    callback.plot_rewards('training_rewards.png')
    
    env.close()
    
    print()
    print("=" * 80)
    print("  Training Complete!")
    print("=" * 80)
    
    return model, callback


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Walking Spider with PPO')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--model', type=str, default='output/trained_spider_ppo.zip', help='Model save path')
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use (cpu or cuda/gpu)')
    
    args = parser.parse_args()
    
    train_spider(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.steps,
        batch_size=args.batch_size,
        model_path=args.model,
        render=args.render,
        device=args.device
    )

