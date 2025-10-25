"""
Debug Logger for Walking Spider Environment

This module provides comprehensive logging for debugging the walking spider robot.
Logs include state information, reward breakdowns, physics parameters, and training metrics.

Usage:
    from debug_logger import DebugLogger
    
    # Initialize with logging level and output directory
    logger = DebugLogger(log_level='INFO', log_dir='logs/debug')
    
    # Log step information
    logger.log_step(step_num, observation, action, reward, done, info)
    
    # Log reward breakdown
    logger.log_reward_breakdown(forward_reward, height_reward, energy_cost, ...)
    
    # Save and close
    logger.close()
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict


class DebugLogger:
    """Comprehensive logging system for Walking Spider environment debugging."""
    
    def __init__(self, log_level='INFO', log_dir='logs/debug', enabled=True):
        """
        Initialize debug logger.
        
        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_dir: Directory to save log files
            enabled: Enable/disable logging
        """
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.log_level = log_level
        self.log_dir = log_dir
        self.start_time = time.time()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'spider_debug_{timestamp}.log')
        self.json_file = os.path.join(log_dir, f'spider_data_{timestamp}.json')
        self.summary_file = os.path.join(log_dir, f'spider_summary_{timestamp}.txt')
        
        # Initialize data structures
        self.episode_data = []
        self.current_episode = {
            'steps': [],
            'total_reward': 0,
            'episode_num': 0,
            'start_time': time.time()
        }
        
        # Statistics tracking
        self.stats = defaultdict(list)
        self.step_count = 0
        self.episode_count = 0
        
        # Open log file
        self.file_handle = open(self.log_file, 'w')
        
        # Write header
        self._write_header()
        
    def _write_header(self):
        """Write log file header with metadata."""
        header = f"""
{'='*80}
Walking Spider Debug Log
{'='*80}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log Level: {self.log_level}
Log File: {self.log_file}
{'='*80}

"""
        self.file_handle.write(header)
        self.file_handle.flush()
        
    def log(self, message, level='INFO'):
        """Write a log message with timestamp and level."""
        if not self.enabled:
            return
            
        levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        if levels.get(level, 1) < levels.get(self.log_level, 1):
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        elapsed = time.time() - self.start_time
        log_line = f"[{timestamp}] [{level:7s}] [+{elapsed:8.3f}s] {message}\n"
        self.file_handle.write(log_line)
        self.file_handle.flush()
        
    def log_episode_start(self, episode_num):
        """Log the start of a new episode."""
        if not self.enabled:
            return
            
        self.episode_count = episode_num
        self.current_episode = {
            'episode_num': episode_num,
            'steps': [],
            'total_reward': 0,
            'start_time': time.time()
        }
        
        self.log(f"\n{'='*80}", 'INFO')
        self.log(f"EPISODE {episode_num} START", 'INFO')
        self.log(f"{'='*80}", 'INFO')
        
    def log_step(self, step_num, observation, action, reward, done, info=None):
        """
        Log a single environment step with all relevant information.
        
        Args:
            step_num: Current step number
            observation: Observation array from environment
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            info: Additional info dictionary
        """
        if not self.enabled:
            return
            
        self.step_count = step_num
        
        # Extract key observation components (assuming standard spider observation)
        obs_dict = self._parse_observation(observation)
        
        # Log step summary
        self.log(f"--- Step {step_num} ---", 'DEBUG')
        self.log(f"  Height: {obs_dict['height']:.4f} m", 'DEBUG')
        self.log(f"  Velocity: [{obs_dict['vel_x']:.3f}, {obs_dict['vel_y']:.3f}, {obs_dict['vel_z']:.3f}]", 'DEBUG')
        self.log(f"  Orientation (RPY): [{obs_dict['roll']:.3f}, {obs_dict['pitch']:.3f}, {obs_dict['yaw']:.3f}]", 'DEBUG')
        self.log(f"  Action: [{', '.join([f'{a:.3f}' for a in action])}]", 'DEBUG')
        self.log(f"  Reward: {reward:.4f}", 'INFO')
        
        if done:
            self.log(f"  Episode DONE at step {step_num}", 'WARNING')
            
        # Store step data
        step_data = {
            'step': step_num,
            'observation': obs_dict,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': float(reward),
            'done': done,
            'info': info or {}
        }
        self.current_episode['steps'].append(step_data)
        self.current_episode['total_reward'] += reward
        
        # Track statistics
        self.stats['rewards'].append(reward)
        self.stats['heights'].append(obs_dict['height'])
        self.stats['velocities'].append(obs_dict['vel_x'])
        
    def log_reward_breakdown(self, **reward_components):
        """
        Log detailed reward component breakdown.
        
        Args:
            **reward_components: Named reward components (forward_reward=0.5, energy_cost=-0.1, etc.)
        """
        if not self.enabled:
            return
            
        self.log("  Reward Breakdown:", 'DEBUG')
        total = 0
        for name, value in reward_components.items():
            self.log(f"    {name:20s}: {value:8.4f}", 'DEBUG')
            total += value
        self.log(f"    {'TOTAL':20s}: {total:8.4f}", 'DEBUG')
        
        # Store in current step
        if self.current_episode['steps']:
            self.current_episode['steps'][-1]['reward_breakdown'] = reward_components
            
    def log_physics_state(self, physics_info):
        """
        Log physics parameters and state.
        
        Args:
            physics_info: Dictionary with physics parameters
                - friction: dict with lateral, spinning, rolling friction
                - contacts: list of contact points
                - joint_states: joint positions, velocities, torques
        """
        if not self.enabled:
            return
            
        self.log("  Physics State:", 'DEBUG')
        
        if 'friction' in physics_info:
            self.log(f"    Friction - Lateral: {physics_info['friction'].get('lateral', 'N/A')}, "
                    f"Spinning: {physics_info['friction'].get('spinning', 'N/A')}, "
                    f"Rolling: {physics_info['friction'].get('rolling', 'N/A')}", 'DEBUG')
                    
        if 'contacts' in physics_info:
            contacts = physics_info['contacts']
            self.log(f"    Contact Points: {len(contacts)}", 'DEBUG')
            for i, contact in enumerate(contacts[:4]):  # Limit to first 4
                self.log(f"      Contact {i}: Link {contact.get('link_id', 'N/A')}, "
                        f"Force: {contact.get('normal_force', 'N/A'):.2f} N", 'DEBUG')
                        
        if 'joint_states' in physics_info:
            joints = physics_info['joint_states']
            self.log(f"    Joint States (8 joints):", 'DEBUG')
            for i, joint in enumerate(joints):
                self.log(f"      Joint {i}: Pos={joint.get('position', 0):.3f}, "
                        f"Vel={joint.get('velocity', 0):.3f}, "
                        f"Torque={joint.get('torque', 0):.3f}", 'DEBUG')
                        
    def log_episode_end(self):
        """Log episode completion and statistics."""
        if not self.enabled:
            return
            
        episode_time = time.time() - self.current_episode['start_time']
        total_reward = self.current_episode['total_reward']
        num_steps = len(self.current_episode['steps'])
        
        self.log(f"\n{'='*80}", 'INFO')
        self.log(f"EPISODE {self.episode_count} END", 'INFO')
        self.log(f"{'='*80}", 'INFO')
        self.log(f"  Duration: {episode_time:.2f} seconds", 'INFO')
        self.log(f"  Total Steps: {num_steps}", 'INFO')
        self.log(f"  Total Reward: {total_reward:.4f}", 'INFO')
        self.log(f"  Average Reward: {total_reward/num_steps if num_steps > 0 else 0:.4f}", 'INFO')
        
        # Save episode data
        self.episode_data.append(self.current_episode.copy())
        
        # Write periodic summary every 10 episodes
        if self.episode_count % 10 == 0:
            self._write_summary()
            
    def log_training_metrics(self, metrics):
        """
        Log training-related metrics.
        
        Args:
            metrics: Dictionary with training metrics
                - loss, learning_rate, entropy, value_loss, policy_loss, etc.
        """
        if not self.enabled:
            return
            
        self.log("Training Metrics:", 'INFO')
        for key, value in metrics.items():
            self.log(f"  {key}: {value}", 'INFO')
            
    def log_error(self, error_msg, exception=None):
        """
        Log error messages and exceptions.
        
        Args:
            error_msg: Error description
            exception: Exception object (optional)
        """
        if not self.enabled:
            return
            
        self.log(f"ERROR: {error_msg}", 'ERROR')
        if exception:
            self.log(f"  Exception: {type(exception).__name__}: {str(exception)}", 'ERROR')
            
    def _parse_observation(self, obs):
        """Parse observation array into meaningful components."""
        if len(obs) < 27:
            return {'raw': obs}
            
        # Based on spider environment observation structure
        obs_dict = {
            'height': obs[0],
            'quat_x': obs[1],
            'quat_y': obs[2],
            'quat_z': obs[3],
            'quat_w': obs[4],
            'joint_positions': obs[5:13].tolist() if hasattr(obs, 'tolist') else list(obs[5:13]),
            'vel_x': obs[13],
            'vel_y': obs[14],
            'vel_z': obs[15],
            'ang_vel_x': obs[16],
            'ang_vel_y': obs[17],
            'ang_vel_z': obs[18],
            'joint_velocities': obs[19:27].tolist() if hasattr(obs, 'tolist') else list(obs[19:27])
        }
        
        # Calculate roll, pitch, yaw from quaternion
        qx, qy, qz, qw = obs[1], obs[2], obs[3], obs[4]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        obs_dict['roll'] = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            obs_dict['pitch'] = np.sign(sinp) * np.pi / 2
        else:
            obs_dict['pitch'] = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        obs_dict['yaw'] = np.arctan2(siny_cosp, cosy_cosp)
        
        return obs_dict
        
    def _write_summary(self):
        """Write summary statistics to summary file."""
        if not self.enabled or not self.episode_data:
            return
            
        with open(self.summary_file, 'w') as f:
            f.write(f"Walking Spider Training Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Episodes: {len(self.episode_data)}\n")
            f.write(f"Total Steps: {self.step_count}\n\n")
            
            # Calculate statistics
            rewards = [ep['total_reward'] for ep in self.episode_data]
            steps = [len(ep['steps']) for ep in self.episode_data]
            
            f.write(f"Reward Statistics:\n")
            f.write(f"  Mean: {np.mean(rewards):.4f}\n")
            f.write(f"  Std:  {np.std(rewards):.4f}\n")
            f.write(f"  Min:  {np.min(rewards):.4f}\n")
            f.write(f"  Max:  {np.max(rewards):.4f}\n\n")
            
            f.write(f"Episode Length Statistics:\n")
            f.write(f"  Mean: {np.mean(steps):.2f}\n")
            f.write(f"  Std:  {np.std(steps):.2f}\n")
            f.write(f"  Min:  {np.min(steps)}\n")
            f.write(f"  Max:  {np.max(steps)}\n\n")
            
            if self.stats['heights']:
                f.write(f"Height Statistics:\n")
                f.write(f"  Mean: {np.mean(self.stats['heights']):.4f} m\n")
                f.write(f"  Std:  {np.std(self.stats['heights']):.4f} m\n\n")
                
            if self.stats['velocities']:
                f.write(f"Forward Velocity Statistics:\n")
                f.write(f"  Mean: {np.mean(self.stats['velocities']):.4f} m/s\n")
                f.write(f"  Std:  {np.std(self.stats['velocities']):.4f} m/s\n\n")
                
            # Last 10 episodes summary
            f.write(f"Last 10 Episodes:\n")
            for ep in self.episode_data[-10:]:
                f.write(f"  Episode {ep['episode_num']}: "
                       f"Reward={ep['total_reward']:.2f}, "
                       f"Steps={len(ep['steps'])}\n")
                       
    def save_json(self):
        """Save all logged data to JSON file for analysis."""
        if not self.enabled:
            return
            
        data = {
            'metadata': {
                'log_file': self.log_file,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'total_episodes': len(self.episode_data),
                'total_steps': self.step_count
            },
            'episodes': self.episode_data,
            'statistics': {
                'rewards': self.stats['rewards'][-1000:],  # Last 1000 for size
                'heights': self.stats['heights'][-1000:],
                'velocities': self.stats['velocities'][-1000:]
            }
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.log(f"JSON data saved to: {self.json_file}", 'INFO')
        
    def close(self):
        """Close log files and save final data."""
        if not self.enabled:
            return
            
        self.log("\n" + "="*80, 'INFO')
        self.log("Logging session ended", 'INFO')
        self.log("="*80, 'INFO')
        
        self._write_summary()
        self.save_json()
        
        self.file_handle.close()
        
        print(f"\n✅ Debug logs saved:")
        print(f"   - Detailed log: {self.log_file}")
        print(f"   - JSON data: {self.json_file}")
        print(f"   - Summary: {self.summary_file}")


# Convenience function for quick logging
def create_logger(enabled=True, log_level='INFO', log_dir='logs/debug'):
    """
    Create and return a debug logger instance.
    
    Args:
        enabled: Enable/disable logging
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_dir: Directory for log files
        
    Returns:
        DebugLogger instance
    """
    return DebugLogger(log_level=log_level, log_dir=log_dir, enabled=enabled)

