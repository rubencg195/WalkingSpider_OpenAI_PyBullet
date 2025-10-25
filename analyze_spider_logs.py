"""
Walking Spider Log Analyzer

Utility to analyze and extract debugging information from spider training logs.
This tool helps identify issues with the robot's behavior for iterative debugging.

Usage:
    # Show summary of latest log
    python analyze_spider_logs.py
    
    # Analyze specific log file
    python analyze_spider_logs.py --file logs/debug/spider_data_20241025_120000.json
    
    # Extract reward breakdown
    python analyze_spider_logs.py --rewards
    
    # Find problematic episodes (low reward, crashes, etc.)
    python analyze_spider_logs.py --problems
    
    # Export specific episode details
    python analyze_spider_logs.py --episode 5 --verbose
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict


class LogAnalyzer:
    """Analyze walking spider debug logs."""
    
    def __init__(self, log_file):
        """Initialize analyzer with log file."""
        self.log_file = log_file
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load JSON log data."""
        if not os.path.exists(self.log_file):
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
            
        with open(self.log_file, 'r') as f:
            self.data = json.load(f)
            
        print(f"✅ Loaded: {self.log_file}")
        print(f"   Episodes: {len(self.data['episodes'])}")
        print(f"   Total Steps: {self.data['metadata']['total_steps']}")
        
    def print_summary(self):
        """Print overall summary statistics."""
        episodes = self.data['episodes']
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Reward statistics
        rewards = [ep['total_reward'] for ep in episodes]
        print(f"\n📊 Reward Statistics:")
        print(f"   Mean:   {np.mean(rewards):8.4f}")
        print(f"   Median: {np.median(rewards):8.4f}")
        print(f"   Std:    {np.std(rewards):8.4f}")
        print(f"   Min:    {np.min(rewards):8.4f}")
        print(f"   Max:    {np.max(rewards):8.4f}")
        
        # Episode length
        lengths = [len(ep['steps']) for ep in episodes]
        print(f"\n📏 Episode Length:")
        print(f"   Mean:   {np.mean(lengths):8.2f} steps")
        print(f"   Median: {np.median(lengths):8.2f} steps")
        print(f"   Min:    {np.min(lengths):8.0f} steps")
        print(f"   Max:    {np.max(lengths):8.0f} steps")
        
        # Velocity statistics (if available)
        if 'velocities' in self.data['statistics'] and self.data['statistics']['velocities']:
            vels = self.data['statistics']['velocities']
            print(f"\n🏃 Forward Velocity:")
            print(f"   Mean: {np.mean(vels):8.4f} m/s")
            print(f"   Max:  {np.max(vels):8.4f} m/s")
            print(f"   Min:  {np.min(vels):8.4f} m/s")
            
        # Height statistics
        if 'heights' in self.data['statistics'] and self.data['statistics']['heights']:
            heights = self.data['statistics']['heights']
            print(f"\n📐 Height:")
            print(f"   Mean: {np.mean(heights):8.4f} m")
            print(f"   Min:  {np.min(heights):8.4f} m (may indicate falling)")
            print(f"   Max:  {np.max(heights):8.4f} m")
            
        print("="*80)
        
    def analyze_rewards(self):
        """Analyze reward components and trends."""
        print("\n" + "="*80)
        print("REWARD ANALYSIS")
        print("="*80)
        
        episodes = self.data['episodes']
        
        # Find episodes with reward breakdown
        breakdown_available = []
        for ep in episodes:
            if ep['steps'] and 'reward_breakdown' in ep['steps'][0]:
                breakdown_available.append(ep)
                
        if not breakdown_available:
            print("\n⚠️  No reward breakdown data found in logs.")
            print("   Enable reward breakdown logging with:")
            print("   logger.log_reward_breakdown(forward=..., energy=..., etc.)")
            return
            
        print(f"\n📈 Reward Component Analysis ({len(breakdown_available)} episodes)")
        
        # Aggregate reward components
        components = defaultdict(list)
        for ep in breakdown_available:
            for step in ep['steps']:
                if 'reward_breakdown' in step:
                    for key, value in step['reward_breakdown'].items():
                        components[key].append(value)
                        
        # Print component statistics
        for component, values in components.items():
            print(f"\n   {component}:")
            print(f"      Mean: {np.mean(values):8.4f}")
            print(f"      Min:  {np.min(values):8.4f}")
            print(f"      Max:  {np.max(values):8.4f}")
            
    def find_problems(self):
        """Identify problematic episodes and patterns."""
        print("\n" + "="*80)
        print("PROBLEM DETECTION")
        print("="*80)
        
        episodes = self.data['episodes']
        problems = []
        
        # Check for low rewards
        rewards = [ep['total_reward'] for ep in episodes]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        for ep in episodes:
            issue_list = []
            
            # Low reward
            if ep['total_reward'] < mean_reward - std_reward:
                issue_list.append("Low reward")
                
            # Short episode (premature termination)
            if len(ep['steps']) < 50:
                issue_list.append("Short episode (crash?)")
                
            # Check for falling (low height)
            if ep['steps']:
                heights = [s['observation']['height'] for s in ep['steps'] if 'observation' in s]
                if heights and np.mean(heights) < 0.04:
                    issue_list.append("Low height (falling)")
                    
            # Check for flipping (extreme orientation)
            if ep['steps']:
                for step in ep['steps'][:10]:  # Check first 10 steps
                    if 'observation' in step:
                        obs = step['observation']
                        if abs(obs.get('roll', 0)) > 1.5 or abs(obs.get('pitch', 0)) > 1.5:
                            issue_list.append("Flipped over")
                            break
                            
            if issue_list:
                problems.append({
                    'episode': ep['episode_num'],
                    'reward': ep['total_reward'],
                    'steps': len(ep['steps']),
                    'issues': issue_list
                })
                
        if problems:
            print(f"\n🚨 Found {len(problems)} problematic episodes:\n")
            for prob in problems[:10]:  # Show first 10
                print(f"   Episode {prob['episode']:3d}: Reward={prob['reward']:7.2f}, "
                      f"Steps={prob['steps']:3d}")
                for issue in prob['issues']:
                    print(f"      ⚠️  {issue}")
                print()
        else:
            print("\n✅ No obvious problems detected!")
            
    def analyze_episode(self, episode_num, verbose=False):
        """Detailed analysis of a specific episode."""
        episodes = self.data['episodes']
        
        # Find episode
        episode = None
        for ep in episodes:
            if ep['episode_num'] == episode_num:
                episode = ep
                break
                
        if not episode:
            print(f"❌ Episode {episode_num} not found")
            return
            
        print("\n" + "="*80)
        print(f"EPISODE {episode_num} ANALYSIS")
        print("="*80)
        
        steps = episode['steps']
        total_reward = episode['total_reward']
        
        print(f"\n📊 Overview:")
        print(f"   Total Steps: {len(steps)}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Avg Reward/Step: {total_reward/len(steps) if steps else 0:.4f}")
        
        if not steps:
            return
            
        # Extract trajectory data
        heights = [s['observation']['height'] for s in steps if 'observation' in s]
        vels_x = [s['observation']['vel_x'] for s in steps if 'observation' in s]
        rewards = [s['reward'] for s in steps]
        
        print(f"\n📐 Height Trajectory:")
        print(f"   Start: {heights[0]:.4f} m")
        print(f"   End:   {heights[-1]:.4f} m")
        print(f"   Mean:  {np.mean(heights):.4f} m")
        print(f"   Min:   {np.min(heights):.4f} m")
        
        print(f"\n🏃 Velocity Profile:")
        print(f"   Mean Forward: {np.mean(vels_x):.4f} m/s")
        print(f"   Max Forward:  {np.max(vels_x):.4f} m/s")
        print(f"   Min Forward:  {np.min(vels_x):.4f} m/s")
        
        print(f"\n💰 Reward Profile:")
        print(f"   Mean: {np.mean(rewards):.4f}")
        print(f"   Max:  {np.max(rewards):.4f}")
        print(f"   Min:  {np.min(rewards):.4f}")
        
        # Check for issues
        print(f"\n🔍 Issue Detection:")
        issues_found = False
        
        if np.mean(heights) < 0.04:
            print(f"   ⚠️  Low average height - robot may be falling")
            issues_found = True
            
        if np.mean(vels_x) < 0.01:
            print(f"   ⚠️  Very low forward velocity - not moving")
            issues_found = True
            
        if np.min(heights) < 0.02:
            print(f"   ⚠️  Height dropped below 0.02m - likely crashed")
            issues_found = True
            
        if not issues_found:
            print(f"   ✅ No obvious issues detected")
            
        # Verbose output
        if verbose and len(steps) <= 20:
            print(f"\n📝 Step-by-Step Details:")
            for i, step in enumerate(steps):
                obs = step['observation']
                print(f"\n   Step {i}:")
                print(f"      Height: {obs['height']:.4f} m")
                print(f"      Velocity: [{obs['vel_x']:.3f}, {obs['vel_y']:.3f}, {obs['vel_z']:.3f}]")
                print(f"      Orientation: R={obs['roll']:.3f}, P={obs['pitch']:.3f}, Y={obs['yaw']:.3f}")
                print(f"      Reward: {step['reward']:.4f}")
                
    def export_snippet(self, episode_num, step_range=None, output_file=None):
        """Export a snippet of episode data for sharing/debugging."""
        episodes = self.data['episodes']
        
        # Find episode
        episode = None
        for ep in episodes:
            if ep['episode_num'] == episode_num:
                episode = ep
                break
                
        if not episode:
            print(f"❌ Episode {episode_num} not found")
            return
            
        # Extract step range
        steps = episode['steps']
        if step_range:
            start, end = step_range
            steps = steps[start:end]
            
        snippet = {
            'episode_num': episode_num,
            'total_reward': episode['total_reward'],
            'steps': steps
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(snippet, f, indent=2)
            print(f"✅ Snippet exported to: {output_file}")
        else:
            print(json.dumps(snippet, indent=2))


def find_latest_log(log_dir='logs/debug'):
    """Find the most recent log file."""
    if not os.path.exists(log_dir):
        return None
        
    json_files = [f for f in os.listdir(log_dir) 
                  if f.startswith('spider_data_') and f.endswith('.json')]
    
    if not json_files:
        return None
        
    latest = sorted(json_files)[-1]
    return os.path.join(log_dir, latest)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze Walking Spider Logs')
    parser.add_argument('--file', type=str, help='Specific log file to analyze')
    parser.add_argument('--log-dir', type=str, default='logs/debug',
                       help='Directory containing log files')
    parser.add_argument('--summary', action='store_true', default=True,
                       help='Show summary statistics')
    parser.add_argument('--rewards', action='store_true',
                       help='Analyze reward components')
    parser.add_argument('--problems', action='store_true',
                       help='Find problematic episodes')
    parser.add_argument('--episode', type=int,
                       help='Analyze specific episode')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose episode analysis')
    parser.add_argument('--export', type=str,
                       help='Export episode snippet to file')
    
    args = parser.parse_args()
    
    # Find log file
    if args.file:
        log_file = args.file
    else:
        log_file = find_latest_log(args.log_dir)
        
    if not log_file:
        print(f"❌ No log files found in {args.log_dir}")
        print(f"   Run training with logging enabled first:")
        print(f"   python test_spider_with_logging.py")
        return
        
    # Create analyzer
    try:
        analyzer = LogAnalyzer(log_file)
    except Exception as e:
        print(f"❌ Error loading log: {e}")
        return
        
    # Run analyses
    if args.summary:
        analyzer.print_summary()
        
    if args.rewards:
        analyzer.analyze_rewards()
        
    if args.problems:
        analyzer.find_problems()
        
    if args.episode is not None:
        analyzer.analyze_episode(args.episode, verbose=args.verbose)
        
        if args.export:
            analyzer.export_snippet(args.episode, output_file=args.export)


if __name__ == '__main__':
    main()

