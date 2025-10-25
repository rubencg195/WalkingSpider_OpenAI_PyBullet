#!/usr/bin/env python
"""
Simple PyBullet Test
Verifies that PyBullet is properly installed and can initialize.
"""

import sys

print("=" * 80)
print("WALKING SPIDER - PYBULLET VERIFICATION TEST")
print("=" * 80)

# Test 1: Import pybullet
print("\n[1/5] Testing PyBullet import...")
try:
    import pybullet as p
    print("    ✅ PyBullet imported successfully")
    print(f"    Version: {p.__version__ if hasattr(p, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"    ❌ Failed to import PyBullet: {e}")
    sys.exit(1)

# Test 2: Import gym
print("\n[2/5] Testing Gym import...")
try:
    import gym
    print("    ✅ Gym imported successfully")
    print(f"    Version: {gym.__version__}")
except ImportError as e:
    print(f"    ❌ Failed to import Gym: {e}")
    sys.exit(1)

# Test 3: Import numpy
print("\n[3/5] Testing NumPy import...")
try:
    import numpy as np
    print("    ✅ NumPy imported successfully")
    print(f"    Version: {np.__version__}")
except ImportError as e:
    print(f"    ❌ Failed to import NumPy: {e}")
    sys.exit(1)

# Test 4: Test PyBullet connection (DIRECT mode, no GUI)
print("\n[4/5] Testing PyBullet connection (DIRECT mode)...")
try:
    client = p.connect(p.DIRECT)
    print("    ✅ PyBullet DIRECT connection successful")
    print(f"    Client ID: {client}")
    p.disconnect()
except Exception as e:
    print(f"    ❌ Failed to connect: {e}")
    sys.exit(1)

# Test 5: Import walking_spider environment
print("\n[5/5] Testing Walking Spider environment import...")
try:
    sys.path.insert(0, 'environment/walking-spider')
    import walking_spider
    print("    ✅ Walking Spider environment imported successfully")
    
    # Try creating env in DIRECT mode (no GUI)
    env = gym.make('WalkingSpider-v0')
    print("    ✅ Walking Spider environment created successfully")
    
    # Test reset
    obs = env.reset()
    print(f"    ✅ Environment reset successful")
    print(f"    Observation shape: {np.array(obs).shape}")
    print(f"    Action space: {env.action_space}")
    
    # Test one step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"    ✅ Environment step successful")
    print(f"    Reward: {reward:.4f}")
    print(f"    Done: {done}")
    
    env.close()
    
except Exception as e:
    print(f"    ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour environment is ready for training!")
print("Next steps:")
print("  1. Run: python run_test_with_full_debug.py")
print("  2. Check logs in: logs/debug/")
print("  3. View GIF snapshots in: videos/")
print("\n" + "=" * 80)
