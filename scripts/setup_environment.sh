#!/bin/bash
# WalkingSpider Environment Setup Script
# This script sets up the Python environment and installs all dependencies
# Run: source scripts/setup_environment.sh

echo "=========================================="
echo "WalkingSpider Environment Setup"
echo "=========================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[1/6] Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
echo ""

# Step 1: Check Python version
echo "[2/6] Checking Python version..."
python3 --version
echo ""

# Step 2: Remove old environment if it exists and is broken
if [ -d "venv_spider" ]; then
    if [ ! -f "venv_spider/bin/activate" ]; then
        echo "[3/6] Removing incomplete virtual environment..."
        rm -rf venv_spider
        python3 -m venv venv_spider
        echo "Virtual environment created: venv_spider/"
    else
        echo "[3/6] Virtual environment already exists."
    fi
else
    echo "[3/6] Creating virtual environment..."
    python3 -m venv venv_spider
    echo "Virtual environment created: venv_spider/"
fi
echo ""

# Step 3: Activate virtual environment
echo "[4/6] Activating virtual environment..."
source venv_spider/bin/activate
echo "Virtual environment activated"
echo "Python executable: $(which python)"
echo ""

# Step 4: Upgrade pip
echo "[5/6] Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 5: Install requirements
echo "[6/6] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

# Verification
echo "=========================================="
echo "Installation Complete! Verifying..."
echo "=========================================="
echo ""

python -c "import pybullet; print('[✓] PyBullet installed')"
python -c "import gym; print('[✓] Gym installed')"
python -c "import stable_baselines3; print('[✓] Stable Baselines3 installed')"
python -c "import numpy; print('[✓] NumPy installed')"
python -c "import matplotlib; print('[✓] Matplotlib installed')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment is now active: (venv_spider)"
echo ""
echo "Next steps:"
echo "1. Run the physics demo:"
echo "   python scripts/demo_trained_spider.py --physics-only"
echo ""
echo "2. Train a model:"
echo "   python scripts/train_ppo.py --timesteps 50000"
echo ""
