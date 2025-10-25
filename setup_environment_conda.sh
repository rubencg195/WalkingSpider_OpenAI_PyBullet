#!/bin/bash
#
# WalkingSpider Environment Setup Script (Conda-based)
#
# This script sets up a complete Conda environment for the Walking Spider project.
# Conda is recommended on Windows because it has pre-built PyBullet wheels.
#
# Usage:
#   bash setup_environment_conda.sh [environment_name]
#
# Example:
#   bash setup_environment_conda.sh walking_spider_env
#

set +H  # Disable history expansion to avoid "bash: !: event not found" error

# Configuration
ENV_NAME="${1:-walking_spider_env}"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "  WalkingSpider Conda Environment Setup"
echo "=========================================="
echo ""
echo "Environment Name: $ENV_NAME"
echo "Python Version: $PYTHON_VERSION"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: Conda is not installed or not in PATH"
    echo ""
    echo "Please install Miniconda or Anaconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Create environment
echo "Creating conda environment: $ENV_NAME ..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

# Activate environment
echo ""
echo "Activating environment..."
source activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate environment"
    echo "Try manually: conda activate $ENV_NAME"
    exit 1
fi

echo "✅ Environment activated"
echo ""

# Install PyBullet from conda-forge (has pre-built wheels for Windows!)
echo "Installing PyBullet from conda-forge (pre-built wheels)..."
conda install -c conda-forge pybullet -y

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: PyBullet installation had issues, attempting pip fallback..."
    pip install pybullet
fi

echo "✅ PyBullet installed"
echo ""

# Install other dependencies
echo "Installing core dependencies..."
pip install gym numpy imageio pillow matplotlib

# Optional ML dependencies
echo ""
echo "Installing optional ML/RL dependencies..."
pip install tensorflow-cpu torch torchvision torchaudio stable-baselines3

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run tests:"
echo "  python test_pybullet_simple.py"
echo "  python run_test_with_full_debug.py"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
