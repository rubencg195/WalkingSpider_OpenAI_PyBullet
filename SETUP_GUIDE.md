# Setup Guide - Walking Spider Project

This guide will help you set up the Walking Spider project with all dependencies using Python's package management tools.

## Prerequisites

- Python 3.8 or higher
- Git
- For Windows users: Visual C++ Build Tools (needed for PyBullet)

## Option 1: Using Poetry (Recommended)

Poetry provides reproducible, lock-based dependency management.

### 1. Install Poetry

**On Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**On macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Using pip:**
```bash
pip install poetry
```

### 2. Install Dependencies

```bash
poetry install
```

This will:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Install the project in editable mode

### 3. Activate Environment

```bash
poetry shell
```

### 4. Run Tests

```bash
poetry run python run_test_with_full_debug.py
```

## Option 2: Using pip with requirements.txt (Simple)

### 1. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Tests

```bash
python run_test_with_full_debug.py
```

## Option 3: Using Conda (If you have Conda installed)

### 1. Create Environment from YAML

```bash
conda env create -f walking_spider.yml
```

### 2. Activate Environment

```bash
conda activate walking_spider
```

### 3. Run Tests

```bash
python run_test_with_full_debug.py
```

## Windows-Specific: Installing PyBullet

PyBullet requires a C++ compiler on Windows. You have two options:

### Option A: Install Visual C++ Build Tools (Recommended)

1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select "Desktop development with C++"
4. Complete installation
5. Run: `pip install pybullet`

### Option B: Use Pre-built PyBullet Wheel

```bash
pip install pybullet --only-binary :all:
```

## Verifying Installation

### Test PyBullet Installation

```python
python -c "import pybullet; print('PyBullet version:', pybullet.__version__)"
```

### Test Gym Installation

```python
python -c "import gym; print('Gym version:', gym.__version__)"
```

### Test Full Environment

Run the comprehensive test:

```bash
python run_test_with_full_debug.py
```

This will:
- ✅ Initialize the walking spider environment with GUI
- ✅ Enable debug logging to `logs/debug/`
- ✅ Capture GIF snapshots to `videos/`
- ✅ Run 3 test episodes with random actions
- ✅ Generate analysis logs

## Troubleshooting

### Issue: "pybullet: command not found"

**Solution:** Make sure you've activated your virtual environment:

```bash
# On Windows:
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### Issue: "ModuleNotFoundError: No module named 'pybullet'"

**Solution:** Reinstall pybullet:

```bash
pip uninstall pybullet -y
pip install pybullet --no-cache-dir
```

### Issue: "Microsoft Visual C++ 14.0 or greater is required" (Windows)

**Solution:** Install Visual C++ Build Tools:
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer and select "Desktop development with C++"
3. Retry pybullet installation

### Issue: Permission denied during pip install

**Solution:** Use the `--user` flag or upgrade pip:

```bash
pip install --user --upgrade pip setuptools wheel
```

### Issue: Out of disk space

**Solution:** PyBullet can be large (~500MB+). Free up disk space or use:

```bash
pip install pybullet --no-cache-dir
```

## Development Setup

For development with code quality tools:

```bash
# Install dev dependencies
pip install black flake8 pylint mypy pytest pytest-cov

# Format code
black . --line-length=100

# Run tests
pytest tests/ -v

# Check type hints
mypy environment/walking-spider/walking_spider/envs/
```

## Project Structure

```
.
├── pyproject.toml              # Poetry configuration
├── requirements.txt            # Pip requirements
├── walking_spider.yml          # Conda environment
├── SETUP_GUIDE.md             # This file
├── README.md                  # Project documentation
├── run_test_with_full_debug.py # Comprehensive test script
├── environment/               # Custom gym environment
│   └── walking-spider/
├── logs/
│   └── debug/                # Debug logs (generated)
├── videos/                   # GIF snapshots (generated)
└── tests/                    # Test files
```

## Next Steps

1. **Setup Environment:** Follow one of the three options above
2. **Verify Installation:** Run `python run_test_with_full_debug.py`
3. **Check Logs:** Review generated logs in `logs/debug/`
4. **View Results:** Check `videos/` for GIF snapshots
5. **Analyze:** Use `python analyze_spider_logs.py --summary`

## Environment Specifications

### Minimum Requirements
- Python 3.8+
- 2GB RAM (for training with logging)
- 1GB disk space (for dependencies + generated files)

### Recommended Specifications
- Python 3.10+
- 8GB+ RAM (for faster training)
- 5GB+ disk space (for large models and logs)
- GPU with CUDA support (for TensorFlow/PyTorch, optional)

## Upgrading Dependencies

### Update all packages

```bash
# Using pip
pip install --upgrade -r requirements.txt

# Using Poetry
poetry update
```

### Update specific package

```bash
# Using pip
pip install --upgrade pybullet gym

# Using Poetry
poetry add pybullet@latest gym@latest
```

## Uninstalling

### Remove virtual environment

**Windows:**
```powershell
Remove-Item .venv -Recurse -Force
```

**macOS/Linux:**
```bash
rm -rf .venv
```

### Remove Poetry environment

```bash
poetry env remove walking-spider
```

## Getting Help

- **Documentation:** See `README.md`
- **Logging Guide:** See `logs/LOGGING_GUIDE.txt`
- **API Documentation:** Run `pydoc environment.walking_spider.envs.walking_spider_env`
- **Issues:** Check GitHub issues or project documentation

## Additional Resources

- [PyBullet Documentation](https://pybullet.org)
- [OpenAI Gym Documentation](https://gym.openai.com)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

**Last Updated:** October 2025
**Status:** Installation Guide Complete ✅
