@echo off
REM WalkingSpider Environment Setup Script for Windows (Conda-based)
REM
REM This script sets up a complete Conda environment for the Walking Spider project.
REM Conda is recommended on Windows because it has pre-built PyBullet wheels.
REM
REM Usage:
REM   setup_environment_conda.bat [environment_name]
REM
REM Example:
REM   setup_environment_conda.bat walking_spider_env
REM

setlocal enabledelayedexpansion

REM Configuration
set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=walking_spider_env
set PYTHON_VERSION=3.10

echo.
echo ==========================================
echo   WalkingSpider Conda Environment Setup
echo ==========================================
echo.
echo Environment Name: %ENV_NAME%
echo Python Version: %PYTHON_VERSION%
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] ERROR: Conda is not installed or not in PATH
    echo.
    echo Please install Miniconda or Anaconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('conda --version') do set CONDA_VERSION=%%i
echo [OK] Conda found: %CONDA_VERSION%
echo.

REM Create environment
echo Creating conda environment: %ENV_NAME% ...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

if %errorlevel% neq 0 (
    echo [X] Failed to create conda environment
    pause
    exit /b 1
)

REM Activate environment
echo.
echo Activating environment...
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo [X] Failed to activate environment
    echo Try manually: conda activate %ENV_NAME%
    pause
    exit /b 1
)

echo [OK] Environment activated
echo.

REM Install PyBullet from conda-forge
echo Installing PyBullet from conda-forge (pre-built wheels)...
call conda install -c conda-forge pybullet -y

if %errorlevel% neq 0 (
    echo [!] Warning: PyBullet installation had issues, attempting pip fallback...
    call pip install pybullet
)

echo [OK] PyBullet installed
echo.

REM Install other dependencies
echo Installing core dependencies...
call pip install gym numpy imageio pillow matplotlib

echo.
echo Installing optional ML/RL dependencies...
call pip install tensorflow-cpu torch torchvision torchaudio stable-baselines3

echo.
echo ==========================================
echo   Installation Complete!
echo ==========================================
echo.
echo To activate the environment in the future:
echo   conda activate %ENV_NAME%
echo.
echo To run tests:
echo   python test_pybullet_simple.py
echo   python run_test_with_full_debug.py
echo.
echo To deactivate:
echo   conda deactivate
echo.
pause
