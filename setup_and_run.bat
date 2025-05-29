@echo off
REM DORA Controls Analyzer - Windows Setup Script
REM This script runs the automated setup and analysis

echo Starting DORA Controls Analyzer Setup...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Run the setup script
python setup_and_run.py

echo.
echo Setup and analysis complete!
echo Check the log file: setup_and_run.log
pause 
