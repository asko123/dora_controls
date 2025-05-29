#!/bin/bash
# DORA Controls Analyzer - Unix/Linux/macOS Setup Script
# This script runs the automated setup and analysis

echo "Starting DORA Controls Analyzer Setup..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.7+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info.major, sys.version_info.minor)")
MAJOR=$(echo $PYTHON_VERSION | cut -d' ' -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d' ' -f2)

if [ "$MAJOR" -lt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -lt 7 ]; then
    echo "Error: Python 3.7+ required. Found Python $MAJOR.$MINOR"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo

# Run the setup script
$PYTHON_CMD setup_and_run.py

echo
echo "Setup and analysis complete!"
echo "Check the log file: setup_and_run.log"

# Keep terminal open on some systems
if [ -t 0 ]; then
    read -p "Press Enter to continue..."
fi 
