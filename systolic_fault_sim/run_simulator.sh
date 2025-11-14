#!/bin/bash
# Runner script for Systolic Fault Simulator

echo "========================================================================"
echo "           Systolic Array Fault Simulator"
echo "               (Based on SCALE-Sim)"
echo "========================================================================"
echo ""

# Activate virtual environment from gtsrb_project
VENV_PATH="../gtsrb_project/gtsrb_env"

if [ -d "$VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "[WARNING] Virtual environment not found at $VENV_PATH"
    echo "[WARNING] Using system Python"
fi

# Run simulator
python fault_simulator.py

echo ""
echo "========================================================================"
echo "Simulation completed!"
echo "========================================================================"
