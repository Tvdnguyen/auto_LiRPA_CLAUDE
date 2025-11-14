#!/bin/bash
# Script to run Systolic Array Fault Simulator

echo "========================================================================"
echo "            Systolic Array Fault Simulator"
echo "========================================================================"
echo ""

# Activate virtual environment if exists
if [ -d "gtsrb_env" ]; then
    echo "[INFO] Activating virtual environment..."
    source gtsrb_env/bin/activate
fi

# Run simulator in interactive mode
python systolic_array_fault_simulator.py --interactive
