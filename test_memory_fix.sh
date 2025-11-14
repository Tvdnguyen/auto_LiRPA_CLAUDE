#!/bin/bash
#
# Quick test to verify the memory fix works
#

echo "========================================================================"
echo "  TESTING MEMORY FIX"
echo "========================================================================"
echo ""
echo "This will test a single PE with the fixed efficient perturbation code."
echo "Expected: Should complete without being killed on 8GB RAM"
echo ""

# Test PE (0,0) - corner PE, typically has medium impact
python3 test_single_pe.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --pe_row 0 \
    --pe_col 0 \
    --array_size 8 \
    --class_id 0 \
    --test_idx 0 \
    --duration 1 \
    --epsilon_max 0.3 \
    --tolerance 0.05

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ SUCCESS - Memory fix verified!"
    echo "========================================================================"
    echo ""
    echo "The program completed without being killed."
    echo "You can now run the full analysis with run_all_pes_sequential.sh"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ FAILED - Still having issues"
    echo "========================================================================"
    echo ""
    echo "Please check:"
    echo "1. Is the checkpoint file valid?"
    echo "2. Is the data directory correct?"
    echo "3. Close all other applications to free RAM"
    echo ""
fi
