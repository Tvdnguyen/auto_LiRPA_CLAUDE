#!/bin/bash

# Script to run integrated PE sensitivity analysis with correct paths

# Set correct paths
DATA_DIR="gtsrb_project/data/GTSRB_data"
CHECKPOINT="gtsrb_project/checkpoints/traffic_sign_net_full.pth"

# Run analysis
python3 integrated_pe_sensitivity_analysis.py \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --array_size 8 \
  --sample_idx 243 \
  --duration 10 \
  --epsilon_max 1.0 \
  --tolerance 0.001 \
  --device cpu

echo ""
echo "Analysis complete!"
