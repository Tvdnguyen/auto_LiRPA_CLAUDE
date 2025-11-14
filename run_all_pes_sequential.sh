#!/bin/bash
#
# Run all PEs sequentially (one at a time) - Safest for low memory
#

DATA_DIR="gtsrb_project/data/GTSRB_data"
CHECKPOINT="gtsrb_project/checkpoints/traffic_sign_net_full.pth"
ARRAY_SIZE=8
CLASS_ID=0
TEST_IDX=0

echo "========================================================================"
echo "  SEQUENTIAL PE SENSITIVITY ANALYSIS"
echo "========================================================================"
echo "Array size: ${ARRAY_SIZE}x${ARRAY_SIZE}"
echo "Total PEs: $((ARRAY_SIZE * ARRAY_SIZE))"
echo ""
echo "This will test each PE one at a time to minimize RAM usage."
echo "Each PE result will be saved separately."
echo ""
read -p "Press Enter to start..."

# Create results directory
RESULTS_DIR="pe_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Counter
count=0
total=$((ARRAY_SIZE * ARRAY_SIZE))

# Loop through all PEs
for pe_row in $(seq 0 $((ARRAY_SIZE - 1))); do
    for pe_col in $(seq 0 $((ARRAY_SIZE - 1))); do
        count=$((count + 1))

        echo ""
        echo "========================================================================"
        echo "[$count/$total] Testing PE($pe_row,$pe_col)"
        echo "========================================================================"

        # Run single PE test
        python3 test_single_pe.py \
            --data_dir "$DATA_DIR" \
            --checkpoint "$CHECKPOINT" \
            --pe_row $pe_row \
            --pe_col $pe_col \
            --array_size $ARRAY_SIZE \
            --class_id $CLASS_ID \
            --test_idx $TEST_IDX \
            --duration 1 \
            --epsilon_max 0.3 \
            --tolerance 0.05

        # Check exit code
        if [ $? -eq 0 ]; then
            # Move result file to results directory
            mv pe_${pe_row}_${pe_col}_result_*.csv "$RESULTS_DIR/" 2>/dev/null
            echo "✓ PE($pe_row,$pe_col) completed"
        else
            echo "✗ PE($pe_row,$pe_col) failed"
        fi

        # Small delay between tests
        sleep 2
    done
done

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE"
echo "========================================================================"
echo "Results saved in: $RESULTS_DIR"
echo ""

# Combine all results into one CSV
echo "Combining results..."
COMBINED_FILE="$RESULTS_DIR/combined_results.csv"

# Write header
echo "pe_row,pe_col,max_epsilon,num_affected_channels,num_affected_spatial,global_idx,class_idx,class_id,true_label,pred_class,array_size" > "$COMBINED_FILE"

# Append all results (skip headers)
for file in "$RESULTS_DIR"/pe_*_*_result_*.csv; do
    if [ -f "$file" ]; then
        tail -n +2 "$file" >> "$COMBINED_FILE"
    fi
done

echo "✓ Combined results saved to: $COMBINED_FILE"
echo ""
echo "Done!"
