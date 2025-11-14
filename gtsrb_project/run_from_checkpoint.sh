#!/bin/bash
# Script to run pipeline from existing checkpoint (skip training)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_status() {
    if [ $? -eq 0 ]; then
        print_info "$1 completed successfully"
    else
        print_error "$1 failed"
        exit 1
    fi
}

echo "========================================================================"
echo "      GTSRB Pipeline Runner (From Existing Checkpoint)"
echo "========================================================================"
echo ""

# Check arguments
if [ $# -lt 2 ]; then
    print_error "Usage: $0 <path_to_gtsrb_data> <checkpoint_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  path_to_gtsrb_data    Path to GTSRB dataset"
    echo "  checkpoint_path       Path to trained model checkpoint (.pth file)"
    echo ""
    echo "Options:"
    echo "  --model [full|simple]    Model architecture (default: full)"
    echo "  --batch-size N           Batch size (default: 128)"
    echo "  --device [cuda|cpu]      Device to use (default: cuda)"
    echo "  --skip-collection        Skip sample collection step"
    echo "  --skip-interactive       Skip interactive testing"
    echo ""
    echo "Examples:"
    echo "  # Run full pipeline from checkpoint"
    echo "  $0 data/GTSRB_data checkpoints/traffic_sign_net.pth"
    echo ""
    echo "  # Skip collection, go directly to interactive"
    echo "  $0 data/GTSRB_data checkpoints/traffic_sign_net.pth --skip-collection"
    echo ""
    exit 1
fi

# Parse arguments
DATA_DIR=$1
CHECKPOINT_PATH=$2
shift 2

# Default values
MODEL="full"
BATCH_SIZE=128
DEVICE="cuda"
SKIP_COLLECTION=false
SKIP_INTERACTIVE=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
            shift
            ;;
        --skip-interactive)
            SKIP_INTERACTIVE=true
            shift
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Print configuration
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Model: $MODEL"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    print_error "Checkpoint file not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found in checkpoints/"
    exit 1
fi

print_info "Checkpoint found: $CHECKPOINT_PATH"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR/Train" ] || [ ! -d "$DATA_DIR/Test" ]; then
    print_error "Invalid GTSRB directory structure"
    exit 1
fi

print_info "Data directory verified"

# Create necessary directories
mkdir -p correct_samples
mkdir -p logs

# Define log file
LOG_FILE="logs/run_from_checkpoint_$(date +%Y%m%d_%H%M%S).log"

# Start logging
exec > >(tee -a "$LOG_FILE")
exec 2>&1

print_info "Logging to: $LOG_FILE"

# Step 1: Collect correct samples (if not skipped)
if [ "$SKIP_COLLECTION" = false ]; then
    echo ""
    echo "========================================================================"
    echo "Step 1: Collecting Correctly Classified Samples"
    echo "========================================================================"
    print_info "Running inference on test set..."

    python collect_correct_samples.py \
        --data_dir "$DATA_DIR" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model "$MODEL" \
        --output_dir correct_samples \
        --batch_size "$BATCH_SIZE"

    check_status "Sample collection"

    # Check if CSV files were created
    CSV_COUNT=$(ls correct_samples/class_*_correct_indices.csv 2>/dev/null | wc -l)
    if [ "$CSV_COUNT" -lt 43 ]; then
        print_warning "Expected 43 CSV files, found $CSV_COUNT"
    else
        print_info "Successfully created $CSV_COUNT CSV files"
    fi
else
    print_warning "Skipping sample collection step"

    # Check if CSV files exist
    if [ ! -d "correct_samples" ] || [ -z "$(ls -A correct_samples/*.csv 2>/dev/null)" ]; then
        print_error "correct_samples/ directory is empty but collection was skipped"
        echo "Please run collection first or remove --skip-collection flag"
        exit 1
    fi
    print_info "Using existing samples in correct_samples/"
fi

# Step 2: Interactive testing (if not skipped)
if [ "$SKIP_INTERACTIVE" = false ]; then
    echo ""
    echo "========================================================================"
    echo "Step 2: Interactive Testing"
    echo "========================================================================"
    print_info "All preparation steps completed successfully!"
    echo ""
    print_info "Starting interactive testing mode..."
    echo ""

    python main_interactive.py \
        --data_dir "$DATA_DIR" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model "$MODEL" \
        --correct_samples_dir correct_samples \
        --device "$DEVICE"
else
    print_warning "Skipping interactive testing"
    echo ""
    print_info "You can run interactive testing later with:"
    echo ""
    echo "  python main_interactive.py \\"
    echo "      --data_dir $DATA_DIR \\"
    echo "      --checkpoint $CHECKPOINT_PATH \\"
    echo "      --model $MODEL \\"
    echo "      --device $DEVICE"
fi

echo ""
echo "========================================================================"
echo "Pipeline Completed Successfully!"
echo "========================================================================"
print_info "Summary:"
echo "  - Checkpoint: $CHECKPOINT_PATH"
echo "  - Correct samples: correct_samples/"
echo "  - Log file: $LOG_FILE"
echo ""
print_info "Done!"
