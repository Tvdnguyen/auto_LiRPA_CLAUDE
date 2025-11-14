#!/bin/bash
# Script to run batch verification for a class

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "========================================================================"
echo "              GTSRB Batch Verification Runner"
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
    echo "  --device [cuda|cpu]      Device to use (default: cuda)"
    echo ""
    echo "Examples:"
    echo "  # Run batch verification"
    echo "  $0 data/GTSRB_data checkpoints/traffic_sign_net_full.pth"
    echo ""
    exit 1
fi

# Parse arguments
DATA_DIR=$1
CHECKPOINT_PATH=$2
shift 2

# Default values
MODEL="full"
DEVICE="cuda"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            shift
            ;;
    esac
done

# Print configuration
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    print_error "Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if correct_samples directory exists
if [ ! -d "correct_samples" ]; then
    print_error "correct_samples/ directory not found"
    echo "Please run collect_correct_samples.py first"
    exit 1
fi

# Create output directory
mkdir -p verification_results

print_info "Starting batch verification..."
echo ""

# Activate virtual environment if exists
if [ -d "gtsrb_env" ]; then
    print_info "Activating virtual environment..."
    source gtsrb_env/bin/activate
fi

# Run batch verification
python batch_verification.py \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT_PATH" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --correct_samples_dir correct_samples \
    --output_dir verification_results

if [ $? -eq 0 ]; then
    print_info "Batch verification completed successfully!"
    echo ""
    echo "Results saved in verification_results/"
else
    print_error "Batch verification failed"
    exit 1
fi
