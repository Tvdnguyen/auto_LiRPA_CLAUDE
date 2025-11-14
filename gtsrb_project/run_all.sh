#!/bin/bash
# Automated script to run the complete GTSRB pipeline

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        print_info "$1 completed successfully"
    else
        print_error "$1 failed"
        exit 1
    fi
}

echo "========================================================================"
echo "         GTSRB Intermediate Perturbation Pipeline Runner"
echo "========================================================================"
echo ""

# Check for required argument
if [ $# -lt 1 ]; then
    print_error "Usage: $0 <path_to_gtsrb_data> [options]"
    echo ""
    echo "Options:"
    echo "  --model [full|simple]    Model architecture (default: full)"
    echo "  --epochs N               Number of epochs (default: 50)"
    echo "  --batch-size N          Batch size (default: 128)"
    echo "  --device [cuda|cpu]     Device to use (default: cuda)"
    echo "  --skip-training         Skip training step"
    echo "  --skip-collection       Skip sample collection step"
    echo ""
    echo "Example:"
    echo "  $0 ~/Documents/GTSRB_data --model full --epochs 50"
    exit 1
fi

# Parse arguments
DATA_DIR=$1
shift

# Default values
MODEL="full"
EPOCHS=50
BATCH_SIZE=128
DEVICE="cuda"
SKIP_TRAINING=false
SKIP_COLLECTION=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
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
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-collection)
            SKIP_COLLECTION=true
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
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR/Train" ] || [ ! -d "$DATA_DIR/Test" ]; then
    print_error "Invalid GTSRB directory structure. Expected Train/ and Test/ subdirectories"
    echo ""
    echo "Current structure:"
    ls -la "$DATA_DIR" 2>/dev/null || echo "Directory not found: $DATA_DIR"
    echo ""
    echo "Please ensure:"
    echo "  1. Dataset is downloaded and extracted"
    echo "  2. Directory structure is:"
    echo "     $DATA_DIR/"
    echo "     ├── Train/"
    echo "     │   ├── 00000/"
    echo "     │   └── ... (43 classes)"
    echo "     └── Test/"
    echo "         ├── GT-final_test.csv"
    echo "         └── Images/*.ppm  (or *.ppm directly in Test/)"
    echo ""
    echo "Try running: bash setup_dataset.sh"
    exit 1
fi

# Additional check for test images
if [ ! -f "$DATA_DIR/Test/GT-final_test.csv" ]; then
    print_error "GT-final_test.csv not found in Test directory"
    echo "Please ensure Test/GT-final_test.csv exists"
    exit 1
fi

# Check if test images exist (either in Test/ or Test/Images/)
TEST_IMG_COUNT=$(ls "$DATA_DIR/Test"/*.ppm 2>/dev/null | wc -l)
TEST_IMG_SUBDIR_COUNT=$(ls "$DATA_DIR/Test/Images"/*.ppm 2>/dev/null | wc -l)

if [ "$TEST_IMG_COUNT" -eq 0 ] && [ "$TEST_IMG_SUBDIR_COUNT" -eq 0 ]; then
    print_error "No test images found"
    echo "Test images should be in either:"
    echo "  - $DATA_DIR/Test/*.ppm"
    echo "  - $DATA_DIR/Test/Images/*.ppm"
    exit 1
fi

print_info "Data directory verified"
if [ "$TEST_IMG_SUBDIR_COUNT" -gt 0 ]; then
    print_info "Found $TEST_IMG_SUBDIR_COUNT test images in Test/Images/"
else
    print_info "Found $TEST_IMG_COUNT test images in Test/"
fi

# Create necessary directories
print_info "Creating directories..."
mkdir -p checkpoints
mkdir -p correct_samples
mkdir -p logs

# Define paths
CHECKPOINT_PATH="checkpoints/traffic_sign_net_${MODEL}.pth"
LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"

# Start logging
exec > >(tee -a "$LOG_FILE")
exec 2>&1

print_info "Logging to: $LOG_FILE"

# Step 0: Test installation
echo ""
echo "========================================================================"
echo "Step 0: Testing Installation"
echo "========================================================================"
print_info "Running installation tests..."

python test_installation.py "$DATA_DIR"
check_status "Installation test"

# Step 1: Training
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "========================================================================"
    echo "Step 1: Training Model"
    echo "========================================================================"
    print_info "Training ${MODEL} model for ${EPOCHS} epochs..."
    print_info "This may take 15-30 minutes with GPU, or 2-4 hours with CPU"

    python train_gtsrb.py \
        --data_dir "$DATA_DIR" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --save_path "$CHECKPOINT_PATH"

    check_status "Model training"

    # Check if checkpoint was created
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        print_error "Checkpoint file not found: $CHECKPOINT_PATH"
        exit 1
    fi

    print_info "Model checkpoint saved to: $CHECKPOINT_PATH"
else
    print_warning "Skipping training step"

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        print_error "Checkpoint file not found: $CHECKPOINT_PATH"
        print_error "Cannot skip training without existing checkpoint"
        exit 1
    fi
fi

# Step 2: Collect correct samples
if [ "$SKIP_COLLECTION" = false ]; then
    echo ""
    echo "========================================================================"
    echo "Step 2: Collecting Correctly Classified Samples"
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
fi

# Step 3: Interactive testing
echo ""
echo "========================================================================"
echo "Step 3: Interactive Testing"
echo "========================================================================"
print_info "All preparation steps completed successfully!"
echo ""
print_info "You can now run interactive testing with:"
echo ""
echo "  python main_interactive.py \\"
echo "      --data_dir $DATA_DIR \\"
echo "      --checkpoint $CHECKPOINT_PATH \\"
echo "      --model $MODEL \\"
echo "      --device $DEVICE"
echo ""

# Ask user if they want to run interactive mode now
read -p "Do you want to start interactive testing now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting interactive testing..."
    python main_interactive.py \
        --data_dir "$DATA_DIR" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model "$MODEL" \
        --correct_samples_dir correct_samples \
        --device "$DEVICE"
else
    print_info "You can run interactive testing later with the command above"
fi

echo ""
echo "========================================================================"
echo "Pipeline Completed Successfully!"
echo "========================================================================"
print_info "Summary:"
echo "  - Model checkpoint: $CHECKPOINT_PATH"
echo "  - Correct samples: correct_samples/"
echo "  - Log file: $LOG_FILE"
echo ""
print_info "Next steps:"
echo "  1. Run: python main_interactive.py --data_dir $DATA_DIR --checkpoint $CHECKPOINT_PATH --model $MODEL"
echo "  2. Select a layer to perturb"
echo "  3. Configure perturbation region and epsilon"
echo "  4. Analyze bounds and robustness results"
echo ""
print_info "For help, see README.md or SETUP_GUIDE.md"
