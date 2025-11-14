#!/bin/bash
# Script to setup GTSRB dataset structure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "========================================================================"
echo "           GTSRB Dataset Setup Script"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "gtsrb_dataset.py" ]; then
    print_error "Please run this script from gtsrb_project directory"
    exit 1
fi

# Create directory structure
print_info "Creating directory structure..."
mkdir -p data/GTSRB_data/Train
mkdir -p data/GTSRB_data/Test
mkdir -p data/GTSRB_data/downloads

cd data/GTSRB_data/downloads

# Check if zip files exist
TRAINING_ZIP="GTSRB-Training_fixed.zip"
TEST_IMAGES_ZIP="GTSRB_Final_Test_Images.zip"
TEST_GT_ZIP="GTSRB_Final_Test_GT.zip"

print_info "Checking for zip files..."

if [ ! -f "$TRAINING_ZIP" ] && [ ! -f "$TEST_IMAGES_ZIP" ] && [ ! -f "$TEST_GT_ZIP" ]; then
    print_warning "No zip files found in downloads/"
    echo ""
    echo "Please download the following files and place them in:"
    echo "  data/GTSRB_data/downloads/"
    echo ""
    echo "Download links:"
    echo "  1. Training set:"
    echo "     https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip"
    echo ""
    echo "  2. Test images:"
    echo "     https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    echo ""
    echo "  3. Test GT:"
    echo "     https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    echo ""
    echo "After downloading, run this script again."
    exit 1
fi

# Unzip Training data
if [ -f "$TRAINING_ZIP" ]; then
    print_info "Extracting training data..."
    unzip -q "$TRAINING_ZIP"

    # Find Train directory and move it
    if [ -d "GTSRB/Train" ]; then
        print_info "Moving training data to Train/..."
        mv GTSRB/Train/* ../Train/
    elif [ -d "Train" ]; then
        mv Train/* ../Train/
    else
        print_error "Cannot find Train directory after extraction"
        exit 1
    fi

    # Check if we have 43 classes
    TRAIN_CLASSES=$(ls -d ../Train/*/ 2>/dev/null | wc -l)
    print_info "Found $TRAIN_CLASSES training classes"

    if [ "$TRAIN_CLASSES" -ne 43 ]; then
        print_warning "Expected 43 classes, found $TRAIN_CLASSES"
    fi
else
    print_warning "Training zip not found, skipping..."
fi

# Unzip Test images
if [ -f "$TEST_IMAGES_ZIP" ]; then
    print_info "Extracting test images..."
    unzip -q "$TEST_IMAGES_ZIP"

    # Find Images directory and move to Test/
    print_info "Setting up test images structure..."

    if [ -d "GTSRB/Final_Test/Images" ]; then
        # Move entire Images folder to Test/
        mv GTSRB/Final_Test/Images ../Test/
    elif [ -d "Images" ]; then
        # Move Images folder to Test/
        mv Images ../Test/
    else
        # Search for .ppm files recursively and create Images folder
        mkdir -p ../Test/Images
        find . -name "*.ppm" -exec mv {} ../Test/Images/ \;
    fi

    # Check number of test images
    TEST_COUNT=$(ls ../Test/Images/*.ppm 2>/dev/null | wc -l)
    print_info "Found $TEST_COUNT test images in Test/Images/"

    if [ "$TEST_COUNT" -ne 12630 ]; then
        print_warning "Expected 12630 test images, found $TEST_COUNT"
    fi
else
    print_warning "Test images zip not found, skipping..."
fi

# Unzip Test GT
if [ -f "$TEST_GT_ZIP" ]; then
    print_info "Extracting test GT..."
    unzip -q "$TEST_GT_ZIP"

    # Find GT CSV and move to Test/
    if [ -f "GT-final_test.csv" ]; then
        print_info "Moving GT-final_test.csv to Test/..."
        mv GT-final_test.csv ../Test/
    else
        find . -name "GT-final_test.csv" -exec mv {} ../Test/ \;
    fi

    if [ ! -f "../Test/GT-final_test.csv" ]; then
        print_error "GT-final_test.csv not found after extraction"
        exit 1
    fi
else
    print_warning "Test GT zip not found, skipping..."
fi

# Clean up
print_info "Cleaning up temporary files..."
cd ..
rm -rf downloads/GTSRB downloads/Images downloads/Train

# Verify structure
print_info "Verifying dataset structure..."
cd ../..

echo ""
echo "========================================================================"
echo "Dataset Structure Verification"
echo "========================================================================"

# Check Train/
if [ -d "data/GTSRB_data/Train" ]; then
    TRAIN_CLASSES=$(ls -d data/GTSRB_data/Train/*/ 2>/dev/null | wc -l)
    echo "✓ Train directory: $TRAIN_CLASSES classes"

    # Sample first class
    FIRST_CLASS=$(ls data/GTSRB_data/Train/ | head -1)
    if [ ! -z "$FIRST_CLASS" ]; then
        SAMPLE_IMAGES=$(ls data/GTSRB_data/Train/$FIRST_CLASS/*.ppm 2>/dev/null | wc -l)
        echo "  Sample class $FIRST_CLASS: $SAMPLE_IMAGES images"
    fi
else
    echo "✗ Train directory not found"
fi

# Check Test/
if [ -d "data/GTSRB_data/Test" ]; then
    # Check for images in Test/Images/ subfolder
    TEST_IMAGES=$(ls data/GTSRB_data/Test/Images/*.ppm 2>/dev/null | wc -l)
    if [ "$TEST_IMAGES" -eq 0 ]; then
        # Fallback: check directly in Test/
        TEST_IMAGES=$(ls data/GTSRB_data/Test/*.ppm 2>/dev/null | wc -l)
        echo "✓ Test directory: $TEST_IMAGES images"
    else
        echo "✓ Test/Images directory: $TEST_IMAGES images"
    fi

    if [ -f "data/GTSRB_data/Test/GT-final_test.csv" ]; then
        TEST_LABELS=$(tail -n +2 data/GTSRB_data/Test/GT-final_test.csv | wc -l)
        echo "✓ GT-final_test.csv: $TEST_LABELS labels"
    else
        echo "✗ GT-final_test.csv not found"
    fi
else
    echo "✗ Test directory not found"
fi

echo ""
echo "========================================================================"
echo "Testing Dataset Loader"
echo "========================================================================"

# Test dataset loader
python gtsrb_dataset.py data/GTSRB_data

if [ $? -eq 0 ]; then
    echo ""
    print_info "Dataset setup completed successfully!"
    echo ""
    echo "You can now run:"
    echo "  bash run_all.sh data/GTSRB_data --model full"
else
    print_error "Dataset loader test failed"
    echo "Please check the error messages above"
fi
