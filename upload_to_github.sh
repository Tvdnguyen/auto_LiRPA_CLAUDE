#!/bin/bash

# Script to upload auto_LiRPA_CLAUDE to GitHub
# Run this script from the project root directory

echo "================================================"
echo "  Upload auto_LiRPA_CLAUDE to GitHub"
echo "================================================"

# Step 1: Initialize git if not already done
if [ ! -d ".git" ]; then
    echo ""
    echo "[Step 1] Initializing git repository..."
    git init
    echo "Git repository initialized."
else
    echo ""
    echo "[Step 1] Git repository already exists."
fi

# Step 2: Add all files
echo ""
echo "[Step 2] Adding files to git..."
git add .
echo "Files added."

# Step 3: Create initial commit
echo ""
echo "[Step 3] Creating commit..."
git commit -m "Initial commit: Auto-LiRPA GTSRB & Systolic Array Fault Analysis

Features:
- GTSRB DNN verification with Auto-LiRPA
- Conv1 sensitivity analysis using binary search
- Systolic array fault simulator (OS/WS/IS dataflows)
- PE position analysis for fault coverage
- Integrated PE-to-DNN sensitivity analysis
- Interactive verification tools"

echo "Commit created."

# Step 4: Instructions for GitHub
echo ""
echo "================================================"
echo "  Next Steps - Create GitHub Repository"
echo "================================================"
echo ""
echo "1. Go to https://github.com/new"
echo "2. Repository name: auto_LiRPA_CLAUDE"
echo "3. Description: Auto-LiRPA GTSRB & Systolic Array Fault Analysis"
echo "4. Choose Public or Private"
echo "5. DO NOT initialize with README (we already have one)"
echo "6. Click 'Create repository'"
echo ""
echo "7. After creating the repository, run these commands:"
echo ""
echo "   git branch -M main"
echo "   git remote add origin https://github.com/YOUR_USERNAME/auto_LiRPA_CLAUDE.git"
echo "   git push -u origin main"
echo ""
echo "Replace YOUR_USERNAME with your GitHub username"
echo ""
echo "================================================"
