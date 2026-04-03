#!/bin/bash
# Athena AI Complete Installation Script
# Run this on the RunPod via SSH

set -e

echo "=========================================="
echo "🚀 ATHENA AI - COMPLETE INSTALLATION"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE="/workspace"
REPO_URL="https://github.com/anzugtp-dot/athena-ai-public"
RAW_URL="https://raw.githubusercontent.com/anzugtp-dot/athena-ai-public/main"

echo -e "${GREEN}📦 Step 1: Creating workspace...${NC}"
mkdir -p ${WORKSPACE}/{datasets/athena,scripts,config,logs,models,checkpoints}

echo -e "${GREEN}📥 Step 2: Downloading training files...${NC}"

# Dataset (320 philosophical examples)
echo "  Downloading dataset..."
curl -s -L "${RAW_URL}/datasets/athena_training_completo.jsonl" -o ${WORKSPACE}/datasets/athena/athena_training_completo.jsonl

# Scripts
echo "  Downloading scripts..."
curl -s -L "${RAW_URL}/scripts/train_athena.py" -o ${WORKSPACE}/scripts/train_athena.py
curl -s -L "${RAW_URL}/scripts/setup_athena_training.py" -o ${WORKSPACE}/scripts/setup_athena_training.py

# Config
echo "  Downloading config..."
curl -s -L "${RAW_URL}/config/training-athena.yaml" -o ${WORKSPACE}/config/training-athena.yaml

# Requirements
echo "  Downloading requirements..."
curl -s -L "${RAW_URL}/requirements.txt" -o ${WORKSPACE}/requirements.txt

echo -e "${GREEN}✅ Files downloaded successfully!${NC}"

# Verify files
echo -e "${YELLOW}📊 File verification:${NC}"
ls -la ${WORKSPACE}/datasets/athena/athena_training_completo.jsonl
ls -la ${WORKSPACE}/scripts/train_athena.py
ls -la ${WORKSPACE}/config/training-athena.yaml

# Check GPU
echo -e "${GREEN}🎯 Step 3: Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${RED}❌ No NVIDIA GPU detected${NC}"
    echo "Trying to install NVIDIA drivers..."
    apt-get update && apt-get install -y nvidia-driver-535 2>/dev/null || echo "Driver installation may require reboot"
fi

# Install Python/pip if missing
echo -e "${GREEN}🐍 Step 4: Setting up Python environment...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo "Installing Python/pip..."
    apt-get update && apt-get install -y python3-pip python3-venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ${WORKSPACE}/venv
source ${WORKSPACE}/venv/bin/activate

# Install dependencies
echo -e "${GREEN}📦 Step 5: Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r ${WORKSPACE}/requirements.txt

echo -e "${GREEN}🔧 Step 6: Running setup...${NC}"
cd ${WORKSPACE}
python scripts/setup_athena_training.py

echo -e "${GREEN}🚀 Step 7: Starting Athena training...${NC}"
echo -e "${YELLOW}⚠️  Training will take 12-16 hours${NC}"
echo -e "${YELLOW}   Estimated cost: €7.58-€9.21${NC}"

# Start training in background
nohup python scripts/train_athena.py > ${WORKSPACE}/logs/training_athena.log 2>&1 &

# Get training process ID
TRAINING_PID=$!
echo -e "${GREEN}✅ Training started with PID: ${TRAINING_PID}${NC}"

echo ""
echo "=========================================="
echo "🎉 ATHENA AI TRAINING STARTED!"
echo "=========================================="
echo ""
echo -e "${GREEN}📊 Monitoring:${NC}"
echo "  Logs:        tail -f ${WORKSPACE}/logs/training_athena.log"
echo "  GPU usage:   nvidia-smi"
echo "  Checkpoints: ${WORKSPACE}/checkpoints/"
echo "  Final model: ${WORKSPACE}/models/athena_finetuned/"
echo ""
echo -e "${YELLOW}⏰ Estimated completion:${NC}"
echo "  Start:   $(date)"
echo "  End:     ~$(date -d '+14 hours')"
echo ""
echo -e "${YELLOW}💰 Cost estimate:${NC}"
echo "  GPU:     RTX 4090 Secure ($0.59/hour)"
echo "  Time:    14 hours"
echo "  Total:   ~$8.26 (€7.58)"
echo ""
echo "=========================================="
echo -e "${GREEN}🤖 Athena is now learning philosophy!${NC}"
echo "=========================================="

# Show initial log output
echo ""
echo -e "${YELLOW}📝 Initial training log (first 10 lines):${NC}"
sleep 2
tail -n 10 ${WORKSPACE}/logs/training_athena.log 2>/dev/null || echo "Log file not yet created, check in 30 seconds..."

# Keep script running
echo ""
echo -e "${YELLOW}🔄 To exit and keep training running:${NC}"
echo "  Press Ctrl+C"
echo "  Training will continue in background"
echo ""
echo -e "${YELLOW}🛑 To stop training:${NC}"
echo "  kill $TRAINING_PID"

# Wait a bit to show logs
sleep 5