#!/bin/bash
# ==============================================================================
# BBoxMaskPose Docker Entrypoint Script
# ==============================================================================
#
# This script performs initialization tasks before running the main application:
#   1. Validates GPU availability
#   2. Downloads SAM2 weights if not present
#   3. Displays usage information when no arguments provided
#
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SAM_WEIGHTS="/app/models/SAM/sam2.1_hiera_base_plus.pt"
SAM_DOWNLOAD_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

# ------------------------------------------------------------------------------
# GPU Validation
# ------------------------------------------------------------------------------
echo "============================================"
echo "  BBoxMaskPose - Container Initialization"
echo "============================================"
echo ""

echo "[INFO] Checking GPU availability..."
GPU_CHECK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$GPU_CHECK" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "[OK] GPU detected: ${GPU_NAME}"
else
    echo "[WARNING] No GPU detected. Running on CPU will be significantly slower."
    echo "         Ensure nvidia-container-toolkit is installed and --gpus all is passed."
fi
echo ""

# ------------------------------------------------------------------------------
# SAM2 Weights Check
# ------------------------------------------------------------------------------
echo "[INFO] Checking SAM2 model weights..."
if [ ! -f "$SAM_WEIGHTS" ]; then
    echo "[INFO] Downloading SAM2 weights (approximately 309MB)..."
    mkdir -p /app/models/SAM
    wget -q --show-progress -O "$SAM_WEIGHTS" "$SAM_DOWNLOAD_URL"
    echo "[OK] SAM2 weights downloaded successfully."
else
    echo "[OK] SAM2 weights found."
fi
echo ""

# ------------------------------------------------------------------------------
# Usage Information
# ------------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "============================================"
    echo "  Usage Examples"
    echo "============================================"
    echo ""
    echo "Run on a single image:"
    echo "  docker run --gpus all -v \$(pwd)/images:/app/input -v \$(pwd)/outputs:/app/outputs \\"
    echo "    bboxmaskpose:latest python demo/bmp_demo.py configs/bmp_D3.yaml /app/input/image.jpg"
    echo ""
    echo "Run with sample image:"
    echo "  docker run --gpus all bboxmaskpose:latest python demo/bmp_demo.py \\"
    echo "    configs/bmp_D3.yaml demo/data/004806.jpg"
    echo ""
    echo "Interactive shell:"
    echo "  docker run --gpus all -it bboxmaskpose:latest bash"
    echo ""
    echo "Generate GIF animation:"
    echo "  docker run --gpus all bboxmaskpose:latest python demo/bmp_demo.py \\"
    echo "    configs/bmp_D3.yaml demo/data/004806.jpg --create-gif"
    echo ""
    exit 0
fi

# ------------------------------------------------------------------------------
# Execute Command
# ------------------------------------------------------------------------------
exec "$@"
