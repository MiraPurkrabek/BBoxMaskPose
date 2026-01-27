# ==============================================================================
# BBoxMaskPose Dockerfile
# Multi-body Detection, Pose Estimation and Segmentation
# ICCV 2025
# 
# Repository: https://github.com/MiraPurkrabek/BBoxMaskPose
# Paper: https://arxiv.org/abs/2412.01562
# ==============================================================================

# ------------------------------------------------------------------------------
# Base Image: PyTorch with CUDA 12.1 and cuDNN 9
# ------------------------------------------------------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

WORKDIR /app

# ------------------------------------------------------------------------------
# System Dependencies
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ------------------------------------------------------------------------------
# Python Package Installation
# ------------------------------------------------------------------------------

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install OpenMMLab ecosystem
# Note: mmcv 2.2.0 is used with pre-built CUDA extensions for faster installation
RUN pip install --no-cache-dir openmim && \
    mim install mmengine

# Install mmcv with pre-built CUDA extensions
# Source: https://download.openmmlab.com/mmcv/dist/
RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# Install detection and pose estimation frameworks
RUN pip install --no-cache-dir mmdet==3.2.0 mmpretrain==1.2.0

# Patch mmdet version check to allow mmcv 2.2.0
# mmdet 3.2.0 requires mmcv<2.2.0 but 2.2.0 is fully compatible
RUN sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    /opt/conda/lib/python3.11/site-packages/mmdet/__init__.py

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    json_tricks \
    munkres \
    scipy \
    tqdm \
    hydra-core \
    iopath \
    platformdirs \
    sparsemax \
    loguru \
    pycocotools

# Install xtcocotools from source for extended COCO API functionality
RUN pip install --no-cache-dir git+https://github.com/jin-s13/xtcocoapi.git

# ------------------------------------------------------------------------------
# Project Installation
# ------------------------------------------------------------------------------

# Copy project files
COPY . /app/

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Create required directories
RUN mkdir -p /app/models/SAM /app/outputs

# ------------------------------------------------------------------------------
# Model Weights
# ------------------------------------------------------------------------------

# Download SAM 2.1 weights (Hiera Base Plus model, approximately 309MB)
# These weights are required for the segmentation component
RUN wget -q --show-progress -O /app/models/SAM/sam2.1_hiera_base_plus.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Note: RTMDet and MaskPose weights are downloaded automatically during runtime
# from HuggingFace: https://huggingface.co/vrg-prague/BBoxMaskPose/

# ------------------------------------------------------------------------------
# Security: Non-root User
# ------------------------------------------------------------------------------
RUN useradd -m -u 1000 bmpuser && \
    chown -R bmpuser:bmpuser /app
USER bmpuser

# ------------------------------------------------------------------------------
# Container Configuration
# ------------------------------------------------------------------------------
# Default command - can be overridden via docker-compose or docker run
CMD ["python", "demo/bmp_demo.py", "--help"]

# ------------------------------------------------------------------------------
# Metadata Labels
# ------------------------------------------------------------------------------
LABEL maintainer="Harsh Tomar"
LABEL description="BBoxMaskPose: Multi-body Detection, Pose Estimation and Segmentation"
LABEL version="1.1.0"
LABEL org.opencontainers.image.source="https://github.com/MiraPurkrabek/BBoxMaskPose"
LABEL org.opencontainers.image.documentation="https://mirapurkrabek.github.io/BBox-Mask-Pose/"
LABEL org.opencontainers.image.licenses="GPL-3.0"
