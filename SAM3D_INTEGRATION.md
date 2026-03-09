# SAM-3D-Body Integration Guide

This guide explains how to integrate and use SAM-3D-Body for 3D human mesh recovery within the BBoxMaskPose pipeline.

## Overview

BBoxMaskPose v2 extends the original BMP pipeline with [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) from Meta AI, enabling full 3D human mesh recovery from single images. The integration leverages BMP's high-quality 2D pose estimates and segmentation masks as prompts to SAM-3D-Body, resulting in accurate 3D reconstructions even in crowded scenes.

**Pipeline Flow:**
```
Input Image
    ↓
BBoxMaskPose (Detection + 2D Pose + Segmentation)
    ↓
2D Bboxes + Masks + Poses
    ↓
SAM-3D-Body (3D Mesh Recovery)
    ↓
3D Human Meshes (vertices, joints, faces)
```

## Installation

### Prerequisites

- BBoxMaskPose must be already installed and working
- CUDA-capable GPU recommended (CPU inference is very slow)
- Python 3.8+ (Python 3.11 recommended for SAM-3D-Body)

### Step 1: Install SAM-3D-Body Dependencies

```bash
# Navigate to BBoxMaskPose root directory
cd /path/to/BBoxMaskPose

# Install SAM-3D-Body dependencies
pip install -r requirements/sam3d.txt
```

### Step 2: Install Detectron2

SAM-3D-Body requires a specific version of Detectron2:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps
```

### Step 3: Install MoGe (Optional but Recommended)

MoGe provides FOV (field-of-view) estimation for better camera calibration:

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

### Step 4: Install SAM-3D-Body

```bash
# Install adapted SAM-3D-Body repository
pip install git+https://github.com/MiraPurkrabek/sam-3d-body.git
```

### Step 5: Get Model Checkpoints

SAM-3D-Body checkpoints are hosted on HuggingFace. You need to:

1. **Request access** at [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
2. **Wait for approval** (usually within 24 hours)
3. **Authenticate** with HuggingFace:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

The BMPv2 demo will auto-download the checkpoint on first use, or you can download manually to the default location for auto-detection:

```bash
# Download checkpoint manually to default location (will be auto-detected)
mkdir -p checkpoints
huggingface-cli download facebook/sam-3d-body-dinov3 \
    --local-dir checkpoints/sam-3d-body-dinov3
```

## Usage

### Basic Usage

Run the BMPv2 demo with automatic checkpoint handling:

```bash
python demos/BMPv2_demo.py --image data/004806.jpg --device cuda
```

**The demo will:**
1. Auto-detect checkpoint in `checkpoints/sam-3d-body-dinov3/` OR download from HuggingFace (~3.5 GB)
2. Run BMP pipeline to get 2D detections, poses, and masks
3. Run SAM-3D-Body to recover 3D meshes
4. Save visualizations to `demos/outputs/bboxmaskpose_v2/`

### Advanced Usage

#### Use Local Checkpoint (Auto-Detection)

Download checkpoint to the default location for automatic detection:

```bash
# The demo automatically detects checkpoints in this location
huggingface-cli download facebook/sam-3d-body-dinov3 \
    --local-dir checkpoints/sam-3d-body-dinov3

# Then just run the demo - no checkpoint arguments needed!
python demos/BMPv2_demo.py --image data/004806.jpg --device cuda
```

#### Use Custom Checkpoint Path

If your checkpoint is in a different location:

```bash
python demos/BMPv2_demo.py \
    --image data/004806.jpg \
    --device cuda \
    --sam3d_checkpoint /path/to/model.ckpt \
    --mhr_path /path/to/mhr_model.pt
```

#### Speed vs Quality Trade-offs

```bash
# Fastest: body-only inference without mask conditioning
python demos/BMPv2_demo.py --image data/004806.jpg \
    --inference_type body --no_mask_conditioning

# Balanced: body-only with mask conditioning
python demos/BMPv2_demo.py --image data/004806.jpg \
    --inference_type body

# Best quality: full inference with mask conditioning (default)
python demos/BMPv2_demo.py --image data/004806.jpg
```

#### Disable Mask Conditioning

Faster but less accurate (doesn't use segmentation masks as prompts):

```bash
python demos/BMPv2_demo.py \
    --image data/004806.jpg \
    --no_mask_conditioning
```

#### Skip 3D Recovery

Run only BMP pipeline (useful for testing BMP without SAM-3D-Body):

```bash
python demos/BMPv2_demo.py \
    --image data/004806.jpg \
    --skip_3d
```

### Output Files

The demo saves the following visualizations:

- `{image_name}_bmp_pose.jpg` - 2D pose estimation results
- `{image_name}_bmp_mask.jpg` - Segmentation mask results
- `{image_name}_3d_mesh.jpg` - 3D mesh overlay on image
- `{image_name}_combined.jpg` - Side-by-side comparison of all results

## Programmatic API

You can also use SAM-3D-Body programmatically:

```python
from bboxmaskpose import BBoxMaskPose
from bboxmaskpose.sam3d_utils import SAM3DBodyWrapper, visualize_3d_meshes

# Step 1: Run BMP pipeline
bmp = BBoxMaskPose(config="bmp_D3", device="cuda")
result = bmp.predict(image="path/to/image.jpg")

# Step 2: Initialize SAM-3D-Body
sam3d = SAM3DBodyWrapper(device="cuda")

# Step 3: Predict 3D meshes from BMP outputs
outputs_3d = sam3d.predict(
    image="path/to/image.jpg",
    bboxes=result['bboxes'],
    masks=result['masks'],
    use_mask=True,
    inference_type="full",  # Options: "full", "body", "hand"
)

# Step 4: Visualize results
import cv2
img = cv2.imread("path/to/image.jpg")
vis = visualize_3d_meshes(img, outputs_3d, sam3d.faces)
cv2.imwrite("output_3d.jpg", vis)
```

### Access 3D Mesh Data

Each element in `outputs_3d` is a dictionary containing:

```python
output_3d[0].keys()
# dict_keys(['vertices', 'joints', 'bbox', 'mask', ...])

# 3D mesh vertices in camera coordinates (V, 3)
vertices = outputs_3d[0]['vertices']

# 3D joint locations (J, 3)
joints_3d = outputs_3d[0]['joints']

# Mesh faces (shared across all people)
faces = sam3d.faces  # (F, 3)
```

## Integration Architecture

### Wrapper Design

The integration follows BBoxMaskPose's modular design pattern:

```
bboxmaskpose/
├── sam3d_utils.py          # SAM-3D-Body wrapper (new)
│   ├── SAM3DBodyWrapper    # Main wrapper class
│   ├── visualize_3d_meshes # Visualization helper
│   └── check_sam3d_available
│
demos/
├── BMP_demo.py             # Original BMP demo
└── BMPv2_demo.py           # New demo with 3D (new)
```

### Why a Wrapper?

The `SAM3DBodyWrapper` class:
- **Simplifies** SAM-3D-Body's complex initialization
- **Adapts** BMP outputs (bboxes, masks) to SAM-3D-Body inputs
- **Handles** optional dependencies gracefully (no hard requirement)
- **Follows** BMP's design patterns (similar to PMPose wrapper)

### Key Design Decisions

1. **Optional Dependency**: SAM-3D-Body is not required for core BMP functionality
2. **No Code Duplication**: Reuses SAM-3D-Body's existing code via wrapper
3. **Mask Conditioning**: Leverages BMP's high-quality masks as prompts
4. **No Internal Detector**: Disables SAM-3D-Body's detector (BMP already detects)

## Troubleshooting

### Import Error: `sam_3d_body` not found

**Solution**: Install SAM-3D-Body following Step 4 above.

### HuggingFace Authentication Error

**Solution**: 
1. Request access at https://huggingface.co/facebook/sam-3d-body-dinov3
2. Login: `huggingface-cli login`

### MoGe Import Error (FOV Estimator)

**Solution**: Either:
- Install MoGe: `pip install git+https://github.com/microsoft/MoGe.git`
- Or disable FOV estimation (uses default FOV instead)

### Detectron2 Build Errors

**Solution**: Make sure you have:
- CUDA toolkit installed and matching PyTorch CUDA version
- GCC/G++ compiler available
- Use the exact commit hash: `@a1ce2f9`

## References

- **SAM-3D-Body**: [GitHub](https://github.com/facebookresearch/sam-3d-body) | [Paper](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/)
- **BBoxMaskPose**: [GitHub](https://github.com/MiraPurkrabek/BBoxMaskPose) | [Paper](https://arxiv.org/abs/2601.15200)

## Citation

If you use this integration, please cite both works:

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint; identifier to be added},
  year={2025}
}

@InProceedings{Purkrabek2026BMPv2,
  author    = {Purkrabek, Miroslav and Kolomiiets, Constantin and Matas, Jiri},
  title     = {BBoxMaskPose v2: Expanding Mutual Conditioning to 3D},
  booktitle = {arXiv preprint arXiv:2601.15200},
  year      = {2026}
}
```
