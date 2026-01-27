</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h1 style="margin-bottom: 0.0em;">
        Detection, Pose Estimation and Segmentation for Multiple Bodies: Closing the Virtuous Circle
      </h1>
    </summary>
  </ul>
</div>
</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h2 style="margin-bottom: 0.2em;">
        ICCV 2025
      </h2>
    </summary>
  </ul>
</div>

<div align="center">
  <img src="images/004806_BMP.gif" alt="BBox-Mask-Pose loop" height="500px">

  [![Paper](https://img.shields.io/badge/Paper-ICCV%202025-blue)](https://arxiv.org/abs/2412.01562) &nbsp;&nbsp;&nbsp;
  [![Website](https://img.shields.io/badge/Website-BBoxMaskPose-green)](https://mirapurkrabek.github.io/BBox-Mask-Pose/) &nbsp;&nbsp;&nbsp;
  [![License](https://img.shields.io/badge/License-GPL%203.0-orange.svg)](LICENSE) &nbsp;&nbsp;&nbsp;
  [![Video](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://youtu.be/U05yUP4b2LQ)
  

  Papers with code:

  [![2D Pose AP on OCHuman: 42.5](https://img.shields.io/badge/OCHuman-2D_Pose:_49.2_AP-blue)](https://paperswithcode.com/sota/2d-human-pose-estimation-on-ochuman?p=detection-pose-estimation-and-segmentation-1) &nbsp;&nbsp;
  [![Human Instance Segmentation AP on OCHuman: 34.0](https://img.shields.io/badge/OCHuman-Human_Instance_Segmentation:_34.0_AP-blue)](https://paperswithcode.com/sota/human-instance-segmentation-on-ochuman?p=detection-pose-estimation-and-segmentation-1)  

</div>

> [!IMPORTANT]
> The new version of <b>BBox-Mask-Pose (BMPv2)</b> is now available on [<b>arXiv</b>](https://arxiv.org/abs/2601.15200v1).
> BMPv2 significantly improves performance; see the quantitative results reported in the preprint.
> One of the key contributions is <b>PMPose</b>, a new top-down pose estimation model, that is already strong on standard benchmarks and in crowded scenes.
> The code will be added to the <code>BMP-v2</code> branch in the following weeks and gradually merged into <code>main</code> as well as to the online demo.


## Overview

The BBox-Mask-Pose (BMP) method integrates detection, pose estimation, and segmentation into a self-improving loop by conditioning these tasks on each other. This approach enhances all three tasks simultaneously. Using segmentation masks instead of bounding boxes improves performance in crowded scenarios, making top-down methods competitive with bottom-up approaches.

Key contributions:
1. **MaskPose**: a pose estimation model conditioned by segmentation masks instead of bounding boxes, boosting performance in dense scenes without adding parameters
    - Download pre-trained weights below
2. **BBox-MaskPose (BMP)**: method linking bounding boxes, segmentation masks, and poses to simultaneously address multi-body detection, segmentation and pose estimation
    - Try the demo!
3. Fine-tuned RTMDet adapted for itterative detection (ignoring 'holes')
    - Download pre-trained weights below
5. Support for multi-dataset training of ViTPose, previously implemented in the official ViTPose repository but absent in MMPose.

For more details, please visit our [project website](https://mirapurkrabek.github.io/BBox-Mask-Pose/).


## News

- **Aug 2025**: [HuggingFace Image Demo](https://huggingface.co/spaces/purkrmir/BBoxMaskPose-demo) is available
- **Jul 2025**: Version 1.1 with easy-to-run image demo released
- **Jun 2025**: Paper accepted to ICCV 2025
- **Dec 2024**: The code is available
- **Nov 2024**: The [project website](https://MiraPurkrabek.github.io/BBox-Mask-Pose) is live


## Installation

### Docker Installation (Recommended)

Docker provides the fastest and most reliable way to get started, eliminating dependency conflicts.

**Prerequisites:**
- Docker Engine 19.03 or later
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with CUDA 12.1 support

**Build and Run:**
```bash
# Clone the repository
git clone https://github.com/mirapurkrabek/BBoxMaskPose.git
cd BBoxMaskPose

# Build the Docker image (approximately 15 minutes on first build)
docker-compose build

# Run on the sample image
docker-compose up
```

**Processing Custom Images:**
```bash
# Process a single image
docker run --gpus all \
  -v $(pwd)/your_images:/app/input \
  -v $(pwd)/outputs:/app/outputs \
  bboxmaskpose:latest \
  python demo/bmp_demo.py configs/bmp_D3.yaml /app/input/your_image.jpg --output-root /app/outputs

# Interactive shell for debugging
docker run --gpus all -it bboxmaskpose:latest bash
```

**Tested Configuration:**

| Component | Version |
|-----------|--------|
| PyTorch | 2.4.0+cu121 |
| CUDA | 12.1 |
| mmcv | 2.2.0 (pre-built) |
| mmdet | 3.2.0 |
| mmengine | 0.10.7 |
| Python | 3.11/3.12 |

---

### Manual Installation
  
This project is built on top of [MMPose](https://github.com/open-mmlab/mmpose) and [SAM 2.1](https://github.com/facebookresearch/sam2).
Please refer to the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) or [SAM installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) for detailed setup instructions.

Basic installation steps:
```bash
# Clone the repository
git clone https://github.com/mirapurkrabek/BBoxMaskPose.git BBoxMaskPose/
cd BBoxMaskPose

# Install PyTorch with CUDA support
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install OpenMMLab libraries
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0" "mmdet==3.2.0" "mmpretrain==1.2.0"

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

> **Note**: For Python 3.12 users, some packages (e.g., chumpy) may have compatibility issues. The Docker installation handles these automatically.

## Demo

**Step 1:** Download SAM2 weights using the [enclosed script](models/SAM/download_ckpts.sh).

**Step 2:** Run the full BBox-Mask-Pose pipeline on an input image:

```bash
python demo/bmp_demo.py configs/bmp_D3.yaml demo/data/004806.jpg
```

The pipeline executes the following stages:
1. **Detection**: RTMDet-L identifies person bounding boxes
2. **Pose Estimation**: MaskPose estimates 17 COCO keypoints per person
3. **Segmentation**: SAM2 generates instance masks using pose keypoints as prompts
4. **Iteration**: The loop repeats, using masks to find additional occluded persons

**Command Line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `configs/bmp_D3.yaml` | BMP configuration file | Required |
| `input_image` | Path to input image | Required |
| `--device` | Inference device | `cuda:0` |
| `--output-root` | Output directory | `demo/outputs` |
| `--create-gif` | Generate iteration GIF | `False` |

**Output Structure:**

After running, outputs are saved in `outputs/<image_name>/`:

```
outputs/004806/
├── 004806_iter1_Detector_(out).jpg    # Detection results
├── 004806_iter1_MaskPose_(in).jpg     # Cropped input to pose estimator
├── 004806_iter1_MaskPose_(out).jpg    # Pose estimation output
├── 004806_iter1_SAM_Masks.jpg         # SAM segmentation masks
├── 004806_iter1_Final_Poses.jpg       # Final pose overlay
├── 004806_iter2_...                   # Second iteration outputs
└── ...
```

<div align="center">
  <a href="images/004806_mask.jpg" target="_blank">
    <img src="images/004806_mask.jpg" alt="Segmentation masks" width="200" />
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="images/004806_pose.jpg" target="_blank">
    <img src="images/004806_pose.jpg" alt="Pose estimation" width="200" />
  </a>
</div>


## Pre-trained Models

Pre-trained models are available on [VRG Hugging Face](https://huggingface.co/vrg-prague/BBoxMaskPose/).
To run the demo, you only need to download SAM weights with the [enclosed script](models/SAM/download_ckpts.sh).
The detector and pose estimator weights are downloaded automatically during runtime.

**Manual Download:**

| Model | Description | Link |
|-------|-------------|------|
| ViTPose-b | Multi-dataset (COCO+MPII+AIC) | [download](https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/ViTPose-b-multi_mmpose20.pth) |
| MaskPose-b | Mask-conditioned pose estimator | [download](https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose-b.pth) |
| RTMDet-L | Fine-tuned detector | [download](https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/rtmdet-ins-l-mask.pth) |

## Acknowledgments

This project builds upon:
- [MMDetection](https://github.com/open-mmlab/mmdetection) - Object detection framework
- [MMPose 2.0](https://github.com/open-mmlab/mmpose) - Pose estimation framework
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) - Vision Transformer for pose estimation
- [SAM 2.1](https://github.com/facebookresearch/sam2) - Segment Anything Model

## Citation

The code was implemented by [Miroslav Purkrábek]([htt]https://mirapurkrabek.github.io/).
If you use this work, kindly cite it using the reference provided below.

For questions, please use the Issues of Discussion.

```
@InProceedings{Purkrabek2025ICCV,
    author    = {Purkrabek, Miroslav and Matas, Jiri},
    title     = {Detection, Pose Estimation and Segmentation for Multiple Bodies: Closing the Virtuous Circle},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {9004-9013}
}
```
