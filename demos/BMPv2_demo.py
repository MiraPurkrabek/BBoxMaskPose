#!/usr/bin/env python3
# Copyright (c) authors of BBoxMaskPose (BMPv2). All rights reserved.
"""
BBoxMaskPose v2 Demo - Demonstrate BBoxMaskPose + SAM-3D-Body integration.

This demo extends BMP_demo.py to additionally predict 3D human meshes
using SAM-3D-Body. The pipeline:
1. Run BMP to get bboxes, masks, and 2D poses
2. Pass masks and bboxes to SAM-3D-Body for 3D mesh recovery
3. Output and visualize all results: masks, 2D poses, and 3D meshes

Usage:
    python demos/BMPv2_demo.py --image <image_path> --output <output_dir>
    
Requirements:
    - SAM-3D-Body must be installed (see installation guide)
    - SAM-3D-Body checkpoints must be available (download from HuggingFace)
    
Examples:
    # Basic usage with default checkpoint (downloads from HuggingFace)
    python demos/BMPv2_demo.py --image data/004806.jpg --device cuda
    
    # With local checkpoint
    python demos/BMPv2_demo.py --image data/004806.jpg --device cuda \
        --sam3d_checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    
    # Without mask conditioning
    python demos/BMPv2_demo.py --image data/004806.jpg --no_mask_conditioning
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from bboxmaskpose import BBoxMaskPose
from pmpose import PMPose


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BBoxMaskPose v2 Demo - Full Pipeline with 3D Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (downloads checkpoint from HuggingFace)
  python demos/BMPv2_demo.py --image data/004806.jpg --device cuda
  
  # With local checkpoint
  python demos/BMPv2_demo.py --image data/004806.jpg --device cuda \\
      --sam3d_checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \\
      --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

Note: First time usage may download models from HuggingFace (requires authentication)
        """,
    )
    parser.add_argument("--image", type=str, default="data/004806.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="demos/outputs/bboxmaskpose_v2", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    parser.add_argument("--config", type=str, default="bmp_D3", help="BMP config (bmp_D3 or bmp_J1)")

    # SAM-3D-Body specific arguments
    parser.add_argument(
        "--sam3d_checkpoint",
        type=str,
        default=None,
        help="Path to SAM-3D-Body checkpoint. If None, auto-downloads from HuggingFace. "
        "To use local checkpoint, download to 'checkpoints/sam-3d-body/' and it will be auto-detected.",
    )
    parser.add_argument(
        "--mhr_path", type=str, default=None, help="Path to MHR model file. Auto-detected if checkpoint is in 'checkpoints/sam-3d-body/'."
    )
    parser.add_argument(
        "--no_mask_conditioning", action="store_true", help="Disable mask-conditioned 3D inference (faster but less accurate)"
    )
    parser.add_argument(
        "--kpts_conditioning",
        action="store_true",
        default=False,
        help="Enable keypoints-conditioned 3D inference (faster but less accurate)",
    )
    parser.add_argument("--no_fov", action="store_true", help="Disable FOV estimator (enabled by default if MoGe is installed)")
    parser.add_argument(
        "--inference_type",
        type=str,
        default="full",
        choices=["full", "body", "hand"],
        help="Type of 3D inference: 'full' (body+hands, slower), 'body' (body only, faster), 'hand' (hands only)",
    )
    parser.add_argument("--skip_3d", action="store_true", help="Skip 3D mesh recovery (only run BMP pipeline)")

    return parser.parse_args()


def check_sam3d_available():
    """Check if SAM-3D-Body is installed."""
    try:
        from bboxmaskpose.sam3d_utils import check_sam3d_available

        return check_sam3d_available()
    except ImportError:
        return False


def print_installation_guide():
    """Print SAM-3D-Body installation instructions."""
    print("\n" + "=" * 70)
    print("SAM-3D-Body Installation Required")
    print("=" * 70)
    print("\nSAM-3D-Body is not installed. To use 3D mesh recovery, install it:")
    print("\n1. Install core dependencies:")
    print("   pip install -r requirements/sam3d.txt")
    print("\n2. Install detectron2:")
    print("   pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \\")
    print("       --no-build-isolation --no-deps")
    print("\n3. Install MoGe (optional, for FOV estimation):")
    print("   pip install git+https://github.com/microsoft/MoGe.git")
    print("\n4. Clone and install SAM-3D-Body:")
    print("   git clone https://github.com/facebookresearch/sam-3d-body.git")
    print("   cd sam-3d-body")
    print("   pip install -e .")
    print("\n5. Request access to model checkpoints:")
    print("   https://huggingface.co/facebook/sam-3d-body-dinov3")
    print("\nFor more details, see:")
    print("https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md")
    print("=" * 70 + "\n")


def bmpv2_demo(
    image_path: str,
    output_dir: Path,
    device: str,
    config: str,
    sam3d_checkpoint: str = None,
    mhr_path: str = None,
    use_mask_conditioning: bool = True,
    use_kpts_conditioning: bool = False,
    use_fov: bool = True,
    inference_type: str = "full",
    skip_3d: bool = False,
):
    """
    Run BBoxMaskPose v2 demo: BMP + SAM-3D-Body pipeline.

    Args:
        image_path: Path to input image.
        output_dir: Output directory for results.
        device: Device for inference ('cuda' or 'cpu').
        config: BMP configuration name.
        sam3d_checkpoint: Path to SAM-3D-Body checkpoint.
        mhr_path: Path to MHR model file.
        use_mask_conditioning: Whether to use mask-conditioned 3D inference.
        use_fov: Whether to use FOV estimator.
        inference_type: Type of 3D inference ('full', 'body', or 'hand').
        skip_3d: If True, skip 3D mesh recovery.
    """
    # Auto-detect checkpoint paths if not provided
    default_checkpoint_dir = Path("checkpoints/sam-3d-body-dinov3")
    if sam3d_checkpoint is None and default_checkpoint_dir.exists():
        checkpoint_file = default_checkpoint_dir / "model.ckpt"
        if checkpoint_file.exists():
            sam3d_checkpoint = str(checkpoint_file)
            print(f"Auto-detected SAM-3D-Body checkpoint: {sam3d_checkpoint}")

    if mhr_path is None and default_checkpoint_dir.exists():
        mhr_file = default_checkpoint_dir / "assets" / "mhr_model.pt"
        if mhr_file.exists():
            mhr_path = str(mhr_file)
            print(f"Auto-detected MHR model: {mhr_path}")

    print("\n" + "=" * 70)
    print("BBoxMaskPose v2 Demo - Full Pipeline with 3D Mesh Recovery")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Run BBoxMaskPose pipeline (detection + pose + segmentation)
    # ========================================================================
    print("\n[STEP 1/3] Running BBoxMaskPose Pipeline")
    print("-" * 70)

    # Initialize BBoxMaskPose (creates internal pose model from config)
    print(f"  • Initializing BBoxMaskPose (config: {config})...")
    bmp_model = BBoxMaskPose(
        config=config,
        device=device,
    )
    print("    ✓ BBoxMaskPose ready")

    # Run full BMP pipeline
    print(f"  • Running full BMP pipeline on: {image_path}")
    result = bmp_model.predict(
        image=image_path,
        bboxes=None,  # Run detector
        return_intermediates=False,
    )

    num_people = len(result["bboxes"])
    print(f"    ✓ Pipeline complete - detected {num_people} people")
    print(f"      - Bboxes: {result['bboxes'].shape}")
    print(f"      - Keypoints: {result['keypoints'].shape}")
    print(f"      - Masks: {result['masks'].shape}")

    # Visualize BMP results
    print("  • Generating BMP visualizations...")
    img = cv2.imread(image_path)
    vis_pose = bmp_model.visualize(image=img, result=result, vis_type="pose")
    vis_mask = bmp_model.visualize(image=img, result=result, vis_type="mask")

    # Save BMP outputs
    bmp_output_path = output_dir / f"{Path(image_path).stem}_bmp_pose.jpg"
    cv2.imwrite(str(bmp_output_path), vis_pose)
    print(f"    ✓ Saved pose visualization: {bmp_output_path}")

    bmp_mask_path = output_dir / f"{Path(image_path).stem}_bmp_mask.jpg"
    cv2.imwrite(str(bmp_mask_path), vis_mask)
    print(f"    ✓ Saved mask visualization: {bmp_mask_path}")

    # ========================================================================
    # STEP 2: Check if 3D mesh recovery should be run
    # ========================================================================
    if skip_3d:
        print("\n[STEP 2/3] Skipping 3D mesh recovery (--skip_3d flag set)")
        print("\n" + "=" * 70)
        print("Demo completed successfully! (BMP only)")
        print("=" * 70)
        return result, None

    if not check_sam3d_available():
        print("\n[STEP 2/3] SAM-3D-Body not available")
        print_installation_guide()
        print("Skipping 3D mesh recovery. Use --skip_3d to suppress this message.")
        print("\n" + "=" * 70)
        print("Demo completed successfully! (BMP only)")
        print("=" * 70)
        return result, None

    # ========================================================================
    # STEP 3: Run SAM-3D-Body for 3D mesh recovery
    # ========================================================================
    print("\n[STEP 2/3] Initializing SAM-3D-Body")
    print("-" * 70)

    from bboxmaskpose.sam3d_utils import SAM3DBodyWrapper, visualize_3d_meshes

    try:
        print("  • Loading SAM-3D-Body model...")
        sam3d = SAM3DBodyWrapper(
            checkpoint_path=sam3d_checkpoint,
            mhr_path=mhr_path,
            device=device,
            use_detector=False,  # We already have detections from BMP
            use_segmentor=False,  # We already have masks from BMP
            use_fov=use_fov,
        )
        print("    ✓ SAM-3D-Body ready")
    except Exception as e:
        # Traceback e
        import traceback

        traceback.print_exc()

        print(f"\n  ✗ Error loading SAM-3D-Body: {e}")
        print("   \nPlease check:")
        print("     1. SAM-3D-Body is installed correctly")
        print("     2. You have HuggingFace access to facebook/sam-3d-body-dinov3")
        print("     3. You are authenticated with HuggingFace (huggingface-cli login)")
        print("\nSkipping 3D mesh recovery...")
        return result, None

    print("\n[STEP 3/3] Running 3D Mesh Recovery")
    print("-" * 70)

    # Prepare inputs for SAM-3D-Body
    bboxes = result["bboxes"]  # (N, 4) in [x1, y1, x2, y2] format
    masks = result["masks"]  # (N, H, W) binary masks
    keypoints = result["keypoints"]  # (N, 17, 2) keypoints
    keypoints = keypoints[:, :17, :]  # Ensure only COCO keypoints are used (if more are present)

    print(f"  • Running SAM-3D-Body on {len(bboxes)} detected people...")
    print(f"    - Using mask conditioning: {use_mask_conditioning}")
    print(f"    - Using keypoints conditioning: {use_kpts_conditioning}")
    print(f"    - Inference type: {inference_type}")

    # Run 3D inference
    outputs_3d = sam3d.predict(
        image=image_path,
        bboxes=bboxes,
        masks=masks if use_mask_conditioning else None,
        keypoints=keypoints if use_kpts_conditioning else None,
        use_mask=use_mask_conditioning,
        inference_type=inference_type,
    )

    print(f"    ✓ 3D mesh recovery complete")
    print(f"      - Generated {len(outputs_3d)} 3D meshes")

    # Visualize 3D meshes
    print("  • Generating 3D mesh visualization...")
    vis_3d_path = output_dir / f"{Path(image_path).stem}_3d_mesh.jpg"
    vis_3d = visualize_3d_meshes(
        image=img,
        outputs=outputs_3d,
        faces=sam3d.faces,
        masks=masks if use_mask_conditioning else None,
        keypoints=keypoints if use_kpts_conditioning else None,
        output_path=str(vis_3d_path),
    )
    print(f"    ✓ Saved 3D visualization: {vis_3d_path}")

    # Create combined visualization (BMP + 3D)
    print("  • Creating combined visualization...")

    # Ensure all images have the same height for horizontal stacking
    target_height = vis_pose.shape[0]

    # Resize vis_mask if needed
    if vis_mask.shape[0] != target_height:
        aspect_ratio = vis_mask.shape[1] / vis_mask.shape[0]
        target_width = int(target_height * aspect_ratio)
        vis_mask = cv2.resize(vis_mask, (target_width, target_height))

    # Resize vis_3d if needed
    if vis_3d.shape[0] != target_height:
        aspect_ratio = vis_3d.shape[1] / vis_3d.shape[0]
        target_width = int(target_height * aspect_ratio)
        vis_3d = cv2.resize(vis_3d, (target_width, target_height))

    combined = np.hstack((vis_pose, vis_mask, vis_3d))
    combined_path = output_dir / f"{Path(image_path).stem}_combined.jpg"
    cv2.imwrite(str(combined_path), combined)
    print(f"    ✓ Saved combined visualization: {combined_path}")

    return result, outputs_3d


def main():
    """Main demo function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    print("\n" + "=" * 70)
    print("BBoxMaskPose v2 Demo Configuration")
    print("=" * 70)
    print(f"Image:                {args.image}")
    print(f"Output directory:     {args.output}")
    print(f"Device:               {args.device}")
    print(f"BMP config:           {args.config}")
    print(f"SAM-3D checkpoint:    {args.sam3d_checkpoint or 'Auto-detect or HuggingFace'}")
    print(f"Mask conditioning:    {not args.no_mask_conditioning}")
    print(f"Keypoints conditioning: {args.kpts_conditioning}")
    print(f"FOV estimation:       {not args.no_fov}")
    print(f"Inference type:       {args.inference_type}")
    print(f"Skip 3D:              {args.skip_3d}")

    # Run demo
    result_bmp, result_3d = bmpv2_demo(
        image_path=args.image,
        output_dir=output_dir,
        device=args.device,
        config=args.config,
        sam3d_checkpoint=args.sam3d_checkpoint,
        mhr_path=args.mhr_path,
        use_mask_conditioning=not args.no_mask_conditioning,
        use_kpts_conditioning=args.kpts_conditioning,
        use_fov=not args.no_fov,
        inference_type=args.inference_type,
        skip_3d=args.skip_3d,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Demo Completed Successfully!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  • BMP pose visualization: {Path(args.image).stem}_bmp_pose.jpg")
    print(f"  • BMP mask visualization: {Path(args.image).stem}_bmp_mask.jpg")
    if result_3d is not None:
        print(f"  • 3D mesh visualization:  {Path(args.image).stem}_3d_mesh.jpg")
        print(f"  • Combined visualization: {Path(args.image).stem}_combined.jpg")

    print(f"\nDetected {len(result_bmp['bboxes'])} people in the image")
    if result_3d is not None:
        print(f"Recovered {len(result_3d)} 3D meshes")
    print()


if __name__ == "__main__":
    main()
