#!/usr/bin/env python3
"""
BBoxMaskPose Demo - Demonstrate BBoxMaskPose public API usage.

This demo shows two usage patterns:
1. BMP with internal pose model
2. BMP with externally provided PMPose model

Usage:
    python demos/BMP_demo.py --image <image_path> --output <output_dir>
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pmpose import PMPose
from bboxmaskpose import BBoxMaskPose


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BBoxMaskPose Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="demo/data/004806.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demos/outputs/bboxmaskpose",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bmp_D3",
        help="BMP config (bmp_D3 or bmp_J1)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["internal", "external", "both"],
        help="Demo mode: internal (BMP creates pose model), external (inject PMPose), or both"
    )
    return parser.parse_args()


def bmp_demo(image_path: str, output_dir: Path, device: str, config: str):
    """
    Demo 2: User creates PMPose model and injects it into BMP.
    
    This pattern is useful when you want to:
    - Reuse the same pose model across multiple BMP instances
    - Pre-configure the pose model with custom settings
    - Have fine-grained control over the pose model
    """
    print("\n" + "=" * 60)
    print("Demo 2: BMP with External PMPose Model")
    print("=" * 60)
    
    # Step 1: Create PMPose model
    print("\n[Step 1] Creating PMPose model...")
    print(f"  Device: {device}")
    
    pose_model = PMPose(
        device=device,
        variant="PMPose-b",
        from_pretrained=True,
    )
    print("  ✓ PMPose model created")
    
    # Step 2: Inject into BBoxMaskPose
    print("\n[Step 2] Initializing BBoxMaskPose with external pose model...")
    print(f"  Config: {config}")
    
    bmp_model = BBoxMaskPose(
        config=config,
        device=device,
        pose_model=pose_model,  # Inject the PMPose instance. If None, BBoxMaskPose creates a default one.
    )
    print("  ✓ BBoxMaskPose initialized with external pose model")
    
    # Step 3: Run pipeline
    print(f"\n[Step 3] Running full BMP pipeline on: {image_path}")
    result = bmp_model.predict(
        image=image_path,
        bboxes=None,  # Run detector
        return_intermediates=False,
    )
    
    print("  ✓ Pipeline complete")
    print(f"\n  Results:")
    print(f"    - Detected {len(result['bboxes'])} people")
    print(f"    - Keypoints shape: {result['keypoints'].shape}")
    print(f"    - Masks shape: {result['masks'].shape}")
    
    # Visualize
    print("\n[Step 4] Visualizing results...")
    img = cv2.imread(image_path)
    vis_pose = bmp_model.visualize(
        image=img,
        result=result,
        vis_type="pose",
    )
    vis_mask = bmp_model.visualize(
        image=img,
        result=result,
        vis_type="mask",
    )
    vis_img = np.hstack((vis_pose, vis_mask))    


    output_path = output_dir / f"{Path(image_path).stem}_bmp.jpg"
    cv2.imwrite(str(output_path), vis_img)
    print(f"  ✓ Saved to: {output_path}")
    
    return result


def main():
    """Main demo function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    print("=" * 60)
    print("BBoxMaskPose Demo - Public API Usage")
    print("=" * 60)
    print(f"\nImage: {args.image}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    
    result1 = bmp_demo(
        args.image,
        output_dir,
        args.device,
        args.config,
    )
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")

    print(result1['bboxes'])


if __name__ == "__main__":
    main()
