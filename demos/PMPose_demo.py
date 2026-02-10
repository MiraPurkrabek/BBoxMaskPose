#!/usr/bin/env python3
"""
PMPose Demo - Demonstrate PMPose public API usage.

This demo shows how to use the PMPose wrapper API for pose estimation
given an image and bounding boxes.

Usage:
    python demos/PMPose_demo.py --image <image_path> --output <output_dir>
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pmpose import PMPose


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PMPose Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="demo/data/004806.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demos/outputs/pmpose",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--bboxes",
        type=str,
        default=None,
        help="Comma-separated bbox coords: x1,y1,x2,y2;x1,y1,x2,y2;..."
    )
    return parser.parse_args()


def parse_bboxes(bbox_str: str):
    """Parse bbox string into numpy array."""
    if bbox_str is None:
        return None
    
    bboxes = []
    for bbox in bbox_str.split(";"):
        coords = [float(x) for x in bbox.split(",")]
        if len(coords) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}")
        bboxes.append(coords)
    
    return np.array(bboxes, dtype=np.float32)


def get_default_bboxes(image_path: str):
    """
    Get some default bboxes for demo purposes.
    
    For the OCHuman 004806.jpg image, we use pre-defined bboxes
    that cover the people in the image.
    """
    # These are approximate bboxes for the people in demo/data/004806.jpg
    # You can adjust these for your specific image
    if "004806" in image_path:
        # OCHuman image with multiple people
        return np.array([
            [  1.343687,  55.028114, 530.4726,   863.68    ],
            [196.49245,   48.729275, 528.9763,   832.8075  ],
        ], dtype=np.float32)
    else:
        # Generic full-image bbox
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        return np.array([[0, 0, w, h]], dtype=np.float32)


def main():
    """Main demo function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PMPose Demo - Public API Usage")
    print("=" * 60)
    
    # Step 1: Initialize PMPose model
    print("\n[Step 1] Initializing PMPose model...")
    print(f"  Device: {args.device}")
    
    pose_model = PMPose(
        device=args.device,
        variant="PMPose-b",
        from_pretrained=True,
    )
    print("  ✓ Model initialized successfully")
    
    # Step 2: Load image and prepare bboxes
    print(f"\n[Step 2] Loading image: {args.image}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Failed to read image: {args.image}")
    
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")
    
    # Get bboxes
    if args.bboxes:
        bboxes = parse_bboxes(args.bboxes)
        print(f"  Using provided bboxes: {len(bboxes)} boxes")
    else:
        bboxes = get_default_bboxes(args.image)
        print(f"  Using default bboxes: {len(bboxes)} boxes")

    # Create segmentation masks (polygon format) as MaskPose requires them
    # Create dummy masks -- all ones covering the bbox area
    masks_binary = (np.ones((bboxes.shape[0], h, w), dtype=np.uint8) * 255)
    masks_polygon = []
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i]
        polygon = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        masks_polygon.append(polygon)
    
    print(f"  Bboxes shape: {bboxes.shape}")
    
    # Step 3: Run pose estimation
    print("\n[Step 3] Running pose estimation...")
    keypoints, presence, visibility, heatmaps = pose_model.predict(
        image=img,
        bboxes=bboxes,
        masks=masks_polygon,
        return_probmaps=False,
    )
    
    print("  ✓ Pose estimation complete")
    print(f"  Keypoints shape: {keypoints.shape}")
    print(f"  Presence shape: {presence.shape}")
    print(f"  Visibility shape: {visibility.shape}")
    
    # Print some statistics
    num_people = keypoints.shape[0]
    num_keypoints = keypoints.shape[1]
    avg_scores = keypoints[:, :, 2].mean(axis=1)
    
    print(f"\n  Results:")
    print(f"    - Detected {num_people} people")
    print(f"    - {num_keypoints} keypoints per person")
    for i, score in enumerate(avg_scores):
        print(f"    - Person {i+1}: avg confidence = {score:.3f}")
    
    # Step 4: Visualize results
    print("\n[Step 4] Visualizing results...")
    vis_img = pose_model.visualize(
        image=img,
        keypoints=keypoints,
        bboxes=bboxes,
        save_path=None,
    )
    
    # Save visualization
    output_path = output_dir / f"{Path(args.image).stem}_pmpose.jpg"
    cv2.imwrite(str(output_path), vis_img)
    print(f"  ✓ Visualization saved to: {output_path}")
    
    # Save keypoints as numpy file
    keypoints_path = output_dir / f"{Path(args.image).stem}_keypoints.npy"
    np.save(str(keypoints_path), {
        'keypoints': keypoints,
        'presence': presence,
        'visibility': visibility,
        'bboxes': bboxes,
    })
    print(f"  ✓ Keypoints saved to: {keypoints_path}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
