# Copyright (c) OpenMMLab. All rights reserved.
"""
BMP Demo script: sequentially runs detection, pose estimation, SAM-based mask refinement, and visualization.
Usage:
    python bmp_demo.py <config.yaml> <input_image> [--output-root <dir>]
"""

import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import mmengine
import numpy as np
import yaml
from demo_utils import DotDict, concat_instances, create_GIF, filter_instances, pose_nms, visualize_itteration
from mm_utils import run_MMDetector, run_MMPose
from mmdet.apis import init_detector
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from sam2_utils import prepare_model as prepare_sam2_model
from sam2_utils import process_image_with_SAM

from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline

# Default thresholds
DEFAULT_DET_CAT_ID: int = 0  # "person"
DEFAULT_BBOX_THR: float = 0.3
DEFAULT_NMS_THR: float = 0.3
DEFAULT_KPT_THR: float = 0.3


def parse_args() -> Namespace:
    """
    Parse command-line arguments for BMP demo.

    Returns:
        Namespace: Contains bmp_config (Path), input (Path), output_root (Path), device (str).
    """
    parser = ArgumentParser(description="BBoxMaskPose demo")
    parser.add_argument("bmp_config", type=Path, help="Path to BMP YAML config file")
    parser.add_argument("input", type=Path, help="Input image file")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory to save outputs (default: ./outputs)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (e.g., cuda:0 or cpu)")
    parser.add_argument("--create-gif", action="store_true", default=False, help="Create GIF of all BMP iterations")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = os.path.join(Path(__file__).parent, "outputs")
    return args


def parse_yaml_config(yaml_path: Path) -> DotDict:
    """
    Load BMP configuration from a YAML file.

    Args:
        yaml_path (Path): Path to YAML config.
    Returns:
        DotDict: Nested config dictionary.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return DotDict(cfg)


def process_one_image(
    args: Namespace,
    bmp_config: DotDict,
    img_path: Path,
    detector: object,
    detector_prime: object,
    pose_estimator: object,
    sam2_model: object,
) -> InstanceData:
    """
    Run the full BMP pipeline on a single image: detection, pose, SAM mask refinement, and visualization.

    Args:
        args (Namespace): Parsed CLI arguments.
        bmp_config (DotDict): Configuration parameters.
        img_path (Path): Path to the input image.
        detector: Primary MMDetection model.
        detector_prime: Secondary MMDetection model for iterations.
        pose_estimator: MMPose model for keypoint estimation.
        sam2_model: SAM model for mask refinement.
    Returns:
        InstanceData: Final merged detections and refined masks.
    """
    # Load image
    img = mmcv.imread(str(img_path), channel_order="bgr")
    if img is None:
        raise ValueError("Failed to read image from {}.".format(img_path))

    # Prepare output directory
    output_dir = os.path.join(args.output_root, img_path.stem)
    shutil.rmtree(str(output_dir), ignore_errors=True)
    mmengine.mkdir_or_exist(str(output_dir))

    img_for_detection = img.copy()
    all_detections = None
    for iteration in range(bmp_config.num_bmp_iters):
        print_log("BMP Iteration {}/{} started".format(iteration + 1, bmp_config.num_bmp_iters), logger="current")

        # Step 1: Detection
        det_instances = run_MMDetector(
            detector if iteration == 0 else detector_prime,
            img_for_detection,
            det_cat_id=DEFAULT_DET_CAT_ID,
            bbox_thr=DEFAULT_BBOX_THR,
            nms_thr=DEFAULT_NMS_THR,
        )
        print_log("Detected {} instances".format(len(det_instances.bboxes)), logger="current")
        if len(det_instances.bboxes) == 0:
            print_log("No detections found, skipping.", logger="current")
            continue

        # Step 2: Pose estimation
        pose_instances = run_MMPose(
            pose_estimator,
            img.copy(),
            detections=det_instances,
            kpt_thr=DEFAULT_KPT_THR,
        )
        # Restrict to first 17 COCO keypoints
        pose_instances.keypoints = pose_instances.keypoints[:, :17, :]
        pose_instances.keypoint_scores = pose_instances.keypoint_scores[:, :17]
        pose_instances.keypoints = np.concatenate(
            [pose_instances.keypoints, pose_instances.keypoint_scores[:, :, None]], axis=-1
        )

        # Step 3: Pose-NMS and SAM refinement
        all_keypoints = (
            pose_instances.keypoints
            if all_detections is None
            else np.concatenate([all_detections.keypoints, pose_instances.keypoints], axis=0)
        )
        all_bboxes = (
            pose_instances.bboxes
            if all_detections is None
            else np.concatenate([all_detections.bboxes, pose_instances.bboxes], axis=0)
        )
        num_valid_kpts = np.sum(all_keypoints[:, :, 2] > bmp_config.sam2.prompting.confidence_thr, axis=1)
        keep_indices = pose_nms(
            DotDict({"confidence_thr": bmp_config.sam2.prompting.confidence_thr, "oks_thr": bmp_config.oks_nms_thr}),
            image_kpts=all_keypoints,
            image_bboxes=all_bboxes,
            num_valid_kpts=num_valid_kpts,
        )
        keep_indices = sorted(keep_indices)  # Sort by original index
        num_old_detections = 0 if all_detections is None else len(all_detections.bboxes)
        keep_new_indices = [i - num_old_detections for i in keep_indices if i >= num_old_detections]
        keep_old_indices = [i for i in keep_indices if i < num_old_detections]
        if len(keep_new_indices) == 0:
            print_log("No new instances passed pose NMS, skipping SAM refinement.", logger="current")
            continue
        # filter new detections and compute scores
        new_dets = filter_instances(pose_instances, keep_new_indices)
        new_dets.scores = pose_instances.keypoint_scores[keep_new_indices].mean(axis=-1)
        old_dets = None
        if len(keep_old_indices) > 0:
            old_dets = filter_instances(all_detections, keep_old_indices)
        print_log(
            "Pose NMS reduced instances to {:d} ({:d}+{:d}) instances".format(
                len(new_dets.bboxes) + num_old_detections, num_old_detections, len(new_dets.bboxes)
            ),
            logger="current",
        )

        new_detections = process_image_with_SAM(
            DotDict(bmp_config.sam2.prompting),
            img.copy(),
            sam2_model,
            new_dets,
            old_dets if old_dets is not None else None,
        )

        # Merge detections
        if all_detections is None:
            all_detections = new_detections
        else:
            all_detections = concat_instances(all_detections, new_dets)

        # Step 4: Visualization
        img_for_detection = visualize_itteration(
            img.copy(),
            all_detections,
            iteration_idx=iteration,
            output_root=str(output_dir),
            img_name=img_path.stem,
        )
        print_log("Iteration {} completed".format(iteration + 1), logger="current")

    # Create GIF of iterations if requested
    if args.create_gif:
        image_file = os.path.join(output_dir, "{:s}.jpg".format(img_path.stem))
        create_GIF(
            img_path=str(image_file),
            output_root=str(output_dir),
            bmp_x=bmp_config.num_bmp_iters,
        )
    return all_detections


def main() -> None:
    """
    Entry point for the BMP demo: loads models and processes one image.
    """
    args = parse_args()
    bmp_config = parse_yaml_config(args.bmp_config)

    # Ensure output root exists
    mmengine.mkdir_or_exist(str(args.output_root))

    # build detectors
    detector = init_detector(bmp_config.detector.det_config, bmp_config.detector.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    if (
        bmp_config.detector.det_config == bmp_config.detector.det_prime_config
        and bmp_config.detector.det_checkpoint == bmp_config.detector.det_prime_checkpoint
    ) or (bmp_config.detector.det_prime_config is None or bmp_config.detector.det_prime_checkpoint is None):
        print_log("Using the same detector as D and D'", logger="current")
        detector_prime = detector
    else:
        detector_prime = init_detector(
            bmp_config.detector.det_prime_config, bmp_config.detector.det_prime_checkpoint, device=args.device
        )
        detector_prime.cfg = adapt_mmdet_pipeline(detector_prime.cfg)
        print_log("Using a different detector for D'", logger="current")

    # build pose estimator
    pose_estimator = init_pose_estimator(
        bmp_config.pose_estimator.pose_config,
        bmp_config.pose_estimator.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )

    sam2 = prepare_sam2_model(
        model_cfg=bmp_config.sam2.sam2_config,
        model_checkpoint=bmp_config.sam2.sam2_checkpoint,
    )

    # Run inference on one image
    _ = process_one_image(args, bmp_config, args.input, detector, detector_prime, pose_estimator, sam2)


if __name__ == "__main__":
    main()
