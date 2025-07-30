"""
This module provides high-level interfaces to run MMDetection and MMPose
models sequentially. Users can call run_MMDetector and run_MMPose from
other scripts (e.g., bmp_demo.py) to perform object detection and
pose estimation in a clean, modular fashion.
"""

import numpy as np
from mmdet.apis import inference_detector
from mmengine.structures import InstanceData

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples


def run_MMDetector(detector, image, det_cat_id: int = 0, bbox_thr: float = 0.3, nms_thr: float = 0.3) -> InstanceData:
    """
    Run an MMDetection model to detect bounding boxes (and masks) in an image.

    Args:
        detector: An initialized MMDetection detector model.
        image: Input image as file path or BGR numpy array.
        det_cat_id: Category ID to filter detections (default is 0 for 'person').
        bbox_thr: Minimum bounding box score threshold.
        nms_thr: IoU threshold for Non-Maximum Suppression (NMS).

    Returns:
        InstanceData: A structure containing filtered bboxes, bbox_scores, and masks (if available).
    """
    # Run detection
    det_result = inference_detector(detector, image)
    pred_instances = det_result.pred_instances.cpu().numpy()

    # Aggregate bboxes and scores into an (N, 5) array
    bboxes_all = np.concatenate((pred_instances.bboxes, pred_instances.scores[:, None]), axis=1)

    # Filter by category and score
    keep_mask = np.logical_and(pred_instances.labels == det_cat_id, pred_instances.scores > bbox_thr)
    if not np.any(keep_mask):
        # Return empty structure if nothing passes threshold
        return InstanceData(bboxes=np.zeros((0, 4)), bbox_scores=np.zeros((0,)), masks=np.zeros((0, 1, 1)))

    bboxes = bboxes_all[keep_mask]
    masks = getattr(pred_instances, "masks", None)
    if masks is not None:
        masks = masks[keep_mask]

    # Sort detections by descending score
    order = np.argsort(bboxes[:, 4])[::-1]
    bboxes = bboxes[order]
    if masks is not None:
        masks = masks[order]

    # Apply Non-Maximum Suppression
    keep_indices = nms(bboxes, nms_thr)
    bboxes = bboxes[keep_indices]
    if masks is not None:
        masks = masks[keep_indices]

    # Construct InstanceData to return
    det_instances = InstanceData(bboxes=bboxes[:, :4], bbox_scores=bboxes[:, 4], masks=masks)
    return det_instances


def run_MMPose(pose_estimator, image, detections: InstanceData, kpt_thr: float = 0.3) -> InstanceData:
    """
    Run an MMPose top-down model to estimate human pose given detected bounding boxes.

    Args:
        pose_estimator: An initialized MMPose model.
        image: Input image as file path or RGB/BGR numpy array.
        detections: InstanceData from run_MMDetector containing bboxes and masks.
        kpt_thr: Minimum keypoint score threshold to filter low-confidence joints.

    Returns:
        InstanceData: A structure containing estimated keypoints, keypoint_scores,
                      original bboxes, and masks (if provided).
    """
    # Extract bounding boxes
    bboxes = detections.bboxes
    if bboxes.shape[0] == 0:
        # No detections => empty pose data
        return InstanceData(
            keypoints=np.zeros((0, 17, 3)),
            keypoint_scores=np.zeros((0, 17)),
            bboxes=bboxes,
            bbox_scores=detections.bbox_scores,
            masks=detections.masks,
        )

    # Run top-down pose estimation
    pose_results = inference_topdown(pose_estimator, image, bboxes, masks=detections.masks)
    data_samples = merge_data_samples(pose_results)

    # Attach masks back into the data_samples if available
    if detections.masks is not None:
        data_samples.pred_instances.pred_masks = detections.masks

    # Filter out low-confidence keypoints
    kp_scores = data_samples.pred_instances.keypoint_scores
    kp_mask = kp_scores >= kpt_thr
    # data_samples.pred_instances.keypoints[~kp_mask] = [0, 0, 0]

    # Return final InstanceData for poses
    return data_samples.pred_instances
