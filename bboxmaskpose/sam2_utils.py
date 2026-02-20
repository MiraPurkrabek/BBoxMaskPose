# Copyright (c) authors of BBoxMaskPose (BMPv2). All rights reserved.

"""
SAM2 utilities for BMP demo:
- Build and prepare SAM model
- Convert poses to segmentation
- Compute mask-pose consistency
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData
from pycocotools import mask as Mask

from bboxmaskpose.sam2.build_sam import build_sam2
from bboxmaskpose.sam2.sam2_image_predictor import SAM2ImagePredictor

# Threshold for keypoint validity in mask-pose consistency
STRICT_KPT_THRESHOLD: float = 0.5


def _validate_sam_args(sam_args):
    """Validate that all required sam_args attributes are present."""
    required = [
        "crop",
        "use_bbox",
        "confidence_thr",
        "ignore_small_bboxes",
        "num_pos_keypoints",
        "num_pos_keypoints_if_crowd",
        "crowd_by_max_iou",
        "batch",
        "exclusive_masks",
        "extend_bbox",
        "pose_mask_consistency",
        "visibility_thr",
    ]
    for param in required:
        if not hasattr(sam_args, param):
            raise AttributeError(f"Missing required arg {param} in sam_args")


def _get_max_ious(bboxes: List[np.ndarray]) -> np.ndarray:
    """Compute maximum IoU for each bbox against others."""
    if len(bboxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    is_crowd = [0] * len(bboxes)
    ious = Mask.iou(bboxes, bboxes, is_crowd)
    mat = np.array(ious)
    np.fill_diagonal(mat, 0)
    return mat.max(axis=1)


def _compute_one_mask_pose_consistency(
    mask: np.ndarray,
    pos_keypoints: Optional[np.ndarray] = None,
    neg_keypoints: Optional[np.ndarray] = None,
) -> float:
    """Compute a consistency score between a mask and given keypoints."""
    if mask is None:
        return 0.0

    def _mean_inside(points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        pts_int = np.floor(points[:, :2]).astype(int)
        pts_int[:, 0] = np.clip(pts_int[:, 0], 0, mask.shape[1] - 1)
        pts_int[:, 1] = np.clip(pts_int[:, 1], 0, mask.shape[0] - 1)
        vals = mask[pts_int[:, 1], pts_int[:, 0]]
        return vals.mean() if vals.size > 0 else 0.0

    pos_mean = 0.0
    if pos_keypoints is not None:
        valid = pos_keypoints[:, 2] > STRICT_KPT_THRESHOLD
        pos_mean = _mean_inside(pos_keypoints[valid])

    neg_mean = 0.0
    if neg_keypoints is not None:
        valid = neg_keypoints[:, 2] > STRICT_KPT_THRESHOLD
        pts = neg_keypoints[valid][:, :2]
        inside = mask[np.floor(pts[:, 1]).astype(int), np.floor(pts[:, 0]).astype(int)]
        neg_mean = (~inside.astype(bool)).mean() if inside.size > 0 else 0.0

    return 0.5 * pos_mean + 0.5 * neg_mean


def _require_instance_keypoint_channels(
    instances: InstanceData,
    role: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and validate keypoint channels from InstanceData.

    Returns:
        coords: (N, K, 2)
        scores: (N, K)
        visibilities: (N, K)
        probabilities: (N, K)
    """
    if not hasattr(instances, "keypoints") or instances.keypoints is None:
        raise AttributeError(f"{role} instances must contain keypoints")
    if not hasattr(instances, "keypoint_vis") or instances.keypoint_vis is None:
        raise AttributeError(f"{role} instances must contain keypoint_vis")
    if not hasattr(instances, "keypoint_prob") or instances.keypoint_prob is None:
        raise AttributeError(f"{role} instances must contain keypoint_prob")

    keypoints = np.asarray(instances.keypoints)
    visibilities = np.asarray(instances.keypoint_vis)
    probabilities = np.asarray(instances.keypoint_prob)

    if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
        raise ValueError(f"{role} keypoints must have shape (N, K, 3)")

    if hasattr(instances, "keypoint_scores") and instances.keypoint_scores is not None:
        scores = np.asarray(instances.keypoint_scores)
    else:
        scores = keypoints[:, :, 2]

    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if visibilities.ndim == 3 and visibilities.shape[-1] == 1:
        visibilities = visibilities[..., 0]
    if probabilities.ndim == 3 and probabilities.shape[-1] == 1:
        probabilities = probabilities[..., 0]

    expected_shape = keypoints.shape[:2]
    if scores.shape != expected_shape:
        raise ValueError(f"{role} keypoint_scores shape {scores.shape} does not match {expected_shape}")
    if visibilities.shape != expected_shape:
        raise ValueError(f"{role} keypoint_vis shape {visibilities.shape} does not match {expected_shape}")
    if probabilities.shape != expected_shape:
        raise ValueError(f"{role} keypoint_prob shape {probabilities.shape} does not match {expected_shape}")

    coords = keypoints[:, :, :2]
    return coords, scores, visibilities, probabilities


def _select_keypoints(
    args: Any,
    coords: np.ndarray,
    scores: np.ndarray,
    visibilities: np.ndarray,
    num_visible: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    method: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select and order keypoints for SAM prompting using explicit channels.

    Visibility thresholding is always performed over visibilities.
    Method-specific ranking uses:
      - k_most_visible: visibility
      - distance+confidence: confidence

    Returns:
        selected_coords: (M, 2)
        selected_drive_values: (M,) method-dependent ranking signal
        selected_indices: (M,) original keypoint indices
    """
    methods = ["k_most_visible", "distance", "distance+confidence", "closest"]
    sel_method = method or args.selection_method
    if sel_method not in methods:
        raise ValueError(f"Unknown method for keypoint selection: {sel_method}")

    if num_visible <= 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    coords = np.asarray(coords, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    visibilities = np.asarray(visibilities, dtype=np.float32).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (K, 2)")
    if not (coords.shape[0] == scores.shape[0] == visibilities.shape[0]):
        raise ValueError("coords, scores and visibilities must share K")

    kept_indices = np.arange(coords.shape[0])

    # Optional face-anchor for non-k_most_visible methods.
    if sel_method != "k_most_visible" and coords.shape[0] >= 3:
        facial_rel_idx = int(np.argmax(visibilities[:3]))
        if visibilities[facial_rel_idx] >= args.visibility_thr:
            facial_abs_idx = facial_rel_idx
            remaining_abs = np.arange(3, coords.shape[0])
            reorder_abs = np.concatenate([np.array([facial_abs_idx]), remaining_abs])
            coords = coords[reorder_abs]
            scores = scores[reorder_abs]
            visibilities = visibilities[reorder_abs]
            kept_indices = kept_indices[reorder_abs]

    # Visibility filtering for all methods.
    vis_mask = visibilities >= args.visibility_thr
    coords = coords[vis_mask]
    scores = scores[vis_mask]
    visibilities = visibilities[vis_mask]
    kept_indices = kept_indices[vis_mask]

    if coords.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    if sel_method == "k_most_visible":
        order = np.argsort(visibilities)[::-1]
        drive_values = visibilities

    elif sel_method == "distance":
        if bbox is None:
            bbox_center = np.array([coords[:, 0].mean(), coords[:, 1].mean()])
        else:
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(coords - bbox_center, axis=1)
        if coords.shape[0] > 1:
            dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            np.fill_diagonal(dist_matrix, np.inf)
            min_inter_dist = np.min(dist_matrix, axis=1)
        else:
            min_inter_dist = np.zeros((coords.shape[0],), dtype=np.float32)
        order = np.argsort(dists + 3 * min_inter_dist)[::-1]
        drive_values = scores

    elif sel_method == "distance+confidence":
        conf_order = np.argsort(scores)[::-1]
        ordered_coords = coords[conf_order]
        ordered_scores = scores[conf_order]
        ordered_vis = visibilities[conf_order]
        ordered_indices = kept_indices[conf_order]

        if ordered_coords.shape[0] <= 1:
            greedy_indices = np.arange(ordered_coords.shape[0])
        else:
            dist_matrix = np.linalg.norm(ordered_coords[:, None, :] - ordered_coords[None, :, :], axis=2)
            greedy_indices = [0]
            available_scores = ordered_scores.copy()
            available_scores[0] = -1
            for _ in range(ordered_coords.shape[0] - 1):
                min_dist = np.min(dist_matrix[:, greedy_indices], axis=1)
                min_dist[available_scores < np.percentile(available_scores, 80)] = -1
                next_idx = int(np.argmax(min_dist))
                greedy_indices.append(next_idx)
                available_scores[next_idx] = -1
            greedy_indices = np.array(greedy_indices, dtype=np.int64)

        coords = ordered_coords[greedy_indices]
        scores = ordered_scores[greedy_indices]
        visibilities = ordered_vis[greedy_indices]
        kept_indices = ordered_indices[greedy_indices]
        order = np.arange(coords.shape[0])
        drive_values = scores

    else:  # closest
        if bbox is None:
            bbox_center = np.array([coords[:, 0].mean(), coords[:, 1].mean()])
        else:
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(coords - bbox_center, axis=1)
        order = np.argsort(dists)
        drive_values = scores

    selected_coords = coords[order]
    selected_drive_values = drive_values[order]
    selected_indices = kept_indices[order]
    return selected_coords, selected_drive_values, selected_indices


def prepare_model(model_cfg: Any, model_checkpoint: str) -> SAM2ImagePredictor:
    """Build and return a SAM2ImagePredictor model on the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    sam2 = build_sam2(model_cfg, model_checkpoint, device=device, apply_postprocessing=True)
    model = SAM2ImagePredictor(
        sam2,
        max_hole_area=10.0,
        max_sprinkle_area=50.0,
    )
    return model


def _compute_mask_pose_consistency(masks: List[np.ndarray], keypoints_list: List[np.ndarray]) -> np.ndarray:
    """Compute mask-pose consistency score for each mask-keypoints pair."""
    scores: List[float] = []
    for idx, (mask, kpts) in enumerate(zip(masks, keypoints_list)):
        other_kpts = np.concatenate([keypoints_list[:idx], keypoints_list[idx + 1 :]], axis=0).reshape(-1, 3)
        score = _compute_one_mask_pose_consistency(mask, kpts, other_kpts)
        scores.append(score)
    return np.array(scores)


def _pose2seg(
    args: Any,
    model: SAM2ImagePredictor,
    bbox_xyxy: Optional[List[float]] = None,
    pos_coords: Optional[np.ndarray] = None,
    pos_scores: Optional[np.ndarray] = None,
    pos_visibilities: Optional[np.ndarray] = None,
    pos_probabilities: Optional[np.ndarray] = None,
    neg_coords: Optional[np.ndarray] = None,
    neg_scores: Optional[np.ndarray] = None,
    neg_visibilities: Optional[np.ndarray] = None,
    neg_probabilities: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    gt_mask: Optional[Any] = None,
    num_pos_keypoints: Optional[int] = None,
    gt_mask_is_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run SAM segmentation conditioned on explicit keypoint channels."""
    num_pos_keypoints = args.num_pos_keypoints if num_pos_keypoints is None else num_pos_keypoints

    # Positive keypoints.
    if pos_coords is not None and pos_scores is not None and pos_visibilities is not None:
        pos_coords = pos_coords.reshape(-1, 2)
        pos_scores = pos_scores.reshape(-1)
        pos_visibilities = pos_visibilities.reshape(-1)
        if pos_probabilities is not None:
            _ = pos_probabilities.reshape(-1)

        valid_pos = (pos_scores > args.confidence_thr) & (pos_visibilities > args.visibility_thr)
        pos_kpts, pos_drive, _ = _select_keypoints(
            args,
            coords=pos_coords,
            scores=pos_scores,
            visibilities=pos_visibilities,
            num_visible=int(np.sum(valid_pos)),
            bbox=bbox_xyxy,
        )
        pos_kpts_backup = (
            np.concatenate([pos_kpts, pos_drive[:, None]], axis=1) if pos_kpts.shape[0] > 0 else np.empty((0, 3), dtype=np.float32)
        )

        if pos_kpts.shape[0] > num_pos_keypoints:
            pos_kpts = pos_kpts[:num_pos_keypoints, :]
            pos_kpts_backup = pos_kpts_backup[:num_pos_keypoints, :]
    else:
        pos_kpts = np.empty((0, 2), dtype=np.float32)
        pos_kpts_backup = np.empty((0, 3), dtype=np.float32)

    # Negative keypoints.
    if neg_coords is not None and neg_scores is not None and neg_visibilities is not None:
        neg_coords = neg_coords.reshape(-1, 2)
        neg_scores = neg_scores.reshape(-1)
        neg_visibilities = neg_visibilities.reshape(-1)
        if neg_probabilities is not None:
            _ = neg_probabilities.reshape(-1)

        valid_neg = (neg_scores > args.confidence_thr) & (neg_visibilities > args.visibility_thr)
        neg_kpts, neg_drive, _ = _select_keypoints(
            args,
            coords=neg_coords,
            scores=neg_scores,
            visibilities=neg_visibilities,
            num_visible=int(np.sum(valid_neg)),
            bbox=bbox_xyxy,
            method="closest",
        )
        selected_neg_kpts = neg_kpts
        neg_kpts_backup = (
            np.concatenate([neg_kpts, neg_drive[:, None]], axis=1) if neg_kpts.shape[0] > 0 else np.empty((0, 3), dtype=np.float32)
        )

        if neg_kpts.shape[0] > args.num_neg_keypoints:
            selected_neg_kpts = neg_kpts[: args.num_neg_keypoints, :]
    else:
        selected_neg_kpts = np.empty((0, 2), dtype=np.float32)
        neg_kpts_backup = np.empty((0, 3), dtype=np.float32)

    # SAM prompts.
    kpts = np.concatenate([pos_kpts, selected_neg_kpts], axis=0)
    kpts_labels = np.concatenate([np.ones(pos_kpts.shape[0]), np.zeros(selected_neg_kpts.shape[0])], axis=0)

    bbox = bbox_xyxy if args.use_bbox else None

    if args.extend_bbox and bbox is not None and pos_kpts.shape[0] > 0:
        pose_bbox = np.array([pos_kpts[:, 0].min() - 2, pos_kpts[:, 1].min() - 2, pos_kpts[:, 0].max() + 2, pos_kpts[:, 1].max() + 2])
        expanded_bbox = np.array(bbox)
        expanded_bbox[:2] = np.minimum(bbox[:2], pose_bbox[:2])
        expanded_bbox[2:] = np.maximum(bbox[2:], pose_bbox[2:])
        bbox = expanded_bbox

    if args.crop and args.use_bbox and image is not None:
        crop_bbox = np.array(bbox)
        bbox_center = np.array([(crop_bbox[0] + crop_bbox[2]) / 2, (crop_bbox[1] + crop_bbox[3]) / 2])
        bbox_size = np.array([crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]])
        bbox_size = 1.5 * bbox_size
        crop_bbox = np.array(
            [
                bbox_center[0] - bbox_size[0] / 2,
                bbox_center[1] - bbox_size[1] / 2,
                bbox_center[0] + bbox_size[0] / 2,
                bbox_center[1] + bbox_size[1] / 2,
            ]
        )
        crop_bbox = np.round(crop_bbox).astype(int)
        crop_bbox = np.clip(crop_bbox, 0, [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        original_image_size = image.shape[:2]
        image = image[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2], :]

        kpts = kpts - crop_bbox[:2]
        bbox[:2] = bbox[:2] - crop_bbox[:2]
        bbox[2:] = bbox[2:] - crop_bbox[:2]

        model.set_image(image)

    masks, scores, logits = model.predict(
        point_coords=kpts,
        point_labels=kpts_labels,
        box=bbox,
        multimask_output=False,
    )
    mask = masks[0]
    scores = scores[0]

    if args.crop and args.use_bbox and image is not None:
        mask_padded = np.zeros(original_image_size, dtype=np.uint8)
        mask_padded[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]] = mask
        mask = mask_padded

        bbox[:2] = bbox[:2] + crop_bbox[:2]
        bbox[2:] = bbox[2:] + crop_bbox[:2]

    if args.pose_mask_consistency:
        if gt_mask_is_binary:
            gt_mask_binary = gt_mask
        else:
            gt_mask_binary = Mask.decode(gt_mask).astype(bool) if gt_mask is not None else None

        gt_mask_pose_consistency = _compute_one_mask_pose_consistency(gt_mask_binary, pos_kpts_backup, neg_kpts_backup)
        dt_mask_pose_consistency = _compute_one_mask_pose_consistency(mask, pos_kpts_backup, neg_kpts_backup)

        tol = 0.1
        dt_is_same = np.abs(dt_mask_pose_consistency - gt_mask_pose_consistency) < tol
        if dt_is_same:
            mask = gt_mask_binary if gt_mask_binary.sum() < mask.sum() else mask
        else:
            mask = gt_mask_binary if gt_mask_pose_consistency > dt_mask_pose_consistency else mask

    return mask, pos_kpts_backup, neg_kpts_backup, scores


def process_image_with_SAM(
    sam_args: Any,
    image: np.ndarray,
    model: SAM2ImagePredictor,
    new_dets: InstanceData,
    old_dets: Optional[InstanceData] = None,
) -> InstanceData:
    """Validate args and route to single or batch processing."""
    _validate_sam_args(sam_args)
    if sam_args.batch:
        return _process_image_batch(sam_args, image, model, new_dets, old_dets)
    return _process_image_single(sam_args, image, model, new_dets, old_dets)


def _process_image_single(
    sam_args: Any,
    image: np.ndarray,
    model: SAM2ImagePredictor,
    new_dets: InstanceData,
    old_dets: Optional[InstanceData] = None,
) -> InstanceData:
    """Refine instance segmentation masks using SAM2 with pose-conditioned prompts."""
    _validate_sam_args(sam_args)

    if not (sam_args.crop and sam_args.use_bbox):
        model.set_image(image)

    new_coords, new_scores, new_visibilities, new_probabilities = _require_instance_keypoint_channels(new_dets, role="new")

    n_new_dets = len(new_dets.bboxes)
    n_old_dets = 0
    if old_dets is not None:
        n_old_dets = len(old_dets.bboxes)
        old_coords, old_scores, old_visibilities, old_probabilities = _require_instance_keypoint_channels(old_dets, role="old")

    all_bboxes = new_dets.bboxes.copy()
    if old_dets is not None:
        all_bboxes = np.concatenate([all_bboxes, old_dets.bboxes], axis=0)

    max_ious = _get_max_ious(all_bboxes)

    new_dets.refined_masks = np.zeros((n_new_dets, image.shape[0], image.shape[1]), dtype=np.uint8)
    new_dets.sam_scores = np.zeros_like(new_dets.bbox_scores)
    new_dets.sam_kpts = np.zeros((len(new_dets.bboxes), sam_args.num_pos_keypoints, 3), dtype=np.float32)

    for instance_idx in range(len(new_dets.bboxes)):
        bbox_xywh = new_dets.bboxes[instance_idx]
        bbox_area = bbox_xywh[2] * bbox_xywh[3]

        if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
            continue

        dt_mask = new_dets.pred_masks[instance_idx] if hasattr(new_dets, "pred_masks") else None

        bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]

        this_coords = new_coords[instance_idx]
        this_scores = new_scores[instance_idx]
        this_vis = new_visibilities[instance_idx]
        this_probs = new_probabilities[instance_idx]

        other_coords = None
        other_scores = None
        other_vis = None
        other_probs = None

        if old_dets is not None:
            other_coords = old_coords.copy().reshape(n_old_dets, -1, 2)
            other_scores = old_scores.copy().reshape(n_old_dets, -1)
            other_vis = old_visibilities.copy().reshape(n_old_dets, -1)
            other_probs = old_probabilities.copy().reshape(n_old_dets, -1)

        if len(new_coords) > 1:
            other_new_coords = np.concatenate([new_coords[:instance_idx], new_coords[instance_idx + 1 :]], axis=0)
            other_new_scores = np.concatenate([new_scores[:instance_idx], new_scores[instance_idx + 1 :]], axis=0)
            other_new_vis = np.concatenate([new_visibilities[:instance_idx], new_visibilities[instance_idx + 1 :]], axis=0)
            other_new_probs = np.concatenate([new_probabilities[:instance_idx], new_probabilities[instance_idx + 1 :]], axis=0)

            other_coords = np.concatenate([other_coords, other_new_coords], axis=0) if other_coords is not None else other_new_coords
            other_scores = np.concatenate([other_scores, other_new_scores], axis=0) if other_scores is not None else other_new_scores
            other_vis = np.concatenate([other_vis, other_new_vis], axis=0) if other_vis is not None else other_new_vis
            other_probs = np.concatenate([other_probs, other_new_probs], axis=0) if other_probs is not None else other_new_probs

        num_pos_keypoints = sam_args.num_pos_keypoints
        if sam_args.crowd_by_max_iou is not None and max_ious[instance_idx] > sam_args.crowd_by_max_iou:
            bbox_xyxy = None
            num_pos_keypoints = sam_args.num_pos_keypoints_if_crowd

        dt_mask, pos_kpts, neg_kpts, scores = _pose2seg(
            sam_args,
            model,
            bbox_xyxy,
            pos_coords=this_coords,
            pos_scores=this_scores,
            pos_visibilities=this_vis,
            pos_probabilities=this_probs,
            neg_coords=other_coords,
            neg_scores=other_scores,
            neg_visibilities=other_vis,
            neg_probabilities=other_probs,
            image=image if (sam_args.crop and sam_args.use_bbox) else None,
            gt_mask=dt_mask,
            num_pos_keypoints=num_pos_keypoints,
            gt_mask_is_binary=True,
        )

        new_dets.refined_masks[instance_idx] = dt_mask
        new_dets.sam_scores[instance_idx] = scores

        if len(pos_kpts) != sam_args.num_pos_keypoints:
            pos_kpts = np.concatenate([pos_kpts, np.zeros((sam_args.num_pos_keypoints - len(pos_kpts), 3), dtype=np.float32)], axis=0)
        new_dets.sam_kpts[instance_idx] = pos_kpts

    n_masks = len(new_dets.refined_masks) + (len(old_dets.refined_masks) if old_dets is not None else 0)

    if sam_args.exclusive_masks and n_masks > 1:
        all_masks = (
            np.concatenate([new_dets.refined_masks, old_dets.refined_masks], axis=0) if old_dets is not None else new_dets.refined_masks
        )
        all_scores = np.concatenate([new_dets.sam_scores, old_dets.sam_scores], axis=0) if old_dets is not None else new_dets.sam_scores
        refined_masks = _apply_exclusive_masks(all_masks, all_scores)
        new_dets.refined_masks = refined_masks[: len(new_dets.refined_masks)]

    return new_dets


def _process_image_batch(
    sam_args: Any,
    image: np.ndarray,
    model: SAM2ImagePredictor,
    new_dets: InstanceData,
    old_dets: Optional[InstanceData] = None,
) -> InstanceData:
    """Batch process multiple detection instances with SAM2 refinement."""
    n_new_dets = len(new_dets.bboxes)

    model.set_image(image)

    new_coords, new_scores, new_visibilities, new_probabilities = _require_instance_keypoint_channels(new_dets, role="new")

    image_coords = []
    image_scores = []
    image_visibilities = []
    image_probabilities = []
    image_bboxes = []
    num_valid_kpts = []

    for instance_idx in range(len(new_dets.bboxes)):
        bbox_xywh = new_dets.bboxes[instance_idx].copy()
        bbox_area = bbox_xywh[2] * bbox_xywh[3]
        if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
            continue

        this_coords = new_coords[instance_idx].copy().reshape(-1, 2)
        this_scores = new_scores[instance_idx].copy().reshape(-1)
        this_vis = new_visibilities[instance_idx].copy().reshape(-1)
        this_probs = new_probabilities[instance_idx].copy().reshape(-1)

        visible_kpts = (this_vis > sam_args.visibility_thr) & (this_scores > sam_args.confidence_thr)
        num_visible = int(visible_kpts.sum())
        if num_visible <= 0:
            continue

        num_valid_kpts.append(num_visible)
        image_bboxes.append(np.array(bbox_xywh))
        image_coords.append(this_coords)
        image_scores.append(this_scores)
        image_visibilities.append(this_vis)
        image_probabilities.append(this_probs)

    if old_dets is not None:
        old_coords, old_scores, old_visibilities, old_probabilities = _require_instance_keypoint_channels(old_dets, role="old")
        for instance_idx in range(len(old_dets.bboxes)):
            bbox_xywh = old_dets.bboxes[instance_idx].copy()
            bbox_area = bbox_xywh[2] * bbox_xywh[3]
            if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
                continue

            this_coords = old_coords[instance_idx].reshape(-1, 2)
            this_scores = old_scores[instance_idx].reshape(-1)
            this_vis = old_visibilities[instance_idx].reshape(-1)
            this_probs = old_probabilities[instance_idx].reshape(-1)

            visible_kpts = (this_vis > sam_args.visibility_thr) & (this_scores > sam_args.confidence_thr)
            num_visible = int(visible_kpts.sum())
            if num_visible <= 0:
                continue

            num_valid_kpts.append(num_visible)
            image_bboxes.append(np.array(bbox_xywh))
            image_coords.append(this_coords)
            image_scores.append(this_scores)
            image_visibilities.append(this_vis)
            image_probabilities.append(this_probs)

    if len(image_bboxes) == 0:
        new_dets.refined_masks = np.zeros((n_new_dets, image.shape[0], image.shape[1]), dtype=np.uint8)
        new_dets.sam_scores = np.zeros((n_new_dets,), dtype=np.float32)
        new_dets.sam_kpts = np.zeros((n_new_dets, sam_args.num_pos_keypoints, 3), dtype=np.float32)
        return new_dets

    image_coords = np.array(image_coords)
    image_scores = np.array(image_scores)
    image_visibilities = np.array(image_visibilities)
    image_probabilities = np.array(image_probabilities)
    image_bboxes = np.array(image_bboxes)
    num_valid_kpts = np.array(num_valid_kpts)

    prepared_kpts = []
    prepared_kpts_backup = []
    for bbox, coords, scores, visibilities, probabilities, num_visible in zip(
        image_bboxes,
        image_coords,
        image_scores,
        image_visibilities,
        image_probabilities,
        num_valid_kpts,
    ):
        _ = probabilities  # extracted for clarity; currently unused in selection
        this_kpts, this_drive_vals, _ = _select_keypoints(
            sam_args,
            coords=coords,
            scores=scores,
            visibilities=visibilities,
            num_visible=int(num_visible),
            bbox=bbox,
        )

        if this_kpts.shape[0] == 0:
            continue

        if this_kpts.shape[0] < num_valid_kpts.max():
            this_kpts = np.concatenate([this_kpts, np.tile(this_kpts[-1], (num_valid_kpts.max() - this_kpts.shape[0], 1))], axis=0)
            this_drive_vals = np.concatenate(
                [this_drive_vals, np.tile(this_drive_vals[-1], (num_valid_kpts.max() - this_drive_vals.shape[0],))],
                axis=0,
            )

        prepared_kpts.append(this_kpts)
        prepared_kpts_backup.append(np.concatenate([this_kpts, this_drive_vals[:, None]], axis=1))

    if len(prepared_kpts) == 0:
        new_dets.refined_masks = np.zeros((n_new_dets, image.shape[0], image.shape[1]), dtype=np.uint8)
        new_dets.sam_scores = np.zeros((n_new_dets,), dtype=np.float32)
        new_dets.sam_kpts = np.zeros((n_new_dets, sam_args.num_pos_keypoints, 3), dtype=np.float32)
        return new_dets

    image_kpts = np.array(prepared_kpts)
    image_kpts_backup = np.array(prepared_kpts_backup)
    kpts_labels = np.ones(image_kpts.shape[:2])

    max_ious = _get_max_ious(image_bboxes)
    num_pos_keypoints = sam_args.num_pos_keypoints
    use_bbox = sam_args.use_bbox
    if sam_args.crowd_by_max_iou is not None and len(max_ious) > 0 and max_ious.max() > sam_args.crowd_by_max_iou:
        use_bbox = False
        num_pos_keypoints = sam_args.num_pos_keypoints_if_crowd

    if num_pos_keypoints > 0 and num_pos_keypoints < image_kpts.shape[1]:
        image_kpts = image_kpts[:, :num_pos_keypoints, :]
        kpts_labels = kpts_labels[:, :num_pos_keypoints]
        image_kpts_backup = image_kpts_backup[:, :num_pos_keypoints, :]
    elif num_pos_keypoints == 0:
        image_kpts = None
        kpts_labels = None
        image_kpts_backup = np.empty((0, 3), dtype=np.float32)

    image_bboxes_xyxy = None
    if use_bbox:
        image_bboxes_xyxy = np.array(image_bboxes)
        image_bboxes_xyxy[:, 2:] += image_bboxes_xyxy[:, :2]

        if sam_args.extend_bbox and image_kpts is not None and image_kpts.size > 0:
            pose_bbox = np.stack(
                [
                    np.min(image_kpts[:, :, 0], axis=1) - 2,
                    np.min(image_kpts[:, :, 1], axis=1) - 2,
                    np.max(image_kpts[:, :, 0], axis=1) + 2,
                    np.max(image_kpts[:, :, 1], axis=1) + 2,
                ],
                axis=1,
            )
            expanded_bbox = np.array(image_bboxes_xyxy)
            expanded_bbox[:, :2] = np.minimum(expanded_bbox[:, :2], pose_bbox[:, :2])
            expanded_bbox[:, 2:] = np.maximum(expanded_bbox[:, 2:], pose_bbox[:, 2:])
            image_bboxes_xyxy = expanded_bbox

    masks, scores, logits = model.predict(
        point_coords=image_kpts,
        point_labels=kpts_labels,
        box=image_bboxes_xyxy,
        multimask_output=False,
    )

    if len(masks.shape) == 3:
        masks = masks[None, :, :, :]
    masks = masks[:, 0, :, :]
    n_masks_out = masks.shape[0]
    scores = scores.reshape(n_masks_out)

    if sam_args.exclusive_masks and n_masks_out > 1:
        masks = _apply_exclusive_masks(masks, scores)

    gt_masks = new_dets.pred_masks.copy() if new_dets.pred_masks is not None else None
    if sam_args.pose_mask_consistency and gt_masks is not None:
        dt_mask_pose_consistency = _compute_mask_pose_consistency(masks, image_kpts_backup)
        gt_mask_pose_consistency = _compute_mask_pose_consistency(gt_masks, image_kpts_backup)

        dt_masks_area = np.array([m.sum() for m in masks])
        gt_masks_area = np.array([m.sum() for m in gt_masks]) if gt_masks is not None else np.zeros_like(dt_masks_area)

        tol = 0.1
        pmc_is_equal = np.isclose(dt_mask_pose_consistency, gt_mask_pose_consistency, atol=tol)
        dt_is_worse = (dt_mask_pose_consistency < (gt_mask_pose_consistency - tol)) | (pmc_is_equal & (dt_masks_area > gt_masks_area))

        new_masks = []
        for dt_mask, gt_mask, dt_worse in zip(masks, gt_masks, dt_is_worse):
            new_masks.append(gt_mask if dt_worse else dt_mask)
        masks = np.array(new_masks)

    new_dets.refined_masks = masks[:n_new_dets]
    new_dets.sam_scores = scores[:n_new_dets]
    new_dets.sam_kpts = image_kpts_backup[:n_new_dets]

    return new_dets


def _apply_exclusive_masks(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Ensure masks are non-overlapping by keeping per pixel the highest-score mask."""
    no_mask = masks.sum(axis=0) == 0
    masked_scores = masks * scores[:, None, None]
    argmax_masks = np.argmax(masked_scores, axis=0)
    new_masks = argmax_masks[None, :, :] == (np.arange(masks.shape[0])[:, None, None])
    new_masks[:, no_mask] = 0
    return new_masks
