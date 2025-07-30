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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Threshold for keypoint validity in mask-pose consistency
STRICT_KPT_THRESHOLD: float = 0.5


def _validate_sam_args(sam_args):
    """
    Validate that all required sam_args attributes are present.
    """
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
    """
    Compute maximum IoU for each bbox against others.
    """
    is_crowd = [0] * len(bboxes)
    ious = Mask.iou(bboxes, bboxes, is_crowd)
    mat = np.array(ious)
    np.fill_diagonal(mat, 0)
    return mat.max(axis=1)


def _compute_one_mask_pose_consistency(
    mask: np.ndarray, pos_keypoints: Optional[np.ndarray] = None, neg_keypoints: Optional[np.ndarray] = None
) -> float:
    """
    Compute a consistency score between a mask and given keypoints.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
        pos_keypoints (Optional[np.ndarray]): Positive keypoints array (N, 3).
        neg_keypoints (Optional[np.ndarray]): Negative keypoints array (M, 3).

    Returns:
        float: Weighted mean of positive and negative keypoint consistency.
    """
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


def _select_keypoints(
    args: Any,
    kpts: np.ndarray,
    num_visible: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    method: Optional[str] = "distance+confidence",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select and order keypoints for SAM prompting based on specified method.

    Args:
        args: Configuration object with selection_method and visibility_thr attributes.
        kpts (np.ndarray): Keypoints array of shape (K, 3).
        num_visible (int): Number of keypoints above visibility threshold.
        bbox (Optional[Tuple]): Optional bbox for distance methods.
        method (Optional[str]): Override selection method.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Selected keypoint coordinates (N,2) and confidences (N,).

    Raises:
        ValueError: If an unknown method is specified.
    """
    if num_visible == 0:
        return kpts[:, :2], kpts[:, 2]

    methods = ["confidence", "distance", "distance+confidence", "closest"]
    sel_method = method or args.selection_method
    if sel_method not in methods:
        raise ValueError("Unknown method for keypoint selection: {}".format(sel_method))

    # Select at maximum keypoint from the face
    facial_kpts = kpts[:3, :]
    facial_conf = kpts[:3, 2]
    facial_point = facial_kpts[np.argmax(facial_conf)]
    if facial_point[-1] >= args.visibility_thr:
        kpts = np.concatenate([facial_point[None, :], kpts[3:]], axis=0)

    conf = kpts[:, 2]
    vis_mask = conf >= args.visibility_thr
    coords = kpts[vis_mask, :2]
    confs = conf[vis_mask]

    if sel_method == "confidence":
        order = np.argsort(confs)[::-1]
        coords = coords[order]
        confs = confs[order]
    elif sel_method == "distance":
        if bbox is None:
            bbox_center = np.array([coords[:, 0].mean(), coords[:, 1].mean()])
        else:
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(coords[:, :2] - bbox_center, axis=1)
        dist_matrix = np.linalg.norm(coords[:, None, :2] - coords[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        order = np.argsort(dists + 3 * min_inter_dist)[::-1]
        coords = coords[order, :2]
        confs = confs[order]
    elif sel_method == "distance+confidence":
        order = np.argsort(confs)[::-1]
        confidences = kpts[order, 2]
        coords = coords[order, :2]
        confs = confs[order]

        dist_matrix = np.linalg.norm(coords[:, None, :2] - coords[None, :, :2], axis=2)

        selected_idx = [0]
        confidences[0] = -1
        for _ in range(coords.shape[0] - 1):
            min_dist = np.min(dist_matrix[:, selected_idx], axis=1)
            min_dist[confidences < np.percentile(confidences, 80)] = -1

            next_idx = np.argmax(min_dist)
            selected_idx.append(next_idx)
            confidences[next_idx] = -1

        coords = coords[selected_idx]
        confs = confs[selected_idx]
    elif sel_method == "closest":
        coords = coords[confs > STRICT_KPT_THRESHOLD, :]
        confs = confs[confs > STRICT_KPT_THRESHOLD]
        if bbox is None:
            bbox_center = np.array([coords[:, 0].mean(), coords[:, 1].mean()])
        else:
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(coords[:, :2] - bbox_center, axis=1)
        order = np.argsort(dists)
        coords = coords[order, :2]
        confs = confs[order]

    return coords, confs


def prepare_model(model_cfg: Any, model_checkpoint: str) -> SAM2ImagePredictor:
    """
    Build and return a SAM2ImagePredictor model on the appropriate device.

    Args:
        model_cfg: Configuration for SAM2 model.
        model_checkpoint (str): Path to model checkpoint.

    Returns:
        SAM2ImagePredictor: Initialized SAM2 image predictor.
    """
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
    """
    Compute mask-pose consistency score for each mask-keypoints pair.

    Args:
        masks (List[np.ndarray]): Binary masks list.
        keypoints_list (List[np.ndarray]): List of keypoint arrays per instance.

    Returns:
        np.ndarray: Consistency scores array of shape (N,).
    """
    scores: List[float] = []
    for mask, kpts in zip(masks, keypoints_list):
        other_kpts = np.concatenate([keypoints_list[:idx], keypoints_list[idx + 1 :]], axis=0).reshape(-1, 3)
        score = _compute_one_mask_pose_consistency(mask, kpts, other_kpts)
        scores.append(score)

    return np.array(scores)


def _pose2seg(
    args: Any,
    model: SAM2ImagePredictor,
    bbox_xyxy: Optional[List[float]] = None,
    pos_kpts: Optional[np.ndarray] = None,
    neg_kpts: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    gt_mask: Optional[Any] = None,
    num_pos_keypoints: Optional[int] = None,
    gt_mask_is_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run SAM segmentation conditioned on pose keypoints and optional ground truth mask.

    Args:
        args: Configuration object with prompting settings.
        model (SAM2ImagePredictor): Prepared SAM2 model.
        bbox_xyxy (Optional[List[float]]): Bounding box coordinates in xyxy format.
        pos_kpts (Optional[np.ndarray]): Positive keypoints array.
        neg_kpts (Optional[np.ndarray]): Negative keypoints array.
        image (Optional[np.ndarray]): Input image array.
        gt_mask (Optional[Any]): Ground truth mask (optional).
        num_pos_keypoints (Optional[int]): Number of positive keypoints to use.
        gt_mask_is_binary (bool): Flag indicating if ground truth mask is binary.

    Returns:
        Tuple of (mask, pos_kpts_backup, neg_kpts_backup, score).
    """
    num_pos_keypoints = args.num_pos_keypoints if num_pos_keypoints is None else num_pos_keypoints

    # Filter-out un-annotated and invisible keypoints
    if pos_kpts is not None:
        pos_kpts = pos_kpts.reshape(-1, 3)
        valid_kpts = pos_kpts[:, 2] > args.visibility_thr

        pose_bbox = np.array([pos_kpts[:, 0].min(), pos_kpts[:, 1].min(), pos_kpts[:, 0].max(), pos_kpts[:, 1].max()])
        pos_kpts, conf = _select_keypoints(args, pos_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy)

        pos_kpts_backup = np.concatenate([pos_kpts, conf[:, None]], axis=1)

        if pos_kpts.shape[0] > num_pos_keypoints:
            pos_kpts = pos_kpts[:num_pos_keypoints, :]
            pos_kpts_backup = pos_kpts_backup[:num_pos_keypoints, :]

    else:
        pose_bbox = None
        pos_kpts = np.empty((0, 2), dtype=np.float32)
        pos_kpts_backup = np.empty((0, 3), dtype=np.float32)

    if neg_kpts is not None:
        neg_kpts = neg_kpts.reshape(-1, 3)
        valid_kpts = neg_kpts[:, 2] > args.visibility_thr

        neg_kpts, conf = _select_keypoints(
            args, neg_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy, method="closest"
        )
        selected_neg_kpts = neg_kpts
        neg_kpts_backup = np.concatenate([neg_kpts, conf[:, None]], axis=1)

        if neg_kpts.shape[0] > args.num_neg_keypoints:
            selected_neg_kpts = neg_kpts[: args.num_neg_keypoints, :]

    else:
        selected_neg_kpts = np.empty((0, 2), dtype=np.float32)
        neg_kpts_backup = np.empty((0, 3), dtype=np.float32)

    # Concatenate positive and negative keypoints
    kpts = np.concatenate([pos_kpts, selected_neg_kpts], axis=0)
    kpts_labels = np.concatenate([np.ones(pos_kpts.shape[0]), np.zeros(selected_neg_kpts.shape[0])], axis=0)

    bbox = bbox_xyxy if args.use_bbox else None

    if args.extend_bbox and not bbox is None:
        # Expand the bbox such that it contains all positive keypoints
        pose_bbox = np.array(
            [pos_kpts[:, 0].min() - 2, pos_kpts[:, 1].min() - 2, pos_kpts[:, 0].max() + 2, pos_kpts[:, 1].max() + 2]
        )
        expanded_bbox = np.array(bbox)
        expanded_bbox[:2] = np.minimum(bbox[:2], pose_bbox[:2])
        expanded_bbox[2:] = np.maximum(bbox[2:], pose_bbox[2:])
        bbox = expanded_bbox

    if args.crop and args.use_bbox and image is not None:
        # Crop the image to the 1.5 * bbox size
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

        # Update the keypoints
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
        # Pad the mask to the original image size
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
    """
    Wrapper that validates args and routes to single or batch processing.
    """
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
    """
    Refine instance segmentation masks using SAM2 with pose-conditioned prompts.

    Args:
        sam_args (Any): DotDict containing required SAM parameters:
            crop (bool), use_bbox (bool), confidence_thr (float),
            ignore_small_bboxes (bool), num_pos_keypoints (int),
            num_pos_keypoints_if_crowd (int), crowd_by_max_iou (Optional[float]),
            batch (bool), exclusive_masks (bool), extend_bbox (bool), pose_mask_consistency (bool).
        image (np.ndarray): BGR image array of shape (H, W, 3).
        model (SAM2ImagePredictor): Initialized SAM2 predictor.
        new_dets (InstanceData): New detections with attributes:
            bboxes, pred_masks, keypoints, bbox_scores.
        old_dets (Optional[InstanceData]): Previous detections for negative prompts.

    Returns:
        InstanceData: `new_dets` updated in-place with
            `.refined_masks`, `.sam_scores`, and `.sam_kpts`.
    """
    _validate_sam_args(sam_args)

    if not (sam_args.crop and sam_args.use_bbox):
        model.set_image(image)

    # Ignore all keypoints with confidence below the threshold
    new_keypoints = new_dets.keypoints.copy()
    for kpts in new_keypoints:
        conf_mask = kpts[:, 2] < sam_args.confidence_thr
        kpts[conf_mask, :] = 0
    if old_dets is not None:
        old_keypoints = old_dets.keypoints.copy()
        for kpts in old_keypoints:
            conf_mask = kpts[:, 2] < sam_args.confidence_thr
            kpts[conf_mask, :] = 0

    all_bboxes = new_dets.bboxes.copy()
    if old_dets is not None:
        all_bboxes = np.concatenate([all_bboxes, old_dets.bboxes], axis=0)

    max_ious = _get_max_ious(all_bboxes)

    gt_bboxes = []
    new_dets.refined_masks = np.zeros_like(new_dets.pred_masks)
    new_dets.sam_scores = np.zeros_like(new_dets.bbox_scores)
    new_dets.sam_kpts = np.zeros((len(new_dets.bboxes), sam_args.num_pos_keypoints, 3), dtype=np.float32)
    for instance_idx in range(len(new_dets.bboxes)):
        bbox_xywh = new_dets.bboxes[instance_idx]
        bbox_area = bbox_xywh[2] * bbox_xywh[3]

        if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
            continue
        dt_mask = new_dets.pred_masks[instance_idx] if new_dets.pred_masks is not None else None

        bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
        gt_bboxes.append(bbox_xyxy)
        this_kpts = new_keypoints[instance_idx].reshape(1, -1, 3)
        other_kpts = None
        if old_dets is not None:
            other_kpts = old_keypoints.copy().reshape(-1, 3)
        if len(new_keypoints) > 1:
            other_new_kpts = np.concatenate(new_keypoints[:instance_idx] + new_keypoints[instance_idx + 1 :], axis=0)
            other_kpts = (
                np.concatenate([other_kpts, other_new_kpts], axis=0) if other_kpts is not None else other_new_kpts
            )

        num_pos_keypoints = sam_args.num_pos_keypoints
        if sam_args.crowd_by_max_iou is not None and max_ious[instance_idx] > sam_args.crowd_by_max_iou:
            bbox_xyxy = None
            num_pos_keypoints = sam_args.num_pos_keypoints_if_crowd

        dt_mask, pos_kpts, neg_kpts, scores = _pose2seg(
            sam_args,
            model,
            bbox_xyxy,
            pos_kpts=this_kpts,
            neg_kpts=other_kpts,
            image=image if (sam_args.crop and sam_args.use_bbox) else None,
            gt_mask=dt_mask,
            num_pos_keypoints=num_pos_keypoints,
            gt_mask_is_binary=True,
        )

        new_dets.refined_masks[instance_idx] = dt_mask
        new_dets.sam_scores[instance_idx] = scores

        # If the number of positive keypoints is less than the required number, fill the rest with zeros
        if len(pos_kpts) != sam_args.num_pos_keypoints:
            pos_kpts = np.concatenate(
                [pos_kpts, np.zeros((sam_args.num_pos_keypoints - len(pos_kpts), 3), dtype=np.float32)], axis=0
            )
        new_dets.sam_kpts[instance_idx] = pos_kpts

    n_masks = len(new_dets.refined_masks) + (len(old_dets.refined_masks) if old_dets is not None else 0)

    if sam_args.exclusive_masks and n_masks > 1:
        all_masks = (
            np.concatenate([new_dets.refined_masks, old_dets.refined_masks], axis=0)
            if old_dets is not None
            else new_dets.refined_masks
        )
        all_scores = (
            np.concatenate([new_dets.sam_scores, old_dets.sam_scores], axis=0)
            if old_dets is not None
            else new_dets.sam_scores
        )
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
    """
    Batch process multiple detection instances with SAM2 refinement.

    Args:
        sam_args (Any): DotDict of SAM parameters (same as `process_image_with_SAM`).
        image (np.ndarray): Input BGR image.
        model (SAM2ImagePredictor): Prepared SAM2 predictor.
        new_dets (InstanceData): New detection instances.
        old_dets (Optional[InstanceData]): Previous detections for negative prompts.

    Returns:
        InstanceData: `new_dets` updated as in `process_image_with_SAM`.
    """
    n_new_dets = len(new_dets.bboxes)

    model.set_image(image)

    image_kpts = []
    image_bboxes = []
    num_valid_kpts = []
    for instance_idx in range(len(new_dets.bboxes)):

        bbox_xywh = new_dets.bboxes[instance_idx].copy()
        bbox_area = bbox_xywh[2] * bbox_xywh[3]
        if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
            continue

        this_kpts = new_dets.keypoints[instance_idx].copy().reshape(-1, 3)
        kpts_vis = np.array(this_kpts[:, 2])
        visible_kpts = (kpts_vis > sam_args.visibility_thr) & (this_kpts[:, 2] > sam_args.confidence_thr)
        num_visible = (visible_kpts).sum()
        if num_visible <= 0:
            continue
        num_valid_kpts.append(num_visible)
        image_bboxes.append(np.array(bbox_xywh))
        this_kpts[~visible_kpts, :2] = 0
        this_kpts[:, 2] = visible_kpts
        image_kpts.append(this_kpts)
    if old_dets is not None:
        for instance_idx in range(len(old_dets.bboxes)):
            bbox_xywh = old_dets.bboxes[instance_idx].copy()
            bbox_area = bbox_xywh[2] * bbox_xywh[3]
            if sam_args.ignore_small_bboxes and bbox_area < 100 * 100:
                continue
            this_kpts = old_dets.keypoints[instance_idx].reshape(-1, 3)
            kpts_vis = np.array(this_kpts[:, 2])
            visible_kpts = (kpts_vis > sam_args.visibility_thr) & (this_kpts[:, 2] > sam_args.confidence_thr)
            num_visible = (visible_kpts).sum()
            if num_visible <= 0:
                continue
            num_valid_kpts.append(num_visible)
            image_bboxes.append(np.array(bbox_xywh))
            this_kpts[~visible_kpts, :2] = 0
            this_kpts[:, 2] = visible_kpts
            image_kpts.append(this_kpts)

    image_kpts = np.array(image_kpts)
    image_bboxes = np.array(image_bboxes)
    num_valid_kpts = np.array(num_valid_kpts)

    image_kpts_backup = image_kpts.copy()

    # Prepare keypoints such that all instances have the same number of keypoints
    # First sort keypoints by their distance to the center of the bounding box
    # If some are missing, duplicate the last one
    prepared_kpts = []
    prepared_kpts_backup = []
    for bbox, kpts, num_visible in zip(image_bboxes, image_kpts, num_valid_kpts):

        this_kpts, this_conf = _select_keypoints(sam_args, kpts, num_visible, bbox)

        # Duplicate the last keypoint if some are missing
        if this_kpts.shape[0] < num_valid_kpts.max():
            this_kpts = np.concatenate(
                [this_kpts, np.tile(this_kpts[-1], (num_valid_kpts.max() - this_kpts.shape[0], 1))], axis=0
            )
            this_conf = np.concatenate(
                [this_conf, np.tile(this_conf[-1], (num_valid_kpts.max() - this_conf.shape[0],))], axis=0
            )

        prepared_kpts.append(this_kpts)
        prepared_kpts_backup.append(np.concatenate([this_kpts, this_conf[:, None]], axis=1))
    image_kpts = np.array(prepared_kpts)
    image_kpts_backup = np.array(prepared_kpts_backup)
    kpts_labels = np.ones(image_kpts.shape[:2])

    # Compute IoUs between all bounding boxes
    max_ious = _get_max_ious(image_bboxes)
    num_pos_keypoints = sam_args.num_pos_keypoints
    use_bbox = sam_args.use_bbox
    if sam_args.crowd_by_max_iou is not None and max_ious[instance_idx] > sam_args.crowd_by_max_iou:
        use_bbox = False
        num_pos_keypoints = sam_args.num_pos_keypoints_if_crowd

    # Threshold the number of positive keypoints
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

        # Expand the bbox to include the positive keypoints
        if sam_args.extend_bbox:
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
            # bbox_expanded = (np.abs(expanded_bbox - image_bboxes_xyxy) > 1e-4).any(axis=1)
            image_bboxes_xyxy = expanded_bbox

    # Process even old detections to get their 'negative' keypoints
    masks, scores, logits = model.predict(
        point_coords=image_kpts,
        point_labels=kpts_labels,
        box=image_bboxes_xyxy,
        multimask_output=False,
    )

    # Reshape the masks to (N, C, H, W). If the model outputs (C, H, W), add a number of masks dimension
    if len(masks.shape) == 3:
        masks = masks[None, :, :, :]
    masks = masks[:, 0, :, :]
    N = masks.shape[0]
    scores = scores.reshape(N)

    if sam_args.exclusive_masks and N > 1:
        # Make sure the masks are non-overlapping
        # If two masks overlap, set the pixel to the one with the highest score
        masks = _apply_exclusive_masks(masks, scores)

    gt_masks = new_dets.pred_masks.copy() if new_dets.pred_masks is not None else None
    if sam_args.pose_mask_consistency and gt_masks is not None:
        # Measure 'mask-pose_conistency' by computing number of keypoints inside the mask
        # Compute for both gt (if available) and predicted masks and then choose the one with higher consistency
        dt_mask_pose_consistency = _compute_mask_pose_consistency(masks, image_kpts_backup)
        gt_mask_pose_consistency = _compute_mask_pose_consistency(gt_masks, image_kpts_backup)

        dt_masks_area = np.array([m.sum() for m in masks])
        gt_masks_area = np.array([m.sum() for m in gt_masks]) if gt_masks is not None else np.zeros_like(dt_masks_area)

        # If PM-c is approx the same, prefer the smaller mask
        tol = 0.1
        pmc_is_equal = np.isclose(dt_mask_pose_consistency, gt_mask_pose_consistency, atol=tol)
        dt_is_worse = (dt_mask_pose_consistency < (gt_mask_pose_consistency - tol)) | pmc_is_equal & (
            dt_masks_area > gt_masks_area
        )

        new_masks = []
        for dt_mask, gt_mask, dt_worse in zip(masks, gt_masks, dt_is_worse):
            if dt_worse:
                new_masks.append(gt_mask)
            else:
                new_masks.append(dt_mask)
        masks = np.array(new_masks)

    new_dets.refined_masks = masks[:n_new_dets]
    new_dets.sam_scores = scores[:n_new_dets]
    new_dets.sam_kpts = image_kpts_backup[:n_new_dets]

    return new_dets


def _apply_exclusive_masks(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Ensure masks are non-overlapping by keeping at each pixel the mask with the highest score.
    """
    no_mask = masks.sum(axis=0) == 0
    masked_scores = masks * scores[:, None, None]
    argmax_masks = np.argmax(masked_scores, axis=0)
    new_masks = argmax_masks[None, :, :] == (np.arange(masks.shape[0])[:, None, None])
    new_masks[:, no_mask] = 0
    return new_masks
