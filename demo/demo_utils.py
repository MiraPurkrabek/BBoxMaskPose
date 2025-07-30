"""
Utilities for the BMP demo:
- Visualization of detections, masks, and poses
- Mask and bounding-box processing
- Pose non-maximum suppression (NMS)
- Animated GIF creation of demo iterations
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from pycocotools import mask as Mask
from sam2.distinctipy import get_colors
from tqdm import tqdm

### Visualization hyperparameters
MIN_CONTOUR_AREA: int = 50
BBOX_WEIGHT: float = 0.9
MASK_WEIGHT: float = 0.6
BACK_MASK_WEIGHT: float = 0.6
POSE_WEIGHT: float = 0.8


"""
posevis is our custom visualization library for pose estimation. For compatibility, we also provide a lite version that has fewer features but still reproduces visualization from the paper.
"""
try:
    from posevis import pose_visualization
except ImportError:
    from posevis_lite import pose_visualization


class DotDict(dict):
    """Dictionary with attribute access and nested dict wrapping."""

    def __getattr__(self, name: str) -> any:
        if name in self:
            val = self[name]
            if isinstance(val, dict):
                val = DotDict(val)
                self[name] = val
            return val
        raise AttributeError("No attribute named {!r}".format(name))

    def __setattr__(self, name: str, value: any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError("No attribute named {!r}".format(name))


def filter_instances(instances: InstanceData, indices):
    """
    Return a new InstanceData containing only the entries of 'instances' at the given indices.
    """
    if instances is None:
        return None
    data = {}
    # Attributes to filter
    for attr in [
        "bboxes",
        "bbox_scores",
        "keypoints",
        "keypoint_scores",
        "scores",
        "pred_masks",
        "refined_masks",
        "sam_scores",
        "sam_kpts",
    ]:
        if hasattr(instances, attr):
            arr = getattr(instances, attr)
            data[attr] = arr[indices] if arr is not None else None
    return InstanceData(**data)


def concat_instances(instances1: InstanceData, instances2: InstanceData):
    """
    Concatenate two InstanceData objects along the first axis, preserving order.
    If instances1 or instances2 is None, returns the other.
    """
    if instances1 is None:
        return instances2
    if instances2 is None:
        return instances1
    data = {}
    for attr in [
        "bboxes",
        "bbox_scores",
        "keypoints",
        "keypoint_scores",
        "scores",
        "pred_masks",
        "refined_masks",
        "sam_scores",
        "sam_kpts",
    ]:
        arr1 = getattr(instances1, attr, None)
        arr2 = getattr(instances2, attr, None)
        if arr1 is None and arr2 is None:
            continue
        if arr1 is None:
            data[attr] = arr2
        elif arr2 is None:
            data[attr] = arr1
        else:
            data[attr] = np.concatenate([arr1, arr2], axis=0)
    return InstanceData(**data)


def _visualize_predictions(
    img: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    masks: List[Optional[List[np.ndarray]]],
    poses: List[Optional[np.ndarray]],
    vis_type: str = "mask",
    mask_is_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render bounding boxes, segmentation masks, and poses on the input image.

    Args:
        img (np.ndarray): BGR image of shape (H, W, 3).
        bboxes (np.ndarray): Array of bounding boxes [x, y, w, h].
        scores (np.ndarray): Confidence scores for each bbox.
        masks (List[Optional[List[np.ndarray]]]): Polygon masks per instance.
        poses (List[Optional[np.ndarray]]): Keypoint arrays per instance.
        vis_type (str): Flags for visualization types separated by '+'.
        mask_is_binary (bool): Whether input masks are binary arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The visualized image and color map.
    """
    vis_types = vis_type.split("+")

    # Exclude white, black, and green colors from the palette as they are not distinctive
    colors = (np.array(get_colors(len(bboxes), exclude_colors=[(0, 1, 0), (0, 0, 0), (1, 1, 1)], rng=0)) * 255).astype(
        int
    )

    if mask_is_binary:
        poly_masks: List[Optional[List[np.ndarray]]] = []
        for binary_mask in masks:
            if binary_mask is not None:
                contours, _ = cv2.findContours(
                    (binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                polys = [cnt.flatten() for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA]
            else:
                polys = None
            poly_masks.append(polys)
        masks = poly_masks  # type: ignore

    if "inv-mask" in vis_types:
        stencil = np.zeros_like(img)

    for bbox, score, mask_poly, pose, color in zip(bboxes, scores, masks, poses, colors):
        bbox = _update_bbox_by_mask(list(map(int, bbox)), mask_poly, img.shape)
        color_list = color.tolist()
        img_copy = img.copy()

        if "bbox" in vis_types:
            x, y, w, h = bbox
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color_list, 2)
            img = cv2.addWeighted(img, 1 - BBOX_WEIGHT, img_copy, BBOX_WEIGHT, 0)

        if mask_poly is not None and "mask" in vis_types:
            for seg in mask_poly:
                seg_pts = np.array(seg).reshape(-1, 1, 2).astype(int)
                cv2.fillPoly(img_copy, [seg_pts], color_list)
            img = cv2.addWeighted(img, 1 - MASK_WEIGHT, img_copy, MASK_WEIGHT, 0)

        if mask_poly is not None and "mask-out" in vis_types:
            for seg in mask_poly:
                seg_pts = np.array(seg).reshape(-1, 1, 2).astype(int)
                cv2.fillPoly(img, [seg_pts], (0, 0, 0))

        if mask_poly is not None and "inv-mask" in vis_types:
            for seg in mask_poly:
                seg = np.array(seg).reshape(-1, 1, 2).astype(int)
                if cv2.contourArea(seg) < MIN_CONTOUR_AREA:
                    continue
                cv2.fillPoly(stencil, [seg], (255, 255, 255))

        if pose is not None and "pose" in vis_types:
            vis_img = pose_visualization(
                img.copy(),
                pose.reshape(-1, 3),
                width_multiplier=8,
                differ_individuals=True,
                color=color_list,
                keep_image_size=True,
            )
            img = cv2.addWeighted(img, 1 - POSE_WEIGHT, vis_img, POSE_WEIGHT, 0)

    if "inv-mask" in vis_types:
        img = cv2.addWeighted(img, 1 - BACK_MASK_WEIGHT, cv2.bitwise_and(img, stencil), BACK_MASK_WEIGHT, 0)

    return img, colors


def visualize_itteration(
    img: np.ndarray, detections: Any, iteration_idx: int, output_root: Path, img_name: str, with_text: bool = True
) -> Optional[np.ndarray]:
    """
    Generate and save visualization images for each BMP iteration.

    Args:
        img (np.ndarray): Original input image.
        detections: InstanceData containing bboxes, scores, masks, keypoints.
        iteration_idx (int): Current iteration index (0-based).
        output_root (Path): Directory to save output images.
        img_name (str): Base name of the image without extension.
        with_text (bool): Whether to overlay text labels.

    Returns:
        Optional[np.ndarray]: The masked-out image if generated, else None.
    """
    bboxes = detections.bboxes
    scores = detections.scores
    pred_masks = detections.pred_masks
    refined_masks = detections.refined_masks
    keypoints = detections.keypoints
    sam_kpts = detections.sam_kpts

    masked_out = None
    for vis_def in [
        {"type": "bbox+mask", "masks": pred_masks, "label": "Detector (out)"},
        {"type": "inv-mask", "masks": pred_masks, "label": "MaskPose (in)"},
        {"type": "inv-mask+pose", "masks": pred_masks, "label": "MaskPose (out)"},
        {"type": "mask", "masks": refined_masks, "label": "SAM Masks"},
        {"type": "mask-out", "masks": refined_masks, "label": "Mask-Out"},
        {"type": "pose", "masks": refined_masks, "label": "Final Poses"},
    ]:
        vis_img, colors = _visualize_predictions(
            img.copy(), bboxes, scores, vis_def["masks"], keypoints, vis_type=vis_def["type"], mask_is_binary=True
        )
        if vis_def["type"] == "mask-out":
            masked_out = vis_img
        if with_text:
            label = "BMP {:d}x: {}".format(iteration_idx + 1, vis_def["label"])
            cv2.putText(vis_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(vis_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out_path = os.path.join(
            output_root, "{}_iter{}_{}.jpg".format(img_name, iteration_idx + 1, vis_def["label"].replace(" ", "_"))
        )
        cv2.imwrite(str(out_path), vis_img)

    # Show prompting keypoints
    tmp_img = img.copy()
    for i, _ in enumerate(bboxes):
        if len(sam_kpts[i]) > 0:
            instance_color = colors[i].astype(int).tolist()
            for kpt in sam_kpts[i]:
                cv2.drawMarker(
                    tmp_img,
                    (int(kpt[0]), int(kpt[1])),
                    instance_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=3,
                )
                # Write the keypoint confidence next to the marker
                cv2.putText(
                    tmp_img,
                    f"{kpt[2]:.2f}",
                    (int(kpt[0]) + 10, int(kpt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    instance_color,
                    1,
                    cv2.LINE_AA,
                )
    if with_text:
        text = "BMP {:d}x: SAM prompts".format(iteration_idx + 1)
        cv2.putText(tmp_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tmp_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite("{:s}/{:s}_iter{:d}_prompting_kpts.jpg".format(output_root, img_name, iteration_idx + 1), tmp_img)

    return masked_out


def create_GIF(
    img_path: Path,
    output_root: Path,
    bmp_x: int = 2,
) -> None:
    """
    Compile iteration images into an animated GIF using ffmpeg.

    Args:
        img_path (Path): Path to a sample iteration image.
        output_root (Path): Directory to save the GIF.
        bmp_x (int): Number of BMP iterations.
        duration_per_frame (int): Frame display duration in ms.

    Raises:
        RuntimeError: If ffmpeg is not available or images are missing.
    """
    display_dur = 1.5  # seconds
    fade_dur = 1.0
    fps = 10
    scale_width = 300  # Resize width for GIF, height will be auto-scaled to maintain aspect ratio

    # Check if ffmpeg is installed. If not, raise warning and return
    if shutil.which("ffmpeg") is None:
        print_log("FFMpeg is not installed. GIF creation will be skipped.", logger="current", level=logging.WARNING)
        return
    print_log("Creating GIF with FFmpeg...", logger="current")

    dirname, filename = os.path.split(img_path)
    img_name_wo_ext, _ = os.path.splitext(filename)

    gif_image_names = [
        "Detector_(out)",
        "MaskPose_(in)",
        "MaskPose_(out)",
        "prompting_kpts",
        "SAM_Masks",
        "Mask-Out",
    ]

    # Create black image of the same size as the last image
    last_img_path = os.path.join(dirname, "{}_iter1_{}".format(img_name_wo_ext, gif_image_names[0]) + ".jpg")
    last_img = cv2.imread(last_img_path)
    if last_img is None:
        print_log("Could not read image {}.".format(last_img_path), logger="current", level=logging.ERROR)
        return
    black_img = np.zeros_like(last_img)
    cv2.imwrite(os.path.join(dirname, "black_image.jpg"), black_img)
    # gif_images.append(os.path.join(dirname, "black_image.jpg"))

    gif_images = []
    for iter in range(bmp_x):
        iter_img_path = os.path.join(dirname, "{}_iter{}_".format(img_name_wo_ext, iter + 1))
        for img_name in gif_image_names:

            if iter + 1 == bmp_x and img_name == "Mask-Out":
                # Skip the last iteration's Mask-Out image
                continue

            img_file = "{}{}.jpg".format(iter_img_path, img_name)
            if not os.path.exists(img_file):
                print_log("{} does not exist, skipping.".format(img_file), logger="current", level=logging.WARNING)
                continue
            gif_images.append(img_file)
            # gif_images.append(cv2.imread(img_file))

    if len(gif_images) == 0:
        print_log("No images found for GIF creation.", logger="current", level=logging.WARNING)
        return

    # Add 'before' and 'after' images
    after1_img = os.path.join(dirname, "{}_iter{}_Final_Poses.jpg".format(img_name_wo_ext, bmp_x))
    after2_img = os.path.join(dirname, "{}_iter{}_SAM_Masks.jpg".format(img_name_wo_ext, bmp_x))
    # gif_images.append(os.path.join(dirname, "black_image.jpg"))  # Add black image at the end
    gif_images.append(after1_img)
    gif_images.append(after2_img)
    gif_images.append(os.path.join(dirname, "black_image.jpg"))  # Add black image at the end

    # Create a GIF from the images
    gif_output_path = os.path.join(output_root, "{}_bmp_{}x.gif".format(img_name_wo_ext, bmp_x))

    # 1. inputs
    in_args = []
    for p in gif_images:
        in_args += ["-loop", "1", "-t", str(display_dur), "-i", p]

    # 2. build xfade chain
    n = len(gif_images)
    parts = []
    for i in range(1, n):
        # left label: first is input [0:v], then [v1], [v2], …
        left = "[{}:v]".format(i - 1) if i == 1 else "[v{}]".format(i - 1)
        right = "[{}:v]".format(i)
        out = "[v{}]".format(i)
        offset = (i - 1) * (display_dur + fade_dur) + display_dur
        parts.append(
            "{}{}xfade=transition=fade:".format(left, right)
            + "duration={}:offset={:.3f}{}".format(fade_dur, offset, out)
        )
    filter_complex = ";".join(parts)

    # 3. make MP4 slideshow
    mp4 = "slideshow.mp4"
    cmd1 = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-v",
        "quiet",
        "-hide_banner",
        "-y",
        *in_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v{}]".format(n - 1),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4,
    ]
    subprocess.run(cmd1, check=True)

    # 4. palette
    palette = "palette.png"
    vf = "fps={}".format(fps)
    if scale_width:
        vf += ",scale={}: -1:flags=lanczos".format(scale_width)

    # 5. generate palette
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-v",
            "quiet",
            "-hide_banner",
            "-y",
            "-i",
            mp4,
            "-vf",
            vf + ",palettegen",
            palette,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # 6. build final GIF
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-v",
            "quiet",
            "-hide_banner",
            "-y",
            "-i",
            mp4,
            "-i",
            palette,
            "-lavfi",
            vf + "[x];[x][1:v]paletteuse",
            gif_output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Clean up temporary files
    os.remove(mp4)
    os.remove(palette)

    print_log(f"GIF saved as '{gif_output_path}'", logger="current")


def _update_bbox_by_mask(
    bbox: List[int], mask_poly: Optional[List[List[int]]], image_shape: Tuple[int, int, int]
) -> List[int]:
    """
    Adjust bounding box to tightly fit mask polygon.

    Args:
        bbox (List[int]): Original [x, y, w, h].
        mask_poly (Optional[List[List[int]]]): Polygon coordinates.
        image_shape (Tuple[int,int,int]): Image shape (H, W, C).

    Returns:
        List[int]: Updated [x, y, w, h] bounding box.
    """
    if mask_poly is None:
        return bbox

    mask_rle = Mask.frPyObjects(mask_poly, image_shape[0], image_shape[1])
    mask_rle = Mask.merge(mask_rle)
    bbox_segm_xywh = Mask.toBbox(mask_rle)
    bbox_segm_xyxy = np.array(
        [
            bbox_segm_xywh[0],
            bbox_segm_xywh[1],
            bbox_segm_xywh[0] + bbox_segm_xywh[2],
            bbox_segm_xywh[1] + bbox_segm_xywh[3],
        ]
    )

    bbox = bbox_segm_xywh

    return bbox.astype(int).tolist()


def pose_nms(config: Any, image_kpts: np.ndarray, image_bboxes: np.ndarray, num_valid_kpts: np.ndarray) -> np.ndarray:
    """
    Perform OKS-based non-maximum suppression on detected poses.

    Args:
        config (Any): Configuration with confidence_thr and oks_thr.
        image_kpts (np.ndarray): Detected keypoints of shape (N, K, 3).
        image_bboxes (np.ndarray): Corresponding bboxes (N,4).
        num_valid_kpts (np.ndarray): Count of valid keypoints per instance.

    Returns:
        np.ndarray: Indices of kept instances.
    """
    # Sort image kpts by average score - lowest first
    # scores = image_kpts[:, :, 2].mean(axis=1)
    # sort_idx = np.argsort(scores)
    # image_kpts = image_kpts[sort_idx, :, :]

    # Compute OKS between all pairs of poses
    oks_matrix = np.zeros((image_kpts.shape[0], image_kpts.shape[0]))
    for i in range(image_kpts.shape[0]):
        for j in range(image_kpts.shape[0]):
            gt_bbox_xywh = image_bboxes[i].copy()
            gt_bbox_xyxy = gt_bbox_xywh.copy()
            gt_bbox_xyxy[2:] += gt_bbox_xyxy[:2]
            gt = {
                "keypoints": image_kpts[i].copy(),
                "bbox": gt_bbox_xyxy,
                "area": gt_bbox_xywh[2] * gt_bbox_xywh[3],
            }
            dt = {"keypoints": image_kpts[j].copy(), "bbox": gt_bbox_xyxy}
            gt["keypoints"][:, 2] = (gt["keypoints"][:, 2] > config.confidence_thr) * 2
            oks = compute_oks(gt, dt)
            if oks > 1:
                breakpoint()
            oks_matrix[i, j] = oks

    np.fill_diagonal(oks_matrix, -1)
    is_subset = oks_matrix > config.oks_thr

    remove_instances = []
    while is_subset.any():
        # Find the pair with the highest OKS
        i, j = np.unravel_index(np.argmax(oks_matrix), oks_matrix.shape)

        # Keep the one with the highest number of keypoints
        if num_valid_kpts[i] > num_valid_kpts[j]:
            remove_idx = j
        else:
            remove_idx = i

        # Remove the column from is_subset
        oks_matrix[:, remove_idx] = 0
        oks_matrix[remove_idx, j] = 0
        remove_instances.append(remove_idx)
        is_subset = oks_matrix > config.oks_thr

    keep_instances = np.setdiff1d(np.arange(image_kpts.shape[0]), remove_instances)

    return keep_instances


def compute_oks(gt: Dict[str, Any], dt: Dict[str, Any], use_area: bool = True, per_kpt: bool = False) -> float:
    """
    Compute Object Keypoint Similarity (OKS) between ground-truth and detected poses.

    Args:
        gt (Dict): Ground-truth keypoints and bbox info.
        dt (Dict): Detected keypoints and bbox info.
        use_area (bool): Whether to normalize by GT area.
        per_kpt (bool): Whether to return per-keypoint OKS array.

    Returns:
        float: OKS score or mean OKS.
    """
    sigmas = (
        np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt["keypoints"]).reshape(k, 3)
    xg = g[:, 0]
    yg = g[:, 1]
    vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt["bbox"]
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2

    d = np.array(dt["keypoints"]).reshape((k, 3))
    xd = d[:, 0]
    yd = d[:, 1]

    if k1 > 0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
    else:
        tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea + np.spacing(1)) / 2

    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e = e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks
