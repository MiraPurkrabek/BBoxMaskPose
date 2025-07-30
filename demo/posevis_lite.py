import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

NEUTRAL_COLOR = (52, 235, 107)

LEFT_ARM_COLOR = (216, 235, 52)
LEFT_LEG_COLOR = (235, 107, 52)
LEFT_SIDE_COLOR = (245, 188, 113)
LEFT_FACE_COLOR = (235, 52, 107)

RIGHT_ARM_COLOR = (52, 235, 216)
RIGHT_LEG_COLOR = (52, 107, 235)
RIGHT_SIDE_COLOR = (52, 171, 235)
RIGHT_FACE_COLOR = (107, 52, 235)

COCO_MARKERS = [
    ["nose", cv2.MARKER_CROSS, NEUTRAL_COLOR],
    ["left_eye", cv2.MARKER_SQUARE, LEFT_FACE_COLOR],
    ["right_eye", cv2.MARKER_SQUARE, RIGHT_FACE_COLOR],
    ["left_ear", cv2.MARKER_CROSS, LEFT_FACE_COLOR],
    ["right_ear", cv2.MARKER_CROSS, RIGHT_FACE_COLOR],
    ["left_shoulder", cv2.MARKER_TRIANGLE_UP, LEFT_ARM_COLOR],
    ["right_shoulder", cv2.MARKER_TRIANGLE_UP, RIGHT_ARM_COLOR],
    ["left_elbow", cv2.MARKER_SQUARE, LEFT_ARM_COLOR],
    ["right_elbow", cv2.MARKER_SQUARE, RIGHT_ARM_COLOR],
    ["left_wrist", cv2.MARKER_CROSS, LEFT_ARM_COLOR],
    ["right_wrist", cv2.MARKER_CROSS, RIGHT_ARM_COLOR],
    ["left_hip", cv2.MARKER_TRIANGLE_UP, LEFT_LEG_COLOR],
    ["right_hip", cv2.MARKER_TRIANGLE_UP, RIGHT_LEG_COLOR],
    ["left_knee", cv2.MARKER_SQUARE, LEFT_LEG_COLOR],
    ["right_knee", cv2.MARKER_SQUARE, RIGHT_LEG_COLOR],
    ["left_ankle", cv2.MARKER_TILTED_CROSS, LEFT_LEG_COLOR],
    ["right_ankle", cv2.MARKER_TILTED_CROSS, RIGHT_LEG_COLOR],
]


COCO_SKELETON = [
    [[16, 14], LEFT_LEG_COLOR],  # Left ankle - Left knee
    [[14, 12], LEFT_LEG_COLOR],  # Left knee - Left hip
    [[17, 15], RIGHT_LEG_COLOR],  # Right ankle - Right knee
    [[15, 13], RIGHT_LEG_COLOR],  # Right knee - Right hip
    [[12, 13], NEUTRAL_COLOR],  # Left hip - Right hip
    [[6, 12], LEFT_SIDE_COLOR],  # Left hip - Left shoulder
    [[7, 13], RIGHT_SIDE_COLOR],  # Right hip - Right shoulder
    [[6, 7], NEUTRAL_COLOR],  # Left shoulder - Right shoulder
    [[6, 8], LEFT_ARM_COLOR],  # Left shoulder - Left elbow
    [[7, 9], RIGHT_ARM_COLOR],  # Right shoulder - Right elbow
    [[8, 10], LEFT_ARM_COLOR],  # Left elbow - Left wrist
    [[9, 11], RIGHT_ARM_COLOR],  # Right elbow - Right wrist
    [[2, 3], NEUTRAL_COLOR],  # Left eye - Right eye
    [[1, 2], LEFT_FACE_COLOR],  # Nose - Left eye
    [[1, 3], RIGHT_FACE_COLOR],  # Nose - Right eye
    [[2, 4], LEFT_FACE_COLOR],  # Left eye - Left ear
    [[3, 5], RIGHT_FACE_COLOR],  # Right eye - Right ear
    [[4, 6], LEFT_FACE_COLOR],  # Left ear - Left shoulder
    [[5, 7], RIGHT_FACE_COLOR],  # Right ear - Right shoulder
]


def _draw_line(
    img: np.ndarray,
    start: Tuple[float, float],
    stop: Tuple[float, float],
    color: Tuple[int, int, int],
    line_type: str,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a line segment on an image, supporting solid, dashed, or dotted styles.

    Args:
        img (np.ndarray): BGR image of shape (H, W, 3).
        start (tuple of float): (x, y) start coordinates.
        stop (tuple of float): (x, y) end coordinates.
        color (tuple of int): BGR color values.
        line_type (str): One of 'solid', 'dashed', or 'doted'.
        thickness (int): Line thickness in pixels.

    Returns:
        np.ndarray: Image with the line drawn.
    """
    start = np.array(start)[:2]
    stop = np.array(stop)[:2]
    if line_type.lower() == "solid":
        img = cv2.line(
            img,
            (int(start[0]), int(start[1])),
            (int(stop[0]), int(stop[1])),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif line_type.lower() == "dashed":
        delta = stop - start
        length = np.linalg.norm(delta)
        frac = np.linspace(0, 1, num=int(length / 5), endpoint=True)
        for i in range(0, len(frac) - 1, 2):
            s = start + frac[i] * delta
            e = start + frac[i + 1] * delta
            img = cv2.line(
                img,
                (int(s[0]), int(s[1])),
                (int(e[0]), int(e[1])),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    elif line_type.lower() == "doted":
        delta = stop - start
        length = np.linalg.norm(delta)
        frac = np.linspace(0, 1, num=int(length / 5), endpoint=True)
        for i in range(0, len(frac)):
            s = start + frac[i] * delta
            img = cv2.circle(
                img,
                (int(s[0]), int(s[1])),
                radius=max(thickness // 2, 1),
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return img


def pose_visualization(
    img: Union[str, np.ndarray],
    keypoints: Union[Dict[str, Any], np.ndarray],
    format: str = "COCO",
    greyness: float = 1.0,
    show_markers: bool = True,
    show_bones: bool = True,
    line_type: str = "solid",
    width_multiplier: float = 1.0,
    bbox_width_multiplier: float = 1.0,
    show_bbox: bool = False,
    differ_individuals: bool = False,
    confidence_thr: float = 0.3,
    errors: Optional[np.ndarray] = None,
    color: Optional[Tuple[int, int, int]] = None,
    keep_image_size: bool = False,
    return_padding: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """
    Overlay pose keypoints and skeleton on an image.

    Args:
        img (str or np.ndarray): Path to image file or BGR image array.
        keypoints (dict or np.ndarray): Either a dict with 'bbox' and 'keypoints' or
            an array of shape (17, 2 or 3) or multiple poses stacked.
        format (str): Keypoint format, currently only 'COCO'.
        greyness (float): Factor for bone/marker color intensity (0.0-1.0).
        show_markers (bool): Whether to draw keypoint markers.
        show_bones (bool): Whether to draw skeleton bones.
        line_type (str): One of 'solid', 'dashed', 'doted' for bone style.
        width_multiplier (float): Line width scaling factor for bones.
        bbox_width_multiplier (float): Line width scaling factor for bounding box.
        show_bbox (bool): Whether to draw bounding box around keypoints.
        differ_individuals (bool): Use distinct color per individual pose.
        confidence_thr (float): Confidence threshold for keypoint visibility.
        errors (np.ndarray or None): Optional array of per-kpt errors (17,1).
        color (tuple or None): Override color for markers and bones.
        keep_image_size (bool): Prevent image padding for out-of-bounds keypoints.
        return_padding (bool): If True, also return padding offsets [top,bottom,left,right].

    Returns:
        np.ndarray or (np.ndarray, list of int): Annotated image, and optional
            padding offsets if `return_padding` is True.
    """

    bbox = None
    if isinstance(keypoints, dict):
        try:
            bbox = np.array(keypoints["bbox"]).flatten()
        except KeyError:
            pass
        keypoints = np.array(keypoints["keypoints"])

    # If keypoints is a list of poses, draw them all
    if len(keypoints) % 17 != 0 or keypoints.ndim == 3:

        if color is not None:
            if not isinstance(color, (list, tuple)):
                color = [color for keypoint in keypoints]
        else:
            color = [None for keypoint in keypoints]

        max_padding = [0, 0, 0, 0]
        for keypoint, clr in zip(keypoints, color):
            img = pose_visualization(
                img,
                keypoint,
                format=format,
                greyness=greyness,
                show_markers=show_markers,
                show_bones=show_bones,
                line_type=line_type,
                width_multiplier=width_multiplier,
                bbox_width_multiplier=bbox_width_multiplier,
                show_bbox=show_bbox,
                differ_individuals=differ_individuals,
                color=clr,
                confidence_thr=confidence_thr,
                keep_image_size=keep_image_size,
                return_padding=return_padding,
            )
            if return_padding:
                img, padding = img
                max_padding = [max(max_padding[i], int(padding[i])) for i in range(4)]

        if return_padding:
            return img, max_padding
        else:
            return img

    keypoints = np.array(keypoints).reshape(17, -1)
    # If keypoint visibility is not provided, assume all keypoints are visible
    if keypoints.shape[1] == 2:
        keypoints = np.hstack([keypoints, np.ones((17, 1)) * 2])

    assert keypoints.shape[1] == 3, "Keypoints should be in the format (x, y, visibility)"
    assert keypoints.shape[0] == 17, "Keypoints should be in the format (x, y, visibility)"

    if errors is not None:
        errors = np.array(errors).reshape(17, -1)
        assert errors.shape[1] == 1, "Errors should be in the format (K, r)"
        assert errors.shape[0] == 17, "Errors should be in the format (K, r)"
    else:
        errors = np.ones((17, 1)) * np.nan

    # If keypoint visibility is float between 0 and 1, it is detection
    # If conf < confidence_thr: conf = 1
    # If conf >= confidence_thr: conf = 2
    vis_is_float = np.any(np.logical_and(keypoints[:, -1] > 0, keypoints[:, -1] < 1))
    if keypoints.shape[1] == 3 and vis_is_float:
        # print("before", keypoints[:, -1])
        lower_idx = keypoints[:, -1] < confidence_thr
        keypoints[lower_idx, -1] = 1
        keypoints[~lower_idx, -1] = 2
        # print("after", keypoints[:, -1])
        # print("-"*20)

    # All visibility values should be ints
    keypoints[:, -1] = keypoints[:, -1].astype(int)

    if isinstance(img, str):
        img = cv2.imread(img)

    if img is None:
        if return_padding:
            return None, [0, 0, 0, 0]
        else:
            return None

    if not (keypoints[:, 2] > 0).any():
        if return_padding:
            return img, [0, 0, 0, 0]
        else:
            return img

    valid_kpts = (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
    num_valid_kpts = np.sum(valid_kpts)

    if num_valid_kpts == 0:
        if return_padding:
            return img, [0, 0, 0, 0]
        else:
            return img

    min_x_kpts = np.min(keypoints[keypoints[:, 2] > 0, 0])
    min_y_kpts = np.min(keypoints[keypoints[:, 2] > 0, 1])
    max_x_kpts = np.max(keypoints[keypoints[:, 2] > 0, 0])
    max_y_kpts = np.max(keypoints[keypoints[:, 2] > 0, 1])
    if bbox is None:
        min_x = min_x_kpts
        min_y = min_y_kpts
        max_x = max_x_kpts
        max_y = max_y_kpts
    else:
        min_x = bbox[0]
        min_y = bbox[1]
        max_x = bbox[2]
        max_y = bbox[3]

    max_area = (max_x - min_x) * (max_y - min_y)
    diagonal = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    line_width = max(int(np.sqrt(max_area) / 500 * width_multiplier), 1)
    bbox_line_width = max(int(np.sqrt(max_area) / 500 * bbox_width_multiplier), 1)
    marker_size = max(int(np.sqrt(max_area) / 80), 1)
    invisible_marker_size = max(int(np.sqrt(max_area) / 100), 1)
    marker_thickness = max(int(np.sqrt(max_area) / 100), 1)

    if differ_individuals:
        if color is not None:
            instance_color = color
        else:
            instance_color = np.random.randint(0, 255, size=(3,)).tolist()
            instance_color = tuple(instance_color)

    # Pad image with dark gray if keypoints are outside the image
    if not keep_image_size:
        padding = [
            max(0, -min_y_kpts),
            max(0, max_y_kpts - img.shape[0]),
            max(0, -min_x_kpts),
            max(0, max_x_kpts - img.shape[1]),
        ]
        padding = [int(p) for p in padding]
        img = cv2.copyMakeBorder(
            img,
            padding[0],
            padding[1],
            padding[2],
            padding[3],
            cv2.BORDER_CONSTANT,
            value=(80, 80, 80),
        )

        # Add padding to bbox and kpts
        value_x_to_add = max(0, -min_x_kpts)
        value_y_to_add = max(0, -min_y_kpts)
        keypoints[keypoints[:, 2] > 0, 0] += value_x_to_add
        keypoints[keypoints[:, 2] > 0, 1] += value_y_to_add
        if bbox is not None:
            bbox[0] += value_x_to_add
            bbox[1] += value_y_to_add
            bbox[2] += value_x_to_add
            bbox[3] += value_y_to_add

    if show_bbox and not (bbox is None):
        pts = [
            (bbox[0], bbox[1]),
            (bbox[0], bbox[3]),
            (bbox[2], bbox[3]),
            (bbox[2], bbox[1]),
            (bbox[0], bbox[1]),
        ]
        for i in range(len(pts) - 1):
            if differ_individuals:
                img = _draw_line(img, pts[i], pts[i + 1], instance_color, "doted", thickness=bbox_line_width)
            else:
                img = _draw_line(img, pts[i], pts[i + 1], (0, 255, 0), line_type, thickness=bbox_line_width)

    if show_markers:
        for kpt, marker_info, err in zip(keypoints, COCO_MARKERS, errors):
            if kpt[0] == 0 and kpt[1] == 0:
                continue

            if kpt[2] != 2:
                color = (140, 140, 140)
            elif differ_individuals:
                color = instance_color
            else:
                color = marker_info[2]

            if kpt[2] == 1:
                img_overlay = img.copy()
                img_overlay = cv2.drawMarker(
                    img_overlay,
                    (int(kpt[0]), int(kpt[1])),
                    color=color,
                    markerType=marker_info[1],
                    markerSize=marker_size,
                    thickness=marker_thickness,
                )
                img = cv2.addWeighted(img_overlay, 0.4, img, 0.6, 0)

            else:
                img = cv2.drawMarker(
                    img,
                    (int(kpt[0]), int(kpt[1])),
                    color=color,
                    markerType=marker_info[1],
                    markerSize=invisible_marker_size if kpt[2] == 1 else marker_size,
                    thickness=marker_thickness,
                )

            if not np.isnan(err).any():
                radius = err * diagonal
                clr = (0, 0, 255) if "solid" in line_type else (0, 255, 0)
                plus = 1 if "solid" in line_type else -1
                img = cv2.circle(
                    img,
                    (int(kpt[0]), int(kpt[1])),
                    radius=int(radius),
                    color=clr,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                dx = np.sqrt(radius**2 / 2)
                img = cv2.line(
                    img,
                    (int(kpt[0]), int(kpt[1])),
                    (int(kpt[0] + plus * dx), int(kpt[1] - dx)),
                    color=clr,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    if show_bones:
        for bone_info in COCO_SKELETON:
            kp1 = keypoints[bone_info[0][0] - 1, :]
            kp2 = keypoints[bone_info[0][1] - 1, :]

            if (kp1[0] == 0 and kp1[1] == 0) or (kp2[0] == 0 and kp2[1] == 0):
                continue

            dashed = kp1[2] == 1 or kp2[2] == 1

            if differ_individuals:
                color = np.array(instance_color)
            else:
                color = np.array(bone_info[1])
            color = (color * greyness).astype(int).tolist()

            if dashed:
                img_overlay = img.copy()
                img_overlay = _draw_line(img_overlay, kp1, kp2, color, line_type, thickness=line_width)
                img = cv2.addWeighted(img_overlay, 0.4, img, 0.6, 0)

            else:
                img = _draw_line(img, kp1, kp2, color, line_type, thickness=line_width)

    if return_padding:
        return img, padding
    else:
        return img


if __name__ == "__main__":
    kpts = np.array(
        [
            344,
            222,
            2,
            356,
            211,
            2,
            330,
            211,
            2,
            372,
            220,
            2,
            309,
            224,
            2,
            413,
            279,
            2,
            274,
            300,
            2,
            444,
            372,
            2,
            261,
            396,
            2,
            398,
            359,
            2,
            316,
            372,
            2,
            407,
            489,
            2,
            185,
            580,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    kpts = kpts.reshape(-1, 3)
    kpts[:, -1] = np.random.randint(1, 3, size=(17,))

    img = pose_visualization("demo/posevis_test.jpg", kpts, show_markers=True, line_type="solid")

    kpts2 = kpts.copy()
    kpts2[kpts2[:, 1] > 0, :2] += 10
    img = pose_visualization(img, kpts2, show_markers=False, line_type="doted")

    os.makedirs("demo/outputs", exist_ok=True)
    cv2.imwrite("demo/outputs/posevis_test_out.jpg", img)
