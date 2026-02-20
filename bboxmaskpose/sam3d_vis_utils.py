# Copyright (c) authors of BBoxMaskPose (BMPv2). All rights reserved.
# and
# Copyright (c) Meta Platforms, Inc. and affiliates.
import cv2
import numpy as np

from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

try:
    import distinctipy

    HAS_DISTINCTIPY = True
except ImportError:
    HAS_DISTINCTIPY = False

try:
    from posevis import pose_visualization

    has_posevis = True
except ImportError:
    has_posevis = False


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
# Amount of yaw used when showing meshes from the "side".
SIDE_VIEW_ROTATION_DEG = 60

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _build_color_palettes(num_colors, pastel_factor=0.5, order=None):
    if num_colors <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    if HAS_DISTINCTIPY:
        rgb = np.array(distinctipy.get_colors(num_colors, exclude_colors=[(0, 1, 0), (0, 0, 0), (1, 1, 1)], rng=0), dtype=np.float32)
        bgr_float = rgb[:, ::-1]
    else:
        random_colors = []
        for _ in range(num_colors):
            random_hsv = np.array([np.random.uniform(0, 255), 255, 255], dtype=np.float32).reshape(1, 1, 3)
            random_bgr = cv2.cvtColor(random_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).flatten().astype(np.float32) / 255.0
            random_colors.append(random_bgr)
        bgr_float = np.stack(random_colors, axis=0)

    if pastel_factor > 0:
        hsv = cv2.cvtColor(
            (bgr_float.reshape(1, num_colors, 3) * 255).astype(np.uint8),
            cv2.COLOR_BGR2HSV,
        )
        # Lower the saturation to get pastel colors
        hsv = hsv.astype(np.float32)
        s_floor = 20.0
        v_target = 240.0
        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - pastel_factor) + pastel_factor * s_floor
        hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - pastel_factor) + pastel_factor * v_target
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        bgr_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(num_colors, 3).astype(np.float32) / 255.0

    if order is not None:
        ordered_bgr_float = np.zeros_like(bgr_float)
        ordered_bgr_float[order, :] = bgr_float
        bgr_float = ordered_bgr_float

    return bgr_float, (bgr_float * 255).astype(np.uint8)


def _merge_mesh_instances(outputs_sorted, faces):
    if not outputs_sorted:
        return None, None, None, []

    merged_vertices = []
    merged_faces = []
    vertex_counts = []
    vertex_offset = 0
    faces_np = np.asarray(faces, dtype=np.int32)
    for person_output in outputs_sorted:
        verts = np.asarray(person_output["pred_vertices"], dtype=np.float32) + np.asarray(person_output["pred_cam_t"], dtype=np.float32)
        merged_vertices.append(verts)
        merged_faces.append(faces_np + vertex_offset)
        vertex_counts.append(verts.shape[0])
        vertex_offset += verts.shape[0]

    merged_vertices = np.concatenate(merged_vertices, axis=0)
    merged_faces = np.concatenate(merged_faces, axis=0)

    verts_per_person = vertex_counts[0] if vertex_counts else 0
    tail_vertices = min(2 * verts_per_person, merged_vertices.shape[0])
    if tail_vertices > 0:
        tail = merged_vertices[-tail_vertices:]
        fake_pred_cam_t = (np.max(tail, axis=0) + np.min(tail, axis=0)) / 2.0
    else:
        fake_pred_cam_t = np.zeros(3, dtype=np.float32)

    merged_vertices = merged_vertices - fake_pred_cam_t
    return merged_vertices, merged_faces, fake_pred_cam_t, vertex_counts


def _expand_vertex_colors(mesh_colors, vertex_counts):
    if not mesh_colors or not vertex_counts:
        return None
    per_vertex_colors = []
    for color, count in zip(mesh_colors, vertex_counts):
        tiled = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
        per_vertex_colors.append(tiled)
    return np.concatenate(per_vertex_colors, axis=0)


def visualize_sample(img_cv2, outputs, faces, distinct_colors=False):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    if distinct_colors:
        palette_float, palette_uint8 = _build_color_palettes(len(outputs))
    else:
        palette_float, palette_uint8 = None, None

    rend_img = []
    for pid, person_output in enumerate(outputs):
        if distinct_colors and palette_float is not None and pid < len(palette_float):
            mesh_color = tuple(palette_float[pid].tolist())
            bbox_color = palette_uint8[pid].tolist()
        else:
            mesh_color = LIGHT_BLUE
            bbox_color = (0, 255, 0)
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate([keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1)
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            bbox_color,
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=mesh_color,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=mesh_color,
                scene_bg_color=(1, 1, 1),
                side_view=True,
                rot_angle=SIDE_VIEW_ROTATION_DEG,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img


def visualize_sample_animation(
    img_cv2,
    outputs,
    faces,
    masks=None,
    keypoints=None,
    distinct_colors=True,
):
    # Render everything together
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    sorted_indices = np.argsort(-all_depths)
    outputs_sorted = [outputs[idx] for idx in sorted_indices]

    mesh_colors = []
    if distinct_colors:

        scores = [out["bbox_score"] for out in outputs_sorted]
        sorted_color_indices = np.argsort(np.array(scores))[::-1]
        palette_float, palette_uint8 = _build_color_palettes(len(outputs_sorted), order=sorted_color_indices, pastel_factor=0.2)
        random_rgb_colors = None
    else:
        palette_float = palette_uint8 = None
        random_rgb_colors = []
        for _ in range(len(outputs_sorted)):
            random_hsv = np.array([np.random.uniform(0, 255), 255, 255]).reshape(1, 1, 3)
            random_rgb = cv2.cvtColor(random_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).flatten().astype(np.uint8)
            random_rgb_colors.append(random_rgb)

    # Then, draw all keypoints and bboxes.
    for pid, person_output in enumerate(outputs_sorted):
        if distinct_colors and palette_float is not None and pid < len(palette_float):
            mesh_color = tuple(palette_float[pid].tolist())
            color_uint8 = palette_uint8[pid].tolist()
        else:
            mesh_color = LIGHT_BLUE
            color_uint8 = random_rgb_colors[pid].tolist()
        mesh_colors.append(mesh_color)

    merged_vertices, merged_faces, fake_pred_cam_t, vertex_counts = _merge_mesh_instances(outputs_sorted, faces)

    rendered_images = []

    if merged_vertices is None:
        img_mesh = img_cv2.copy()
        img_mesh_side = np.ones_like(img_cv2) * 255
    else:
        renderer = Renderer(focal_length=outputs_sorted[-1]["focal_length"], faces=merged_faces)
        vertex_colors = _expand_vertex_colors(mesh_colors, vertex_counts) if distinct_colors else None
        img_mesh = (
            renderer(
                merged_vertices,
                fake_pred_cam_t,
                img_mesh.astype(np.float32),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                vertex_colors=vertex_colors,
            )
            * 255
        ).astype(np.uint8)
        white_img = np.ones_like(img_cv2, dtype=np.float32) * 255.0

        rendered_images.append(img_mesh)

        num_frames_per_transition = 10
        angles = np.concatenate(
            [
                np.linspace(0, SIDE_VIEW_ROTATION_DEG, num=num_frames_per_transition, endpoint=True),
                np.linspace(SIDE_VIEW_ROTATION_DEG, 0, num=num_frames_per_transition, endpoint=True),
                np.linspace(0, -SIDE_VIEW_ROTATION_DEG, num=num_frames_per_transition, endpoint=True),
                np.linspace(-SIDE_VIEW_ROTATION_DEG, 0, num=num_frames_per_transition, endpoint=True),
            ],
            axis=0,
        )

        for angle in angles:
            img_mesh_side = (
                renderer(
                    merged_vertices,
                    fake_pred_cam_t,
                    white_img,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    vertex_colors=vertex_colors,
                    side_view=True,
                    rot_angle=angle,
                )
                * 255
            ).astype(np.uint8)
            rendered_images.append(img_mesh_side)

        rendered_images.append(img_mesh)

    return rendered_images


def visualize_sample_together(
    img_cv2,
    outputs,
    faces,
    masks=None,
    keypoints=None,
    distinct_colors=True,
):
    # Render everything together
    img_bboxes = img_cv2.copy()
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
    sorted_indices = np.argsort(-all_depths)
    outputs_sorted = [outputs[idx] for idx in sorted_indices]

    if masks is not None:
        masks = [masks[idx] for idx in sorted_indices]
    if keypoints is not None:
        keypoints = [keypoints[idx] for idx in sorted_indices]

    mesh_colors = []
    if distinct_colors:

        scores = [out["bbox_score"] for out in outputs_sorted]
        sorted_color_indices = np.argsort(np.array(scores))[::-1]
        palette_float, palette_uint8 = _build_color_palettes(len(outputs_sorted), order=sorted_color_indices, pastel_factor=0.2)
        random_rgb_colors = None
    else:
        palette_float = palette_uint8 = None
        random_rgb_colors = []
        for _ in range(len(outputs_sorted)):
            random_hsv = np.array([np.random.uniform(0, 255), 255, 255]).reshape(1, 1, 3)
            random_rgb = cv2.cvtColor(random_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).flatten().astype(np.uint8)
            random_rgb_colors.append(random_rgb)

    # Then, draw all keypoints and bboxes.
    for pid, person_output in enumerate(outputs_sorted):
        if distinct_colors and palette_float is not None and pid < len(palette_float):
            mesh_color = tuple(palette_float[pid].tolist())
            color_uint8 = palette_uint8[pid].tolist()[::-1]
        else:
            mesh_color = LIGHT_BLUE
            color_uint8 = random_rgb_colors[pid].tolist()[::-1]
        mesh_colors.append(mesh_color)
        mask = masks[pid] if masks is not None else None

        bbox = person_output["bbox"]

        if masks is not None:
            masks_image = img_bboxes.copy()
            mask = masks[pid]
            binary_mask = mask > 0.5
            masks_image[binary_mask, :] = color_uint8
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = [cmin, rmin, cmax, rmax]
            img_bboxes = cv2.addWeighted(img_bboxes, 0.4, masks_image, 0.6, 0)

        bbox_score = person_output.get("bbox_score", 0.0)
        img_bboxes = cv2.rectangle(
            img_bboxes,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color_uint8,
            2,
        )
        img_bboxes = cv2.putText(
            img_bboxes,
            f"{bbox_score:.2f}",
            (int(bbox[0]), int(bbox[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_uint8,
            2,
        )

        # Draw masks semi-transparently if available

        # Draw keypoints if available
        if keypoints is not None:
            keypoints_2d = keypoints[pid]
            if has_posevis:
                img_bboxes = pose_visualization(
                    img_bboxes,
                    keypoints_2d,
                    format="COCO",
                    greyness=1.0,
                    show_markers=True,
                    show_bones=True,
                    line_type="solid",
                    width_multiplier=1.0,
                    bbox_width_multiplier=1.0,
                    show_bbox=False,
                    differ_individuals=True,
                    conf_thr=0.3,
                    errors=None,
                    color=color_uint8,
                    keep_image_size=True,
                    return_padding=False,
                )
            else:
                for kp in keypoints_2d:
                    cv2.circle(img_bboxes, (int(kp[0]), int(kp[1])), 3, color_uint8, -1)

        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate([keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1)
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    merged_vertices, merged_faces, fake_pred_cam_t, vertex_counts = _merge_mesh_instances(outputs_sorted, faces)

    if merged_vertices is None:
        img_mesh = img_cv2.copy()
        img_mesh_side = np.ones_like(img_cv2) * 255
    else:
        renderer = Renderer(focal_length=outputs_sorted[-1]["focal_length"], faces=merged_faces)
        vertex_colors = _expand_vertex_colors(mesh_colors, vertex_counts) if distinct_colors else None
        img_mesh = (
            renderer(
                merged_vertices,
                fake_pred_cam_t,
                img_mesh.astype(np.float32),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                vertex_colors=vertex_colors,
            )
            * 255
        ).astype(np.uint8)

        white_img = np.ones_like(img_cv2, dtype=np.float32) * 255.0
        img_mesh_side = (
            renderer(
                merged_vertices,
                fake_pred_cam_t,
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                vertex_colors=vertex_colors,
                side_view=True,
                rot_angle=SIDE_VIEW_ROTATION_DEG,
            )
            * 255
        ).astype(np.uint8)

    cur_img = np.concatenate([img_bboxes, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img
