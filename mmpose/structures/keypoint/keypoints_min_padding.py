import numpy as np

def find_min_padding_exact(bbox, kpts, aspect_ratio=3/4, bbox_format='xywh'):
    '''Find the minimum padding to make keypoint inside bbox'''
    assert bbox_format.lower() in ['xywh', 'xyxy'], f"Invalid bbox format {bbox_format}. Only 'xyxy' or 'xywh' are supported."

    if kpts.size % 2 == 0:
        kpts = kpts.reshape(-1, 2)
        vis = np.ones(kpts.shape[0])
    elif kpts.size % 3 == 0:
        kpts = kpts.reshape(-1, 3)
        vis = kpts[:, 2].flatten()
        kpts = kpts[:, :2]
    else:
        raise ValueError('Keypoints should have 2 or 3 values each')

    if bbox_format.lower() == 'xyxy':
        bbox = np.array([
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ])

    if aspect_ratio is not None:
        # Fix the aspect ratio of the bounding box
        bbox = fix_bbox_aspect_ratio(bbox, aspect_ratio=aspect_ratio, padding=1.0, bbox_format='xywh')
    
    x0, y0, w, h = np.hsplit(bbox, [1, 2, 3])

    x1 = x0 + w
    y1 = y0 + h
    x_bbox_distances = np.max(np.stack([
        np.clip(x0 - kpts[:, 0], a_min=0, a_max=None),
        np.clip(kpts[:, 0] - x1, a_min=0, a_max=None),
    ]), axis=0)
    y_bbox_distances = np.max(np.stack([
        np.clip(y0 - kpts[:, 1], a_min=0, a_max=None),
        np.clip(kpts[:, 1] - y1, a_min=0, a_max=None),
    ]), axis=0)

    padding_x = 2 * x_bbox_distances / w
    padding_y = 2 * y_bbox_distances / h
    padding = 1 + np.maximum(padding_x, padding_y)
    padding = np.array(padding).flatten()

    padding[vis <= 0] = -1.0
    
    return padding

def fix_bbox_aspect_ratio(bbox, aspect_ratio=3/4, padding=1.25, bbox_format='xywh'):
    assert bbox_format.lower() in ['xywh', 'xyxy'], f"Invalid bbox format {bbox_format}. Only 'xyxy' or 'xywh' are supported."

    in_shape = bbox.shape
    bbox = bbox.reshape((-1, 4))

    if bbox_format.lower() == 'xywh':
        bbox_xyxy = np.array([
            bbox[:, 0],
            bbox[:, 1],
            bbox[:, 0] + bbox[:, 2],
            bbox[:, 1] + bbox[:, 3],
        ]).T
    else:
        bbox_xyxy = np.array(bbox)
    
    centers = bbox_xyxy[:, :2] + (bbox_xyxy[:, 2:] - bbox_xyxy[:, :2]) / 2
    widths = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
    heights = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
    
    new_widths = widths.copy().astype(np.float32)
    new_heights = heights.copy().astype(np.float32)

    for i in range(bbox_xyxy.shape[0]):
        if widths[i] == 0:
            widths[i] =+ 1
        if heights[i] == 0:
            heights[i] =+ 1

        if widths[i] / heights[i] > aspect_ratio:
            new_heights[i] = widths[i] / aspect_ratio
        else:
            new_widths[i] = heights[i] * aspect_ratio
    new_widths *= padding
    new_heights *= padding

    new_bbox_xyxy = np.array([
        centers[:, 0] - new_widths / 2,
        centers[:, 1] - new_heights / 2,
        centers[:, 0] + new_widths / 2,
        centers[:, 1] + new_heights / 2,
    ]).T

    if bbox_format.lower() == 'xywh':
        new_bbox = np.array([
            new_bbox_xyxy[:, 0],
            new_bbox_xyxy[:, 1],
            new_bbox_xyxy[:, 2] - new_bbox_xyxy[:, 0],
            new_bbox_xyxy[:, 3] - new_bbox_xyxy[:, 1],
        ]).T
    else:
        new_bbox = new_bbox_xyxy


    new_bbox = new_bbox.reshape(in_shape)

    return new_bbox