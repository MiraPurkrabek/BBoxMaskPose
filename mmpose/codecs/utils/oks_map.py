# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist


def generate_oks_maps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    keypoints_visibility: np.ndarray,
    sigma: float = 0.55,
    increase_sigma_with_padding: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `UDP`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value of the Gaussian heatmap
        keypoints_visibility (np.ndarray): The visibility bit for each keypoint (N, K)
        increase_sigma_with_padding (bool): Whether to increase the sigma
            value with padding. Default: False

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    # The default sigmas are used for COCO dataset.
    sigmas = np.array(
        [2.6, 2.5, 2.5, 3.5, 3.5, 7.9, 7.9, 7.2, 7.2, 6.2, 6.2, 10.7, 10.7, 8.7, 8.7, 8.9, 8.9])/100
    # sigmas = sigmas * 2 / sigmas.mean()
    # sigmas = np.round(sigmas).astype(int)
    # sigmas = np.clip(sigmas, 1, 10)
    
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # bbox_area = W/1.25 * H/1.25
    # bbox_area = W * H * 0.53
    bbox_area = np.sqrt(H/1.25 * W/1.25)

    # print(scales_arr)
    # print(scaled_sigmas)

    for n, k in product(range(N), range(K)):
        kpt_sigma = sigmas[k]
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        y_idx, x_idx = np.indices((H, W))
        dx = x_idx - keypoints[n, k, 0]
        dy = y_idx - keypoints[n, k, 1]
        dist = np.sqrt(dx**2 + dy**2)

        # e_map = (dx**2 + dy**2) / ((kpt_sigma*100)**2 * sigma)
        vars = (kpt_sigma*2)**2
        s = vars * bbox_area * 2
        s = np.clip(s, 0.55, 3.0)
        if sigma is not None and sigma > 0:
            s = sigma
        e_map = dist**2 / (2*s)
        oks_map = np.exp(-e_map)

        keypoint_weights[n, k] = (oks_map.max() > 0).astype(int)
        
        # Scale such that there is always 1 at the maximum
        if oks_map.max() > 1e-3:
            oks_map = oks_map / oks_map.max()

        # Scale OKS map such that 1 stays 1 and 0.5 becomes 0
        # oks_map[oks_map < 0.5] = 0
        # oks_map = 2 * oks_map - 1


        # oks_map[oks_map > 0.95] = 1
        # print("{:.4f}, {:7.1f}, {:9.3f}, {:9.3f}, {:4.2f}".format(vars, bbox_area, vars * bbox_area* 2, s, oks_map.max()))
        # if np.all(oks_map < 0.1):
        #     print("\t{:d} --> {:.4f}".format(k, s))
        heatmaps[k] = oks_map 
        # breakpoint()

    return heatmaps, keypoint_weights
