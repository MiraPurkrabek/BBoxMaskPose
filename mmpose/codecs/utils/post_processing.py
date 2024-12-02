# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scipy.signal import convolve2d


def get_simcc_normalized(batch_pred_simcc, sigma=None):
    """Normalize the predicted SimCC.

    Args:
        batch_pred_simcc (torch.Tensor): The predicted SimCC.
        sigma (float): The sigma of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized SimCC.
    """
    B, K, _ = batch_pred_simcc.shape

    # Scale and clamp the tensor
    if sigma is not None:
        batch_pred_simcc = batch_pred_simcc / (sigma * np.sqrt(np.pi * 2))
    batch_pred_simcc = batch_pred_simcc.clamp(min=0)

    # Compute the binary mask
    mask = (batch_pred_simcc.amax(dim=-1) > 1).reshape(B, K, 1)

    # Normalize the tensor using the maximum value
    norm = (batch_pred_simcc / batch_pred_simcc.amax(dim=-1).reshape(B, K, 1))

    # Apply normalization
    batch_pred_simcc = torch.where(mask, norm, batch_pred_simcc)

    return batch_pred_simcc


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray,
                      apply_softmax: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_x.ndim == simcc_y.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals


def get_heatmap_3d_maximum(heatmaps: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap dimension: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, D, H, W) or
            (B, K, D, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 3) or (B, K, 3)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4 or heatmaps.ndim == 5, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 4:
        K, D, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, D, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    z_locs, y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(D, H, W))
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 3)
        vals = vals.reshape(B, K)

    return locs, vals


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / (np.max(heatmaps[k])+1e-12)
    return heatmaps


def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc


def batch_heatmap_nms(batch_heatmaps: Tensor, kernel_size: int = 5):
    """Apply NMS on a batch of heatmaps.

    Args:
        batch_heatmaps (Tensor): batch heatmaps in shape (B, K, H, W)
        kernel_size (int): The kernel size of the NMS which should be
            a odd integer. Defaults to 5

    Returns:
        Tensor: The batch heatmaps after NMS.
    """

    assert isinstance(kernel_size, int) and kernel_size % 2 == 1, \
        f'The kernel_size should be an odd integer, got {kernel_size}'

    padding = (kernel_size - 1) // 2

    maximum = F.max_pool2d(
        batch_heatmaps, kernel_size, stride=1, padding=padding)
    maximum_indicator = torch.eq(batch_heatmaps, maximum)
    batch_heatmaps = batch_heatmaps * maximum_indicator.float()

    return batch_heatmaps


def get_heatmap_expected_value(heatmaps: np.ndarray, parzen_size: float = 0.1, return_heatmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')
    
    assert parzen_size >= 0.0 and parzen_size <= 1.0, (
        f'Invalid parzen_size {parzen_size}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = 1
        FIRST_DIM = K
        heatmaps_flatten = heatmaps.reshape(1, K, H, W)
    else:
        B, K, H, W = heatmaps.shape
        FIRST_DIM = K*B
        heatmaps_flatten = heatmaps.reshape(B, K, H, W)

    # Blur heatmaps with Gaussian
    # heatmaps_flatten = gaussian_blur(heatmaps_flatten, kernel=9)
    
    # Zero out pixels far from the maximum for each heatmap
    # heatmaps_tmp = heatmaps_flatten.copy().reshape(B*K, H*W)
    # y_locs, x_locs = np.unravel_index(
    #     np.argmax(heatmaps_tmp, axis=1), shape=(H, W))
    # locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    # heatmaps_flatten = heatmaps_flatten.reshape(B*K, H, W)
    # for i, x in enumerate(x_locs):
    #     y = y_locs[i]
    #     start_x = int(max(0, x - 0.2*W))
    #     end_x = int(min(W, x + 0.2*W))
    #     start_y = int(max(0, y - 0.2*H))
    #     end_y = int(min(H, y + 0.2*H))
    #     mask = np.zeros((H, W))
    #     mask[start_y:end_y, start_x:end_x] = 1
    #     heatmaps_flatten[i] = heatmaps_flatten[i] * mask
    # heatmaps_flatten = heatmaps_flatten.reshape(B, K, H, W)


    bbox_area = np.sqrt(H/1.25 * W/1.25)

    kpt_sigmas = np.array(
        [2.6, 2.5, 2.5, 3.5, 3.5, 7.9, 7.9, 7.2, 7.2, 6.2, 6.2, 10.7, 10.7, 8.7, 8.7, 8.9, 8.9])/100
    
    heatmaps_covolved = np.zeros_like(heatmaps_flatten)
    for k in range(K):
        vars = (kpt_sigmas[k]*2)**2
        s = vars * bbox_area * 2
        s = np.clip(s, 0.55, 3.0)
        radius = np.ceil(s * 3).astype(int)
        diameter = 2*radius + 1
        diameter = np.ceil(diameter).astype(int)
        # kernel_sizes[kernel_sizes % 2 == 0] += 1
        center = diameter // 2
        dist_x = np.arange(diameter) - center
        dist_y = np.arange(diameter) - center
        dist_x, dist_y = np.meshgrid(dist_x, dist_y)
        dist = np.sqrt(dist_x**2 + dist_y**2)
        oks_kernel = np.exp(-dist**2 / (2 * s))
        oks_kernel = oks_kernel / oks_kernel.sum()
        
        htm = heatmaps_flatten[:, k, :, :].reshape(-1, H, W)
        # htm = np.pad(htm, ((0, 0), (radius, radius), (radius, radius)), mode='symmetric')
        # htm = torch.from_numpy(htm).float()
        # oks_kernel = torch.from_numpy(oks_kernel).float().to(htm.device).reshape(1, diameter, diameter)
        oks_kernel = oks_kernel.reshape(1, diameter, diameter)
        htm_conv = np.zeros_like(htm)
        for b in range(B):
            htm_conv[b, :, :] = convolve2d(htm[b, :, :], oks_kernel[b, :, :], mode='same', boundary='symm')
        # htm_conv = F.conv2d(htm.unsqueeze(1), oks_kernel.unsqueeze(1), padding='same')
        # htm_conv = htm_conv[:, :, radius:-radius, radius:-radius]
        htm_conv = htm_conv.reshape(-1, 1, H, W)
        heatmaps_covolved[:, k, :, :] = htm_conv

    
    heatmaps_covolved = heatmaps_covolved.reshape(B*K, H*W)
    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_covolved, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    # Apply mean-shift to get sub-pixel locations
    locs = _get_subpixel_maximums(heatmaps_covolved.reshape(B*K, H, W), locs)
    # breakpoint()


    # heatmaps_sums = heatmaps_flatten.sum(axis=(1, 2))
    # norm_heatmaps = heatmaps_flatten.copy()
    # norm_heatmaps[heatmaps_sums > 0] = heatmaps_flatten[heatmaps_sums > 0] / heatmaps_sums[heatmaps_sums > 0, None, None]
    

    # # Compute Parzen window with Gaussian blur along the edge instead of simple mirroring
    # x_pad = int(parzen_size * W + 0.5)
    # y_pad = int(parzen_size * H + 0.5)
    # # x_pad = 0
    # # y_pad = 0
    # kernel_size = int(min(H, W)*parzen_size + 0.5)
    # if kernel_size % 2 == 0:
    #     kernel_size += 1
    # # norm_heatmaps_pad_blur = np.pad(norm_heatmaps, ((0, 0), (x_pad, x_pad), (y_pad, y_pad)), mode='symmetric')
    # norm_heatmaps_pad = np.pad(norm_heatmaps, ((0, 0), (y_pad, y_pad), (x_pad, x_pad)), mode='constant', constant_values=0)
    # norm_heatmaps_pad_blur = gaussian_blur(norm_heatmaps_pad, kernel=kernel_size)
        
    # # norm_heatmaps_pad_blur[:, x_pad:-x_pad, y_pad:-y_pad] = norm_heatmaps
    
    # norm_heatmaps_pad_sum = norm_heatmaps_pad_blur.sum(axis=(1, 2))
    # norm_heatmaps_pad_blur[norm_heatmaps_pad_sum>0] = norm_heatmaps_pad_blur[norm_heatmaps_pad_sum>0] / norm_heatmaps_pad_sum[norm_heatmaps_pad_sum>0, None, None]
    
    # # # Save the blurred heatmaps
    # # for i in range(heatmaps.shape[0]):
    # #     tmp_htm = norm_heatmaps_pad_blur[i].copy()
    # #     tmp_htm = (tmp_htm - tmp_htm.min()) / (tmp_htm.max() - tmp_htm.min())
    # #     tmp_htm = (tmp_htm*255).astype(np.uint8)
    # #     tmp_htm = cv2.cvtColor(tmp_htm, cv2.COLOR_GRAY2BGR)
    # #     tmp_htm = cv2.applyColorMap(tmp_htm, cv2.COLORMAP_JET)

    # #     tmp_htm2 = norm_heatmaps_pad[i].copy()
    # #     tmp_htm2 = (tmp_htm2 - tmp_htm2.min()) / (tmp_htm2.max() - tmp_htm2.min())
    # #     tmp_htm2 = (tmp_htm2*255).astype(np.uint8)
    # #     tmp_htm2 = cv2.cvtColor(tmp_htm2, cv2.COLOR_GRAY2BGR)
    # #     tmp_htm2 = cv2.applyColorMap(tmp_htm2, cv2.COLORMAP_JET)

    # #     tmp_htm = cv2.addWeighted(tmp_htm, 0.5, tmp_htm2, 0.5, 0)

    # #     cv2.imwrite(f'heatmaps_blurred_{i}.png', tmp_htm)

    # # norm_heatmaps_pad = np.pad(norm_heatmaps, ((0, 0), (x_pad, x_pad), (y_pad, y_pad)), mode='edge')

    # y_idx, x_idx = np.indices(norm_heatmaps_pad_blur.shape[1:])

    # # breakpoint()
    # x_locs = np.sum(norm_heatmaps_pad_blur * x_idx, axis=(1, 2)) - x_pad
    # y_locs = np.sum(norm_heatmaps_pad_blur * y_idx, axis=(1, 2)) - y_pad
    
    # # mean_idx = np.argmax(heatmaps_flatten, axis=1)
    # # x_locs, y_locs = np.unravel_index(mean_idx, shape=(H, W))
    # # locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    # # breakpoint()
    # # vals = heatmaps_flatten[np.arange(heatmaps_flatten.shape[0]), mean_idx]
    # # locs[vals <= 0.] = -1

    # # mean_idx = np.argmax(norm_heatmaps, axis=1)
    # # y_locs, x_locs = np.unravel_index(
    # #     mean_idx, shape=(H, W))
    
    # locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    # # vals = np.amax(heatmaps_flatten, axis=1)
    
    
    x_locs_int = np.round(x_locs).astype(int)
    x_locs_int = np.clip(x_locs_int, 0, W-1)
    y_locs_int = np.round(y_locs).astype(int)
    y_locs_int = np.clip(y_locs_int, 0, H-1)
    vals = heatmaps_flatten[np.arange(B), np.arange(K), y_locs_int, x_locs_int]
    # breakpoint()
    # locs[vals <= 0.] = -1

    # print(mean_idx)
    # print(x_locs)
    # print(y_locs)
    # print(locs)
    heatmaps_covolved = heatmaps_covolved.reshape(B, K, H, W)

    if B > 1:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)
        heatmaps_covolved = heatmaps_covolved.reshape(B, K, H, W)
    else:
        locs = locs.reshape(K, 2)
        vals = vals.reshape(K)
        heatmaps_covolved = heatmaps_covolved.reshape(K, H, W)

    if return_heatmap:
        return locs, vals, heatmaps_covolved
    else:
        return locs, vals       



def _get_subpixel_maximums(heatmaps, locs):
    # Extract integer peak locations
    x_locs = locs[:, 0].astype(np.int32)
    y_locs = locs[:, 1].astype(np.int32)

    # Ensure we are not near the boundaries (avoid boundary issues)
    valid_mask = (x_locs > 0) & (x_locs < heatmaps.shape[2] - 1) & \
                 (y_locs > 0) & (y_locs < heatmaps.shape[1] - 1)

    # Initialize the output array with the integer locations
    subpixel_locs = locs.copy()

    if np.any(valid_mask):
        # Extract valid locations
        x_locs_valid = x_locs[valid_mask]
        y_locs_valid = y_locs[valid_mask]

        # Compute gradients (dx, dy) and second derivatives (dxx, dyy)
        dx = (heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1] - 
              heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1]) / 2.0
        dy = (heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid] - 
              heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid]) / 2.0
        dxx = heatmaps[valid_mask, y_locs_valid, x_locs_valid + 1] + \
              heatmaps[valid_mask, y_locs_valid, x_locs_valid - 1] - \
              2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]
        dyy = heatmaps[valid_mask, y_locs_valid + 1, x_locs_valid] + \
              heatmaps[valid_mask, y_locs_valid - 1, x_locs_valid] - \
              2 * heatmaps[valid_mask, y_locs_valid, x_locs_valid]

        # Avoid division by zero by setting a minimum threshold for the second derivatives
        dxx = np.where(dxx != 0, dxx, 1e-6)
        dyy = np.where(dyy != 0, dyy, 1e-6)

        # Calculate the sub-pixel shift
        subpixel_x_shift = -dx / dxx
        subpixel_y_shift = -dy / dyy

        # Update subpixel locations for valid indices
        subpixel_locs[valid_mask, 0] += subpixel_x_shift
        subpixel_locs[valid_mask, 1] += subpixel_y_shift

    return subpixel_locs

