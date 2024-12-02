# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_offset_heatmap, generate_udp_gaussian_heatmaps,
                    get_heatmap_maximum, refine_keypoints_dark_udp)


@KEYPOINT_CODECS.register_module()
class UDPHeatmap(BaseKeypointCodec):
    r"""Generate keypoint heatmaps by Unbiased Data Processing (UDP).
    See the paper: `The Devil is in the Details: Delving into Unbiased Data
    Processing for Human Pose Estimation`_ by Huang et al (2020) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmap (np.ndarray): The generated heatmap in shape (C_out, H, W)
            where [W, H] is the `heatmap_size`, and the C_out is the output
            channel number which depends on the `heatmap_type`. If
            `heatmap_type=='gaussian'`, C_out equals to keypoint number K;
            if `heatmap_type=='combined'`, C_out equals to K*3
            (x_offset, y_offset and class label)
        - keypoint_weights (np.ndarray): The target weights in shape (K,)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        heatmap_type (str): The heatmap type to encode the keypoitns. Options
            are:

            - ``'gaussian'``: Gaussian heatmap
            - ``'combined'``: Combination of a binary label map and offset
                maps for X and Y axes.

        sigma (float): The sigma value of the Gaussian heatmap when
            ``heatmap_type=='gaussian'``. Defaults to 2.0
        radius_factor (float): The radius factor of the binary label
            map when ``heatmap_type=='combined'``. The positive region is
            defined as the neighbor of the keypoit with the radius
            :math:`r=radius_factor*max(W, H)`. Defaults to 0.0546875
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. Defaults to 11

    .. _`The Devil is in the Details: Delving into Unbiased Data Processing for
    Human Pose Estimation`: https://arxiv.org/abs/1911.07524
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 heatmap_type: str = 'gaussian',
                 sigma: float = 2.,
                 radius_factor: float = 0.0546875,
                 blur_kernel_size: int = 11,
                 increase_sigma_with_padding=False,
                 amap_scale: float = 1.0,
                 normalize=None,
                 ) -> None:
        super().__init__()
        self.input_size = np.array(input_size)
        self.heatmap_size = np.array(heatmap_size)
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size
        self.increase_sigma_with_padding = increase_sigma_with_padding
        self.normalize = normalize

        self.amap_size = self.input_size * amap_scale
        self.scale_factor = ((self.amap_size - 1) /
                             (self.heatmap_size - 1)).astype(np.float32)
        self.input_center = self.input_size / 2
        self.top_left = self.input_center - self.amap_size / 2
        
        if self.heatmap_type not in {'gaussian', 'combined'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')

    def _kpts_to_activation_pts(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Transform the keypoint coordinates to the activation space.
        In the original UDPHeatmap, activation map is the same as the input image space with
        different resolution but in this case we allow the activation map to have different
        size (padding) than the input image space.
        Centers of activation map and input image space are aligned.
        """
        transformed_keypoints = keypoints - self.top_left
        transformed_keypoints = transformed_keypoints / self.scale_factor
        return transformed_keypoints
    
    def _activation_pts_to_kpts(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Transform the points in activation map to the keypoint coordinates.
        In the original UDPHeatmap, activation map is the same as the input image space with
        different resolution but in this case we allow the activation map to have different
        size (padding) than the input image space.
        Centers of activation map and input image space are aligned.
        """
        W, H = self.heatmap_size
        transformed_keypoints = keypoints / [W - 1, H - 1] * self.amap_size
        transformed_keypoints += self.top_left
        return transformed_keypoints

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               id_similarity: Optional[float] = 0.0,
               keypoints_visibility: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            id_similarity (float): The usefulness of the identity information
                for the whole pose. Defaults to 0.0
            keypoints_visibility (np.ndarray): The visibility bit for each
                keypoint (N, K). Defaults to None

        Returns:
            dict:
            - heatmap (np.ndarray): The generated heatmap in shape
                (C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """
        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')
        
        if keypoints_visibility is None:
            keypoints_visibility = np.zeros(keypoints.shape[:2], dtype=np.float32)

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.heatmap_type == 'gaussian':
            heatmaps, keypoint_weights = generate_udp_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints),
                keypoints_visible=keypoints_visible,
                sigma=self.sigma,
                keypoints_visibility=keypoints_visibility,
                increase_sigma_with_padding=self.increase_sigma_with_padding)
        elif self.heatmap_type == 'combined':
            heatmaps, keypoint_weights = generate_offset_heatmap(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints),
                keypoints_visible=keypoints_visible,
                radius_factor=self.radius_factor)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')
        
        if self.normalize is not None:
            heatmaps_sum = np.sum(heatmaps, axis=(1, 2), keepdims=False)
            mask = heatmaps_sum > 0
            heatmaps[mask, :, :] = heatmaps[mask, :, :] / (heatmaps_sum[mask, None, None] + np.finfo(np.float32).eps)
            heatmaps = heatmaps * self.normalize

        annotated = keypoints_visible > 0
        
        heatmap_keypoints = self._kpts_to_activation_pts(keypoints)
        in_image = np.logical_and(
            heatmap_keypoints[:, :, 0] >= 0,
            heatmap_keypoints[:, :, 0] < self.heatmap_size[0],
        )
        in_image = np.logical_and(
            in_image,
            heatmap_keypoints[:, :, 1] >= 0,
        )
        in_image = np.logical_and(
            in_image,
            heatmap_keypoints[:, :, 1] < self.heatmap_size[1],
        )
        
        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            annotated=annotated,
            in_image=in_image,
            keypoints_scaled=keypoints,
            heatmap_keypoints=heatmap_keypoints,
            identification_similarity=id_similarity,
        )

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()

        if self.heatmap_type == 'gaussian':
            keypoints, scores = get_heatmap_maximum(heatmaps)
            # unsqueeze the instance dimension for single-instance results
            keypoints = keypoints[None]
            scores = scores[None]

            keypoints = refine_keypoints_dark_udp(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        elif self.heatmap_type == 'combined':
            _K, H, W = heatmaps.shape
            K = _K // 3

            for cls_heatmap in heatmaps[::3]:
                # Apply Gaussian blur on classification maps
                ks = 2 * self.blur_kernel_size + 1
                cv2.GaussianBlur(cls_heatmap, (ks, ks), 0, cls_heatmap)

            # valid radius
            radius = self.radius_factor * max(W, H)

            x_offset = heatmaps[1::3].flatten() * radius
            y_offset = heatmaps[2::3].flatten() * radius
            keypoints, scores = get_heatmap_maximum(heatmaps=heatmaps[::3])
            index = (keypoints[..., 0] + keypoints[..., 1] * W).flatten()
            index += W * H * np.arange(0, K)
            index = index.astype(int)
            keypoints += np.stack((x_offset[index], y_offset[index]), axis=-1)
            # unsqueeze the instance dimension for single-instance results
            keypoints = keypoints[None].astype(np.float32)
            scores = scores[None]

        keypoints = self._activation_pts_to_kpts(keypoints)

        return keypoints, scores
