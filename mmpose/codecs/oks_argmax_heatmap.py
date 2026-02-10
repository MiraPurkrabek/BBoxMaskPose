# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import numpy as np
import os

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_offset_heatmap, generate_udp_gaussian_heatmaps,
                    get_heatmap_maximum, refine_keypoints_dark_udp, generate_oks_maps, get_heatmap_expected_value, gaussian_blur)
from mmpose.utils import get_root_logger


@KEYPOINT_CODECS.register_module()
class OKSArgMaxHeatmap(BaseKeypointCodec):
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
                 sigma: float = -1,
                 radius_factor: float = 0.0546875,
                 blur_kernel_size: int = 11,
                 increase_sigma_with_padding=False,
                 ) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = ((np.array(input_size) - 1) /
                             (np.array(heatmap_size) - 1)).astype(np.float32)
        self.increase_sigma_with_padding = increase_sigma_with_padding
        self.sigma = sigma

        if self.heatmap_type not in {'gaussian', 'combined'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')

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

        
        heatmaps, keypoint_weights = generate_oks_maps(
            heatmap_size=self.heatmap_size,
            keypoints=keypoints / self.scale_factor,
            keypoints_visible=keypoints_visible,
            keypoints_visibility=keypoints_visibility,
            sigma=self.sigma,
            increase_sigma_with_padding=self.increase_sigma_with_padding)

        annotated = keypoints_visible > 0
        
        in_image = np.logical_and(
            keypoints[:, :, 0] >= 0,
            keypoints[:, :, 0] < self.input_size[0],
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] >= 0,
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] < self.input_size[1],
        )
        
        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            annotated=annotated,
            in_image=in_image,
            keypoints_scaled=keypoints,
            # identification_similarity=id_similarity,
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
        W, H = self.heatmap_size

        logger = get_root_logger()

        if self.heatmap_type == 'gaussian':

            # Set the bottom 20% of the heatmaps to zero to filter-out low values (~noise)
            # bottom_threshold = np.percentile(heatmaps, 20)
            # heatmaps[heatmaps < bottom_threshold] = 0

            keypoints_max, scores = get_heatmap_maximum(heatmaps)
            # unsqueeze the instance dimension for single-instance results
            keypoints_max = keypoints_max[None]
            scores = scores[None]

            try:
                keypoints = refine_keypoints_dark_udp(
                    keypoints_max.copy(), heatmaps, blur_kernel_size=self.blur_kernel_size)
                # print(keypoints_max.shape, keypoints_max.dtype, keypoints.shape, keypoints.dtype)
            except np.linalg.LinAlgError:
                keypoints = keypoints_max.copy()
                logger.warning('LinAlgError in np.linalg.solve, '
                               'use the non-refined keypoints instead')

            draw_comparison = False
            if draw_comparison:

                keypoints_exp, _, oks_maps = get_heatmap_expected_value(encoded.copy(), return_heatmap=True)
                keypoints_exp = keypoints_exp.reshape(keypoints.shape)

                dist = np.linalg.norm(keypoints - keypoints_exp, axis=-1)

                # keypoints = keypoints_exp
                # keypoints[dist > 1] = keypoints_exp[dist > 1]


                for k in range(keypoints.shape[1]):
                    # continue
                    d = dist[0, k]

                    # 1/4 of heatmap pixel is 1 pixel in image space
                    if 0.5 < d:
                        
                        # Skip 80% of the heatmaps to save time and space
                        if np.random.rand() < 0.8:
                            continue

                        # size = np.array([W, H])
                        size = self.input_size

                        # Draw heatmaps with estimated values
                        htm = encoded.copy()[k, :, :]
                        # kpt_max, _ = get_heatmap_maximum(htm.reshape(1, H, W).copy())
                        # kpt_max = np.array(kpt_max).reshape(2)
                        htm = cv2.resize(htm, (size[0], size[1]))
                        htm = cv2.cvtColor(htm, cv2.COLOR_GRAY2BGR)
                        htm /= htm.max()
                        htm = cv2.applyColorMap((htm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        htm_exp = htm.copy()
                        htm_max = htm.copy()
                        kpt = keypoints[0, k, :]
                        kpt_max = keypoints_max[0, k, :]
                        kpt_exp = keypoints_exp[0, k, :]

                        kpt = (kpt / [W - 1, H - 1] * size).flatten()
                        kpt_exp = (kpt_exp / [W - 1, H - 1] * size).flatten()
                        kpt_max = (kpt_max / [W - 1, H - 1] * size).flatten()

                        # kpt[0] = np.clip(kpt[0], 0, size[0] - 1)
                        # kpt[1] = np.clip(kpt[1], 0, size[1] - 1)
                        # kpt_exp[0] = np.clip(kpt_exp[0], 0, size[0] - 1)
                        # kpt_exp[1] = np.clip(kpt_exp[1], 0, size[1] - 1)
                        # kpt_max[0] = np.clip(kpt_max[0], 0, size[0] - 1)
                        # kpt_max[1] = np.clip(kpt_max[1], 0, size[1] - 1)

                        kpt = kpt.astype(int)
                        kpt_exp = kpt_exp.astype(int)
                        kpt_max = kpt_max.astype(int)

                        htm_raw = htm.copy()

                        
                        htm_center = np.array(size) // 2
                        htm = cv2.arrowedLine(htm, htm_center, kpt, (191, 64, 191), thickness=1, tipLength=0.05)
                        htm_exp = cv2.arrowedLine(htm_exp, htm_center, kpt_exp, (191, 64, 191), thickness=1, tipLength=0.05)
                        htm_max = cv2.arrowedLine(htm_max, htm_center, kpt_max, (191, 64, 191), thickness=1, tipLength=0.05)
                        
                        white_column = np.ones((size[1], 3, 3), dtype=np.uint8) * 150
                        save_img = np.hstack((htm_max, white_column, htm, white_column, htm_exp))
                        
                        oksm = oks_maps[k, :, :]
                        oksm = cv2.resize(oksm, (size[0], size[1]))
                        oksm /= oksm.max()
                        oksm = cv2.cvtColor(oksm, cv2.COLOR_GRAY2BGR)
                        oksm = cv2.applyColorMap((oksm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        

                        raw_htm = encoded[k, :, :].copy().reshape(1, H, W)
                        blur_htm = gaussian_blur(raw_htm.copy(), self.blur_kernel_size).squeeze()
                        blur_htm = cv2.resize(blur_htm, (size[0], size[1]))
                        blur_htm /= blur_htm.max()
                        blur_htm = cv2.cvtColor(blur_htm, cv2.COLOR_GRAY2BGR)
                        blur_htm = cv2.applyColorMap((blur_htm * 255).astype(np.uint8), cv2.COLORMAP_JET)

                        raw_htm = cv2.resize(raw_htm.squeeze(), (size[0], size[1]))
                        raw_htm /= raw_htm.max()
                        raw_htm = cv2.cvtColor(raw_htm, cv2.COLOR_GRAY2BGR)
                        raw_htm = cv2.applyColorMap((raw_htm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        
                        oksm_merge = oksm.copy()
                        oksm_merge = cv2.drawMarker(oksm_merge, kpt_exp, (191, 64, 191), cv2.MARKER_CROSS, 10, 2)
                        
                        htm_merge = blur_htm.copy()
                        htm_merge = cv2.drawMarker(htm_merge, kpt, (255, 159, 207), cv2.MARKER_CROSS, 10, 2)
                        htm_merge = cv2.drawMarker(htm_merge, kpt_max, (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
                        
                        save_heatmaps = np.hstack((raw_htm, white_column, blur_htm, white_column, oksm))
                        white_row = np.ones((3, save_img.shape[1], 3), dtype=np.uint8) * 150
                        save_img = np.vstack((save_img, white_row, save_heatmaps))
                        
                        os.makedirs('debug', exist_ok=True)
                        save_path = "debug/{:04.1f}_{:d}_{:06d}.png".format(d, k, abs(hash(str(keypoints[0, k, :])) % (10 ** 6)))
                        cv2.imwrite(save_path, save_img)
                        save_path = "debug/{:04.1f}_{:d}_{:06d}_merge.png".format(d, k, abs(hash(str(keypoints[0, k, :])) % (10 ** 6)))
                        cv2.imwrite(save_path, np.hstack((htm_merge, white_column, oksm_merge)))
                        save_path = "debug/{:04.1f}_{:d}_{:06d}_blur.png".format(d, k, abs(hash(str(keypoints[0, k, :])) % (10 ** 6)))
                        cv2.imwrite(save_path, htm_merge)
                        save_path = "debug/{:04.1f}_{:d}_{:06d}_oks.png".format(d, k, abs(hash(str(keypoints[0, k, :])) % (10 ** 6)))
                        cv2.imwrite(save_path, oksm_merge)

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

        keypoints = keypoints / [W - 1, H - 1] * self.input_size

        return keypoints, scores
