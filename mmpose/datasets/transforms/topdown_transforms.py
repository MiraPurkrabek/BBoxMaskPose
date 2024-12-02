# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix, bbox_cs2xyxy, bbox_xyxy2cs


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints
        - bbox_mask

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 input_padding: float = 1.25,
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp
        self.input_padding = input_padding

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))
        img_h, img_w = results['img'].shape[:2]

        bbox_xyxy = results['bbox_xyxy_wrt_input'].flatten()
        bbox_xyxy[:2] = np.maximum(bbox_xyxy[:2], 0)
        bbox_xyxy[2:4] = np.minimum(bbox_xyxy[2:4], [img_w, img_h])
        # bbox_xyxy[2] = np.minimum(bbox_xyxy[2], img_w)
        # bbox_xyxy[3] = np.minimum(bbox_xyxy[3], img_h)
        x0, y0, x1, y1 = bbox_xyxy[:4].astype(int)
        bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bbox_mask[y0:y1, x0:x1] = 1
        # bbox_img = np.tile(bbox_mask[:, :, None], (1, 1, 3))
        # img = results['img']
        # bbox_img = bbox_img * 255
        # transparent = cv2.addWeighted(img, 0.5, bbox_img, 0.5, 0)
        # cv2.imwrite("transparent.jpg", transparent)
        # breakpoint()

        
        # assert x0 >= 0 and x0 < img_w, f'{x0} {img_w}'
        # assert x1 >= 0 and x1 < img_w, f'{x1} {img_w}'
        # assert y0 >= 0 and y0 < img_h, f'{y0} {img_h}'
        # assert y1 >= 0 and y1 < img_h, f'{y1} {img_h}'
        # assert bbox_mask.any(), 'bbox_mask is empty'

        # Take the bbox wrt the input
        bbox_xyxy_wrt_input = results.get('bbox_xyxy_wrt_input', None)
        if bbox_xyxy_wrt_input is not None:
            _c, _s = bbox_xyxy2cs(bbox_xyxy_wrt_input, padding=self.input_padding)
            results['bbox_center'] = _c.reshape(1, 2)
            results['bbox_scale'] = _s.reshape(1, 2)

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)
            bbox_mask = cv2.warpAffine(
                bbox_mask, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
            bbox_mask = bbox_mask.reshape(1, h, w)
            results['bbox_mask'] = bbox_mask

        if results.get('keypoints', None) is not None:
            if results.get('transformed_keypoints', None) is not None:
                transformed_keypoints = results['transformed_keypoints'].copy()
            else:
                transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
               transformed_keypoints[..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        if results.get('bbox_xyxy_wrt_input', None) is not None:
            bbox_xyxy_wrt_input = results['bbox_xyxy_wrt_input'].copy()
            bbox_xyxy_wrt_input = bbox_xyxy_wrt_input.reshape(1, 2, 2)
            bbox_xyxy_wrt_input = cv2.transform(
                bbox_xyxy_wrt_input, warp_mat)
            results['bbox_xyxy_wrt_input'] = bbox_xyxy_wrt_input.reshape(1, 4)

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str
