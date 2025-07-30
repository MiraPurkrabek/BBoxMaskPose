# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS

from mmpose.structures.keypoint import fix_bbox_aspect_ratio


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self, pad_to_aspect_ratio=False, **kwargs):
        super().__init__(**kwargs)
        self.pad_to_aspect_ratio = pad_to_aspect_ratio

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]

            if self.pad_to_aspect_ratio:
                # Pad image with zeros to ensure activation map is not cut off
                abox_xyxy = fix_bbox_aspect_ratio(
                    results['bbox'], aspect_ratio=3/4, padding=1.25, bbox_format='xyxy').flatten()
                
                x_pad = np.array([max(0, -abox_xyxy[0]), max(0, abox_xyxy[2] - results['img_shape'][1])], dtype=int)
                y_pad = np.array([max(0, -abox_xyxy[1]), max(0, abox_xyxy[3] - results['img_shape'][0])], dtype=int)

                img = results['img']
                img = np.pad(img, ((y_pad[0], y_pad[1]), (x_pad[0], x_pad[1]), (0, 0)), mode='constant', constant_values=255)
                results['img'] = img
                
                # Update bbox
                bbox = np.array(results['bbox']).flatten()
                bbox[:2] += np.array([x_pad[0], y_pad[0]])
                bbox[2:] += np.array([x_pad[0], y_pad[0]])
                results['bbox'] = bbox.reshape(np.array(results['bbox']).shape)

                # Update keypoints
                kpts = np.array(results['keypoints']).reshape(-1, 2)
                kpts[:, :2] += np.array([x_pad[0], y_pad[0]])
                results['keypoints'] = kpts.reshape(np.array(results['keypoints']).shape)

                # Update img_shape and ori_shape
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]

        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results
