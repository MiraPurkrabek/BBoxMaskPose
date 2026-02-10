# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.image import imflip
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine import is_list_of
from mmengine.dist import get_dist_info
from scipy.stats import truncnorm
from scipy.ndimage import distance_transform_edt

from mmpose.codecs import *  # noqa: F401, F403
from mmpose.registry import KEYPOINT_CODECS, TRANSFORMS
from mmpose.structures.bbox import bbox_xyxy2cs, flip_bbox, bbox_cs2xyxy
from mmpose.structures.keypoint import flip_keypoints
from mmpose.utils.typing import MultiConfig

from pycocotools import mask as Mask

try:
    import albumentations
except ImportError:
    albumentations = None

Number = Union[int, float]


@TRANSFORMS.register_module()
class GetBBoxCenterScale(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        
        # Save the original bbox wrt. input
        results['bbox_xyxy_wrt_input'] = results['bbox']
        
        if 'bbox_center' in results and 'bbox_scale' in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('Use the existing "bbox_center" and "bbox_scale"'
                              '. The padding will still be applied.')
            results['bbox_scale'] = results['bbox_scale'] * self.padding

        else:
            bbox = results['bbox']
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str


@TRANSFORMS.register_module()
class MaskBackground(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self,
        continue_on_failure: bool = True,
        prob: float = 1.0,
        alpha: float = 1.0,
        erode_prob: float = 0.0,
        erode_amount: float = 0.5,
        dilate_prob: float = 0.0,
        dilate_amount: float = 0.5,
        patches_computation_method: str = 'voronoi',
        context_size: float = 1.25,
    ) -> None:
        
        super().__init__()
        
        assert 0 <= alpha <= 1, 'alpha should be in [0, 1]'
        assert 0 <= prob <= 1, 'prob should be in [0, 1]'

        self.continue_on_failure = continue_on_failure
        self.alpha = alpha
        self.prob = prob
        
        assert 0 <= erode_prob <= 1, 'erode_prob should be in [0, 1]'
        assert 0 <= dilate_prob <= 1, 'dilate_prob should be in [0, 1]'
        assert 0 < erode_amount < 1, 'erode_amount should be in [0, 1]'
        assert 0 < dilate_amount < 1, 'dilate_amount should be in [0, 1]'
        assert erode_prob + dilate_prob <= 1, 'erode_prob + dilate_prob should be less than or equal to 1'
        self.noise_prob = erode_prob + dilate_prob
        if self.noise_prob > 0:
            self.erode_prob = erode_prob / (self.noise_prob)
            self.dilate_prob = dilate_prob / (self.noise_prob)
        else:
            self.erode_prob = 0
            self.dilate_prob = 0

        self.erode_amount = erode_amount
        self.dilate_amount = dilate_amount
        self.patches_computation_method = patches_computation_method
        self.context_size = context_size

    def _perturb_by_dilation(self, mask: np.ndarray) -> np.ndarray:
        """Perturb the mask to simulate real-world detector."""
        mask_shape = mask.shape

        mask_area = (mask>0).sum()

        # Close the mask to erase small holes
        k = max(mask_area // 1000, 5)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Dilate the mask to increase it a bit
        k = max(mask_area // 3000, 5)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
                    
        return mask.reshape(mask_shape)

    def _perturb_by_erosion(self, mask: np.ndarray) -> np.ndarray:
        """Perturb the mask to simulate real-world detector."""
        mask_shape = mask.shape

        mask_area = (mask>0).sum()

        # Close the mask to erase small holes
        k = max(mask_area // 1000, 5)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Erode the mask to decrease it a bit and cut-off limbs
        k = max(mask_area // 3000, 5)
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
                    
        return mask.reshape(mask_shape)

    @cache_randomness
    def _perturb_by_patches(self, mask: np.ndarray, amount: float, num_patches: int = 10) -> np.ndarray:
        mask_shape = mask.shape
        
        # Generate 10 random seeds uniformly distributed in the mask
        mask_idx = np.where(mask.flatten() > 0)[0]
        seeds = np.random.choice(mask_idx, num_patches, replace=False)
        sx, sy = np.unravel_index(seeds, mask.shape)

        # For each pixel, label it by it nearest seed
        labels = np.ones_like(mask)
        seed_labels = np.zeros_like(mask)
        seed_labels[sx, sy] = np.arange(num_patches) + 1

        _, indices = distance_transform_edt(seed_labels == 0, return_indices=True)
        labels = seed_labels[indices[0], indices[1]]
        labels = labels * mask

        # Select labels for removal
        random_remove_amount = np.random.uniform(0.0, amount)
        random_remove_ratio = int(num_patches * random_remove_amount)
        remove_labels = np.random.choice(np.unique(labels), random_remove_ratio, replace=False)
        binary_labels = np.isin(labels, remove_labels, invert=True)

        mask = (binary_labels > 0).astype(np.uint8) * mask

        return mask.reshape(mask_shape)

    @cache_randomness
    def _coin_flip(self) -> bool:
        return np.random.rand() < 0.5

    @cache_randomness
    def _perturb_mask(self, mask: np.ndarray) -> np.ndarray:
        """Perturb the mask to simulate real-world detector."""

        mask_shape = mask.shape

        if not np.random.rand() < self.noise_prob:
            return mask

        # Erode and dilate the mask to increase smoothness
        kernel = np.ones((5, 5), np.uint8)       
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        increase_mask = np.random.choice([False, True], p=[self.erode_prob, self.dilate_prob])
        
        if increase_mask:
            if self._coin_flip():
                try:
                    mask = self._perturb_by_patches(
                        mask=1-mask,
                        amount=self.dilate_amount,
                        num_patches=50,
                    )
                    mask = 1-mask
                except ValueError:
                    pass
            else:
                mask = self._perturb_by_dilation(mask)

        else:
            if self._coin_flip():
                try:
                    mask = self._perturb_by_patches(
                        mask=mask,
                        amount=self.erode_amount,
                        num_patches=10,
                    )
                except ValueError:
                    pass

            else:
                mask = self._perturb_by_erosion(mask)

        mask = (mask>0).astype(np.uint8)
        return mask.reshape(mask_shape)

    @cache_randomness
    def _do_masking(self):
        return np.random.rand() < self.prob

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        # Try to load the mask from the results        
        mask = results.get('segmentation', None)
        # print("\nMaskBackground: ", mask is not None)

        if mask is None and not self.continue_on_failure:
            raise ValueError('No mask found in the results and self.continue_on_failure is set to False.')

        if mask is not None and self._do_masking():
            # Convert mask from polygons to binary mask
            try:
                try:
                    mask_rle = Mask.frPyObjects(mask, results['img_shape'][0], results['img_shape'][1])
                    mask_rle = Mask.merge(mask_rle)
                except ValueError:
                    mask_rle = mask
            except IndexError:
                # breakpoint()
                # print("Mask shape:", mask.shape)
                # print("Mask max:", mask.max())
                # print("Mask min:", mask.min())
                # print("Image shape:", results['img_shape'])

                return results
            img = results['img'].copy()
            masked_image = results['img'].copy()
            mask = Mask.decode(mask_rle).reshape((img.shape[0], img.shape[1], 1))
            binary_mask = (mask > 0).astype(np.uint8)

            # Perturb the mask to simulate real-world detector
            # print("Here I would perturb the mask")
            old_mask = mask.copy()
            binary_mask = self._perturb_mask(binary_mask)

            masked_image = masked_image * binary_mask
            results['img'] = cv2.addWeighted(img, 1 - self.alpha, masked_image, self.alpha, 0)

            # hash_id = abs(hash(555))
            # cv2.imwrite("tmp_visualization/_perturbed_mask_{:d}.jpg".format(hash_id), mask * 255)
            # cv2.imwrite("tmp_visualization/_old_mask_{:d}.jpg".format(hash_id), old_mask * 255)
            # cv2.imwrite("tmp_visualization/_weighted_masked_image_{:d}.jpg".format(hash_id), results['img'])
            # breakpoint()
            # Save the mask as a binary mask

        # Save the image
        img = results['img']
        
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("tmp_visualization/masked_image_{:d}.jpg".format(abs(hash(555))), img)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(continue_on_failure={self.continue_on_failure})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
        - flip_indices
        - input_size (optional)
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - keypoints_visibility (optional)
        - img_mask (optional)

    Modified Keys:

        - img
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - keypoints_visibility (optional)
        - img_mask (optional)

    Added Keys:

        - flip
        - flip_direction

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
        direction (str | list[str]): The flipping direction. Options are
            ``'horizontal'``, ``'vertical'`` and ``'diagonal'``. If a list is
            is given, each data sample's flipping direction will be sampled
            from a distribution determined by the argument ``prob``. Defaults
            to ``'horizontal'``.
    """

    def __init__(self,
                 prob: Union[float, List[float]] = 0.5,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      List) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`RandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        flip_dir = self._choose_direction()

        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results.get('input_size', results['img_shape'])
            # flip image and mask
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            if 'img_mask' in results:
                results['img_mask'] = imflip(
                    results['img_mask'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)
            
            # flip bboxes
            if results.get('bbox_xyxy_wrt_input', None) is not None:
                results['bbox_xyxy_wrt_input'] = flip_bbox(
                    results['bbox_xyxy_wrt_input'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)

            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visible', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)
                _, keypoints_visibility = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visibility', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)

                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible
                results['keypoints_visibility'] = keypoints_visibility

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'
        return repr_str


@TRANSFORMS.register_module()
class RandomHalfBody(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 9,
                 min_upper_keypoints: int = 2,
                 min_lower_keypoints: int = 3,
                 padding: float = 1.5,
                 prob: float = 0.3,
                 upper_prioritized_prob: float = 0.7) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        return center, scale
    
    def _get_half_body_exact_bbox(self, keypoints: np.ndarray,
                                half_body_ids: List[int],
                                bbox: np.ndarray,
                                ) -> np.ndarray:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        x1, y1 = center - scale / 2
        x2, y2 = center + scale / 2

        # Do not exceed the original bbox
        x1 = np.maximum(x1, bbox[0])
        y1 = np.maximum(y1, bbox[1])
        x2 = np.minimum(x2, bbox[2])
        y2 = np.minimum(y2, bbox[3])

        return np.array([x1, y1, x2, y2])

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (num_upper < self.min_upper_keypoints
                        and num_lower < self.min_lower_keypoints):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_valid_ids if prefer_upper else lower_valid_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'],
            upper_body_ids=results['upper_body_ids'],
            lower_body_ids=results['lower_body_ids'])

        bbox_center = []
        bbox_scale = []

        bbox_xyxy_wrt_input = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
                bbox_xyxy_wrt_input.append(results['bbox_xyxy_wrt_input'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)
                exact_bbox = self._get_half_body_exact_bbox(
                    results['keypoints'][i], indices, results['bbox_xyxy_wrt_input'][i])
                bbox_xyxy_wrt_input.append(exact_bbox)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
        results['bbox_xyxy_wrt_input'] = np.stack(bbox_xyxy_wrt_input)
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(min_total_keypoints={self.min_total_keypoints}, '
        repr_str += f'min_upper_keypoints={self.min_upper_keypoints}, '
        repr_str += f'min_lower_keypoints={self.min_lower_keypoints}, '
        repr_str += f'padding={self.padding}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'upper_prioritized_prob={self.upper_prioritized_prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomPatchesBlackout(BaseTransform):
    """Data augmentation that divide image into patches and set color of random
        pathes to black. In AID paper marked as 'hide and seek'.

    Required Keys:

        - keypoints
        - keypoints_visible
        - keypoint_visibility

    Modified Keys:

        - img
        - keypoint_visibility

    Args:
        grid_size (tuple(int, int)): Grid size of the patches. Defaults to
            (8, 6)
        mask_ratio (float): Ratio of patches to blackout. Defaults to 0.3
        prob (float): The probability to apply black patches. Defaults to 0.8
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (8, 6),
                 mask_ratio: float = 0.3,
                 prob: float = 0.8) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.mask_ratio = mask_ratio
        self.prob = prob

    @cache_randomness
    def _get_random_patches(self, grid_h, grid_w) -> np.ndarray:
        black_patches = np.zeros((grid_h, grid_w), dtype=bool)

        if np.random.rand() < self.prob:
        
            # Split image into grid
            num_patches = int(self.grid_size[0] * self.grid_size[1])

            # Randomly choose patches to blackout
            black_patches = np.random.choice(
                [0, 1],
                num_patches,
                p=[1 - self.mask_ratio, self.mask_ratio]
            )
            black_patches = black_patches.reshape(grid_h, grid_w).astype(bool)

        return black_patches


    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        img = results['img']
        
        if "transformed_keypoints" in results:
            kpts = results['transformed_keypoints'].squeeze()
        else:
            kpts = results['keypoints'].squeeze()
        
        h, w = img.shape[:2]
        grid_h, grid_w = self.grid_size
        dh = np.ceil(h / grid_h).astype(int)
        dw = np.ceil(w / grid_w).astype(int)

        black_patches = self._get_random_patches(grid_h, grid_w)

        for i in range(grid_h):
            for j in range(grid_w):
                if black_patches[i, j]:
                    # Set all pixel in the patch to black
                    img[i*dh : (i+1)*dh, j*dw : (j+1)*dw, :] = 0


                    # Set keypoints in the patch to invisible
                    in_black = (
                        (kpts[:, 0] >= j*dw) &
                        (kpts[:, 0] < (j+1)*dw) &
                        (kpts[:, 1] >= i*dh) &
                        (kpts[:, 1] < (i+1)*dh)
                    )
                    results['keypoints_visibility'][:, in_black] = 0
                        
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(grid_size={self.grid_size}, '
        repr_str += f'mask_ratio={self.mask_ratio}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomEdgesBlackout(BaseTransform):
    """Data augmentation that masks edged of the image with black color
        simulating image edge or random texture.

    Required Keys:

        - keypoints
        - keypoints_visible
        - keypoint_visibility

    Modified Keys:

        - img
        - keypoint_visibility

    Args:
        mask_ratio_range (tuple[float, float]): Range or mask-to-image ratio. Defaults to
            (0.1, 0.3)
        prob (float): The probability to apply black patches. Defaults to 0.8
        texture_prob (float): The probability to apply texture to the blackout area. Defaults to 0.0
    """

    def __init__(self,
                 mask_ratio_range: tuple[float, float] = (0.1, 0.3),
                 prob: float = 0.8,
                 texture_prob: float = 0.0,
                 context_size:float = 1.25) -> None:
        super().__init__()
        self.mask_ratio_range = mask_ratio_range
        self.prob = prob
        self.texture_prob = texture_prob
        self.context_size = context_size

    @cache_randomness
    def _get_random_mask(self, w, h, bbox_xyxy) -> float:
        """Get random mask ratio.

        Args:
            w (int): Width of the image
            h (int): Height of the image
            bbox_xyxy (tuple): Bounding box (x1, y1, x2, y2)

        Returns:
            np.array: mask (1 for blackout, 0 for keep)
            tuple: bounds of the blackout area (x1, y1, x2, y2)
        """
        mask = np.zeros((h, w), dtype=bool)
        bbox_c, bbox_s = bbox_xyxy2cs(bbox_xyxy, padding=self.context_size)
        x0, y0, x1, y1 = bbox_cs2xyxy(bbox_c, bbox_s)

        # Clip the bounding box to the image
        x0 = np.maximum(x0, 0).astype(int)
        y0 = np.maximum(y0, 0).astype(int)
        x1 = np.minimum(x1, w).astype(int)
        y1 = np.minimum(y1, h).astype(int)
        
        # Set default values
        x = 0
        y = 0
        dw = w
        dh = h
        is_textured = False

        if np.random.rand() < self.prob:
            
            # Generate random rectangle to keep
            rh, rw = np.random.uniform(
                1-self.mask_ratio_range[1],
                1-self.mask_ratio_range[0],
                2
            )
            dh = int((y1-y0) * rh)
            dw = int((x1-x0) * rw)
            x_end = x1-dw if x1-dw > x0 else x0+1
            y_end = y1-dh if y1-dh > y0 else y0+1
            try:
                x = np.random.randint(x0, x_end)
                y = np.random.randint(y0, y_end)
            except ValueError:
                print(x, x0, dw, x1, x1-dw, x_end)
                print(y, y0, dh, y1, y1-dh, y_end)
                raise ValueError

            # Set all pixel outside of the rectangle to black
            mask[y:y+dh, x:x+dw] = True
            
            # Invert the mask. True means blackout
            mask = ~mask

            # Add texture
            is_textured = np.random.rand() < self.texture_prob

        return mask, (x, y, dw+x, dh+y), is_textured

    def _get_random_color(self) -> np.ndarray:
        """Get random color.

        Returns:
            np.array: color
        """
        h = np.random.randint(0, 360)
        s = np.random.uniform(0.75, 1)
        l = np.random.uniform(0.3, 0.7)
        hls_color = np.array([h, l, s])
        rgb_color = cv2.cvtColor(
            np.array([[hls_color]], dtype=np.float32),
            cv2.COLOR_HLS2RGB
        ).squeeze() * 255
        color = rgb_color.astype(np.uint8)
        return color.tolist()

    def _get_random_texture(self, w, h) -> np.ndarray:
        """Get random texture.

        Args:
            w (int): Width of the image
            h (int): Height of the image

        Returns:
            np.array: texture
        """
        mode = np.random.choice([
            'lines',
            'squares',
            'circles',
            # 'noise',
            # 'uniform',
        ])

        if mode == 'lines':
            texture = np.zeros((h, w, 3), dtype=np.uint8)
            texture[:, :, :] = self._get_random_color()
            num_lines = np.random.randint(1, 20)
            for _ in range(num_lines):
                x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
                x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
                line_width = np.random.randint(1, 10)
                color = self._get_random_color()
                cv2.line(texture, (x1, y1), (x2, y2), color, line_width)
        elif mode == 'squares':
            texture = np.zeros((h, w, 3), dtype=np.uint8)
            texture[:, :, :] = self._get_random_color()
            num_squares = np.random.randint(1, 20)
            for _ in range(num_squares):
                x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
                x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
                color = self._get_random_color()
                cv2.rectangle(texture, (x1, y1), (x2, y2), color, -1)
        elif mode == 'circles':
            texture = np.zeros((h, w, 3), dtype=np.uint8)
            texture[:, :, :] = self._get_random_color()
            num_circles = np.random.randint(1, 20)
            for _ in range(num_circles):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                r = np.random.randint(1, min(w, h) // 2)
                color = self._get_random_color()
                cv2.circle(texture, (x, y), r, color, -1)
        elif mode == 'noise':
            texture = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        elif mode == 'uniform':
            texture = np.zeros((h, w, 3), dtype=np.uint8)
            texture[:, :, :] = self._get_random_color()

        return texture

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        img = results['img']
        
        if "transformed_keypoints" in results:
            kpts = results['transformed_keypoints'].squeeze()
        else:
            kpts = results['keypoints'].squeeze()

        # Generate random mask
        mask, (x1, y1, x2, y2), is_textured = self._get_random_mask(img.shape[1], img.shape[0], results['bbox_xyxy_wrt_input'].flatten())
        # breakpoint()
        # print("img shape", img.shape)
        # print("results", results.keys())

        # Apply the mask
        if is_textured:
            textured_img = self._get_random_texture(img.shape[1], img.shape[0])
            textured_img[~mask, :] = img[~mask, :]
            img = textured_img
        else:
            # Set all pixel outside of the rectangle to black
            img[mask, :] = 0
        results['img'] = img

        # Set keypoints outside of the rectangle to invisible
        in_rect = (
            (kpts[:, 0] >= x1) &
            (kpts[:, 0] < x2) &
            (kpts[:, 1] >= y1) &
            (kpts[:, 1] < y2)
        )
        results['keypoints_visibility'][:, ~in_rect] = 0

        # Create new entry describing keypoints in the 'cropped' area
        results['keypoints_in_image'] = in_rect.squeeze().astype(int)

        # Crop the bbox_xyxy_wrt_input according to the blackout area
        if 'bbox_xyxy_wrt_input' in results:
            bbox_xyxy = results['bbox_xyxy_wrt_input'].flatten()
            bbox_xyxy[0] = np.maximum(bbox_xyxy[0], x1)
            bbox_xyxy[1] = np.maximum(bbox_xyxy[1], y1)
            bbox_xyxy[2] = np.minimum(bbox_xyxy[2], x2)
            bbox_xyxy[3] = np.minimum(bbox_xyxy[3], y2)
            results['bbox_xyxy_wrt_input'] = bbox_xyxy.reshape(-1, 4)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(mask_ratio_range={self.mask_ratio_range}, '
        repr_str += f'prob={self.prob}), '
        repr_str += f'texture_prob={self.texture_prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomBBoxTransform(BaseTransform):
    r"""Rnadomly shift, resize and rotate the bounding boxes.

    Required Keys:

        - bbox_center
        - bbox_scale

    Modified Keys:

        - bbox_center
        - bbox_scale

    Added Keys:
        - bbox_rotation

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 80.0
        rotate_prob (float): Probability of applying random rotation. Defaults
            to 0.6
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: Tuple[float, float] = (0.5, 1.5),
                 scale_prob: float = 1.0,
                 rotate_factor: float = 80.0,
                 rotate_prob: float = 0.6) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    @cache_randomness
    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        random_v = self._truncnorm(size=(num_bboxes, 4))
        offset_v = random_v[:, :2]
        scale_v = random_v[:, 2:3]
        rotate_v = random_v[:, 3]

        # Get shift parameters
        offset = offset_v * self.shift_factor
        offset = np.where(
            np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = scale_v * sigma + mu
        scale = np.where(
            np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        rotate = rotate_v * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`RandomBboxTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        bbox_scale = results['bbox_scale']
        num_bboxes = bbox_scale.shape[0]

        offset, scale, rotate = self._get_transform_params(num_bboxes)

        results['bbox_center'] = results['bbox_center'] + offset * bbox_scale
        results['bbox_scale'] = results['bbox_scale'] * scale
        results['bbox_rotation'] = rotate

        bbox_xyxy_wrt_input = results.get('bbox_xyxy_wrt_input', None)
        if bbox_xyxy_wrt_input is not None:
            _c, _s = bbox_xyxy2cs(bbox_xyxy_wrt_input, padding=1.0)
            _c = _c + offset * _s
            _s = _s * scale
            results['bbox_xyxy_wrt_input'] = bbox_cs2xyxy(_c, _s).flatten()

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(shift_prob={self.shift_prob}, '
        repr_str += f'shift_factor={self.shift_factor}, '
        repr_str += f'scale_prob={self.scale_prob}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'rotate_prob={self.rotate_prob}, '
        repr_str += f'rotate_factor={self.rotate_factor})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class Albumentation(BaseTransform):
    """Albumentation augmentation (pixel-level transforms only).

    Adds custom pixel-level transformations from Albumentations library.
    Please visit `https://albumentations.ai/docs/`
    to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        transforms (List[dict]): A list of Albumentation transforms.
            An example of ``transforms`` is as followed:
            .. code-block:: python

                [
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                ]
        keymap (dict | None): key mapping from ``input key`` to
            ``albumentation-style key``.
            Defaults to None, which will use {'img': 'image'}.
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            albumentations.BasicTransform: The constructed transform object
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            rank, _ = get_dist_info()
            if rank == 0 and not hasattr(
                    albumentations.augmentations.transforms, obj_type):
                warnings.warn(
                    f'{obj_type} is not pixel-level transformations. '
                    'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`Albumentation` to apply
        albumentations transforms.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): Result dict from the data pipeline.

        Return:
            dict: updated result dict.
        """
        # map result dict to albumentations format
        results_albu = {}
        for k, v in self.keymap_to_albu.items():
            assert k in results, \
                f'The `{k}` is required to perform albumentations transforms'
            results_albu[v] = results[k]

        # Apply albumentations transforms
        results_albu = self.aug(**results_albu)

        # map the albu results back to the original format
        for k, v in self.keymap_to_albu.items():
            results[k] = results_albu[v]

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@TRANSFORMS.register_module()
class PhotometricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        """Generate the random flags for subsequent transforms.

        Returns:
            Sequence[Number]: a sequence of numbers that indicate whether to
                do the corresponding transforms.
        """
        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        contrast_mode = np.random.randint(2)
        # whether to apply brightness distortion
        brightness_flag = np.random.randint(2)
        # whether to apply contrast distortion
        contrast_flag = np.random.randint(2)
        # the mode to convert color from BGR to HSV
        hsv_mode = np.random.randint(4)
        # whether to apply channel swap
        swap_flag = np.random.randint(2)

        # the beta in `self._convert` to be added to image array
        # in brightness distortion
        brightness_beta = np.random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
        # the alpha in `self._convert` to be multiplied to image array
        # in contrast distortion
        contrast_alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        # the alpha in `self._convert` to be multiplied to image array
        # in saturation distortion to hsv-formatted img
        saturation_alpha = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        # delta of hue to add to image array in hue distortion
        hue_delta = np.random.randint(-self.hue_delta, self.hue_delta)
        # the random permutation of channel order
        swap_channel_order = np.random.permutation(3)

        return (contrast_mode, brightness_flag, contrast_flag, hsv_mode,
                swap_flag, brightness_beta, contrast_alpha, saturation_alpha,
                hue_delta, swap_channel_order)

    def _convert(self,
                 img: np.ndarray,
                 alpha: float = 1,
                 beta: float = 0) -> np.ndarray:
        """Multiple with alpha and add beta with clip.

        Args:
            img (np.ndarray): The image array.
            alpha (float): The random multiplier.
            beta (float): The random offset.

        Returns:
            np.ndarray: The updated image array.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PhotometricDistortion` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        assert 'img' in results, '`img` is not found in results'
        img = results['img']

        (contrast_mode, brightness_flag, contrast_flag, hsv_mode, swap_flag,
         brightness_beta, contrast_alpha, saturation_alpha, hue_delta,
         swap_channel_order) = self._random_flags()

        # random brightness distortion
        if brightness_flag:
            img = self._convert(img, beta=brightness_beta)

        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        if hsv_mode:
            # random saturation/hue distortion
            img = mmcv.bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                # apply saturation distortion to hsv-formatted img
                img[:, :, 1] = self._convert(
                    img[:, :, 1], alpha=saturation_alpha)
            if hsv_mode == 2 or hsv_mode == 3:
                # apply hue distortion to hsv-formatted img
                img[:, :, 0] = img[:, :, 0].astype(int) + hue_delta
            img = mmcv.hsv2bgr(img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_channel_order]

        results['img'] = img
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
        target_type (str, deprecated): This argument is deprecated and has no
            effect. Defaults to ``None``
    """

    def __init__(self,
                 encoder: MultiConfig,
                 target_type: Optional[str] = None,
                 multilevel: bool = False,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()

        if target_type is not None:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn(
                    'The argument `target_type` is deprecated in'
                    ' GenerateTarget. The target type and encoded '
                    'keys will be determined by encoder(s).',
                    DeprecationWarning)

        self.encoder_cfg = deepcopy(encoder)
        self.multilevel = multilevel
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        if isinstance(self.encoder_cfg, list):
            self.encoder = [
                KEYPOINT_CODECS.build(cfg) for cfg in self.encoder_cfg
            ]
        else:
            assert not self.multilevel, (
                'Need multiple encoder configs if ``multilevel==True``')
            self.encoder = KEYPOINT_CODECS.build(self.encoder_cfg)

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """

        if results.get('transformed_keypoints', None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results['transformed_keypoints']
        elif results.get('keypoints', None) is not None:
            # use original keypoints
            keypoints = results['keypoints']
        else:
            raise ValueError(
                'GenerateTarget requires \'transformed_keypoints\' or'
                ' \'keypoints\' in the results.')

        keypoints_visible = results['keypoints_visible']
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = \
                keypoints_visible[..., 0], keypoints_visible[..., 1]
            results['keypoints_visible'] = keypoints_visible
            results['keypoints_visible_weights'] = keypoints_visible_weights
        
        id_similarity = results.get('id_similarity', np.array([0]))
        keypoints_visibility = results.get("keypoints_visibility", None)    
    
        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        if not isinstance(self.encoder, list):
            # For single encoding, the encoded items will be directly added
            # into results.
            auxiliary_encode_kwargs = {
                key: results[key]
                for key in self.encoder.auxiliary_encode_keys
            }
            encoded = self.encoder.encode(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                keypoints_visibility=keypoints_visibility,
                id_similarity=id_similarity,
                **auxiliary_encode_kwargs)

            if self.encoder.field_mapping_table:
                encoded[
                    'field_mapping_table'] = self.encoder.field_mapping_table
            if self.encoder.instance_mapping_table:
                encoded['instance_mapping_table'] = \
                    self.encoder.instance_mapping_table
            if self.encoder.label_mapping_table:
                encoded[
                    'label_mapping_table'] = self.encoder.label_mapping_table

        else:
            encoded_list = []
            _field_mapping_table = dict()
            _instance_mapping_table = dict()
            _label_mapping_table = dict()
            for _encoder in self.encoder:
                auxiliary_encode_kwargs = {
                    key: results[key]
                    for key in _encoder.auxiliary_encode_keys
                }
                encoded_list.append(
                    _encoder.encode(
                        keypoints=keypoints,
                        keypoints_visible=keypoints_visible,
                        keypoints_visibility=keypoints_visibility,
                        id_similarity=id_similarity,
                        **auxiliary_encode_kwargs))

                _field_mapping_table.update(_encoder.field_mapping_table)
                _instance_mapping_table.update(_encoder.instance_mapping_table)
                _label_mapping_table.update(_encoder.label_mapping_table)

            if self.multilevel:
                # For multilevel encoding, the encoded items from each encoder
                # should have the same keys.

                keys = encoded_list[0].keys()
                if not all(_encoded.keys() == keys
                           for _encoded in encoded_list):
                    raise ValueError(
                        'Encoded items from all encoders must have the same '
                        'keys if ``multilevel==True``.')

                encoded = {
                    k: [_encoded[k] for _encoded in encoded_list]
                    for k in keys
                }

            else:
                # For combined encoding, the encoded items from different
                # encoders should have no overlapping items, except for
                # `keypoint_weights`. If multiple `keypoint_weights` are given,
                # they will be multiplied as the final `keypoint_weights`.

                encoded = dict()
                keypoint_weights = []

                for _encoded in encoded_list:
                    for key, value in _encoded.items():
                        if key == 'keypoint_weights':
                            keypoint_weights.append(value)
                        elif key not in encoded:
                            encoded[key] = value
                        else:
                            raise ValueError(
                                f'Overlapping item "{key}" from multiple '
                                'encoders, which is not supported when '
                                '``multilevel==False``')

                if keypoint_weights:
                    encoded['keypoint_weights'] = keypoint_weights

            if _field_mapping_table:
                encoded['field_mapping_table'] = _field_mapping_table
            if _instance_mapping_table:
                encoded['instance_mapping_table'] = _instance_mapping_table
            if _label_mapping_table:
                encoded['label_mapping_table'] = _label_mapping_table

        if self.use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
            if isinstance(encoded['keypoint_weights'], list):
                for w in encoded['keypoint_weights']:
                    w = w * results['dataset_keypoint_weights']
            else:
                encoded['keypoint_weights'] = encoded[
                    'keypoint_weights'] * results['dataset_keypoint_weights']

        results.update(encoded)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str


@TRANSFORMS.register_module()
class YOLOXHSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def transform(self, results: dict) -> dict:
        img = results['img']
        hsv_gains = self._get_hsv_gains()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@TRANSFORMS.register_module()
class FilterAnnotations(BaseTransform):
    """Eliminate undesirable annotations based on specific conditions.

    This class is designed to sift through annotations by examining multiple
    factors such as the size of the bounding box, the visibility of keypoints,
    and the overall area. Users can fine-tune the criteria to filter out
    instances that have excessively small bounding boxes, insufficient area,
    or an inadequate number of visible keypoints.

    Required Keys:

    - bbox (np.ndarray) (optional)
    - area (np.int64) (optional)
    - keypoints_visible (np.ndarray) (optional)

    Modified Keys:

    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground
            truth boxes. Default: (1., 1.)
        min_gt_area (int): Minimum foreground area of instances.
            Default: 1
        min_kpt_vis (int): Minimum number of visible keypoints. Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: False
        by_area (bool): Filter instances with area less than min_gt_area
            threshold. Default: False
        by_kpt (bool): Filter instances with keypoints_visible not meeting the
            min_kpt_vis threshold. Default: True
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_area: int = 1,
                 min_kpt_vis: int = 1,
                 by_box: bool = False,
                 by_area: bool = False,
                 by_kpt: bool = True,
                 keep_empty: bool = True) -> None:

        assert by_box or by_kpt or by_area
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_area = min_gt_area
        self.min_kpt_vis = min_kpt_vis
        self.by_box = by_box
        self.by_area = by_area
        self.by_kpt = by_kpt
        self.keep_empty = keep_empty

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'keypoints' in results
        kpts = results['keypoints']
        if kpts.shape[0] == 0:
            return results

        tests = []
        if self.by_box and 'bbox' in results:
            bbox = results['bbox']
            tests.append(
                ((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) &
                 (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
        if self.by_area and 'area' in results:
            area = results['area']
            tests.append(area >= self.min_gt_area)
        if self.by_kpt:
            kpts_vis = results['keypoints_visible']
            if kpts_vis.ndim == 3:
                kpts_vis = kpts_vis[..., 0]
            tests.append(kpts_vis.sum(axis=1) >= self.min_kpt_vis)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = ('bbox', 'bbox_score', 'category_id', 'keypoints',
                'keypoints_visible', 'area')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'min_gt_bbox_wh={self.min_gt_bbox_wh}, '
                f'min_gt_area={self.min_gt_area}, '
                f'min_kpt_vis={self.min_kpt_vis}, '
                f'by_box={self.by_box}, '
                f'by_area={self.by_area}, '
                f'by_kpt={self.by_kpt}, '
                f'keep_empty={self.keep_empty})')


def compute_paddings(bbox, bbox_s, kpts):
    """Compute the padding of the bbox to fit the keypoints."""
    bbox = np.array(bbox).flatten()
    bbox_s = np.array(bbox_s).flatten()
    if kpts.size % 2 == 0:
        kpts = kpts.reshape(-1, 2)
    else:
        kpts = kpts.reshape(-1, 3)
    
    x0, y0, x1, y1 = bbox
    x_bbox_distances = np.max(np.stack([
        np.clip(x0 - kpts[:, 0], a_min=0, a_max=None),
        np.clip(kpts[:, 0] - x1, a_min=0, a_max=None),
    ]), axis=0)
    y_bbox_distances = np.max(np.stack([
        np.clip(y0 - kpts[:, 1], a_min=0, a_max=None),
        np.clip(kpts[:, 1] - y1, a_min=0, a_max=None),
    ]), axis=0)

    padding_x = 2 * x_bbox_distances / bbox_s[0]
    padding_y = 2 * y_bbox_distances / bbox_s[1]
    padding = 1 + np.maximum(padding_x, padding_y)

    padding = np.maximum(x_bbox_distances, y_bbox_distances)

    return padding.flatten()
