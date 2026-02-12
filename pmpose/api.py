"""
Public API for PMPose (MaskPose) wrapper.

This module provides a stable, user-friendly interface for pose estimation
using the MaskPose model. It handles model initialization, inference,
and visualization while preparing for future PMPose model integration.

Note: Current implementation uses MaskPose and returns dummy presence/visibility
values to maintain API compatibility with future PMPose model.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from .mm_utils import run_MMPose as _internal_run_mmpose
from .posevis_lite import pose_visualization
from mmengine.structures import InstanceData

BMP_ROOT = Path(__file__).parent.parent

# Pretrained model URLs
PRETRAINED_URLS = {
    # MaskPose
    "MaskPose-s": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose/MaskPose-s-1.1.0.pth",
    "MaskPose-b": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose/MaskPose-b-1.1.0.pth",
    "MaskPose-l": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose/MaskPose-l-1.1.0.pth",
    "MaskPose-h": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/MaskPose/MaskPose-h-1.1.0.pth",
    # MaskPose whole body
    "MaskPose-s-wb": "TBD",
    "MaskPose-b-wb": "TBD",
    "MaskPose-l-wb": "TBD",
    "MaskPose-h-wb": "TBD",
    # PMPose
    "PMPose-s": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/PMPose/PMPose-s-1.0.0.pth",
    "PMPose-b": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/PMPose/PMPose-b-1.0.0.pth",
    "PMPose-l": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/PMPose/PMPose-l-1.0.0.pth",
    "PMPose-h": "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/PMPose/PMPose-h-1.0.0.pth",
}

# Default config paths (relative to package root)
DEFAULT_CONFIGS = {
    # MaskPose
    "MaskPose-s": "mmpose/configs/MaskPose/MaskPose-s-1.1.0.py",
    "MaskPose-b": "mmpose/configs/MaskPose/MaskPose-b-1.1.0.py",
    "MaskPose-l": "mmpose/configs/MaskPose/MaskPose-l-1.1.0.py",
    "MaskPose-h": "mmpose/configs/MaskPose/MaskPose-h-1.1.0.py",
    # MaskPose-wb (whole body)
    "MaskPose-s-wb": "mmpose/configs/MaskPose/MaskPose-s-wb-1.1.0.py",
    "MaskPose-b-wb": "mmpose/configs/MaskPose/MaskPose-b-wb-1.1.0.py",
    "MaskPose-l-wb": "mmpose/configs/MaskPose/MaskPose-l-wb-1.1.0.py",
    "MaskPose-h-wb": "mmpose/configs/MaskPose/MaskPose-h-wb-1.1.0.py",
    # PMPose
    "PMPose-s": "mmpose/configs/ProbMaskPose/PMPose-s-1.0.0.py",
    "PMPose-b": "mmpose/configs/ProbMaskPose/PMPose-b-1.0.0.py",
    "PMPose-l": "mmpose/configs/ProbMaskPose/PMPose-l-1.0.0.py",
    "PMPose-h": "mmpose/configs/ProbMaskPose/PMPose-h-1.0.0.py",
}


class PMPose:
    """
    Public wrapper API for PMPose (currently MaskPose) pose estimation.
    
    This class provides a torch.hub-like interface for pose estimation,
    handling model initialization, inference, and visualization.
    
    Example:
        >>> pose_model = PMPose(device="cuda")
        >>> keypoints, presence, visibility, heatmaps = pose_model.predict(
        ...     image="path/to/image.jpg",
        ...     bboxes=[[100, 100, 300, 400]],
        ...     return_probmaps=False
        ... )
    """
    
    def __init__(
        self,
        device: str = "cuda",
        variant: str = "PMPose-b",
        from_pretrained: bool = False,
        pretrained_id: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize PMPose model.
        
        Args:
            device (str): Device for inference ('cuda' or 'cpu'). Defaults to 'cuda'.
            variant (str): Model variant to use. Defaults to 'PMPose-b'.
            from_pretrained (bool): Whether to load pretrained weights. Defaults to False.
            pretrained_id (str, optional): ID for pretrained model from PRETRAINED_URLS.
            config_path (str, optional): Path to custom config file. Overrides variant.
        """
        self.device = device
        self.variant = variant
        self._model = None
        
        # Determine config path
        if config_path is not None:
            self.config_path = config_path
        elif variant in DEFAULT_CONFIGS:
            self.config_path = os.path.join(BMP_ROOT, DEFAULT_CONFIGS[variant])
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(DEFAULT_CONFIGS.keys())}"
            )
        
        # Determine checkpoint path
        checkpoint_path = None
        if from_pretrained:
            if pretrained_id is None:
                pretrained_id = variant  # Use variant as default pretrained_id
            if pretrained_id not in PRETRAINED_URLS:
                raise ValueError(
                    f"Unknown pretrained_id '{pretrained_id}'. "
                    f"Available: {list(PRETRAINED_URLS.keys())}"
                )
            checkpoint_path = PRETRAINED_URLS[pretrained_id]
        
        # Initialize model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Load the pose estimation model."""
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        self._model = init_pose_estimator(
            self.config_path,
            checkpoint_path,
            device=self.device,
            cfg_options=cfg_options,
        )
    
    def to(self, device: str):
        """
        Move model to specified device.
        
        Args:
            device (str): Target device ('cuda' or 'cpu').
        
        Returns:
            PMPose: Self for chaining.
        """
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self
    
    def device(self, device_str: str):
        """Alias for to() method for compatibility."""
        return self.to(device_str)
    
    def load_from_file(self, path: str) -> None:
        """
        Load model weights from a local file.
        
        Args:
            path (str): Path to checkpoint file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        # Reload model with new checkpoint
        self._load_model(checkpoint_path=path)
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        bboxes: Union[List, np.ndarray],
        masks: Optional[Union[List, np.ndarray]] = None,
        return_probmaps: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run pose estimation on image given bounding boxes.
        
        Args:
            image: Image path (str) or BGR numpy array.
            bboxes: List/array of N bounding boxes in [x1, y1, x2, y2] format.
            masks: Optional instance masks. Can be:
                - List of (H, W) boolean/0-1 numpy arrays
                - numpy array of shape (N, H, W)
                - None (no masks)
            return_probmaps: If True, return heatmaps. Defaults to False.
        
        Returns:
            Tuple containing:
                - keypoints: (N, K, 3) array with [x, y, score] in image coordinates
                - presence: (N, K) array with presence probabilities (dummy for MaskPose)
                - visibility: (N, K) array with visibility flags (dummy for MaskPose)
                - heatmaps: (N, K, H, W) if return_probmaps=True, else None
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        else:
            img = image
        
        # Convert bboxes to numpy array
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes, dtype=np.float32)
        
        # Handle empty bboxes
        if len(bboxes) == 0:
            num_kpts = 17  # COCO format
            return (
                np.zeros((0, num_kpts, 3)),
                np.zeros((0, num_kpts)),
                np.zeros((0, num_kpts)),
                None,
            )
        
        # Prepare masks
        if masks is not None:
            if isinstance(masks, list):
                masks = np.array(masks)
            # Ensure masks have correct shape (N, H, W)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
        
        # Update model config to return heatmaps if requested
        if return_probmaps:
            self._model.cfg.model.test_cfg.output_heatmaps = True
        
        # Run inference using mmpose API
        pose_results = inference_topdown(
            self._model,
            img,
            bboxes=bboxes,
            masks=masks,
            bbox_format='xyxy'
        )
        
        # Reset heatmap output setting
        if return_probmaps:
            self._model.cfg.model.test_cfg.output_heatmaps = False
        
        # Extract results
        N = len(pose_results)
        keypoints_list = []
        scores_list = []
        heatmaps_list = [] if return_probmaps else None
        
        K = pose_results[0].pred_instances.keypoints.shape[-2] if N > 0 else 17  # COCO keypoints

        for result in pose_results:
            pred_instances = result.pred_instances
            kpts = pred_instances.keypoints.reshape(K, 2)  # (K, 2)
            kpt_scores = pred_instances.keypoint_scores.reshape(K, 1)  # (K, 1)
            
            # Combine into (K, 3) format
            kpts_with_scores = np.concatenate(
                [kpts, kpt_scores], axis=1
            )
            keypoints_list.append(kpts_with_scores)
            scores_list.append(kpt_scores)
            
            if return_probmaps and hasattr(pred_instances, 'heatmaps'):
                heatmaps_list.append(pred_instances.heatmaps)
        
        # Stack results
        keypoints = np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0, K, 3))
        keypoint_scores = np.stack(scores_list, axis=0) if scores_list else np.zeros((0, K, 1))
        
        # Create dummy presence and visibility for MaskPose compatibility
        # In future PMPose, these will be real model outputs
        presence = np.copy(keypoint_scores)  # Use scores as presence probability
        visibility = (keypoint_scores > 0.3).astype(np.float32)  # Binary visibility
        
        # Process heatmaps if requested
        heatmaps = None
        if return_probmaps and heatmaps_list:
            heatmaps = np.stack(heatmaps_list, axis=0)
        
        return keypoints, presence, visibility, heatmaps
    
    def get_features(
        self,
        image: Union[str, np.ndarray],
        bboxes: Union[List, np.ndarray],
        masks: Optional[Union[List, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Extract backbone features for given bounding boxes.
        
        Args:
            image: Image path (str) or BGR numpy array.
            bboxes: List/array of N bounding boxes in [x1, y1, x2, y2] format.
            masks: Optional instance masks (same format as predict).
        
        Returns:
            np.ndarray: Backbone features of shape (N, C, H, W).
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        else:
            img = image
        
        # Convert bboxes to numpy array
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes, dtype=np.float32)
        
        if len(bboxes) == 0:
            # Return empty features
            return np.zeros((0, 768, 16, 16))  # Typical ViT-B features
        
        # Prepare masks
        if masks is not None:
            if isinstance(masks, list):
                masks = np.array(masks)
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
        
        # Run inference and extract features
        # This requires accessing model internals
        pose_results = inference_topdown(
            self._model,
            img,
            bboxes=bboxes,
            masks=masks,
            bbox_format='xyxy'
        )
        
        # Extract features from results if available
        # Note: This is a simplified implementation
        # Real feature extraction would require model hooks
        features_list = []
        for result in pose_results:
            if hasattr(result, 'features'):
                features_list.append(result.features)
        
        if features_list:
            return np.stack(features_list, axis=0)
        else:
            # Return placeholder if features not available
            N = len(pose_results)
            return np.zeros((N, 768, 16, 16))
    
    def visualize(
        self,
        image: Union[str, np.ndarray],
        keypoints: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize pose estimation results on image.
        
        Args:
            image: Image path (str) or BGR numpy array.
            keypoints: (N, K, 3) array with [x, y, score].
            bboxes: Optional (N, 4) bounding boxes.
            masks: Optional (N, H, W) binary masks.
            save_path: Optional path to save visualization.
        
        Returns:
            np.ndarray: Visualization image (BGR).
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        else:
            img = image.copy()
        
        # Visualize each person's pose
        # for i, kpts in enumerate(keypoints):
        keypoints_17 = keypoints[:, :17, :]  # Assuming COCO format for visualization. For now, the visualization supports only 17 keypoints.
        img = pose_visualization(
            img,
            keypoints_17,
            width_multiplier=8,
            differ_individuals=True,
            keep_image_size=True,
        )
        
        # Save if requested
        if save_path is not None:
            cv2.imwrite(save_path, img)
        
        return img
