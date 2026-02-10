"""
Public API for BBoxMaskPose wrapper.

This module provides a stable, user-friendly interface for the full
BBoxMaskPose pipeline: detection, pose estimation, and mask refinement.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import mmengine
import numpy as np
import yaml
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData

# Import from BBoxMaskPose package
from .sam2_utils import prepare_model as prepare_sam2_model
from .sam2_utils import process_image_with_SAM
from .demo_utils import (
    DotDict,
    concat_instances,
    filter_instances,
    pose_nms,
    _visualize_predictions,
)
from .posevis_lite import pose_visualization

BMP_ROOT = Path(__file__).parent.parent

# Note: PMPose will be imported when needed to avoid circular imports
# from pmpose import PMPose

# Default detector and pose config
DEFAULT_DET_CAT_ID: int = 0
DEFAULT_BBOX_THR: float = 0.3
DEFAULT_NMS_THR: float = 0.3
DEFAULT_KPT_THR: float = 0.3

# Pretrained config URLs (for future use)
PRETRAINED_CONFIGS = {
    "bmp-d3": "BMP_D3",
    "bmp-j1": "BMP_J1",
}


class BBoxMaskPose:
    """
    Public wrapper API for BBoxMaskPose pipeline.
    
    This class provides a complete pipeline for detection, pose estimation,
    and mask refinement using SAM2.
    
    Example:
        >>> bmp_model = BBoxMaskPose(config="BMP_D3", device="cuda")
        >>> result = bmp_model.predict(
        ...     image="path/to/image.jpg",
        ...     return_intermediates=True
        ... )
    """
    
    def __init__(
        self,
        config: str = "BMP_D3",
        device: str = "cuda",
        config_path: Optional[str] = None,
        pose_model=None,  # Type hint removed to avoid import at module level
        pretrained_id: Optional[str] = None,
    ):
        """
        Initialize BBoxMaskPose model.
        
        Args:
            config (str): Config alias ('BMP_D3', 'BMP_J1'). Defaults to 'BMP_D3'.
            device (str): Device for inference. Defaults to 'cuda'.
            config_path (str, optional): Path to custom YAML config file.
            pose_model (PMPose, optional): Pre-initialized PMPose instance.
                If None, will create internal pose model.
            pretrained_id (str, optional): Alias for pretrained config.
        """
        self.device = device
        self.config_name = config
        
        # Determine config path
        if config_path is not None:
            self.config_path = config_path
        else:
            bmp_configs_root = os.path.join(BMP_ROOT, "bboxmaskpose", "configs") 
            config_file = f"{config}.yaml"
            self.config_path = os.path.join(bmp_configs_root, config_file)
            
            if not os.path.exists(self.config_path):
                available_configs = glob.glob(os.path.join(bmp_configs_root, "*.yaml"))
                available_configs = [os.path.basename(f).replace(".yaml", "") for f in available_configs]
                raise FileNotFoundError(
                    f"Config file not found: {self.config_path}. "
                    f"Available configs: {', '.join(available_configs)}"
                )
        
        # Load config
        self.config = self._load_config(self.config_path)
        
        # Initialize or use provided pose model
        if pose_model is not None:
            self.pose_model = pose_model
            self._owns_pose_model = False
        else:
            # Create internal PMPose instance
            self.pose_model = self._create_pose_model()
            self._owns_pose_model = True
        
        # Initialize detector and SAM2
        self.detector = None
        self.detector_prime = None
        self.sam2_model = None
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> DotDict:
        """Load BMP configuration from YAML file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return DotDict(cfg)
    
    def _create_pose_model(self):
        """Create internal PMPose model from config."""
        # Import PMPose here to avoid circular imports
        from pmpose import PMPose
        
        # Extract pose config from BMP config
        pose_config = self.config.pose_estimator.pose_config
        pose_checkpoint = self.config.pose_estimator.pose_checkpoint
        
        # Create PMPose instance with custom config
        full_pose_config = str(BMP_ROOT / pose_config)
        
        pose_model = PMPose(
            device=self.device,
            config_path=full_pose_config,
            from_pretrained=True,
        )
        
        # Load checkpoint if it's a local path
        if not pose_checkpoint.startswith("http"):
            pose_model.load_from_file(pose_checkpoint)
        
        return pose_model
    
    def _initialize_models(self):
        """Initialize detector and SAM2 models."""
        # Initialize detector
        self.detector = init_detector(
            self.config.detector.det_config,
            self.config.detector.det_checkpoint,
            device=self.device
        )
        
        # Adapt detector pipeline
        from mmpose.utils import adapt_mmdet_pipeline
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
        # Initialize detector prime (may be same as detector)
        if (self.config.detector.det_config == self.config.detector.det_prime_config and
            self.config.detector.det_checkpoint == self.config.detector.det_prime_checkpoint) or \
           (self.config.detector.det_prime_config is None or 
            self.config.detector.det_prime_checkpoint is None):
            self.detector_prime = self.detector
        else:
            self.detector_prime = init_detector(
                self.config.detector.det_prime_config,
                self.config.detector.det_prime_checkpoint,
                device=self.device
            )
            self.detector_prime.cfg = adapt_mmdet_pipeline(self.detector_prime.cfg)
        
        # Initialize SAM2
        sam2_config_path = os.path.join(BMP_ROOT, "bboxmaskpose", "sam2", self.config.sam2.sam2_config)
        self.sam2_model = prepare_sam2_model(
            model_cfg=sam2_config_path,
            model_checkpoint=self.config.sam2.sam2_checkpoint,
        )
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        return_intermediates: bool = False,
        return_probmaps: bool = False,
    ) -> Dict:
        """
        Run full BBoxMaskPose pipeline on image.
        
        Args:
            image: Image path (str) or BGR numpy array.
            bboxes: Optional (N, 4) bboxes in [x1, y1, x2, y2] format.
                If None, run detector.
            return_intermediates: If True, return intermediate outputs.
            return_probmaps: If True, request heatmaps from pose model.
        
        Returns:
            Dict with keys:
                - 'bboxes': (N, 4) final bounding boxes
                - 'masks': (N, H, W) refined binary masks
                - 'keypoints': (N, K, 3) keypoints with scores
                - 'presence': (N, K) presence probabilities
                - 'visibility': (N, K) visibility flags
                - 'detector': (optional) raw detector outputs
                - 'sam2': (optional) intermediate SAM outputs
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        else:
            img = image.copy()
        
        # Run BMP iterations
        all_detections = None
        intermediate_results = [] if return_intermediates else None
        
        for iteration in range(self.config.num_bmp_iters):
            # Step 1: Detection
            if iteration == 0 and bboxes is not None:
                # Use provided bboxes for first iteration
                det_instances = InstanceData(
                    bboxes=bboxes,
                    bbox_scores=np.ones(len(bboxes)),
                    masks=None
                )
            else:
                # Run detector
                det_instances = self._run_detector(
                    self.detector if iteration == 0 else self.detector_prime,
                    img if all_detections is None else self._mask_out_image(img, all_detections),
                )
            
            if len(det_instances.bboxes) == 0:
                continue
            
            # Step 2: Pose estimation using PMPose wrapper
            pose_results = self._run_pose_estimation(
                img,
                det_instances,
                return_probmaps=return_probmaps
            )
            
            # Step 3: Pose NMS and SAM refinement
            new_detections, old_detections = self._refine_with_sam(
                img,
                pose_results,
                all_detections,
            )
            
            # Merge detections
            if all_detections is None:
                all_detections = new_detections
            else:
                all_detections = concat_instances(old_detections, new_detections)
            
            # Store intermediates if requested
            if return_intermediates:
                intermediate_results.append({
                    'iteration': iteration,
                    'detections': det_instances,
                    'poses': pose_results,
                    'refined': new_detections,
                })
        
        # Prepare final result
        result = self._format_result(all_detections, img.shape[:2])
        
        if return_intermediates:
            result['intermediates'] = intermediate_results
        
        return result
    
    def _run_detector(
        self,
        detector,
        img: np.ndarray,
    ) -> InstanceData:
        """Run MMDetection detector."""
        from mmpose.evaluation.functional import nms
        
        # Run detection
        det_result = inference_detector(detector, img)
        pred_instances = det_result.pred_instances.cpu().numpy()
        
        # Aggregate bboxes and scores
        bboxes_all = np.concatenate(
            (pred_instances.bboxes, pred_instances.scores[:, None]), axis=1
        )
        
        # Filter by category and score
        keep_mask = np.logical_and(
            pred_instances.labels == DEFAULT_DET_CAT_ID,
            pred_instances.scores > DEFAULT_BBOX_THR
        )
        
        if not np.any(keep_mask):
            return InstanceData(
                bboxes=np.zeros((0, 4)),
                bbox_scores=np.zeros((0,)),
                masks=np.zeros((0, 1, 1))
            )
        
        bboxes = bboxes_all[keep_mask]
        masks = getattr(pred_instances, "masks", None)
        if masks is not None:
            masks = masks[keep_mask]
        
        # Sort by score
        order = np.argsort(bboxes[:, 4])[::-1]
        bboxes = bboxes[order]
        if masks is not None:
            masks = masks[order]
        
        # Apply NMS
        keep_indices = nms(bboxes, DEFAULT_NMS_THR)
        bboxes = bboxes[keep_indices]
        if masks is not None:
            masks = masks[keep_indices]
        
        return InstanceData(
            bboxes=bboxes[:, :4],
            bbox_scores=bboxes[:, 4],
            masks=masks
        )
    
    def _run_pose_estimation(
        self,
        img: np.ndarray,
        det_instances: InstanceData,
        return_probmaps: bool = False,
    ) -> InstanceData:
        """Run pose estimation using PMPose wrapper."""
        bboxes = det_instances.bboxes
        masks = det_instances.masks
        
        if len(bboxes) == 0:
            return InstanceData(
                keypoints=np.zeros((0, 17, 3)),
                keypoint_scores=np.zeros((0, 17)),
                bboxes=bboxes,
                bbox_scores=det_instances.bbox_scores,
                masks=masks,
            )
        
        # Call PMPose public API
        keypoints, presence, visibility, heatmaps = self.pose_model.predict(
            img,
            bboxes,
            masks=masks,
            return_probmaps=return_probmaps,
        )
        
        # Restrict to first 17 COCO keypoints
        keypoints = keypoints[:, :17, :]
        presence = presence[:, :17]
        visibility = visibility[:, :17]
        
        # Create InstanceData with results
        result = InstanceData(
            keypoints=keypoints,
            keypoint_scores=keypoints[:, :, 2],
            bboxes=bboxes,
            bbox_scores=det_instances.bbox_scores,
            masks=masks,
        )
        
        return result
    
    def _refine_with_sam(
        self,
        img: np.ndarray,
        pose_instances: InstanceData,
        all_detections: Optional[InstanceData],
    ) -> tuple:
        """Perform Pose-NMS and SAM refinement."""
        # Combine keypoints with scores
        keypoints_with_scores = pose_instances.keypoints
        
        # Perform Pose-NMS
        all_keypoints = (
            keypoints_with_scores
            if all_detections is None
            else np.concatenate([all_detections.keypoints, keypoints_with_scores], axis=0)
        )
        all_bboxes = (
            pose_instances.bboxes
            if all_detections is None
            else np.concatenate([all_detections.bboxes, pose_instances.bboxes], axis=0)
        )
        
        num_valid_kpts = np.sum(
            all_keypoints[:, :, 2] > self.config.sam2.prompting.confidence_thr,
            axis=1
        )
        
        keep_indices = pose_nms(
            DotDict({
                "confidence_thr": self.config.sam2.prompting.confidence_thr,
                "oks_thr": self.config.oks_nms_thr
            }),
            image_kpts=all_keypoints,
            image_bboxes=all_bboxes,
            num_valid_kpts=num_valid_kpts,
        )
        
        keep_indices = sorted(keep_indices)
        num_old_detections = 0 if all_detections is None else len(all_detections.bboxes)
        keep_new_indices = [i - num_old_detections for i in keep_indices if i >= num_old_detections]
        keep_old_indices = [i for i in keep_indices if i < num_old_detections]
        
        if len(keep_new_indices) == 0:
            return None, all_detections
        
        # Filter new detections
        new_dets = filter_instances(pose_instances, keep_new_indices)
        new_dets.scores = pose_instances.keypoint_scores[keep_new_indices].mean(axis=-1)
        
        old_dets = None
        if len(keep_old_indices) > 0:
            old_dets = filter_instances(all_detections, keep_old_indices)
        
        # Run SAM refinement
        new_detections = process_image_with_SAM(
            DotDict(self.config.sam2.prompting),
            img.copy(),
            self.sam2_model,
            new_dets,
            old_dets if old_dets is not None else None,
        )
        
        return new_detections, old_dets
    
    def _mask_out_image(
        self,
        img: np.ndarray,
        detections: InstanceData,
    ) -> np.ndarray:
        """Mask out detected instances from image for next iteration."""
        masked_img = img.copy()
        if hasattr(detections, 'refined_masks') and detections.refined_masks is not None:
            for mask in detections.refined_masks:
                if mask is not None:
                    masked_img[mask.astype(bool)] = 0
        return masked_img
    
    def _format_result(
        self,
        detections: Optional[InstanceData],
        img_shape: tuple,
    ) -> Dict:
        """Format detection results into standard output dict."""
        if detections is None or len(detections.bboxes) == 0:
            return {
                'bboxes': np.zeros((0, 4)),
                'masks': np.zeros((0, img_shape[0], img_shape[1])),
                'keypoints': np.zeros((0, 17, 3)),
                'presence': np.zeros((0, 17)),
                'visibility': np.zeros((0, 17)),
            }
        
        # Extract refined masks if available
        if hasattr(detections, 'refined_masks') and detections.refined_masks is not None:
            masks = detections.refined_masks
        elif hasattr(detections, 'pred_masks') and detections.pred_masks is not None:
            masks = detections.pred_masks
        elif hasattr(detections, 'masks') and detections.masks is not None:
            masks = detections.masks
        else:
            masks = np.zeros((len(detections.bboxes), img_shape[0], img_shape[1]))
        
        # Create presence and visibility from keypoint scores (dummy for MaskPose)
        keypoint_scores = detections.keypoint_scores
        presence = np.copy(keypoint_scores)
        visibility = (keypoint_scores > 0.3).astype(np.float32)
        
        return {
            'bboxes': detections.bboxes,
            'masks': masks,
            'keypoints': detections.keypoints,
            'presence': presence,
            'visibility': visibility,
        }
    
    def visualize(
        self,
        image: Union[str, np.ndarray],
        result: Dict,
        save_path: Optional[str] = None,
        vis_type: str = "pose",
    ) -> np.ndarray:
        """
        Visualize BBoxMaskPose results on image.
        
        Args:
            image: Image path (str) or BGR numpy array.
            result: Result dict from predict().
            save_path: Optional path to save visualization.
            vis_type: Type of visualization ("pose" or "mask").
        Returns:
            np.ndarray: Visualization image (BGR).
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        else:
            img = image.copy()
        
        if vis_type == "mask":
            vis_img, _ = _visualize_predictions(
                img,
                bboxes = result['bboxes'],
                scores = np.ones(len(result['bboxes'])),
                masks = result['masks'],
                poses = result['keypoints'],
                vis_type = "mask",
                mask_is_binary = True,
            )
            img = vis_img
        else:
            # Visualize using posevis_lite
            keypoints = result['keypoints']
            keypoints = keypoints[:, :17, :]  # Use first 17 COCO keypoints
            # for i, kpts in enumerate(keypoints):
            img = pose_visualization(
                img,
                keypoints,
                width_multiplier=8,
                differ_individuals=True,
                keep_image_size=True,
            )
            
        # Save if requested
        if save_path is not None:
            cv2.imwrite(save_path, img)
        
        return img
