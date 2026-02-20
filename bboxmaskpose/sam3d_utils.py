# Copyright (c) authors of BBoxMaskPose (BMPv2). All rights reserved.

"""
SAM-3D-Body integration utilities for BBoxMaskPose.

This module provides a lightweight wrapper for integrating SAM-3D-Body
(3D human mesh recovery) into the BBoxMaskPose pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


def check_sam3d_available() -> bool:
    """
    Check if SAM-3D-Body package is available.

    Returns:
        bool: True if sam_3d_body can be imported, False otherwise.
    """
    try:
        import sam_3d_body

        return True
    except ImportError:
        return False


class SAM3DBodyWrapper:
    """
    Wrapper class for SAM-3D-Body model.

    This class provides a simplified interface for 3D human mesh recovery
    that integrates seamlessly with BBoxMaskPose outputs.

    Example:
        >>> sam3d = SAM3DBodyWrapper(device="cuda")
        >>> meshes = sam3d.predict(
        ...     image="path/to/image.jpg",
        ...     bboxes=bmp_result['bboxes'],
        ...     masks=bmp_result['masks']
        ... )
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        mhr_path: Optional[str] = None,
        device: str = "cuda",
        use_detector: bool = False,
        use_segmentor: bool = False,
        use_fov: bool = True,
        fov_name: str = "moge2",
    ):
        """
        Initialize SAM-3D-Body wrapper.

        Args:
            checkpoint_path: Path to SAM-3D-Body checkpoint. If None, will attempt
                to load from HuggingFace (facebook/sam-3d-body-dinov3).
            mhr_path: Path to MHR (Momentum Human Rig) model file.
            device: Device for inference ('cuda' or 'cpu').
            use_detector: Whether to use built-in human detector (not needed with BMP).
            use_segmentor: Whether to use built-in segmentor (not needed with BMP).
            use_fov: Whether to use FOV estimator for camera calibration.
            fov_name: FOV estimator name ('moge2' recommended).
        """
        if not check_sam3d_available():
            raise ImportError(
                "SAM-3D-Body package not found. Please install it following:\n"
                "https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md\n\n"
                "Quick install:\n"
                "pip install pytorch-lightning pyrender opencv-python yacs scikit-image "
                "einops timm dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 "
                "roma joblib seaborn appdirs ffmpeg cython jsonlines loguru optree fvcore "
                "black pycocotools huggingface_hub\n"
                "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' "
                "--no-build-isolation --no-deps\n"
                "pip install git+https://github.com/microsoft/MoGe.git"
            )

        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf

        self.device = torch.device(device) if isinstance(device, str) else device

        # Load SAM-3D-Body model
        if checkpoint_path is not None:
            print(f"Loading SAM-3D-Body from checkpoint: {checkpoint_path}")
            self.model, self.model_cfg = load_sam_3d_body(checkpoint_path, device=self.device, mhr_path=mhr_path)
        else:
            # Load from HuggingFace
            print("Loading SAM-3D-Body from HuggingFace (facebook/sam-3d-body-dinov3)")
            print("Note: This requires HuggingFace authentication and access approval.")
            self.model, self.model_cfg = load_sam_3d_body_hf(repo_id="facebook/sam-3d-body-dinov3", device=self.device)

        print("✓ SAM-3D-Body model loaded successfully")

        # Initialize optional components
        human_detector = None
        human_segmentor = None
        fov_estimator = None

        # if use_detector:
        #     from sam_3d_body.tools.build_detector import HumanDetector
        #     human_detector = HumanDetector(name="vitdet", device=self.device)

        # if use_segmentor:
        #     from sam_3d_body.tools.build_sam import HumanSegmentor
        #     human_segmentor = HumanSegmentor(name="sam2", device=self.device)

        if use_fov:
            try:
                from .sam3d_build_fov_estimator import FOVEstimator

                # fov_estimator = FOVEstimator(name=fov_name, device=self.device)
            except Exception as e:
                print(f"Warning: Could not load FOV estimator: {e}")
                print("Continuing without FOV estimation (will use default FOV)")

        # Create estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=human_detector,
            human_segmentor=human_segmentor,
            fov_estimator=fov_estimator,
        )

        print("✓ SAM-3D-Body initialized successfully")

    def predict(
        self,
        image: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = True,
        inference_type: str = "full",
    ) -> List[Dict]:
        """
        Predict 3D human meshes from image.

        Args:
            image: Input image (path or numpy array in BGR format).
            bboxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format.
                If None, will use internal detector (if available).
            masks: Binary masks (N, H, W) or (N, H, W, 1).
                If provided, will be used for mask-conditioned inference.
            bbox_thr: Bounding box detection threshold (only if using internal detector).
            nms_thr: NMS threshold (only if using internal detector).
            use_mask: Whether to use mask-conditioned inference.
            inference_type: Type of inference to run:
                - "full": Full-body inference with both body and hand decoders (default)
                - "body": Inference with body decoder only (faster)
                - "hand": Inference with hand decoder only

        Returns:
            List of prediction dicts, one per detected person. Each dict contains:
                - 'vertices': (V, 3) 3D mesh vertices in camera coordinates
                - 'faces': (F, 3) mesh face indices
                - 'joints': (J, 3) 3D joint locations
                - 'bbox': (4,) bounding box [x1, y1, x2, y2]
                - 'mask': (H, W) binary mask (if provided)
                And other intermediate outputs from SAM-3D-Body
        """
        # Handle different image input formats
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image

        # Process masks if provided
        processed_masks = None
        if masks is not None:
            # Ensure masks are in correct format (N, H, W)
            if masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.squeeze(-1)

            # Ensure masks are in [0, 255] range
            # BMP outputs binary masks [0, 1], so we need to scale
            if masks.max() <= 1.0:
                processed_masks = (masks * 255).astype(np.uint8)
            else:
                # Already in [0, 255] range
                processed_masks = np.clip(masks, 0, 255).astype(np.uint8)
            use_mask = True

        # Run SAM-3D-Body inference
        outputs = self.estimator.process_one_image(
            img,
            bboxes=bboxes,
            masks=processed_masks,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            keypoints=keypoints,
            inference_type=inference_type,
        )

        return outputs

    @property
    def faces(self) -> np.ndarray:
        """Get mesh face indices for visualization."""
        return self.estimator.faces


def visualize_3d_meshes(
    image: np.ndarray,
    outputs: List[Dict],
    faces: np.ndarray,
    masks: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize 3D mesh predictions on the input image.

    Args:
        image: Input image (BGR format).
        outputs: List of prediction dicts from SAM3DBodyWrapper.predict().
        faces: Mesh face indices.
        masks: Optional binary masks for each detected person.
        output_path: Optional path to save the visualization.

    Returns:
        Visualization image with rendered 3D meshes (BGR format).
    """
    try:
        from .sam3d_vis_utils import visualize_sample_together

        vis_img = visualize_sample_together(image, outputs, faces, masks=masks, keypoints=keypoints)

        if output_path is not None:
            cv2.imwrite(output_path, vis_img.astype(np.uint8))

        return vis_img.astype(np.uint8)
    except ImportError:
        print("Warning: SAM-3D-Body visualization tools not available")
        print("Returning original image")
        return image
