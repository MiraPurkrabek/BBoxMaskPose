# Copyright (c) OpenMMLab. All rights reserved.
from .coco_metric import CocoMetric
from .coco_wholebody_metric import CocoWholeBodyMetric
from .hand_metric import InterHandMetric
from .keypoint_2d_metrics import (AUC, EPE, NME, JhmdbPCKAccuracy,
                                  MpiiPCKAccuracy, PCKAccuracy)
from .keypoint_3d_metrics import MPJPE
from .keypoint_partition_metric import KeypointPartitionMetric
from .posetrack18_metric import PoseTrack18Metric
from .simple_keypoint_3d_metrics import SimpleMPJPE
from .identity_metric import IdentityMetric
from .binary_identity_metric import BinaryIdentityMetric

__all__ = [
    'CocoMetric', 'PCKAccuracy', 'MpiiPCKAccuracy', 'JhmdbPCKAccuracy', 'AUC',
    'EPE', 'NME', 'PoseTrack18Metric', 'CocoWholeBodyMetric',
    'KeypointPartitionMetric', 'MPJPE', 'InterHandMetric', 'SimpleMPJPE',
    'IdentityMetric', 'BinaryIdentityMetric'
]
