# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .rtmo_head import RTMOHead
from .vis_head import VisPredictHead
from .yoloxpose_head import YOLOXPoseHead
from .multi_head import MultiHead
from .double_head import DoubleHead
from .double_head_doubleUDP import DoubleHeadDoubleUDP
from .poseid_head import PoseIDHead
from .iterative_head import IterativeHead
from .oks_head import OKSHead
from .calibration_head import CalibrationHead

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead', 'RTMOHead', 'MultiHead', 'DoubleHead', 'PoseIDHead',
           'DoubleHeadDoubleUDP', 'IterativeHead', 'OKSHead', 'CalibrationHead']
