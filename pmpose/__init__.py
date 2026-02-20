# Copyright (c) authors of BBoxMaskPose (BMPv2). All rights reserved.

"""
PMPose package - Public API for pose estimation.

This package provides a stable wrapper around MaskPose (preparing for PMPose)
with a user-friendly interface for pose estimation tasks.
"""

from .api import PMPose

__all__ = ["PMPose"]
