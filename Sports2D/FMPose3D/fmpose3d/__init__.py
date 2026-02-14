"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

__version__ = "0.0.7"
__author__ = "Ti Wang, Xiaohang Yu, Mackenzie Weygandt Mathis"
__license__ = "Apache 2.0"

# Import key components for easy access
from .aggregation_methods import (
    average_aggregation,
    aggregation_select_single_best_hypothesis_by_2D_error,
    aggregation_RPEA_joint_level,
)

# Optional 2D pose detection utilities.
# These are not required for Sports2D's 2D->3D lifting path and can pull
# additional demo-only dependencies (e.g. skimage, yolox, transformers).
try:
    from .lib.gen_kpts import gen_video_kpts
    from .lib.preprocess import h36m_coco_format, revise_kpts
except Exception:  # pragma: no cover - optional demo dependencies
    gen_video_kpts = None
    h36m_coco_format = None
    revise_kpts = None

# Make commonly used classes/functions available at package level
__all__ = [
    # Aggregation methods
    "average_aggregation",
    "aggregation_select_single_best_hypothesis_by_2D_error",
    "aggregation_RPEA_joint_level",
    # 2D pose detection
    "gen_video_kpts",
    "h36m_coco_format",
    "revise_kpts",
    # Version
    "__version__",
]

