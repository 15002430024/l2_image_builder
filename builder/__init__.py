"""
图像构建模块

将Level2数据转换为 [15, 8, 8] 的三维图像
"""

from .image_builder import (
    Level2ImageBuilder,
    build_l2_image,
    build_l2_image_with_stats,
)
from .normalizer import (
    normalize_image, 
    log1p_normalize,
    channel_max_normalize,
    ImageNormalizer,
    compute_channel_statistics,
)
from .sh_builder import (
    SHImageBuilder,
    build_sh_image,
    build_sh_image_with_stats,
)
from .sz_builder import (
    SZImageBuilder,
    build_sz_image,
    build_sz_image_with_stats,
    build_active_seqs_from_trade,
)

__all__ = [
    # 统一入口
    "Level2ImageBuilder",
    "build_l2_image",
    "build_l2_image_with_stats",
    # 归一化
    "normalize_image",
    "log1p_normalize",
    "channel_max_normalize",
    "ImageNormalizer",
    "compute_channel_statistics",
    # 上交所
    "SHImageBuilder",
    "build_sh_image",
    "build_sh_image_with_stats",
    # 深交所
    "SZImageBuilder",
    "build_sz_image",
    "build_sz_image_with_stats",
    "build_active_seqs_from_trade",
]
