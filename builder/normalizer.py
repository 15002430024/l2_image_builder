"""
图像归一化模块

Log1p + 通道内 Max 归一化
"""

import numpy as np
from typing import Optional


def log1p_normalize(image: np.ndarray) -> np.ndarray:
    """
    Log1p 变换
    
    处理长尾分布问题，压缩极值
    
    Args:
        image: 原始计数图像
    
    Returns:
        Log1p 变换后的图像
    
    Formula:
        X_log = log(1 + X)
    """
    return np.log1p(image)


def channel_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    通道内 Max 归一化
    
    每个通道独立归一化到 [0, 1]
    
    Args:
        image: [15, 8, 8] 图像
    
    Returns:
        归一化后的图像，值域 [0, 1]
    """
    normalized = np.zeros_like(image)
    
    for ch in range(image.shape[0]):
        max_val = image[ch].max()
        if max_val > 0:
            normalized[ch] = image[ch] / max_val
        # else: 保持全0
    
    return normalized


def normalize_image(
    image: np.ndarray,
    apply_log: bool = True,
    apply_channel_norm: bool = True,
) -> np.ndarray:
    """
    图像归一化（完整流程）
    
    默认流程: Log1p → 通道内 Max 归一化
    
    Args:
        image: [15, 8, 8] 原始计数图像
        apply_log: 是否应用 log1p 变换
        apply_channel_norm: 是否应用通道内归一化
    
    Returns:
        [15, 8, 8] 归一化后的图像，值域 [0, 1]
    
    Formula:
        X_final = log(1 + X) / max(log(1 + X))
    
    Example:
        >>> image = np.random.randint(0, 1000, (15, 8, 8)).astype(np.float32)
        >>> normalized = normalize_image(image)
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    result = image.astype(np.float32)
    
    # 1. Log 变换：解决长尾问题
    if apply_log:
        result = log1p_normalize(result)
    
    # 2. 通道内归一化
    if apply_channel_norm:
        result = channel_max_normalize(result)
    
    return result.astype(np.float32)


def denormalize_image(
    normalized_image: np.ndarray,
    original_max_values: np.ndarray,
) -> np.ndarray:
    """
    反归一化（用于可视化或调试）
    
    Args:
        normalized_image: 归一化后的图像
        original_max_values: 每个通道的原始最大值
    
    Returns:
        近似还原的原始图像
    
    Note:
        由于 log 变换的非线性，无法完全还原
    """
    result = np.zeros_like(normalized_image)
    
    for ch in range(normalized_image.shape[0]):
        if original_max_values[ch] > 0:
            # 反归一化
            result[ch] = normalized_image[ch] * original_max_values[ch]
            # 反 log
            result[ch] = np.expm1(result[ch])
    
    return result


class ImageNormalizer:
    """
    图像归一化器类
    
    支持保存归一化参数，用于后续反归一化
    """
    
    def __init__(
        self,
        apply_log: bool = True,
        apply_channel_norm: bool = True,
    ):
        self.apply_log = apply_log
        self.apply_channel_norm = apply_channel_norm
        
        # 保存归一化参数
        self.log_max_values: Optional[np.ndarray] = None
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像
        """
        result = image.astype(np.float32)
        
        if self.apply_log:
            result = log1p_normalize(result)
        
        if self.apply_channel_norm:
            # 保存最大值用于反归一化
            self.log_max_values = np.array([result[ch].max() for ch in range(result.shape[0])])
            result = channel_max_normalize(result)
        
        return result.astype(np.float32)
    
    def denormalize(self, normalized_image: np.ndarray) -> np.ndarray:
        """
        反归一化图像
        """
        if self.log_max_values is None:
            raise RuntimeError("归一化参数未保存，无法反归一化")
        
        return denormalize_image(normalized_image, self.log_max_values)


def compute_channel_statistics(image: np.ndarray) -> dict:
    """
    计算图像通道统计信息
    
    Args:
        image: [15, 8, 8] 图像
    
    Returns:
        各通道的统计信息
    """
    stats = {}
    
    for ch in range(image.shape[0]):
        channel_data = image[ch]
        stats[ch] = {
            'sum': float(channel_data.sum()),
            'mean': float(channel_data.mean()),
            'max': float(channel_data.max()),
            'min': float(channel_data.min()),
            'std': float(channel_data.std()),
            'nonzero_count': int(np.count_nonzero(channel_data)),
            'fill_rate': float(np.count_nonzero(channel_data) / channel_data.size),
        }
    
    return stats
