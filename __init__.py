"""
Level2 数据图像化处理模块

将Level2逐笔成交与逐笔委托数据转换为标准化的三维图像格式 [15, 8, 8]，
用于 Vision Transformer (ViT) 和 Video Vision Transformer (ViViT) 模型训练。

核心模块：
- config: 配置管理
- data_loader: 数据加载（上交所/深交所）
- cleaner: 数据清洗
- calculator: 分位数与大单计算
- builder: 图像构建
- storage: LMDB存储
- diagnostics: 诊断报告
- dataset: PyTorch数据集
"""

__version__ = "1.0.0"
__author__ = "L2 Image Builder Team"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
