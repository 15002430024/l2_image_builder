"""
计算模块

Prompt 2.1 增强版本
提供分位数计算、大单还原与阈值计算

分位数计算功能：
- 联合计算：成交+委托数据合并计算
- 沪深分离：针对不同交易所的专用函数
- 验证诊断：分布验证和可视化
"""

from .quantile import (
    # 基础函数
    compute_quantile_bins,
    get_bin_index,
    get_bin_indices_vectorized,
    QuantileCalculator,
    # Prompt 2.1 - Polars 向量化版本
    compute_quantile_bins_sh_polars,
    compute_quantile_bins_sz_polars,
    compute_quantile_bins_sh_pandas,
    compute_quantile_bins_sz_pandas,
    compute_quantile_bins_auto,
    # 验证和诊断
    validate_quantile_bins,
    compute_bin_distribution,
    visualize_quantile_distribution,
    print_quantile_summary,
)
from .big_order import (
    restore_parent_orders,
    compute_threshold,
    BigOrderCalculator,
)

__all__ = [
    # 分位数基础
    "compute_quantile_bins",
    "get_bin_index",
    "get_bin_indices_vectorized",
    "QuantileCalculator",
    # 分位数 Polars 向量化（Prompt 2.1）
    "compute_quantile_bins_sh_polars",
    "compute_quantile_bins_sz_polars",
    "compute_quantile_bins_sh_pandas",
    "compute_quantile_bins_sz_pandas",
    "compute_quantile_bins_auto",
    # 验证和诊断
    "validate_quantile_bins",
    "compute_bin_distribution",
    "visualize_quantile_distribution",
    "print_quantile_summary",
    # 大单
    "restore_parent_orders",
    "compute_threshold",
    "BigOrderCalculator",
]
