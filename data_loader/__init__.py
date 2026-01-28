"""
数据加载模块

提供上交所、深交所数据的统一加载接口

增强功能（Prompt 1.2）：
- 懒加载支持：scan_parquet_with_filter, read_parquet_lazy
- 批量处理：batch_load_stocks, iter_stocks_lazy
- 股票列表获取：get_stock_list_from_parquet
"""

from .sh_loader import SHDataLoader
from .sz_loader import SZDataLoader
from .sz_data_reconstructor import (
    reconstruct_sz_parquet,
    batch_reconstruct_sz_parquet,
    verify_reconstruction,
)
from .polars_utils import (
    # 基础读取
    read_parquet_auto,
    read_parquet_lazy,
    read_multiple_parquets,
    # 懒加载
    scan_parquet_with_filter,
    collect_lazy,
    # 类型转换
    to_pandas,
    to_polars,
    to_numpy,
    # 过滤
    filter_time_range,
    filter_by_values,
    filter_positive,
    # 聚合
    get_unique_stocks,
    groupby_sum,
    compute_percentiles,
    # 迭代
    split_by_stock,
    iter_by_stock,
    iter_stocks_lazy,
    # 批量加载
    get_stock_list_from_parquet,
    batch_load_stocks,
    # 工具
    is_polars_df,
    is_pandas_df,
    estimate_memory,
    describe_df,
    POLARS_AVAILABLE,
)

__all__ = [
    # 加载器
    "SHDataLoader",
    "SZDataLoader",
    # 深交所数据重构 (REQ-004)
    "reconstruct_sz_parquet",
    "batch_reconstruct_sz_parquet",
    "verify_reconstruction",
    # 基础读取
    "read_parquet_auto",
    "read_parquet_lazy",
    "read_multiple_parquets",
    # 懒加载
    "scan_parquet_with_filter",
    "collect_lazy",
    # 类型转换
    "to_pandas",
    "to_polars",
    "to_numpy",
    # 过滤
    "filter_time_range",
    "filter_by_values",
    "filter_positive",
    # 聚合
    "get_unique_stocks",
    "groupby_sum",
    "compute_percentiles",
    # 迭代
    "split_by_stock",
    "iter_by_stock",
    "iter_stocks_lazy",
    # 批量加载
    "get_stock_list_from_parquet",
    "batch_load_stocks",
    # 工具
    "is_polars_df",
    "is_pandas_df",
    "estimate_memory",
    "describe_df",
    "POLARS_AVAILABLE",
]
