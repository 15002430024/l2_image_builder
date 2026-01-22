"""
数据清洗模块

Prompt 1.3 增强版本
提供时间过滤、异常值过滤、深交所撤单价格关联、数据清洗整合等功能

清洗规则：
1. 时间过滤：只保留连续竞价时段 09:30-11:30, 13:00-14:57
2. 异常值过滤：
   - 非撤单：Price > 0 AND Qty > 0
   - 撤单：Qty > 0
"""

from .time_filter import (
    is_continuous_auction,
    filter_continuous_auction,
    filter_continuous_auction_polars,
    filter_continuous_auction_pandas,
    filter_continuous_auction_auto,
    TimeFilter,
)
from .anomaly_filter import (
    filter_anomalies,
    filter_anomalies_polars,
    filter_anomalies_pandas,
    filter_anomalies_auto,
    filter_zero_price,
    filter_zero_qty,
    AnomalyFilter,
)
from .sz_cancel_enricher import (
    enrich_sz_cancel_price,
    SZCancelEnricher,
)
from .data_cleaner import (
    DataCleaner,
    clean_l2_data,
)

__all__ = [
    # 时间过滤
    "is_continuous_auction",
    "filter_continuous_auction",
    "filter_continuous_auction_polars",
    "filter_continuous_auction_pandas",
    "filter_continuous_auction_auto",
    "TimeFilter",
    # 异常值过滤
    "filter_anomalies",
    "filter_anomalies_polars",
    "filter_anomalies_pandas",
    "filter_anomalies_auto",
    "filter_zero_price",
    "filter_zero_qty",
    "AnomalyFilter",
    # 深交所撤单价格关联
    "enrich_sz_cancel_price",
    "SZCancelEnricher",
    # 数据清洗整合（Prompt 1.3）
    "DataCleaner",
    "clean_l2_data",
]
