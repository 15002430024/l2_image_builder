"""
数据清洗整合模块

Prompt 1.3 - 数据清洗模块（简化版）
整合时间过滤和异常值过滤，提供统一的数据清洗接口

清洗规则：
1. 时间过滤：只保留连续竞价时段 09:30-11:30, 13:00-14:57
2. 异常值过滤：
   - 非撤单：Price > 0 AND Qty > 0
   - 撤单：Qty > 0（价格可能为0）
"""

from typing import Union, Optional
import logging

import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from .time_filter import (
    filter_continuous_auction_auto,
    filter_continuous_auction_polars,
    filter_continuous_auction_pandas,
)
from .anomaly_filter import (
    filter_anomalies_auto,
    filter_anomalies_polars,
    filter_anomalies_pandas,
)

logger = logging.getLogger(__name__)

# 类型别名
DataFrame = Union[pd.DataFrame, "pl.DataFrame"]


def is_polars_df(df: DataFrame) -> bool:
    """检查是否为 Polars DataFrame"""
    if POLARS_AVAILABLE and pl is not None:
        return isinstance(df, pl.DataFrame)
    return False


class DataCleaner:
    """
    Level2 数据清洗器
    
    整合时间过滤和异常值过滤，支持不同交易所、不同数据类型的清洗
    
    支持的数据类型：
    - 上交所 (SH): Trade（逐笔成交）、Order（逐笔委托）
    - 深交所 (SZ): Order（逐笔委托）、Trade（逐笔成交）
    
    清洗规则：
    1. 时间过滤：只保留连续竞价时段
       - 上午: 09:30:00 - 11:30:00 (93000000 - 113000000)
       - 下午: 13:00:00 - 14:57:00 (130000000 - 145700000)
    2. 异常值过滤：
       - 非撤单记录：Price > 0 AND Qty > 0
       - 撤单记录：仅检查 Qty > 0（撤单价格可能为0）
    
    列名映射：
    - SH Trade: TickTime, Price, Qty
    - SH Order: TickTime, Price, Qty, OrdType
    - SZ Order: TransactTime, LastPx, LastQty
    - SZ Trade: TransactTime, LastPx, LastQty, ExecType
    
    Example:
        >>> cleaner = DataCleaner()
        >>> cleaned_df = cleaner.clean_sh_trade(df)
        >>> cleaned_df = cleaner.clean_sz_order(df)
    """
    
    # 交易所列名配置
    # R3.2 更新: 深交所使用归一化后的标准列名 (TickTime, Price, Qty)
    COLUMN_CONFIG = {
        "sh_trade": {
            "time_column": "TickTime",
            "price_column": "Price",
            "qty_column": "Qty",
        },
        "sh_order": {
            "time_column": "TickTime",
            "price_column": "Price",
            "qty_column": "Qty",
            "cancel_column": "OrdType",
            "cancel_value": "Cancel",  # 撤单标识
        },
        # R3.2: 深交所使用归一化后的标准列名
        "sz_order": {
            "time_column": "TickTime",  # 原 TransactTime
            "price_column": "Price",     # 原 LastPx (委托表本身就是 Price)
            "qty_column": "Qty",         # 原 OrderQty
        },
        "sz_trade": {
            "time_column": "TickTime",   # 原 TransactTime
            "price_column": "Price",     # 原 LastPx
            "qty_column": "Qty",         # 原 LastQty
            "cancel_column": "ExecType",
            "cancel_value": "52",  # 深交所撤单类型
        },
    }
    
    def __init__(self, verbose: bool = True):
        """
        初始化清洗器
        
        Args:
            verbose: 是否输出清洗日志
        """
        self.verbose = verbose
    
    def _log(self, message: str, original_count: int, cleaned_count: int):
        """输出清洗日志"""
        if self.verbose:
            removed = original_count - cleaned_count
            pct = (removed / original_count * 100) if original_count > 0 else 0
            logger.info(f"{message}: {original_count} -> {cleaned_count} (移除 {removed} 条, {pct:.2f}%)")
    
    def clean_sh_trade(self, df: DataFrame) -> DataFrame:
        """
        清洗上交所逐笔成交数据
        
        步骤：
        1. 时间过滤：只保留连续竞价时段
        2. 异常值过滤：Price > 0 AND Qty > 0
        
        Args:
            df: 上交所逐笔成交数据
        
        Returns:
            清洗后的 DataFrame
        """
        config = self.COLUMN_CONFIG["sh_trade"]
        original_count = len(df)
        
        # 1. 时间过滤
        df = filter_continuous_auction_auto(df, config["time_column"])
        after_time = len(df)
        self._log("SH Trade 时间过滤", original_count, after_time)
        
        # 2. 异常值过滤（非撤单）
        df = filter_anomalies_auto(
            df, 
            price_column=config["price_column"],
            qty_column=config["qty_column"],
            is_cancel=False
        )
        after_anomaly = len(df)
        self._log("SH Trade 异常值过滤", after_time, after_anomaly)
        
        return df
    
    def clean_sh_order(self, df: DataFrame) -> DataFrame:
        """
        清洗上交所逐笔委托数据
        
        步骤：
        1. 时间过滤：只保留连续竞价时段
        2. 异常值过滤（区分新单和撤单）：
           - 新单：Price > 0 AND Qty > 0
           - 撤单 (OrdType='Cancel'): Qty > 0
        
        Args:
            df: 上交所逐笔委托数据
        
        Returns:
            清洗后的 DataFrame
        """
        config = self.COLUMN_CONFIG["sh_order"]
        original_count = len(df)
        
        # 1. 时间过滤
        df = filter_continuous_auction_auto(df, config["time_column"])
        after_time = len(df)
        self._log("SH Order 时间过滤", original_count, after_time)
        
        # 2. 分离新单和撤单，分别过滤异常值
        df = self._filter_order_with_cancel(
            df,
            price_column=config["price_column"],
            qty_column=config["qty_column"],
            cancel_column=config["cancel_column"],
            cancel_value=config["cancel_value"],
        )
        after_anomaly = len(df)
        self._log("SH Order 异常值过滤", after_time, after_anomaly)
        
        return df
    
    def clean_sz_order(self, df: DataFrame) -> DataFrame:
        """
        清洗深交所逐笔委托数据
        
        深交所逐笔委托不包含撤单信息，统一检查 Price > 0 AND Qty > 0
        
        Args:
            df: 深交所逐笔委托数据
        
        Returns:
            清洗后的 DataFrame
        """
        config = self.COLUMN_CONFIG["sz_order"]
        original_count = len(df)
        
        # 1. 时间过滤
        df = filter_continuous_auction_auto(df, config["time_column"])
        after_time = len(df)
        self._log("SZ Order 时间过滤", original_count, after_time)
        
        # 2. 异常值过滤
        df = filter_anomalies_auto(
            df,
            price_column=config["price_column"],
            qty_column=config["qty_column"],
            is_cancel=False
        )
        after_anomaly = len(df)
        self._log("SZ Order 异常值过滤", after_time, after_anomaly)
        
        return df
    
    def clean_sz_trade(self, df: DataFrame) -> DataFrame:
        """
        清洗深交所逐笔成交数据
        
        步骤：
        1. 时间过滤：只保留连续竞价时段
        2. 异常值过滤（区分成交和撤单）：
           - 成交：Price > 0 AND Qty > 0
           - 撤单 (ExecType='52'): Qty > 0
        
        Args:
            df: 深交所逐笔成交数据
        
        Returns:
            清洗后的 DataFrame
        """
        config = self.COLUMN_CONFIG["sz_trade"]
        original_count = len(df)
        
        # 1. 时间过滤
        df = filter_continuous_auction_auto(df, config["time_column"])
        after_time = len(df)
        self._log("SZ Trade 时间过滤", original_count, after_time)
        
        # 2. 分离成交和撤单，分别过滤异常值
        df = self._filter_order_with_cancel(
            df,
            price_column=config["price_column"],
            qty_column=config["qty_column"],
            cancel_column=config["cancel_column"],
            cancel_value=config["cancel_value"],
        )
        after_anomaly = len(df)
        self._log("SZ Trade 异常值过滤", after_time, after_anomaly)
        
        return df
    
    def _filter_order_with_cancel(
        self,
        df: DataFrame,
        price_column: str,
        qty_column: str,
        cancel_column: str,
        cancel_value: str,
    ) -> DataFrame:
        """
        过滤包含撤单的数据（区分新单和撤单）
        
        Args:
            df: 输入 DataFrame
            price_column: 价格列名
            qty_column: 数量列名
            cancel_column: 撤单标识列名
            cancel_value: 撤单标识值
        
        Returns:
            过滤后的 DataFrame
        """
        if is_polars_df(df):
            return self._filter_order_with_cancel_polars(
                df, price_column, qty_column, cancel_column, cancel_value
            )
        else:
            return self._filter_order_with_cancel_pandas(
                df, price_column, qty_column, cancel_column, cancel_value
            )
    
    def _filter_order_with_cancel_polars(
        self,
        df: "pl.DataFrame",
        price_column: str,
        qty_column: str,
        cancel_column: str,
        cancel_value: str,
    ) -> "pl.DataFrame":
        """Polars 版本：过滤包含撤单的数据"""
        # 撤单条件：只检查 Qty > 0
        cancel_mask = (pl.col(cancel_column) == cancel_value) & (pl.col(qty_column) > 0)
        
        # 非撤单条件：Price > 0 AND Qty > 0
        non_cancel_mask = (
            (pl.col(cancel_column) != cancel_value) & 
            (pl.col(price_column) > 0) & 
            (pl.col(qty_column) > 0)
        )
        
        return df.filter(cancel_mask | non_cancel_mask)
    
    def _filter_order_with_cancel_pandas(
        self,
        df: pd.DataFrame,
        price_column: str,
        qty_column: str,
        cancel_column: str,
        cancel_value: str,
    ) -> pd.DataFrame:
        """Pandas 版本：过滤包含撤单的数据"""
        is_cancel = df[cancel_column] == cancel_value
        
        # 撤单条件：只检查 Qty > 0
        cancel_valid = is_cancel & (df[qty_column] > 0)
        
        # 非撤单条件：Price > 0 AND Qty > 0
        non_cancel_valid = (
            ~is_cancel & 
            (df[price_column] > 0) & 
            (df[qty_column] > 0)
        )
        
        return df[cancel_valid | non_cancel_valid]
    
    def clean(
        self,
        df: DataFrame,
        exchange: str,
        data_type: str,
    ) -> DataFrame:
        """
        通用清洗接口
        
        Args:
            df: 输入 DataFrame
            exchange: 交易所代码 ('sh' 或 'sz')
            data_type: 数据类型 ('trade' 或 'order')
        
        Returns:
            清洗后的 DataFrame
        
        Example:
            >>> cleaner = DataCleaner()
            >>> cleaned = cleaner.clean(df, exchange='sh', data_type='trade')
        """
        exchange = exchange.lower()
        data_type = data_type.lower()
        
        method_map = {
            ("sh", "trade"): self.clean_sh_trade,
            ("sh", "order"): self.clean_sh_order,
            ("sz", "trade"): self.clean_sz_trade,
            ("sz", "order"): self.clean_sz_order,
        }
        
        key = (exchange, data_type)
        if key not in method_map:
            raise ValueError(f"不支持的交易所/数据类型组合: {exchange}/{data_type}")
        
        return method_map[key](df)


# 便捷函数
def clean_l2_data(
    df: DataFrame,
    exchange: str,
    data_type: str,
    verbose: bool = True,
) -> DataFrame:
    """
    清洗 Level2 数据的便捷函数
    
    Args:
        df: 输入 DataFrame
        exchange: 交易所代码 ('sh' 或 'sz')
        data_type: 数据类型 ('trade' 或 'order')
        verbose: 是否输出日志
    
    Returns:
        清洗后的 DataFrame
    
    Example:
        >>> from l2_image_builder.cleaner import clean_l2_data
        >>> cleaned = clean_l2_data(df, exchange='sh', data_type='trade')
    """
    cleaner = DataCleaner(verbose=verbose)
    return cleaner.clean(df, exchange, data_type)
