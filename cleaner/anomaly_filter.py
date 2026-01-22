"""
异常值过滤模块

过滤价格为0、数量为0等异常数据

增强功能（Prompt 1.3）：
- Polars 向量化过滤
- 支持 Pandas 兼容
- 区分撤单和非撤单的过滤逻辑
- 提供数据质量验证工具
"""

from typing import List, Optional
from ..data_loader.polars_utils import (
    is_polars_df,
    filter_positive,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd


def filter_zero_price(
    df: DataFrame,
    price_column: str = "Price",
) -> DataFrame:
    """
    过滤价格为0的记录
    
    Args:
        df: 输入 DataFrame
        price_column: 价格列名
    """
    if is_polars_df(df):
        return df.filter(pl.col(price_column) > 0)
    else:
        return df[df[price_column] > 0]


def filter_zero_qty(
    df: DataFrame,
    qty_column: str = "Qty",
) -> DataFrame:
    """
    过滤数量为0的记录
    
    Args:
        df: 输入 DataFrame
        qty_column: 数量列名
    """
    if is_polars_df(df):
        return df.filter(pl.col(qty_column) > 0)
    else:
        return df[df[qty_column] > 0]


def filter_anomalies(
    df: DataFrame,
    price_column: Optional[str] = "Price",
    qty_column: Optional[str] = "Qty",
    is_cancel: bool = False,
) -> DataFrame:
    """
    过滤异常记录
    
    Args:
        df: 输入 DataFrame
        price_column: 价格列名，None 表示不检查价格
        qty_column: 数量列名，None 表示不检查数量
        is_cancel: 是否为撤单数据（撤单价格可能为0，不检查）
    
    Returns:
        过滤后的 DataFrame
    """
    columns_to_check = []
    
    # 撤单记录只检查数量，不检查价格
    if not is_cancel and price_column:
        columns_to_check.append(price_column)
    
    if qty_column:
        columns_to_check.append(qty_column)
    
    if not columns_to_check:
        return df
    
    return filter_positive(df, columns_to_check)


def filter_anomalies_polars(
    df: "pl.DataFrame",
    price_column: str = "Price",
    qty_column: str = "Qty",
    is_cancel: bool = False,
) -> "pl.DataFrame":
    """
    过滤异常值（Polars 向量化版本）
    
    Args:
        df: Polars DataFrame
        price_column: 价格列名
        qty_column: 数量列名
        is_cancel: 是否为撤单（撤单只检查数量）
    
    Returns:
        过滤后的 DataFrame
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    if is_cancel:
        # 撤单只检查数量
        return df.filter(pl.col(qty_column) > 0)
    else:
        # 非撤单检查价格和数量
        return df.filter(
            (pl.col(price_column) > 0) & 
            (pl.col(qty_column) > 0)
        )


def filter_anomalies_pandas(
    df: pd.DataFrame,
    price_column: str = "Price",
    qty_column: str = "Qty",
    is_cancel: bool = False,
) -> pd.DataFrame:
    """
    过滤异常值（Pandas 向量化版本）
    
    Args:
        df: Pandas DataFrame
        price_column: 价格列名
        qty_column: 数量列名
        is_cancel: 是否为撤单
    
    Returns:
        过滤后的 DataFrame
    """
    if is_cancel:
        # 撤单只检查数量
        return df[df[qty_column] > 0]
    else:
        # 非撤单检查价格和数量
        mask = (df[price_column] > 0) & (df[qty_column] > 0)
        return df[mask]


def filter_anomalies_auto(
    df: DataFrame,
    price_column: str = "Price",
    qty_column: str = "Qty",
    is_cancel: bool = False,
) -> DataFrame:
    """
    自动选择引擎过滤异常值
    """
    if is_polars_df(df):
        return filter_anomalies_polars(df, price_column, qty_column, is_cancel)
    else:
        return filter_anomalies_pandas(df, price_column, qty_column, is_cancel)


class AnomalyFilter:
    """
    异常值过滤器类
    
    支持上交所和深交所的不同字段名
    """
    
    # 上交所字段映射
    SH_COLUMNS = {
        'trade_price': 'Price',
        'trade_qty': 'Qty',
        'order_price': 'Price',
        'order_qty': 'Qty',
    }
    
    # 深交所字段映射
    SZ_COLUMNS = {
        'trade_price': 'LastPx',
        'trade_qty': 'LastQty',
        'order_price': 'Price',
        'order_qty': 'OrderQty',
    }
    
    def __init__(self, exchange: str = "SH"):
        """
        Args:
            exchange: 交易所标识，'SH' 或 'SZ'
        """
        self.exchange = exchange.upper()
        self.columns = self.SH_COLUMNS if self.exchange == "SH" else self.SZ_COLUMNS
    
    def filter_trade(
        self,
        df: DataFrame,
        check_price: bool = True,
        check_qty: bool = True,
    ) -> DataFrame:
        """
        过滤成交数据异常值
        
        Args:
            df: 成交数据
            check_price: 是否检查价格
            check_qty: 是否检查数量
        """
        columns_to_check = []
        
        if check_price:
            columns_to_check.append(self.columns['trade_price'])
        if check_qty:
            columns_to_check.append(self.columns['trade_qty'])
        
        if not columns_to_check:
            return df
        
        return filter_positive(df, columns_to_check)
    
    def filter_order(
        self,
        df: DataFrame,
        check_price: bool = True,
        check_qty: bool = True,
    ) -> DataFrame:
        """
        过滤委托数据异常值
        """
        columns_to_check = []
        
        if check_price:
            columns_to_check.append(self.columns['order_price'])
        if check_qty:
            columns_to_check.append(self.columns['order_qty'])
        
        if not columns_to_check:
            return df
        
        return filter_positive(df, columns_to_check)
    
    def filter_cancel(
        self,
        df: DataFrame,
    ) -> DataFrame:
        """
        过滤撤单数据异常值
        
        注意：撤单的价格可能为0（深交所），只检查数量
        """
        qty_col = self.columns['trade_qty']
        
        if is_polars_df(df):
            return df.filter(pl.col(qty_col) > 0)
        else:
            return df[df[qty_col] > 0]


def validate_data_quality(
    df: DataFrame,
    price_column: str,
    qty_column: str,
) -> dict:
    """
    验证数据质量，返回统计信息
    
    Args:
        df: DataFrame
        price_column: 价格列名
        qty_column: 数量列名
    
    Returns:
        包含统计信息的字典
    """
    if is_polars_df(df):
        total = df.height
        zero_price = df.filter(pl.col(price_column) <= 0).height
        zero_qty = df.filter(pl.col(qty_column) <= 0).height
        null_price = df.filter(pl.col(price_column).is_null()).height
        null_qty = df.filter(pl.col(qty_column).is_null()).height
    else:
        total = len(df)
        zero_price = (df[price_column] <= 0).sum()
        zero_qty = (df[qty_column] <= 0).sum()
        null_price = df[price_column].isna().sum()
        null_qty = df[qty_column].isna().sum()
    
    return {
        'total_rows': total,
        'zero_price_count': zero_price,
        'zero_price_ratio': zero_price / total if total > 0 else 0,
        'zero_qty_count': zero_qty,
        'zero_qty_ratio': zero_qty / total if total > 0 else 0,
        'null_price_count': null_price,
        'null_qty_count': null_qty,
    }
