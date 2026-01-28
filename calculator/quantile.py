"""
分位数计算模块

Prompt 2.1 增强版本 - Polars 向量化分位数计算

计算价格和量的分位数边界，用于将连续值映射到离散的 bin

功能：
1. 联合计算：成交数据 + 委托数据合并后统一计算分位数
2. 沪深分离：针对上交所/深交所的字段差异提供专用函数
3. 撤单过滤：上交所过滤 OrdType='Cancel'，深交所过滤 ExecType='52'
4. 验证诊断：提供分布验证和可视化功能
"""

from typing import Tuple, Optional, Union, List, Dict
import numpy as np
import logging

from ..data_loader.polars_utils import (
    is_polars_df,
    compute_percentiles,
    concat_series,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


def compute_quantile_bins(
    df_trade: DataFrame,
    df_order: DataFrame,
    percentiles: Tuple[float, ...] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5),
    price_col_trade: str = "Price",
    price_col_order: str = "Price",
    qty_col_trade: str = "Qty",
    qty_col_order: str = "OrderQty",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算价格和量的分位数边界（联合计算）
    
    将成交数据和委托数据合并后统一计算分位数边界，保留绝对体量差异信息。
    
    Args:
        df_trade: 成交数据
        df_order: 委托数据
        percentiles: 分位数百分比，7个切割点定义8个区间
        price_col_trade: 成交数据的价格列名
        price_col_order: 委托数据的价格列名
        qty_col_trade: 成交数据的数量列名
        qty_col_order: 委托数据的数量列名
    
    Returns:
        (price_bins, qty_bins): 
            - price_bins: 7个价格切割点
            - qty_bins: 7个量切割点
    
    Example:
        >>> price_bins, qty_bins = compute_quantile_bins(df_trade, df_order)
        >>> price_bins
        array([10.5, 11.2, 12.0, 12.8, 13.5, 14.2, 15.0])
    """
    use_polars = is_polars_df(df_trade) and is_polars_df(df_order)
    
    if use_polars:
        # Polars 方式
        all_prices = pl.concat([
            df_trade[price_col_trade],
            df_order[price_col_order],
        ])
        all_volumes = pl.concat([
            df_trade[qty_col_trade],
            df_order[qty_col_order],
        ])
    else:
        # Pandas 方式
        all_prices = pd.concat([
            df_trade[price_col_trade],
            df_order[price_col_order],
        ], ignore_index=True)
        all_volumes = pd.concat([
            df_trade[qty_col_trade],
            df_order[qty_col_order],
        ], ignore_index=True)
    
    # 计算分位数边界
    price_bins = compute_percentiles(all_prices, percentiles)
    qty_bins = compute_percentiles(all_volumes, percentiles)
    
    return price_bins, qty_bins


def get_bin_index(value: float, bins: np.ndarray) -> int:
    """
    根据值获取所属的 bin 索引 (0-7)
    
    Args:
        value: 要查找的值
        bins: 7个切割点数组
    
    Returns:
        bin 索引，范围 0-7
    
    Example:
        >>> bins = np.array([10, 20, 30, 40, 50, 60, 70])
        >>> get_bin_index(5, bins)    # 小于第一个切割点
        0
        >>> get_bin_index(25, bins)   # 在20-30之间
        2
        >>> get_bin_index(100, bins)  # 大于最后一个切割点
        7
    """
    idx = int(np.digitize(value, bins))
    return min(idx, 7)  # 确保不超过7


def get_bin_indices_vectorized(
    values: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    向量化计算 bin 索引
    
    Args:
        values: 值数组
        bins: 7个切割点数组
    
    Returns:
        bin 索引数组，值范围 0-7
    """
    indices = np.digitize(values, bins)
    return np.clip(indices, 0, 7)


class QuantileCalculator:
    """
    分位数计算器类
    
    支持缓存分位数边界，提高批量处理效率
    """
    
    def __init__(
        self,
        percentiles: Tuple[float, ...] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5),
    ):
        """
        Args:
            percentiles: 分位数百分比
        """
        self.percentiles = percentiles
        self.price_bins: Optional[np.ndarray] = None
        self.qty_bins: Optional[np.ndarray] = None
        self._computed_date: Optional[str] = None
    
    def compute(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        date: Optional[str] = None,
        price_col_trade: str = "Price",
        price_col_order: str = "Price",
        qty_col_trade: str = "Qty",
        qty_col_order: str = "OrderQty",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算并缓存分位数边界
        
        Args:
            df_trade: 成交数据
            df_order: 委托数据
            date: 日期标识（用于缓存管理）
            price_col_trade: 成交价格列名
            price_col_order: 委托价格列名
            qty_col_trade: 成交数量列名
            qty_col_order: 委托数量列名
        
        Returns:
            (price_bins, qty_bins)
        """
        self.price_bins, self.qty_bins = compute_quantile_bins(
            df_trade, df_order,
            self.percentiles,
            price_col_trade, price_col_order,
            qty_col_trade, qty_col_order,
        )
        self._computed_date = date
        
        return self.price_bins, self.qty_bins
    
    def compute_for_sz(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        date: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        为深交所数据计算分位数
        
        R3.2 更新: 使用归一化后的标准列名 (Price, Qty)
        """
        return self.compute(
            df_trade, df_order, date,
            price_col_trade="Price",    # R3.2: 原 LastPx -> Price
            price_col_order="Price",
            qty_col_trade="Qty",        # R3.2: 原 LastQty -> Qty
            qty_col_order="Qty",        # R3.2: 原 OrderQty -> Qty
        )
    
    def get_price_bin(self, price: float) -> int:
        """获取价格的 bin 索引"""
        if self.price_bins is None:
            raise RuntimeError("分位数尚未计算，请先调用 compute()")
        return get_bin_index(price, self.price_bins)
    
    def get_qty_bin(self, qty: float) -> int:
        """获取数量的 bin 索引"""
        if self.qty_bins is None:
            raise RuntimeError("分位数尚未计算，请先调用 compute()")
        return get_bin_index(qty, self.qty_bins)
    
    def get_bins(
        self,
        price: float,
        qty: float,
    ) -> Tuple[int, int]:
        """
        同时获取价格和数量的 bin 索引
        
        Returns:
            (price_bin, qty_bin)
        """
        return self.get_price_bin(price), self.get_qty_bin(qty)
    
    def get_bins_vectorized(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化获取 bin 索引
        
        Args:
            prices: 价格数组
            quantities: 数量数组
        
        Returns:
            (price_bins, qty_bins) 索引数组
        """
        if self.price_bins is None or self.qty_bins is None:
            raise RuntimeError("分位数尚未计算，请先调用 compute()")
        
        price_indices = get_bin_indices_vectorized(prices, self.price_bins)
        qty_indices = get_bin_indices_vectorized(quantities, self.qty_bins)
        
        return price_indices, qty_indices
    
    def clear(self):
        """清除缓存"""
        self.price_bins = None
        self.qty_bins = None
        self._computed_date = None
    
    def get_bin_info(self) -> dict:
        """
        获取分位数信息（用于诊断）
        """
        return {
            'percentiles': self.percentiles,
            'price_bins': self.price_bins.tolist() if self.price_bins is not None else None,
            'qty_bins': self.qty_bins.tolist() if self.qty_bins is not None else None,
            'computed_date': self._computed_date,
        }


def compute_quantile_bins_single_source(
    df: DataFrame,
    price_col: str = "Price",
    qty_col: str = "Qty",
    percentiles: Tuple[float, ...] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单一数据源计算分位数边界
    
    适用于只有成交数据或只有委托数据的场景
    """
    if is_polars_df(df):
        prices = df[price_col].to_numpy()
        quantities = df[qty_col].to_numpy()
    else:
        prices = df[price_col].values
        quantities = df[qty_col].values
    
    price_bins = np.percentile(prices[~np.isnan(prices)], percentiles)
    qty_bins = np.percentile(quantities[~np.isnan(quantities)], percentiles)
    
    return price_bins, qty_bins


# ==============================================================================
# Prompt 2.1 - Polars 向量化分位数计算
# ==============================================================================

def compute_quantile_bins_sh_polars(
    df_trade: "pl.DataFrame",
    df_order: "pl.DataFrame",
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    上交所分位数计算（Polars 向量化版本）
    
    特点：
    1. 只取新增委托（OrdType='New'），排除撤单
    2. 联合计算成交价格/数量和委托价格/数量
    
    Args:
        df_trade: 成交表（已清洗）
        df_order: 委托表（包含 OrdType 列）
        percentiles: 分位数百分比，默认 7 个切割点
    
    Returns:
        (price_bins, qty_bins): 各 7 个切割点的数组
    
    Example:
        >>> price_bins, qty_bins = compute_quantile_bins_sh_polars(df_trade, df_order)
        >>> print(f"价格分位数: {price_bins}")
        >>> print(f"数量分位数: {qty_bins}")
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用，请安装 polars 或使用 Pandas 版本")
    
    # 只取新增委托（不含撤单）
    df_order_new = df_order.filter(pl.col('OrdType') == 'New')
    
    # 联合价格
    all_prices = pl.concat([
        df_trade.select('Price'),
        df_order_new.select('Price')
    ])['Price'].drop_nulls().to_numpy()
    
    # 联合数量
    all_volumes = pl.concat([
        df_trade.select('Qty'),
        df_order_new.select('Qty')
    ])['Qty'].drop_nulls().to_numpy()
    
    # 过滤无效值
    all_prices = all_prices[all_prices > 0]
    all_volumes = all_volumes[all_volumes > 0]
    
    # 计算分位数
    price_bins = np.percentile(all_prices, percentiles)
    qty_bins = np.percentile(all_volumes, percentiles)
    
    logger.debug(f"SH 分位数计算: 价格样本={len(all_prices)}, 数量样本={len(all_volumes)}")
    
    return price_bins, qty_bins


def compute_quantile_bins_sz_polars(
    df_trade: "pl.DataFrame",
    df_order: "pl.DataFrame",
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    深交所分位数计算（Polars 向量化版本）
    
    R3.2 更新: 使用归一化后的标准列名 (Price, Qty)
    
    特点：
    1. 只取成交记录（ExecType='70'），排除撤单
    2. R3.2 后使用统一的标准列名 Price/Qty
    
    Args:
        df_trade: 成交表（包含 ExecType 列）
        df_order: 委托表
        percentiles: 分位数百分比
    
    Returns:
        (price_bins, qty_bins)
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 只取成交记录（排除撤单）
    df_trade_exec = df_trade.filter(pl.col('ExecType') == '70')
    
    # 联合价格 - R3.2: 使用标准列名 Price
    all_prices = pl.concat([
        df_trade_exec.select('Price'),  # R3.2: 原 LastPx -> Price
        df_order.select('Price')
    ])['Price'].drop_nulls().to_numpy()
    
    # 联合数量 - R3.2: 使用标准列名 Qty
    all_volumes = pl.concat([
        df_trade_exec.select('Qty'),  # R3.2: 原 LastQty -> Qty
        df_order.select('Qty')        # R3.2: 原 OrderQty -> Qty
    ])['Qty'].drop_nulls().to_numpy()
    
    # 过滤无效值
    all_prices = all_prices[all_prices > 0]
    all_volumes = all_volumes[all_volumes > 0]
    
    # 计算分位数
    price_bins = np.percentile(all_prices, percentiles)
    qty_bins = np.percentile(all_volumes, percentiles)
    
    logger.debug(f"SZ 分位数计算: 价格样本={len(all_prices)}, 数量样本={len(all_volumes)}")
    
    return price_bins, qty_bins


def compute_quantile_bins_sh_pandas(
    df_trade: pd.DataFrame,
    df_order: pd.DataFrame,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    上交所分位数计算（Pandas 版本）
    """
    # 只取新增委托
    df_order_new = df_order[df_order['OrdType'] == 'New']
    
    # 联合价格
    all_prices = pd.concat([
        df_trade['Price'],
        df_order_new['Price']
    ], ignore_index=True).dropna().values
    
    # 联合数量
    all_volumes = pd.concat([
        df_trade['Qty'],
        df_order_new['Qty']
    ], ignore_index=True).dropna().values
    
    # 过滤无效值
    all_prices = all_prices[all_prices > 0]
    all_volumes = all_volumes[all_volumes > 0]
    
    price_bins = np.percentile(all_prices, percentiles)
    qty_bins = np.percentile(all_volumes, percentiles)
    
    return price_bins, qty_bins


def compute_quantile_bins_sz_pandas(
    df_trade: pd.DataFrame,
    df_order: pd.DataFrame,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    深交所分位数计算（Pandas 版本）
    
    R3.2 更新: 使用归一化后的标准列名 (Price, Qty)
    """
    # 只取成交记录
    df_trade_exec = df_trade[df_trade['ExecType'] == '70']
    
    # 联合价格 - R3.2: 使用标准列名 Price
    all_prices = pd.concat([
        df_trade_exec['Price'],  # R3.2: 原 LastPx -> Price
        df_order['Price']
    ], ignore_index=True).dropna().values
    
    # 联合数量 - R3.2: 使用标准列名 Qty
    all_volumes = pd.concat([
        df_trade_exec['Qty'],    # R3.2: 原 LastQty -> Qty
        df_order['Qty']          # R3.2: 原 OrderQty -> Qty
    ], ignore_index=True).dropna().values
    
    # 过滤无效值
    all_prices = all_prices[all_prices > 0]
    all_volumes = all_volumes[all_volumes > 0]
    
    price_bins = np.percentile(all_prices, percentiles)
    qty_bins = np.percentile(all_volumes, percentiles)
    
    return price_bins, qty_bins


# ==============================================================================
# 分离分位数计算（成交/委托独立计算）
# ==============================================================================

def _compute_percentiles_safe(
    values: np.ndarray,
    percentiles: List[float],
) -> np.ndarray:
    """
    安全计算分位数，处理空数据情况
    
    Args:
        values: 值数组
        percentiles: 分位数百分比列表
    
    Returns:
        分位数边界数组，若数据为空返回全 0 数组
    """
    if len(values) == 0:
        logger.warning("分位数计算: 数据为空，返回兜底值 [0, 0, ...]")
        return np.zeros(len(percentiles))
    
    # 过滤无效值
    valid = values[(~np.isnan(values)) & (values > 0)]
    if len(valid) == 0:
        logger.warning("分位数计算: 无有效数据（全为NaN或<=0），返回兜底值")
        return np.zeros(len(percentiles))
    
    return np.percentile(valid, percentiles)


def compute_separate_quantile_bins_sh_polars(
    df_trade: "pl.DataFrame",
    df_order: "pl.DataFrame",
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    上交所分离分位数计算（Polars 版本）
    
    成交和委托分别独立计算分位数，保留各自的分布特征。
    
    Args:
        df_trade: 成交表（已清洗）
        df_order: 委托表（包含 OrdType 列）
        percentiles: 分位数百分比
    
    Returns:
        (trade_price_bins, trade_qty_bins, order_price_bins, order_qty_bins)
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 1. 成交分位数
    trade_prices = df_trade['Price'].drop_nulls().to_numpy()
    trade_qtys = df_trade['Qty'].drop_nulls().to_numpy()
    
    tp_bins = _compute_percentiles_safe(trade_prices, percentiles)
    tq_bins = _compute_percentiles_safe(trade_qtys, percentiles)
    
    # 2. 委托分位数（只取新增委托，排除撤单）
    df_order_new = df_order.filter(pl.col('OrdType') == 'New')
    order_prices = df_order_new['Price'].drop_nulls().to_numpy()
    order_qtys = df_order_new['Qty'].drop_nulls().to_numpy()
    
    op_bins = _compute_percentiles_safe(order_prices, percentiles)
    oq_bins = _compute_percentiles_safe(order_qtys, percentiles)
    
    logger.debug(
        f"SH 分离分位数: Trade 样本=({len(trade_prices)}, {len(trade_qtys)}), "
        f"Order 样本=({len(order_prices)}, {len(order_qtys)})"
    )
    
    return tp_bins, tq_bins, op_bins, oq_bins


def compute_separate_quantile_bins_sh_pandas(
    df_trade: pd.DataFrame,
    df_order: pd.DataFrame,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    上交所分离分位数计算（Pandas 版本）
    """
    # 1. 成交分位数
    trade_prices = df_trade['Price'].dropna().values
    trade_qtys = df_trade['Qty'].dropna().values
    
    tp_bins = _compute_percentiles_safe(trade_prices, percentiles)
    tq_bins = _compute_percentiles_safe(trade_qtys, percentiles)
    
    # 2. 委托分位数
    df_order_new = df_order[df_order['OrdType'] == 'New']
    order_prices = df_order_new['Price'].dropna().values
    order_qtys = df_order_new['Qty'].dropna().values
    
    op_bins = _compute_percentiles_safe(order_prices, percentiles)
    oq_bins = _compute_percentiles_safe(order_qtys, percentiles)
    
    return tp_bins, tq_bins, op_bins, oq_bins


def compute_separate_quantile_bins_sz_polars(
    df_trade: "pl.DataFrame",
    df_order: "pl.DataFrame",
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    深交所分离分位数计算（Polars 版本）
    
    成交和委托分别独立计算分位数。
    R3.2: 使用标准列名 Price/Qty
    
    Args:
        df_trade: 成交表（包含 ExecType 列）
        df_order: 委托表
        percentiles: 分位数百分比
    
    Returns:
        (trade_price_bins, trade_qty_bins, order_price_bins, order_qty_bins)
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 1. 成交分位数（只取成交记录，排除撤单）
    df_trade_exec = df_trade.filter(pl.col('ExecType') == '70')
    trade_prices = df_trade_exec['Price'].drop_nulls().to_numpy()
    trade_qtys = df_trade_exec['Qty'].drop_nulls().to_numpy()
    
    tp_bins = _compute_percentiles_safe(trade_prices, percentiles)
    tq_bins = _compute_percentiles_safe(trade_qtys, percentiles)
    
    # 2. 委托分位数
    order_prices = df_order['Price'].drop_nulls().to_numpy()
    order_qtys = df_order['Qty'].drop_nulls().to_numpy()
    
    op_bins = _compute_percentiles_safe(order_prices, percentiles)
    oq_bins = _compute_percentiles_safe(order_qtys, percentiles)
    
    logger.debug(
        f"SZ 分离分位数: Trade 样本=({len(trade_prices)}, {len(trade_qtys)}), "
        f"Order 样本=({len(order_prices)}, {len(order_qtys)})"
    )
    
    return tp_bins, tq_bins, op_bins, oq_bins


def compute_separate_quantile_bins_sz_pandas(
    df_trade: pd.DataFrame,
    df_order: pd.DataFrame,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    深交所分离分位数计算（Pandas 版本）
    """
    # 1. 成交分位数
    df_trade_exec = df_trade[df_trade['ExecType'] == '70']
    trade_prices = df_trade_exec['Price'].dropna().values
    trade_qtys = df_trade_exec['Qty'].dropna().values
    
    tp_bins = _compute_percentiles_safe(trade_prices, percentiles)
    tq_bins = _compute_percentiles_safe(trade_qtys, percentiles)
    
    # 2. 委托分位数
    order_prices = df_order['Price'].dropna().values
    order_qtys = df_order['Qty'].dropna().values
    
    op_bins = _compute_percentiles_safe(order_prices, percentiles)
    oq_bins = _compute_percentiles_safe(order_qtys, percentiles)
    
    return tp_bins, tq_bins, op_bins, oq_bins


def compute_separate_quantile_bins_auto(
    df_trade: DataFrame,
    df_order: DataFrame,
    exchange: str,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    自动选择引擎的分离分位数计算
    
    Args:
        df_trade: 成交表
        df_order: 委托表
        exchange: 交易所代码 ('sh' 或 'sz')
        percentiles: 分位数百分比
    
    Returns:
        (trade_price_bins, trade_qty_bins, order_price_bins, order_qty_bins)
    """
    exchange = exchange.lower()
    use_polars = is_polars_df(df_trade) and is_polars_df(df_order)
    
    if exchange == 'sh':
        if use_polars:
            return compute_separate_quantile_bins_sh_polars(df_trade, df_order, percentiles)
        else:
            return compute_separate_quantile_bins_sh_pandas(df_trade, df_order, percentiles)
    elif exchange == 'sz':
        if use_polars:
            return compute_separate_quantile_bins_sz_polars(df_trade, df_order, percentiles)
        else:
            return compute_separate_quantile_bins_sz_pandas(df_trade, df_order, percentiles)
    else:
        raise ValueError(f"不支持的交易所: {exchange}")


def compute_quantile_bins_auto(
    df_trade: DataFrame,
    df_order: DataFrame,
    exchange: str,
    percentiles: List[float] = [12.5, 25, 37.5, 50, 62.5, 75, 87.5],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    自动选择引擎的分位数计算
    
    Args:
        df_trade: 成交表
        df_order: 委托表
        exchange: 交易所代码 ('sh' 或 'sz')
        percentiles: 分位数百分比
    
    Returns:
        (price_bins, qty_bins)
    """
    exchange = exchange.lower()
    use_polars = is_polars_df(df_trade) and is_polars_df(df_order)
    
    if exchange == 'sh':
        if use_polars:
            return compute_quantile_bins_sh_polars(df_trade, df_order, percentiles)
        else:
            return compute_quantile_bins_sh_pandas(df_trade, df_order, percentiles)
    elif exchange == 'sz':
        if use_polars:
            return compute_quantile_bins_sz_polars(df_trade, df_order, percentiles)
        else:
            return compute_quantile_bins_sz_pandas(df_trade, df_order, percentiles)
    else:
        raise ValueError(f"不支持的交易所: {exchange}")


# ==============================================================================
# 验证和诊断函数
# ==============================================================================

def validate_quantile_bins(
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
) -> Dict[str, bool]:
    """
    验证分位数边界的有效性
    
    检查项：
    1. 长度正确（7个切割点）
    2. 单调递增
    3. 无 NaN/Inf
    4. 值为正数
    
    Returns:
        验证结果字典
    """
    results = {
        'price_valid': True,
        'qty_valid': True,
        'price_issues': [],
        'qty_issues': [],
    }
    
    # 检查价格分位数
    if len(price_bins) != 7:
        results['price_valid'] = False
        results['price_issues'].append(f"长度错误: {len(price_bins)} != 7")
    
    if not np.all(np.diff(price_bins) >= 0):
        results['price_valid'] = False
        results['price_issues'].append("非单调递增")
    
    if np.any(np.isnan(price_bins)) or np.any(np.isinf(price_bins)):
        results['price_valid'] = False
        results['price_issues'].append("包含 NaN 或 Inf")
    
    if np.any(price_bins <= 0):
        results['price_valid'] = False
        results['price_issues'].append("包含非正数")
    
    # 检查数量分位数
    if len(qty_bins) != 7:
        results['qty_valid'] = False
        results['qty_issues'].append(f"长度错误: {len(qty_bins)} != 7")
    
    if not np.all(np.diff(qty_bins) >= 0):
        results['qty_valid'] = False
        results['qty_issues'].append("非单调递增")
    
    if np.any(np.isnan(qty_bins)) or np.any(np.isinf(qty_bins)):
        results['qty_valid'] = False
        results['qty_issues'].append("包含 NaN 或 Inf")
    
    if np.any(qty_bins <= 0):
        results['qty_valid'] = False
        results['qty_issues'].append("包含非正数")
    
    return results


def compute_bin_distribution(
    values: np.ndarray,
    bins: np.ndarray,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    计算值在各 bin 中的分布
    
    Args:
        values: 值数组
        bins: 7 个切割点
    
    Returns:
        分布统计信息
    """
    indices = np.clip(np.digitize(values, bins), 0, 7)
    
    # 统计各 bin 的数量
    bin_counts = np.bincount(indices, minlength=8)
    bin_percentages = bin_counts / len(values) * 100
    
    return {
        'bin_counts': bin_counts,
        'bin_percentages': bin_percentages,
        'total_count': len(values),
        'bins': bins,
        'min_value': np.min(values),
        'max_value': np.max(values),
        'mean_value': np.mean(values),
        'std_value': np.std(values),
    }


def visualize_quantile_distribution(
    values: np.ndarray,
    bins: np.ndarray,
    title: str = "分位数分布",
    ax=None,
) -> None:
    """
    可视化分位数分布（需要 matplotlib）
    
    Args:
        values: 值数组
        bins: 7 个切割点
        title: 图表标题
        ax: matplotlib axes（可选）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
        return
    
    dist = compute_bin_distribution(values, bins)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    bin_labels = [f"Bin {i}" for i in range(8)]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 8))
    
    ax.bar(bin_labels, dist['bin_counts'], color=colors, edgecolor='black')
    
    # 添加百分比标签
    for i, (count, pct) in enumerate(zip(dist['bin_counts'], dist['bin_percentages'])):
        ax.text(i, count + max(dist['bin_counts']) * 0.02, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # 添加分位数边界信息
    info_text = f"Bins: [{', '.join([f'{b:.2f}' for b in bins])}]"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return ax


def print_quantile_summary(
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
    stock_code: str = "",
    date: str = "",
) -> None:
    """
    打印分位数摘要信息
    
    Args:
        price_bins: 价格分位数边界
        qty_bins: 数量分位数边界
        stock_code: 股票代码（可选）
        date: 日期（可选）
    """
    header = "分位数摘要"
    if stock_code:
        header += f" [{stock_code}]"
    if date:
        header += f" - {date}"
    
    print(f"\n{'='*60}")
    print(f"{header}")
    print(f"{'='*60}")
    
    # 价格分位数
    print("\n📊 价格分位数边界:")
    percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    for i, (pct, val) in enumerate(zip(percentiles, price_bins)):
        print(f"  P{pct:5.1f}%: {val:12.4f}")
    
    # 数量分位数
    print("\n📊 数量分位数边界:")
    for i, (pct, val) in enumerate(zip(percentiles, qty_bins)):
        print(f"  P{pct:5.1f}%: {val:12.0f}")
    
    # 验证
    validation = validate_quantile_bins(price_bins, qty_bins)
    print(f"\n✅ 验证结果:")
    print(f"  价格有效: {'✓' if validation['price_valid'] else '✗'}")
    if not validation['price_valid']:
        for issue in validation['price_issues']:
            print(f"    - {issue}")
    print(f"  数量有效: {'✓' if validation['qty_valid'] else '✗'}")
    if not validation['qty_valid']:
        for issue in validation['qty_issues']:
            print(f"    - {issue}")
    
    print(f"{'='*60}\n")
