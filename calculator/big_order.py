"""
大单还原与阈值计算模块

Prompt 2.2 增强版本 - 母单还原与当日阈值（简化版）

功能：
1. 母单还原：通过聚合订单编号还原真实委托意图
2. 动态阈值计算：当日 Mean + N×Std（无需回溯历史）
3. 沪深分离：针对不同交易所的字段差异
4. 验证诊断：大单占比检查、阈值验证

计算公式：Threshold = Mean(V) + std_multiplier × Std(V)

优点：
- 无需回溯历史数据
- 无冷启动问题
- 计算简单，性能好
"""

from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import logging

from ..data_loader.polars_utils import (
    is_polars_df,
    groupby_sum,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    # 原有函数
    'restore_parent_orders',
    'compute_threshold',
    'compute_combined_threshold',
    'is_big_order_by_amount',
    'BigOrderCalculator',
    # Prompt 2.2 新增
    'restore_parent_orders_sh_polars',
    'restore_parent_orders_sz_polars',
    'restore_parent_orders_sh_pandas',
    'restore_parent_orders_sz_pandas',
    'compute_threshold_daily',
    'compute_all',
    # 验证和统计
    'validate_threshold',
    'compute_big_order_statistics',
    'print_big_order_summary',
]


def restore_parent_orders(
    df_trade: DataFrame,
    exchange: str = "SH",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    母单还原 - 聚合订单编号还原真实委托金额
    
    机构主力常采用拆单策略，需要通过聚合订单编号还原真实委托意图。
    
    Args:
        df_trade: 成交数据
        exchange: 交易所标识，'SH' 或 'SZ'
    
    Returns:
        (buy_parent_amount, sell_parent_amount):
            - buy_parent_amount: {买方订单号: 累计成交金额}
            - sell_parent_amount: {卖方订单号: 累计成交金额}
    
    Example:
        >>> buy_parent, sell_parent = restore_parent_orders(df_trade, 'SH')
        >>> buy_parent[12345]  # 订单号12345的累计成交金额
        500000.0
    """
    if exchange.upper() == "SH":
        return _restore_parent_orders_sh(df_trade)
    elif exchange.upper() == "SZ":
        return _restore_parent_orders_sz(df_trade)
    else:
        raise ValueError(f"未知交易所: {exchange}")


def _restore_parent_orders_sh(
    df_trade: DataFrame,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    上交所母单还原
    
    上交所成交表包含 TradeMoney 字段，直接聚合即可
    """
    buy_parent = groupby_sum(df_trade, "BuyOrderNO", "TradeMoney")
    sell_parent = groupby_sum(df_trade, "SellOrderNO", "TradeMoney")
    return buy_parent, sell_parent


def _restore_parent_orders_sz(
    df_trade: DataFrame,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    深交所母单还原
    
    深交所没有 TradeMoney 字段，需要计算 LastPx × LastQty
    """
    if is_polars_df(df_trade):
        # Polars: 先计算 TradeMoney，再聚合
        df = df_trade.with_columns([
            (pl.col("LastPx") * pl.col("LastQty")).alias("TradeMoney")
        ])
        
        buy_result = df.group_by("BidApplSeqNum").agg(
            pl.col("TradeMoney").sum()
        )
        sell_result = df.group_by("OfferApplSeqNum").agg(
            pl.col("TradeMoney").sum()
        )
        
        buy_parent = dict(zip(
            buy_result["BidApplSeqNum"].to_list(),
            buy_result["TradeMoney"].to_list(),
        ))
        sell_parent = dict(zip(
            sell_result["OfferApplSeqNum"].to_list(),
            sell_result["TradeMoney"].to_list(),
        ))
    else:
        # Pandas
        df = df_trade.copy()
        df["TradeMoney"] = df["LastPx"] * df["LastQty"]
        
        buy_parent = df.groupby("BidApplSeqNum")["TradeMoney"].sum().to_dict()
        sell_parent = df.groupby("OfferApplSeqNum")["TradeMoney"].sum().to_dict()
    
    return buy_parent, sell_parent


def compute_threshold(
    parent_amounts: Dict[int, float],
    std_multiplier: float = 1.0,
) -> float:
    """
    计算大单阈值
    
    公式: Threshold = Mean(V) + std_multiplier × Std(V)
    
    Args:
        parent_amounts: {订单号: 累计金额} 字典
        std_multiplier: 标准差倍数，默认1.0
    
    Returns:
        大单阈值
    
    Example:
        >>> threshold = compute_threshold(parent_amounts, std_multiplier=1.0)
        >>> threshold
        250000.0
    """
    if not parent_amounts:
        return float('inf')  # 无数据时，所有单都是小单
    
    amounts = np.array(list(parent_amounts.values()))
    
    if len(amounts) == 0:
        return float('inf')
    
    mean_amount = np.mean(amounts)
    std_amount = np.std(amounts)
    threshold = mean_amount + std_multiplier * std_amount
    
    return threshold


def compute_combined_threshold(
    buy_parent_amounts: Dict[int, float],
    sell_parent_amounts: Dict[int, float],
    std_multiplier: float = 1.0,
) -> float:
    """
    合并买卖双方母单计算统一阈值
    
    Args:
        buy_parent_amounts: 买方母单金额
        sell_parent_amounts: 卖方母单金额
        std_multiplier: 标准差倍数
    
    Returns:
        统一的大单阈值
    """
    # 合并所有母单金额
    all_amounts = np.concatenate([
        np.array(list(buy_parent_amounts.values())) if buy_parent_amounts else np.array([]),
        np.array(list(sell_parent_amounts.values())) if sell_parent_amounts else np.array([]),
    ])
    
    if len(all_amounts) == 0:
        return float('inf')
    
    mean_amount = np.mean(all_amounts)
    std_amount = np.std(all_amounts)
    threshold = mean_amount + std_multiplier * std_amount
    
    return threshold


class BigOrderCalculator:
    """
    大单计算器类
    
    封装母单还原和阈值计算的完整流程
    """
    
    def __init__(
        self,
        std_multiplier: float = 1.0,
    ):
        """
        Args:
            std_multiplier: 标准差倍数
        """
        self.std_multiplier = std_multiplier
        
        # 缓存
        self.buy_parent_amounts: Optional[Dict[int, float]] = None
        self.sell_parent_amounts: Optional[Dict[int, float]] = None
        self.threshold: Optional[float] = None
        self._computed_date: Optional[str] = None
    
    def compute(
        self,
        df_trade: DataFrame,
        exchange: str = "SH",
        date: Optional[str] = None,
    ) -> float:
        """
        计算大单阈值
        
        Args:
            df_trade: 成交数据
            exchange: 交易所标识
            date: 日期标识
        
        Returns:
            大单阈值
        """
        # 1. 母单还原
        self.buy_parent_amounts, self.sell_parent_amounts = restore_parent_orders(
            df_trade, exchange
        )
        
        # 2. 计算阈值
        self.threshold = compute_combined_threshold(
            self.buy_parent_amounts,
            self.sell_parent_amounts,
            self.std_multiplier,
        )
        
        self._computed_date = date
        
        return self.threshold
    
    def is_big_order(
        self,
        order_no: int,
        side: str,
    ) -> bool:
        """
        判断是否为大单
        
        Args:
            order_no: 订单号
            side: 'buy' 或 'sell'
        
        Returns:
            是否为大单
        """
        if self.threshold is None:
            raise RuntimeError("阈值尚未计算，请先调用 compute()")
        
        if side.lower() == 'buy':
            amount = self.buy_parent_amounts.get(order_no, 0)
        elif side.lower() == 'sell':
            amount = self.sell_parent_amounts.get(order_no, 0)
        else:
            raise ValueError(f"无效的 side: {side}")
        
        return amount >= self.threshold
    
    def get_order_amount(
        self,
        order_no: int,
        side: str,
    ) -> float:
        """
        获取订单的累计金额
        
        Args:
            order_no: 订单号
            side: 'buy' 或 'sell'
        
        Returns:
            累计金额，未找到返回 0
        """
        if side.lower() == 'buy':
            return self.buy_parent_amounts.get(order_no, 0) if self.buy_parent_amounts else 0
        elif side.lower() == 'sell':
            return self.sell_parent_amounts.get(order_no, 0) if self.sell_parent_amounts else 0
        else:
            raise ValueError(f"无效的 side: {side}")
    
    def clear(self):
        """清除缓存"""
        self.buy_parent_amounts = None
        self.sell_parent_amounts = None
        self.threshold = None
        self._computed_date = None
    
    def get_statistics(self) -> dict:
        """
        获取大单统计信息
        """
        if self.buy_parent_amounts is None or self.sell_parent_amounts is None:
            return {'computed': False}
        
        buy_amounts = np.array(list(self.buy_parent_amounts.values()))
        sell_amounts = np.array(list(self.sell_parent_amounts.values()))
        all_amounts = np.concatenate([buy_amounts, sell_amounts])
        
        # 计算大单数量
        big_buy_count = np.sum(buy_amounts >= self.threshold) if self.threshold else 0
        big_sell_count = np.sum(sell_amounts >= self.threshold) if self.threshold else 0
        
        return {
            'computed': True,
            'date': self._computed_date,
            'threshold': self.threshold,
            'std_multiplier': self.std_multiplier,
            'total_buy_orders': len(buy_amounts),
            'total_sell_orders': len(sell_amounts),
            'big_buy_count': int(big_buy_count),
            'big_sell_count': int(big_sell_count),
            'big_buy_ratio': big_buy_count / len(buy_amounts) if len(buy_amounts) > 0 else 0,
            'big_sell_ratio': big_sell_count / len(sell_amounts) if len(sell_amounts) > 0 else 0,
            'mean_amount': float(np.mean(all_amounts)) if len(all_amounts) > 0 else 0,
            'std_amount': float(np.std(all_amounts)) if len(all_amounts) > 0 else 0,
            'max_amount': float(np.max(all_amounts)) if len(all_amounts) > 0 else 0,
            'min_amount': float(np.min(all_amounts)) if len(all_amounts) > 0 else 0,
        }


def is_big_order_by_amount(
    amount: float,
    threshold: float,
) -> bool:
    """
    根据金额判断是否为大单
    
    Args:
        amount: 订单金额
        threshold: 大单阈值
    
    Returns:
        是否为大单
    """
    return amount >= threshold


# ==============================================================================
# Prompt 2.2 - 母单还原与当日阈值（Polars 向量化版本）
# ==============================================================================

def restore_parent_orders_sh_polars(df_trade: "pl.DataFrame") -> Tuple[Dict, Dict]:
    """
    上交所母单还原（Polars 向量化版本）
    
    特点：
    1. 直接使用 TradeMoney 字段
    2. 无需过滤撤单（成交表无撤单）
    
    Args:
        df_trade: 上交所成交表（已清洗）
    
    Returns:
        buy_parent: {BuyOrderNO: 累计成交金额}
        sell_parent: {SellOrderNO: 累计成交金额}
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 买方母单
    buy_parent = (
        df_trade
        .group_by('BuyOrderNO')
        .agg(pl.col('TradeMoney').sum().alias('amount'))
    )
    buy_dict = dict(zip(
        buy_parent['BuyOrderNO'].to_list(),
        buy_parent['amount'].to_list()
    ))
    
    # 卖方母单
    sell_parent = (
        df_trade
        .group_by('SellOrderNO')
        .agg(pl.col('TradeMoney').sum().alias('amount'))
    )
    sell_dict = dict(zip(
        sell_parent['SellOrderNO'].to_list(),
        sell_parent['amount'].to_list()
    ))
    
    logger.debug(f"SH 母单还原: 买方={len(buy_dict)}, 卖方={len(sell_dict)}")
    
    return buy_dict, sell_dict


def restore_parent_orders_sz_polars(df_trade: "pl.DataFrame") -> Tuple[Dict, Dict]:
    """
    深交所母单还原（Polars 向量化版本）
    
    特点：
    1. 只处理成交记录（ExecType='70'），排除撤单
    2. 计算 TradeMoney = LastPx × LastQty
    
    Args:
        df_trade: 深交所成交表（可包含撤单）
    
    Returns:
        buy_parent: {BidApplSeqNum: 累计成交金额}
        sell_parent: {OfferApplSeqNum: 累计成交金额}
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 过滤成交记录（排除撤单）
    df_exec = df_trade.filter(pl.col('ExecType') == '70')
    
    # 计算成交金额
    df_exec = df_exec.with_columns(
        (pl.col('LastPx') * pl.col('LastQty')).alias('TradeMoney')
    )
    
    # 买方母单
    buy_parent = (
        df_exec
        .group_by('BidApplSeqNum')
        .agg(pl.col('TradeMoney').sum().alias('amount'))
    )
    buy_dict = dict(zip(
        buy_parent['BidApplSeqNum'].to_list(),
        buy_parent['amount'].to_list()
    ))
    
    # 卖方母单
    sell_parent = (
        df_exec
        .group_by('OfferApplSeqNum')
        .agg(pl.col('TradeMoney').sum().alias('amount'))
    )
    sell_dict = dict(zip(
        sell_parent['OfferApplSeqNum'].to_list(),
        sell_parent['amount'].to_list()
    ))
    
    logger.debug(f"SZ 母单还原: 买方={len(buy_dict)}, 卖方={len(sell_dict)}")
    
    return buy_dict, sell_dict


def restore_parent_orders_sh_pandas(df_trade: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    上交所母单还原（Pandas 版本）
    """
    buy_dict = df_trade.groupby('BuyOrderNO')['TradeMoney'].sum().to_dict()
    sell_dict = df_trade.groupby('SellOrderNO')['TradeMoney'].sum().to_dict()
    return buy_dict, sell_dict


def restore_parent_orders_sz_pandas(df_trade: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    深交所母单还原（Pandas 版本）
    """
    # 过滤成交记录
    df_exec = df_trade[df_trade['ExecType'] == '70'].copy()
    
    # 计算成交金额
    df_exec['TradeMoney'] = df_exec['LastPx'] * df_exec['LastQty']
    
    buy_dict = df_exec.groupby('BidApplSeqNum')['TradeMoney'].sum().to_dict()
    sell_dict = df_exec.groupby('OfferApplSeqNum')['TradeMoney'].sum().to_dict()
    
    return buy_dict, sell_dict


def compute_threshold_daily(
    buy_parent: Dict,
    sell_parent: Dict,
    std_multiplier: float = 1.0,
) -> float:
    """
    计算当日大单阈值
    
    公式: Threshold = Mean(V) + std_multiplier × Std(V)
    
    优点:
    - 无需回溯历史数据
    - 计算简单，性能好
    - 无冷启动问题
    
    Args:
        buy_parent: 买方母单 {订单号: 金额}
        sell_parent: 卖方母单 {订单号: 金额}
        std_multiplier: 标准差倍数，默认 1.0
    
    Returns:
        大单阈值
    """
    # 合并所有母单金额
    all_amounts = np.array(
        list(buy_parent.values()) + list(sell_parent.values())
    )
    
    if len(all_amounts) == 0:
        return float('inf')
    
    # 过滤无效值
    all_amounts = all_amounts[all_amounts > 0]
    
    if len(all_amounts) == 0:
        return float('inf')
    
    mean_amount = np.mean(all_amounts)
    std_amount = np.std(all_amounts)
    threshold = mean_amount + std_multiplier * std_amount
    
    logger.debug(f"阈值计算: mean={mean_amount:.2f}, std={std_amount:.2f}, threshold={threshold:.2f}")
    
    return threshold


def compute_all(
    df_trade: DataFrame,
    exchange: str,
    std_multiplier: float = 1.0,
) -> Tuple[Dict, Dict, float]:
    """
    一次性完成母单还原和阈值计算
    
    Args:
        df_trade: 成交表
        exchange: 交易所代码 ('sh' 或 'sz')
        std_multiplier: 标准差倍数
    
    Returns:
        (buy_parent, sell_parent, threshold)
    
    Example:
        >>> buy_parent, sell_parent, threshold = compute_all(df_trade, 'sh')
        >>> print(f"阈值: {threshold:.2f}")
    """
    exchange = exchange.lower()
    use_polars = is_polars_df(df_trade)
    
    if exchange == 'sh':
        if use_polars:
            buy_parent, sell_parent = restore_parent_orders_sh_polars(df_trade)
        else:
            buy_parent, sell_parent = restore_parent_orders_sh_pandas(df_trade)
    elif exchange == 'sz':
        if use_polars:
            buy_parent, sell_parent = restore_parent_orders_sz_polars(df_trade)
        else:
            buy_parent, sell_parent = restore_parent_orders_sz_pandas(df_trade)
    else:
        raise ValueError(f"不支持的交易所: {exchange}")
    
    threshold = compute_threshold_daily(buy_parent, sell_parent, std_multiplier)
    
    return buy_parent, sell_parent, threshold


# ==============================================================================
# 验证函数
# ==============================================================================

def validate_threshold(
    threshold: float,
    buy_parent: Dict,
    sell_parent: Dict,
    expected_ratio_range: Tuple[float, float] = (0.05, 0.30),
) -> Dict[str, Union[bool, float, str]]:
    """
    验证大单阈值的合理性
    
    检查项：
    1. 阈值为正数
    2. 大单占比在合理范围内（默认 5%-30%）
    
    Args:
        threshold: 大单阈值
        buy_parent: 买方母单
        sell_parent: 卖方母单
        expected_ratio_range: 期望的大单占比范围
    
    Returns:
        验证结果字典
    """
    results = {
        'valid': True,
        'threshold': threshold,
        'issues': [],
    }
    
    # 检查阈值是否为正数
    if threshold <= 0 or np.isinf(threshold):
        results['valid'] = False
        results['issues'].append(f"阈值无效: {threshold}")
        return results
    
    # 计算大单数量
    all_amounts = np.array(
        list(buy_parent.values()) + list(sell_parent.values())
    )
    
    if len(all_amounts) == 0:
        results['valid'] = False
        results['issues'].append("无母单数据")
        return results
    
    # 过滤正数
    all_amounts = all_amounts[all_amounts > 0]
    
    big_order_count = np.sum(all_amounts >= threshold)
    total_count = len(all_amounts)
    big_order_ratio = big_order_count / total_count if total_count > 0 else 0
    
    results['big_order_count'] = int(big_order_count)
    results['total_count'] = int(total_count)
    results['big_order_ratio'] = float(big_order_ratio)
    
    # 检查大单占比是否在合理范围
    min_ratio, max_ratio = expected_ratio_range
    if big_order_ratio < min_ratio:
        results['issues'].append(
            f"大单占比过低: {big_order_ratio:.2%} < {min_ratio:.0%}"
        )
    elif big_order_ratio > max_ratio:
        results['issues'].append(
            f"大单占比过高: {big_order_ratio:.2%} > {max_ratio:.0%}"
        )
    
    if results['issues']:
        results['valid'] = False
    
    return results


def compute_big_order_statistics(
    buy_parent: Dict,
    sell_parent: Dict,
    threshold: float,
) -> Dict[str, Union[int, float]]:
    """
    计算大单统计信息
    
    Args:
        buy_parent: 买方母单
        sell_parent: 卖方母单
        threshold: 大单阈值
    
    Returns:
        统计信息字典
    """
    buy_amounts = np.array(list(buy_parent.values()))
    sell_amounts = np.array(list(sell_parent.values()))
    all_amounts = np.concatenate([buy_amounts, sell_amounts]) if len(buy_amounts) > 0 or len(sell_amounts) > 0 else np.array([])
    
    # 过滤正数
    buy_amounts = buy_amounts[buy_amounts > 0] if len(buy_amounts) > 0 else np.array([])
    sell_amounts = sell_amounts[sell_amounts > 0] if len(sell_amounts) > 0 else np.array([])
    all_amounts = all_amounts[all_amounts > 0] if len(all_amounts) > 0 else np.array([])
    
    # 大单统计
    big_buy_count = int(np.sum(buy_amounts >= threshold)) if len(buy_amounts) > 0 else 0
    big_sell_count = int(np.sum(sell_amounts >= threshold)) if len(sell_amounts) > 0 else 0
    
    # 大单金额
    big_buy_amount = float(np.sum(buy_amounts[buy_amounts >= threshold])) if len(buy_amounts) > 0 else 0
    big_sell_amount = float(np.sum(sell_amounts[sell_amounts >= threshold])) if len(sell_amounts) > 0 else 0
    
    return {
        'threshold': float(threshold),
        # 数量统计
        'total_buy_orders': len(buy_amounts),
        'total_sell_orders': len(sell_amounts),
        'total_orders': len(all_amounts),
        'big_buy_count': big_buy_count,
        'big_sell_count': big_sell_count,
        'big_order_count': big_buy_count + big_sell_count,
        # 占比
        'big_buy_ratio': big_buy_count / len(buy_amounts) if len(buy_amounts) > 0 else 0,
        'big_sell_ratio': big_sell_count / len(sell_amounts) if len(sell_amounts) > 0 else 0,
        'big_order_ratio': (big_buy_count + big_sell_count) / len(all_amounts) if len(all_amounts) > 0 else 0,
        # 金额统计
        'total_buy_amount': float(np.sum(buy_amounts)) if len(buy_amounts) > 0 else 0,
        'total_sell_amount': float(np.sum(sell_amounts)) if len(sell_amounts) > 0 else 0,
        'big_buy_amount': big_buy_amount,
        'big_sell_amount': big_sell_amount,
        # 分布统计
        'mean_amount': float(np.mean(all_amounts)) if len(all_amounts) > 0 else 0,
        'std_amount': float(np.std(all_amounts)) if len(all_amounts) > 0 else 0,
        'median_amount': float(np.median(all_amounts)) if len(all_amounts) > 0 else 0,
        'max_amount': float(np.max(all_amounts)) if len(all_amounts) > 0 else 0,
        'min_amount': float(np.min(all_amounts)) if len(all_amounts) > 0 else 0,
    }


def print_big_order_summary(
    buy_parent: Dict,
    sell_parent: Dict,
    threshold: float,
    stock_code: str = "",
    date: str = "",
) -> None:
    """
    打印大单摘要信息
    
    Args:
        buy_parent: 买方母单
        sell_parent: 卖方母单
        threshold: 大单阈值
        stock_code: 股票代码（可选）
        date: 日期（可选）
    """
    stats = compute_big_order_statistics(buy_parent, sell_parent, threshold)
    
    header = "大单统计摘要"
    if stock_code:
        header += f" [{stock_code}]"
    if date:
        header += f" - {date}"
    
    print(f"\n{'='*60}")
    print(f"{header}")
    print(f"{'='*60}")
    
    print(f"\n📊 阈值信息:")
    print(f"  大单阈值: {stats['threshold']:,.2f}")
    print(f"  均值: {stats['mean_amount']:,.2f}")
    print(f"  标准差: {stats['std_amount']:,.2f}")
    
    print(f"\n📊 数量统计:")
    print(f"  总母单数: {stats['total_orders']:,}")
    print(f"  大单数: {stats['big_order_count']:,} ({stats['big_order_ratio']:.2%})")
    print(f"  - 大买单: {stats['big_buy_count']:,} ({stats['big_buy_ratio']:.2%})")
    print(f"  - 大卖单: {stats['big_sell_count']:,} ({stats['big_sell_ratio']:.2%})")
    
    print(f"\n📊 金额统计:")
    print(f"  总成交金额: {stats['total_buy_amount'] + stats['total_sell_amount']:,.2f}")
    print(f"  大单金额: {stats['big_buy_amount'] + stats['big_sell_amount']:,.2f}")
    print(f"  最大单笔: {stats['max_amount']:,.2f}")
    print(f"  最小单笔: {stats['min_amount']:,.2f}")
    
    # 验证
    validation = validate_threshold(threshold, buy_parent, sell_parent)
    print(f"\n✅ 验证结果:")
    print(f"  有效: {'✓' if validation['valid'] else '✗'}")
    if not validation['valid']:
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    print(f"{'='*60}\n")
