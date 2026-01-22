"""
深交所撤单价格关联模块

Prompt 2.3 增强版本 - 深交所撤单价格关联

问题背景：
深交所撤单记录(ExecType='52')的 LastPx = 0，直接使用会导致所有撤单都映射到 price_bin=0。

解决方案：
将撤单记录与委托表关联，获取原始委托价格。

关联逻辑：
- BidApplSeqNum > 0 → 撤买单 → 用 BidApplSeqNum 关联委托表
- OfferApplSeqNum > 0 → 撤卖单 → 用 OfferApplSeqNum 关联委托表
"""

from typing import Dict, Optional, Tuple
import logging

from ..data_loader.polars_utils import (
    is_polars_df,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    'enrich_sz_cancel_price',
    'enrich_sz_cancel_price_polars',
    'enrich_sz_cancel_price_pandas',
    'validate_cancel_prices',
    'get_cancel_statistics',
    'SZCancelEnricher',
]


def enrich_sz_cancel_price(
    trade_df: DataFrame,
    order_df: DataFrame,
    exec_type_cancel: str = "52",
    exec_type_trade: str = "70",
    sort_by_time: bool = True,
) -> DataFrame:
    """
    深交所撤单价格关联（自动选择引擎）
    
    将撤单记录的 LastPx=0 替换为原始委托价格
    
    Args:
        trade_df: 成交/撤单数据
        order_df: 委托数据
        exec_type_cancel: 撤单的 ExecType 值
        exec_type_trade: 成交的 ExecType 值
        sort_by_time: 是否按时间排序返回
    
    Returns:
        价格补全后的成交/撤单数据
    """
    if is_polars_df(trade_df):
        return enrich_sz_cancel_price_polars(
            trade_df, order_df, exec_type_cancel, exec_type_trade, sort_by_time
        )
    else:
        return enrich_sz_cancel_price_pandas(
            trade_df, order_df, exec_type_cancel, exec_type_trade, sort_by_time
        )


def enrich_sz_cancel_price_polars(
    trade_df: "pl.DataFrame",
    order_df: "pl.DataFrame",
    exec_type_cancel: str = "52",
    exec_type_trade: str = "70",
    sort_by_time: bool = True,
) -> "pl.DataFrame":
    """
    深交所撤单价格关联（Polars 向量化版本）
    
    将撤单记录的 LastPx=0 替换为原始委托价格
    
    关联逻辑：
    - BidApplSeqNum > 0 → 撤买单 → 用 BidApplSeqNum 关联委托表
    - OfferApplSeqNum > 0 → 撤卖单 → 用 OfferApplSeqNum 关联委托表
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 分离成交和撤单
    df_exec = trade_df.filter(pl.col("ExecType") == exec_type_trade)
    df_cancel = trade_df.filter(pl.col("ExecType") == exec_type_cancel)
    
    if df_cancel.height == 0:
        logger.debug("无撤单记录，跳过价格关联")
        return trade_df
    
    # 构建委托价格映射
    order_price = order_df.select([
        pl.col("ApplSeqNum"),
        pl.col("Price").alias("_OrderPrice"),
    ])
    
    # 分离撤买单和撤卖单
    df_cancel_buy = df_cancel.filter(pl.col("BidApplSeqNum") > 0)
    df_cancel_sell = df_cancel.filter(
        (pl.col("OfferApplSeqNum") > 0) & (pl.col("BidApplSeqNum") <= 0)
    )
    
    result_parts = [df_exec]
    
    # 处理撤买单
    if df_cancel_buy.height > 0:
        df_cancel_buy = df_cancel_buy.join(
            order_price,
            left_on="BidApplSeqNum",
            right_on="ApplSeqNum",
            how="left",
        ).with_columns(
            pl.coalesce([pl.col("_OrderPrice"), pl.col("LastPx")]).alias("LastPx")
        ).drop("_OrderPrice")
        result_parts.append(df_cancel_buy)
        logger.debug(f"处理撤买单: {df_cancel_buy.height} 条")
    
    # 处理撤卖单
    if df_cancel_sell.height > 0:
        df_cancel_sell = df_cancel_sell.join(
            order_price,
            left_on="OfferApplSeqNum",
            right_on="ApplSeqNum",
            how="left",
        ).with_columns(
            pl.coalesce([pl.col("_OrderPrice"), pl.col("LastPx")]).alias("LastPx")
        ).drop("_OrderPrice")
        result_parts.append(df_cancel_sell)
        logger.debug(f"处理撤卖单: {df_cancel_sell.height} 条")
    
    # 合并
    result = pl.concat(result_parts)
    
    # 按时间排序
    if sort_by_time and "TransactTime" in result.columns:
        result = result.sort("TransactTime")
    
    return result


def enrich_sz_cancel_price_pandas(
    trade_df: pd.DataFrame,
    order_df: pd.DataFrame,
    exec_type_cancel: str = "52",
    exec_type_trade: str = "70",
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """
    深交所撤单价格关联（Pandas 版本）
    """
    # 分离成交和撤单
    df_exec = trade_df[trade_df["ExecType"] == exec_type_trade].copy()
    df_cancel = trade_df[trade_df["ExecType"] == exec_type_cancel].copy()
    
    if len(df_cancel) == 0:
        logger.debug("无撤单记录，跳过价格关联")
        return trade_df
    
    # 构建委托价格映射
    order_price_map = order_df.set_index("ApplSeqNum")["Price"].to_dict()
    
    # 分离撤买单和撤卖单
    mask_buy = df_cancel["BidApplSeqNum"] > 0
    mask_sell = (df_cancel["OfferApplSeqNum"] > 0) & (~mask_buy)
    
    df_cancel_buy = df_cancel[mask_buy].copy()
    df_cancel_sell = df_cancel[mask_sell].copy()
    
    result_parts = [df_exec]
    
    # 处理撤买单
    if len(df_cancel_buy) > 0:
        original_prices = df_cancel_buy["BidApplSeqNum"].map(order_price_map)
        # 只替换 LastPx 为 0 或接近 0 的
        mask_zero = df_cancel_buy["LastPx"] <= 0.001
        df_cancel_buy.loc[mask_zero, "LastPx"] = original_prices[mask_zero]
        result_parts.append(df_cancel_buy)
        logger.debug(f"处理撤买单: {len(df_cancel_buy)} 条")
    
    # 处理撤卖单
    if len(df_cancel_sell) > 0:
        original_prices = df_cancel_sell["OfferApplSeqNum"].map(order_price_map)
        mask_zero = df_cancel_sell["LastPx"] <= 0.001
        df_cancel_sell.loc[mask_zero, "LastPx"] = original_prices[mask_zero]
        result_parts.append(df_cancel_sell)
        logger.debug(f"处理撤卖单: {len(df_cancel_sell)} 条")
    
    # 合并
    result = pd.concat(result_parts, ignore_index=True)
    
    # 按时间排序
    if sort_by_time and "TransactTime" in result.columns:
        result = result.sort_values("TransactTime").reset_index(drop=True)
    
    return result


# ==============================================================================
# 原有兼容函数（保持向后兼容）
# ==============================================================================

def _enrich_polars(
    trade_df: "pl.DataFrame",
    order_df: "pl.DataFrame",
    exec_type_cancel: str,
    exec_type_trade: str,
) -> "pl.DataFrame":
    """Polars 实现（向后兼容）"""
    return enrich_sz_cancel_price_polars(
        trade_df, order_df, exec_type_cancel, exec_type_trade, sort_by_time=False
    )


def _enrich_pandas(
    trade_df: pd.DataFrame,
    order_df: pd.DataFrame,
    exec_type_cancel: str,
    exec_type_trade: str,
) -> pd.DataFrame:
    """Pandas 实现（向后兼容）"""
    return enrich_sz_cancel_price_pandas(
        trade_df, order_df, exec_type_cancel, exec_type_trade, sort_by_time=False
    )


# ==============================================================================
# 验证和统计函数
# ==============================================================================

def validate_cancel_prices(
    trade_df: DataFrame,
    exec_type_cancel: str = "52",
) -> bool:
    """
    验证撤单价格是否已正确关联
    
    Args:
        trade_df: 处理后的成交/撤单数据
        exec_type_cancel: 撤单的 ExecType 值
    
    Returns:
        True 如果所有撤单价格都已关联
    """
    stats = get_cancel_statistics(trade_df, exec_type_cancel)
    
    if stats['zero_price_count'] > 0:
        logger.warning(f"仍有 {stats['zero_price_count']} 条撤单价格为 0")
        return False
    
    return True


def get_cancel_statistics(
    trade_df: DataFrame,
    exec_type_cancel: str = "52",
) -> dict:
    """
    获取撤单统计信息
    
    Args:
        trade_df: 处理后的成交/撤单数据
        exec_type_cancel: 撤单的 ExecType 值
    
    Returns:
        统计信息字典
    """
    if is_polars_df(trade_df):
        cancels = trade_df.filter(pl.col("ExecType") == exec_type_cancel)
        total = cancels.height
        
        if total == 0:
            return {
                'total_cancels': 0,
                'cancel_buy_count': 0,
                'cancel_sell_count': 0,
                'zero_price_count': 0,
                'zero_price_ratio': 0.0,
                'is_valid': True,
            }
        
        zero_price = cancels.filter(pl.col("LastPx") <= 0.001).height
        cancel_buy = cancels.filter(pl.col("BidApplSeqNum") > 0).height
        cancel_sell = cancels.filter(
            (pl.col("OfferApplSeqNum") > 0) & (pl.col("BidApplSeqNum") <= 0)
        ).height
    else:
        cancels = trade_df[trade_df["ExecType"] == exec_type_cancel]
        total = len(cancels)
        
        if total == 0:
            return {
                'total_cancels': 0,
                'cancel_buy_count': 0,
                'cancel_sell_count': 0,
                'zero_price_count': 0,
                'zero_price_ratio': 0.0,
                'is_valid': True,
            }
        
        zero_price = (cancels["LastPx"] <= 0.001).sum()
        cancel_buy = (cancels["BidApplSeqNum"] > 0).sum()
        cancel_sell = ((cancels["OfferApplSeqNum"] > 0) & (cancels["BidApplSeqNum"] <= 0)).sum()
    
    return {
        'total_cancels': int(total),
        'cancel_buy_count': int(cancel_buy),
        'cancel_sell_count': int(cancel_sell),
        'zero_price_count': int(zero_price),
        'zero_price_ratio': zero_price / total if total > 0 else 0.0,
        'is_valid': zero_price == 0,
    }


def print_cancel_summary(
    trade_df: DataFrame,
    exec_type_cancel: str = "52",
    stock_code: str = "",
    date: str = "",
) -> None:
    """
    打印撤单处理摘要
    """
    stats = get_cancel_statistics(trade_df, exec_type_cancel)
    
    header = "撤单价格关联摘要"
    if stock_code:
        header += f" [{stock_code}]"
    if date:
        header += f" - {date}"
    
    print(f"\n{'='*50}")
    print(f"{header}")
    print(f"{'='*50}")
    
    print(f"\n📊 撤单统计:")
    print(f"  总撤单数: {stats['total_cancels']:,}")
    print(f"  - 撤买单: {stats['cancel_buy_count']:,}")
    print(f"  - 撤卖单: {stats['cancel_sell_count']:,}")
    
    print(f"\n📊 价格关联:")
    print(f"  未关联数: {stats['zero_price_count']:,} ({stats['zero_price_ratio']:.2%})")
    print(f"  状态: {'✓ 全部关联' if stats['is_valid'] else '✗ 存在未关联'}")
    
    print(f"{'='*50}\n")


class SZCancelEnricher:
    """
    深交所撤单价格关联器
    
    支持缓存委托价格映射，提高批量处理效率
    
    Example:
        >>> enricher = SZCancelEnricher()
        >>> enricher.build_cache(order_df, date='2026-01-21')
        >>> df_enriched = enricher.enrich(trade_df)
        >>> print(enricher.get_statistics(df_enriched))
    """
    
    def __init__(self):
        self._order_price_cache: Optional[Dict[int, float]] = None
        self._cache_date: Optional[str] = None
        self._cache_size: int = 0
    
    def build_cache(
        self,
        order_df: DataFrame,
        date: Optional[str] = None,
    ) -> int:
        """
        构建委托价格缓存
        
        Args:
            order_df: 委托数据
            date: 日期标识（用于缓存失效判断）
        
        Returns:
            缓存的委托数量
        """
        if is_polars_df(order_df):
            self._order_price_cache = dict(zip(
                order_df["ApplSeqNum"].to_list(),
                order_df["Price"].to_list(),
            ))
        else:
            self._order_price_cache = order_df.set_index("ApplSeqNum")["Price"].to_dict()
        
        self._cache_date = date
        self._cache_size = len(self._order_price_cache)
        logger.debug(f"构建委托价格缓存: {self._cache_size} 条")
        
        return self._cache_size
    
    def clear_cache(self):
        """清除缓存"""
        self._order_price_cache = None
        self._cache_date = None
        self._cache_size = 0
    
    def enrich(
        self,
        trade_df: DataFrame,
        order_df: Optional[DataFrame] = None,
        sort_by_time: bool = True,
    ) -> DataFrame:
        """
        关联撤单价格
        
        Args:
            trade_df: 成交/撤单数据
            order_df: 委托数据（如果缓存未构建则必须提供）
            sort_by_time: 是否按时间排序返回
        
        Returns:
            价格补全后的数据
        """
        # 如果没有缓存，则构建
        if self._order_price_cache is None:
            if order_df is None:
                raise ValueError("缓存未构建，必须提供 order_df")
            self.build_cache(order_df)
        
        if order_df is None:
            # 使用缓存进行快速关联
            return self._enrich_with_cache(trade_df, sort_by_time)
        else:
            return enrich_sz_cancel_price(trade_df, order_df, sort_by_time=sort_by_time)
    
    def _enrich_with_cache(
        self,
        trade_df: DataFrame,
        sort_by_time: bool = True,
    ) -> DataFrame:
        """使用缓存进行快速关联"""
        if self._order_price_cache is None:
            raise ValueError("缓存未构建")
        
        if is_polars_df(trade_df):
            return self._enrich_with_cache_polars(trade_df, sort_by_time)
        else:
            return self._enrich_with_cache_pandas(trade_df, sort_by_time)
    
    def _enrich_with_cache_polars(
        self,
        trade_df: "pl.DataFrame",
        sort_by_time: bool,
    ) -> "pl.DataFrame":
        """Polars 版本的缓存关联"""
        df_exec = trade_df.filter(pl.col("ExecType") == "70")
        df_cancel = trade_df.filter(pl.col("ExecType") == "52")
        
        if df_cancel.height == 0:
            return trade_df
        
        # 使用缓存构建价格列
        def get_price(bid_seq, offer_seq):
            if bid_seq > 0:
                return self._order_price_cache.get(bid_seq, 0)
            elif offer_seq > 0:
                return self._order_price_cache.get(offer_seq, 0)
            return 0
        
        # 向量化处理
        bid_seqs = df_cancel["BidApplSeqNum"].to_list()
        offer_seqs = df_cancel["OfferApplSeqNum"].to_list()
        original_prices = [get_price(b, o) for b, o in zip(bid_seqs, offer_seqs)]
        
        df_cancel = df_cancel.with_columns([
            pl.Series("_OriginalPrice", original_prices)
        ]).with_columns([
            pl.when(pl.col("LastPx") <= 0.001)
            .then(pl.col("_OriginalPrice"))
            .otherwise(pl.col("LastPx"))
            .alias("LastPx")
        ]).drop("_OriginalPrice")
        
        result = pl.concat([df_exec, df_cancel])
        
        if sort_by_time and "TransactTime" in result.columns:
            result = result.sort("TransactTime")
        
        return result
    
    def _enrich_with_cache_pandas(
        self,
        trade_df: pd.DataFrame,
        sort_by_time: bool,
    ) -> pd.DataFrame:
        """Pandas 版本的缓存关联"""
        df_exec = trade_df[trade_df["ExecType"] == "70"].copy()
        df_cancel = trade_df[trade_df["ExecType"] == "52"].copy()
        
        if len(df_cancel) == 0:
            return trade_df
        
        def get_price(row):
            if row["BidApplSeqNum"] > 0:
                return self._order_price_cache.get(row["BidApplSeqNum"], 0)
            elif row["OfferApplSeqNum"] > 0:
                return self._order_price_cache.get(row["OfferApplSeqNum"], 0)
            return 0
        
        original_prices = df_cancel.apply(get_price, axis=1)
        mask = df_cancel["LastPx"] <= 0.001
        df_cancel.loc[mask, "LastPx"] = original_prices[mask]
        
        result = pd.concat([df_exec, df_cancel], ignore_index=True)
        
        if sort_by_time and "TransactTime" in result.columns:
            result = result.sort_values("TransactTime").reset_index(drop=True)
        
        return result
    
    def get_original_price(self, seq_num: int) -> float:
        """
        从缓存获取原始委托价格
        
        Args:
            seq_num: 委托序列号
        
        Returns:
            原始委托价格，未找到返回 0
        """
        if self._order_price_cache is None:
            return 0
        return self._order_price_cache.get(seq_num, 0)
    
    def get_statistics(
        self,
        trade_df: DataFrame,
        exec_type_cancel: str = "52",
    ) -> dict:
        """获取撤单统计"""
        stats = get_cancel_statistics(trade_df, exec_type_cancel)
        stats['cache_size'] = self._cache_size
        stats['cache_date'] = self._cache_date
        return stats
    
    @property
    def is_cached(self) -> bool:
        """是否已构建缓存"""
        return self._order_price_cache is not None
    
    @property
    def cache_size(self) -> int:
        """缓存大小"""
        return self._cache_size
