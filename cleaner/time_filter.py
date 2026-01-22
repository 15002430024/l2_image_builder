"""
时间过滤模块

仅保留连续竞价时段数据

增强功能（Prompt 1.3）：
- Polars 向量化过滤（使用整数运算提取 HHMM）
- 支持 Pandas 兼容
- 提供函数式和类式两种 API
"""

from typing import Union, List, Tuple, Optional
from ..data_loader.polars_utils import (
    is_polars_df,
    filter_time_range,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd


def is_continuous_auction(tick_time: int) -> bool:
    """
    判断是否在连续竞价时段
    
    Args:
        tick_time: 时间格式 HHMMSSmmm (如 93000000 表示 09:30:00.000)
    
    Returns:
        是否在连续竞价时段
    
    时段说明：
    - 09:15 - 09:25: 剔除（开盘集合竞价）
    - 09:25 - 09:30: 剔除（集合竞价撮合期）
    - 09:30 - 11:30: 保留（上午连续竞价）
    - 11:30 - 13:00: 剔除（午间休市）
    - 13:00 - 14:57: 保留（下午连续竞价）
    - 14:57 - 15:00: 剔除（收盘集合竞价-深交所）
    """
    # 提取小时分钟: HHMMSSmmm -> HHMM
    hhmm = tick_time // 100000
    
    # 上午连续竞价: 09:30 - 11:30
    if 930 <= hhmm < 1130:
        return True
    
    # 下午连续竞价: 13:00 - 14:57
    if 1300 <= hhmm < 1457:
        return True
    
    return False


def filter_continuous_auction(
    df: DataFrame,
    time_column: str = "TickTime",
    am_start: int = 93000000,
    am_end: int = 113000000,
    pm_start: int = 130000000,
    pm_end: int = 145700000,
) -> DataFrame:
    """
    过滤连续竞价时段数据
    
    Args:
        df: 输入 DataFrame
        time_column: 时间列名
        am_start: 上午开始时间 (HHMMSSmmm)
        am_end: 上午结束时间
        pm_start: 下午开始时间
        pm_end: 下午结束时间
    
    Returns:
        过滤后的 DataFrame
    """
    time_ranges = [
        (am_start, am_end),
        (pm_start, pm_end),
    ]
    
    return filter_time_range(df, time_column, time_ranges)


def filter_continuous_auction_polars(
    df: "pl.DataFrame",
    time_column: str = "TickTime",
    am_start_hhmm: int = 930,
    am_end_hhmm: int = 1130,
    pm_start_hhmm: int = 1300,
    pm_end_hhmm: int = 1457,
) -> "pl.DataFrame":
    """
    过滤连续竞价时段数据（Polars 向量化版本）
    
    使用整数除法直接提取 HHMM，避免字符串操作，性能更优
    
    Args:
        df: Polars DataFrame
        time_column: 时间列名，格式 HHMMSSmmm
        am_start_hhmm: 上午开始时间 (HHMM)，默认 930 (09:30)
        am_end_hhmm: 上午结束时间 (HHMM)，默认 1130 (11:30)
        pm_start_hhmm: 下午开始时间 (HHMM)，默认 1300 (13:00)
        pm_end_hhmm: 下午结束时间 (HHMM)，默认 1457 (14:57)
    
    Returns:
        过滤后的 DataFrame
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    # 提取 HHMM: HHMMSSmmm // 100000 = HHMM
    hhmm = pl.col(time_column) // 100000
    
    return df.filter(
        ((hhmm >= am_start_hhmm) & (hhmm < am_end_hhmm)) |  # 上午
        ((hhmm >= pm_start_hhmm) & (hhmm < pm_end_hhmm))    # 下午
    )


def filter_continuous_auction_pandas(
    df: pd.DataFrame,
    time_column: str = "TickTime",
    am_start_hhmm: int = 930,
    am_end_hhmm: int = 1130,
    pm_start_hhmm: int = 1300,
    pm_end_hhmm: int = 1457,
) -> pd.DataFrame:
    """
    过滤连续竞价时段数据（Pandas 向量化版本）
    
    Args:
        df: Pandas DataFrame
        time_column: 时间列名，格式 HHMMSSmmm
    
    Returns:
        过滤后的 DataFrame
    """
    # 提取 HHMM
    hhmm = df[time_column] // 100000
    
    mask = (
        ((hhmm >= am_start_hhmm) & (hhmm < am_end_hhmm)) |  # 上午
        ((hhmm >= pm_start_hhmm) & (hhmm < pm_end_hhmm))    # 下午
    )
    
    return df[mask]


def filter_continuous_auction_auto(
    df: DataFrame,
    time_column: str = "TickTime",
) -> DataFrame:
    """
    自动选择引擎过滤连续竞价时段
    
    根据 DataFrame 类型自动选择 Polars 或 Pandas 实现
    """
    if is_polars_df(df):
        return filter_continuous_auction_polars(df, time_column)
    else:
        return filter_continuous_auction_pandas(df, time_column)


class TimeFilter:
    """
    时间过滤器类
    
    支持自定义时间段配置
    """
    
    # 默认连续竞价时段
    DEFAULT_RANGES = [
        (93000000, 113000000),   # 上午 09:30 - 11:30
        (130000000, 145700000),  # 下午 13:00 - 14:57
    ]
    
    def __init__(
        self,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        time_column_sh: str = "TickTime",
        time_column_sz: str = "TransactTime",
    ):
        """
        Args:
            time_ranges: 时间范围列表，格式 [(start, end), ...]
            time_column_sh: 上交所时间列名
            time_column_sz: 深交所时间列名
        """
        self.time_ranges = time_ranges or self.DEFAULT_RANGES
        self.time_column_sh = time_column_sh
        self.time_column_sz = time_column_sz
    
    def filter_sh(self, df: DataFrame) -> DataFrame:
        """过滤上交所数据"""
        return filter_time_range(df, self.time_column_sh, self.time_ranges)
    
    def filter_sz(self, df: DataFrame) -> DataFrame:
        """过滤深交所数据"""
        return filter_time_range(df, self.time_column_sz, self.time_ranges)
    
    def filter(
        self,
        df: DataFrame,
        exchange: str,
    ) -> DataFrame:
        """
        根据交易所过滤
        
        Args:
            df: 输入 DataFrame
            exchange: 交易所标识，'SH' 或 'SZ'
        """
        if exchange.upper() == "SH":
            return self.filter_sh(df)
        elif exchange.upper() == "SZ":
            return self.filter_sz(df)
        else:
            raise ValueError(f"未知交易所: {exchange}")
    
    def get_time_range_str(self) -> str:
        """获取时间范围的可读字符串"""
        ranges_str = []
        for start, end in self.time_ranges:
            start_str = self._format_time(start)
            end_str = self._format_time(end)
            ranges_str.append(f"{start_str} - {end_str}")
        return ", ".join(ranges_str)
    
    @staticmethod
    def _format_time(tick_time: int) -> str:
        """格式化时间为可读字符串"""
        hh = tick_time // 10000000
        mm = (tick_time // 100000) % 100
        ss = (tick_time // 1000) % 100
        ms = tick_time % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def parse_tick_time(tick_time: int) -> dict:
    """
    解析时间戳
    
    Args:
        tick_time: 格式 HHMMSSmmm
    
    Returns:
        {'hour': int, 'minute': int, 'second': int, 'millisecond': int}
    """
    return {
        'hour': tick_time // 10000000,
        'minute': (tick_time // 100000) % 100,
        'second': (tick_time // 1000) % 100,
        'millisecond': tick_time % 1000,
    }


def format_tick_time(tick_time: int) -> str:
    """
    格式化时间戳为可读字符串
    
    Args:
        tick_time: 格式 HHMMSSmmm
    
    Returns:
        "HH:MM:SS.mmm" 格式字符串
    """
    t = parse_tick_time(tick_time)
    return f"{t['hour']:02d}:{t['minute']:02d}:{t['second']:02d}.{t['millisecond']:03d}"
