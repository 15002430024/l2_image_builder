"""
上交所数据加载器

加载上交所逐笔成交、逐笔委托数据
注意：上交所委托数据已预处理还原，Qty字段为完整母单量

v3 架构更新（2026-01-26）：
- 委托表必须包含 BizIndex 和 IsAggressive 字段
- BizIndex 用于排序（同一毫秒内区分先后顺序）
- IsAggressive 用于区分主动/被动委托（通道9-12分流）
- 成交表新增 ActiveSide 统一主动方向标识

增强功能（Prompt 1.2）：
- 使用 pl.scan_parquet() 进行懒加载，只读取需要的列
- 支持批量加载多只股票数据
- 优化内存使用，延迟执行查询
"""

import logging
from typing import Optional, Tuple, List, Iterator, Union
from pathlib import Path

from .polars_utils import (
    read_parquet_auto,
    read_parquet_lazy,
    scan_parquet_with_filter,
    collect_lazy,
    filter_time_range,
    filter_positive,
    is_polars_df,
    iter_stocks_lazy,
    get_stock_list_from_parquet,
    batch_load_stocks,
    DataFrame,
    POLARS_AVAILABLE,
)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd


logger = logging.getLogger(__name__)


# 默认连续竞价时段
DEFAULT_TIME_RANGES = [
    (93000000, 113000000),   # 上午 9:30:00.000 - 11:30:00.000
    (130000000, 145700000),  # 下午 13:00:00.000 - 14:57:00.000
]


# ==================== v3 列常量定义 ====================

# v3: 委托表必需字段（含 BizIndex 用于排序，IsAggressive 用于通道分流）
ORDER_COLUMNS_V3_MINIMAL = [
    "SecurityID", "TickTime", "BizIndex",  # ⭐ BizIndex 必需！
    "OrdID", "OrdType", "Side", "Price", "Qty", "IsAggressive"
]

# v3: 完整委托列
ORDER_COLUMNS_V3_FULL = [
    "SecurityID", "TickTime", "BizIndex", 
    "OrdID", "OrdType", "Side", "Price", "Qty", "IsAggressive"
]

# v3: 成交表必需字段（含 BizIndex 和 ActiveSide）
TRADE_COLUMNS_V3_MINIMAL = [
    "SecurityID", "TickTime", "BizIndex",  # ⭐ BizIndex 必需！
    "BuyOrderNO", "SellOrderNO", "Price", "Qty", "TradeMoney", 
    "ActiveSide",  # ⭐ 统一主动方向（1=主动买, 2=主动卖, 0=未知）
    "TickBSFlag"   # 兼容字段
]

# v3: 完整成交列
TRADE_COLUMNS_V3_FULL = [
    "SecurityID", "TickTime", "BizIndex", 
    "BuyOrderNO", "SellOrderNO", "Price", "Qty", "TradeMoney",
    "ActiveSide", "TickBSFlag"
]

# v3: 委托表必需字段名称（用于验证）
V3_REQUIRED_ORDER_FIELDS = ['BizIndex', 'OrdType', 'Side', 'Price', 'Qty', 'IsAggressive']

# v3: 成交表必需字段名称（用于验证）
V3_REQUIRED_TRADE_FIELDS = ['BizIndex', 'BuyOrderNO', 'SellOrderNO', 'Price', 'Qty']


class SHDataLoader:
    """
    上交所数据加载器
    
    数据特点：
    - 成交表: sh_trade_data，包含 TickBSFlag 标识主动方向
    - 委托表: sh_order_data，已预处理还原，Qty为完整母单量
    
    v3 架构要求（2026-01-26）：
    - 委托表必须包含 BizIndex（排序）和 IsAggressive（通道分流）
    - 成交表应包含 ActiveSide 统一主动方向标识
    - 加载时自动验证必需字段
    
    增强功能：
    - 懒加载模式：使用 scan_parquet 延迟加载，减少内存占用
    - 列选择：只读取需要的列，提高I/O效率
    - 批量处理：支持按批次加载多只股票数据
    - 时间过滤下推：在数据读取时就进行过滤
    """
    
    # 成交表字段映射（v3 更新）
    TRADE_COLUMNS = {
        "SecurityID": "str",      # 证券代码
        "TickTime": "int",        # 时间 HHMMSSmmm
        "BizIndex": "int",        # ⭐ v3: 逐笔序号（排序必需）
        "BuyOrderNO": "int",      # 买方订单号
        "SellOrderNO": "int",     # 卖方订单号
        "Price": "float",         # 成交价格
        "Qty": "int",             # 成交数量
        "TradeMoney": "float",    # 成交金额
        "ActiveSide": "int",      # ⭐ v3: 统一主动方向（1=买, 2=卖, 0=未知）
        "TickBSFlag": "str",      # B=主动买, S=主动卖（兼容字段）
    }
    
    # 构建 Image 所需的最小成交列（v3 版本）
    TRADE_COLUMNS_MINIMAL = TRADE_COLUMNS_V3_MINIMAL
    
    # 委托表字段映射（v3 更新：包含 IsAggressive）
    ORDER_COLUMNS = {
        "SecurityID": "str",      # 证券代码
        "TickTime": "int",        # 委托时间
        "BizIndex": "int",        # ⭐ v3: 逐笔序号（排序必需）
        "OrdID": "int",           # 委托单号
        "OrdType": "str",         # New=新增, Cancel=撤单
        "Side": "str",            # B=买入, S=卖出
        "Price": "float",         # 委托价格（撤单已补全）
        "Qty": "int",             # 委托数量（已还原完整母单量）
        "IsAggressive": "bool",   # ⭐ v3: 进攻性标识（True=主动, False=被动, None=撤单）
    }
    
    # 构建 Image 所需的最小委托列（v3 版本）
    ORDER_COLUMNS_MINIMAL = ORDER_COLUMNS_V3_MINIMAL
    
    def __init__(
        self,
        raw_data_dir: str,
        use_polars: bool = True,
        default_time_filter: bool = True,
    ):
        """
        Args:
            raw_data_dir: 原始数据目录
            use_polars: 是否使用 Polars（推荐）
            default_time_filter: 默认是否进行连续竞价时段过滤
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.use_polars = use_polars and POLARS_AVAILABLE
        self.default_time_filter = default_time_filter
    
    def get_trade_path(self, date: str) -> Path:
        """获取成交数据文件路径"""
        return self.raw_data_dir / f"{date}_sh_trade_data.parquet"
    
    def get_order_path(self, date: str) -> Path:
        """获取委托数据文件路径"""
        return self.raw_data_dir / f"{date}_sh_order_data.parquet"
    
    def load_trade(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        time_filter: Optional[bool] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
    ) -> DataFrame:
        """
        加载上交所成交数据
        
        Args:
            date: 日期字符串，如 "20230101"
            columns: 指定加载的列，None 表示全部
            time_filter: 是否进行时间过滤，None 使用默认设置
            time_ranges: 时间范围，默认连续竞价时段
            minimal_columns: 是否只加载构建 Image 所需的最小列
        
        Returns:
            成交数据 DataFrame
        """
        filepath = self.get_trade_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"上交所成交数据不存在: {filepath}")
        
        # 确定要加载的列
        if minimal_columns:
            columns = self.TRADE_COLUMNS_MINIMAL
        
        # 读取数据
        df = read_parquet_auto(
            filepath,
            columns=columns,
            use_polars=self.use_polars,
        )
        
        # 确定是否进行时间过滤
        do_time_filter = time_filter if time_filter is not None else self.default_time_filter
        
        # 时间过滤
        if do_time_filter:
            ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
            df = filter_time_range(df, "TickTime", ranges)
        
        # 过滤异常值
        df = filter_positive(df, ["Price", "Qty"])
        
        return df
    
    def load_order(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        time_filter: Optional[bool] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
        validate_v3_fields: bool = True,
    ) -> DataFrame:
        """
        加载上交所委托数据（已预处理）
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            time_filter: 是否进行时间过滤，None 使用默认设置
            time_ranges: 时间范围
            minimal_columns: 是否只加载构建 Image 所需的最小列
            validate_v3_fields: 是否验证 v3 必需字段（默认 True）
        
        Returns:
            委托数据 DataFrame
        
        Raises:
            ValueError: 缺少 v3 必需字段时抛出
        """
        filepath = self.get_order_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"上交所委托数据不存在: {filepath}")
        
        # 确定要加载的列
        if minimal_columns:
            columns = self.ORDER_COLUMNS_MINIMAL
        
        # 读取数据
        df = read_parquet_auto(
            filepath,
            columns=columns,
            use_polars=self.use_polars,
        )
        
        # v3: 验证必需字段
        if validate_v3_fields:
            self._validate_order_v3_fields(df)
        
        # 确定是否进行时间过滤
        do_time_filter = time_filter if time_filter is not None else self.default_time_filter
        
        # 时间过滤
        if do_time_filter:
            ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
            df = filter_time_range(df, "TickTime", ranges)
        
        # 过滤异常值（撤单价格已补全，但仍需检查数量）
        df = filter_positive(df, ["Qty"])
        
        return df
    
    def _validate_order_v3_fields(self, df: DataFrame) -> None:
        """
        验证委托表的 v3 必需字段
        
        v3 必需字段：BizIndex, OrdType, Side, Price, Qty, IsAggressive
        
        Args:
            df: 委托数据 DataFrame
        
        Raises:
            ValueError: 缺少必需字段时抛出
        """
        if is_polars_df(df):
            existing_cols = set(df.columns)
        else:
            existing_cols = set(df.columns.tolist())
        
        missing = [f for f in V3_REQUIRED_ORDER_FIELDS if f not in existing_cols]
        
        if missing:
            raise ValueError(
                f"委托表缺少 v3 必需字段: {missing}。"
                f"请确保已使用数据预处理脚本生成 derived_sh_orders。"
                f"特别注意：BizIndex 是排序必需字段，IsAggressive 是通道分流必需字段！"
            )
    
    def load_both(
        self,
        date: str,
        time_filter: Optional[bool] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        同时加载成交和委托数据
        
        Args:
            date: 日期字符串
            time_filter: 是否进行时间过滤
            time_ranges: 时间范围
            minimal_columns: 是否只加载最小列
        
        Returns:
            (trade_df, order_df) 元组
        """
        trade_df = self.load_trade(
            date, 
            time_filter=time_filter, 
            time_ranges=time_ranges,
            minimal_columns=minimal_columns,
        )
        order_df = self.load_order(
            date, 
            time_filter=time_filter, 
            time_ranges=time_ranges,
            minimal_columns=minimal_columns,
        )
        return trade_df, order_df
    
    # ========== 懒加载方法（Prompt 1.2 新增）==========
    
    def load_trade_lazy(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
    ) -> "pl.LazyFrame":
        """
        懒加载成交数据（仅 Polars）
        
        使用 scan_parquet 进行懒加载，支持谓词下推优化。
        适用于大数据量场景，可以显著减少内存占用。
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            stock_codes: 要过滤的股票代码列表（谓词下推）
            time_ranges: 时间范围（谓词下推），None 使用默认连续竞价时段
            minimal_columns: 是否只加载最小列
        
        Returns:
            pl.LazyFrame 懒加载帧
        """
        if not self.use_polars:
            raise RuntimeError("懒加载仅支持 Polars，请设置 use_polars=True")
        
        filepath = self.get_trade_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"上交所成交数据不存在: {filepath}")
        
        # 确定要加载的列
        if minimal_columns:
            columns = self.TRADE_COLUMNS_MINIMAL
        
        # 确定时间范围
        ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
        
        # 使用带过滤的懒加载
        lf = scan_parquet_with_filter(
            filepath,
            columns=columns,
            stock_codes=stock_codes,
            code_column="SecurityID",
            time_ranges=ranges,
            time_column="TickTime",
        )
        
        # 添加正值过滤
        lf = lf.filter((pl.col("Price") > 0) & (pl.col("Qty") > 0))
        
        return lf
    
    def load_order_lazy(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
    ) -> "pl.LazyFrame":
        """
        懒加载委托数据（仅 Polars）
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            stock_codes: 要过滤的股票代码列表
            time_ranges: 时间范围，None 使用默认连续竞价时段
            minimal_columns: 是否只加载最小列
        
        Returns:
            pl.LazyFrame 懒加载帧
        """
        if not self.use_polars:
            raise RuntimeError("懒加载仅支持 Polars，请设置 use_polars=True")
        
        filepath = self.get_order_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"上交所委托数据不存在: {filepath}")
        
        # 确定要加载的列
        if minimal_columns:
            columns = self.ORDER_COLUMNS_MINIMAL
        
        # 确定时间范围
        ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
        
        # 使用带过滤的懒加载
        lf = scan_parquet_with_filter(
            filepath,
            columns=columns,
            stock_codes=stock_codes,
            code_column="SecurityID",
            time_ranges=ranges,
            time_column="TickTime",
        )
        
        # 添加正值过滤
        lf = lf.filter(pl.col("Qty") > 0)
        
        return lf
    
    def load_both_lazy(
        self,
        date: str,
        stock_codes: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
    ) -> Tuple["pl.LazyFrame", "pl.LazyFrame"]:
        """
        懒加载成交和委托数据
        
        Returns:
            (trade_lf, order_lf) LazyFrame 元组
        """
        trade_lf = self.load_trade_lazy(
            date,
            stock_codes=stock_codes,
            time_ranges=time_ranges,
            minimal_columns=minimal_columns,
        )
        order_lf = self.load_order_lazy(
            date,
            stock_codes=stock_codes,
            time_ranges=time_ranges,
            minimal_columns=minimal_columns,
        )
        return trade_lf, order_lf
    
    # ========== 单只股票加载 ==========
    
    def load_trade_for_stock(
        self,
        date: str,
        stock_code: str,
        time_filter: Optional[bool] = None,
        minimal_columns: bool = False,
    ) -> DataFrame:
        """
        加载单只股票的成交数据
        
        Args:
            date: 日期
            stock_code: 股票代码，如 "600519"
            time_filter: 是否进行时间过滤
            minimal_columns: 是否只加载最小列
        
        Returns:
            该股票的成交数据
        """
        # 使用懒加载方式更高效
        if self.use_polars:
            do_time_filter = time_filter if time_filter is not None else self.default_time_filter
            time_ranges = DEFAULT_TIME_RANGES if do_time_filter else None
            
            lf = self.load_trade_lazy(
                date,
                stock_codes=[stock_code],
                time_ranges=time_ranges,
                minimal_columns=minimal_columns,
            )
            return lf.collect()
        else:
            # Pandas 回退：加载全量后过滤
            df = self.load_trade(date, time_filter=time_filter, minimal_columns=minimal_columns)
            return df[df["SecurityID"] == stock_code]
    
    def load_order_for_stock(
        self,
        date: str,
        stock_code: str,
        time_filter: Optional[bool] = None,
        minimal_columns: bool = False,
    ) -> DataFrame:
        """
        加载单只股票的委托数据
        
        Args:
            date: 日期
            stock_code: 股票代码
            time_filter: 是否进行时间过滤
            minimal_columns: 是否只加载最小列
        
        Returns:
            该股票的委托数据
        """
        if self.use_polars:
            do_time_filter = time_filter if time_filter is not None else self.default_time_filter
            time_ranges = DEFAULT_TIME_RANGES if do_time_filter else None
            
            lf = self.load_order_lazy(
                date,
                stock_codes=[stock_code],
                time_ranges=time_ranges,
                minimal_columns=minimal_columns,
            )
            return lf.collect()
        else:
            df = self.load_order(date, time_filter=time_filter, minimal_columns=minimal_columns)
            return df[df["SecurityID"] == stock_code]
    
    def load_both_for_stock(
        self,
        date: str,
        stock_code: str,
        time_filter: Optional[bool] = None,
        minimal_columns: bool = False,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        加载单只股票的成交和委托数据
        
        Returns:
            (trade_df, order_df) 元组
        """
        trade_df = self.load_trade_for_stock(
            date, stock_code, time_filter=time_filter, minimal_columns=minimal_columns
        )
        order_df = self.load_order_for_stock(
            date, stock_code, time_filter=time_filter, minimal_columns=minimal_columns
        )
        return trade_df, order_df
    
    # ========== 批量加载方法（Prompt 1.2 新增）==========
    
    def get_stock_list(self, date: str, data_type: str = "trade") -> List[str]:
        """
        获取指定日期数据中的股票代码列表
        
        Args:
            date: 日期字符串
            data_type: "trade" 或 "order"
        
        Returns:
            股票代码列表
        """
        if data_type == "trade":
            filepath = self.get_trade_path(date)
        else:
            filepath = self.get_order_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        return get_stock_list_from_parquet(filepath, code_column="SecurityID")
    
    def iter_stocks_trade(
        self,
        date: str,
        stock_codes: Optional[List[str]] = None,
        minimal_columns: bool = True,
    ) -> Iterator[Tuple[str, DataFrame]]:
        """
        按股票迭代成交数据
        
        适用于需要逐个处理每只股票数据的场景，内存高效。
        
        Args:
            date: 日期字符串
            stock_codes: 要处理的股票代码列表，None 表示全部
            minimal_columns: 是否只加载最小列
        
        Yields:
            (stock_code, trade_df) 元组
        """
        if not self.use_polars:
            # Pandas 回退
            df = self.load_trade(date, minimal_columns=minimal_columns)
            codes = stock_codes if stock_codes else df["SecurityID"].unique().tolist()
            for code in codes:
                yield code, df[df["SecurityID"] == code]
            return
        
        # 获取股票列表
        if stock_codes is None:
            stock_codes = self.get_stock_list(date, "trade")
        
        # 使用懒加载迭代
        lf = self.load_trade_lazy(date, minimal_columns=minimal_columns)
        yield from iter_stocks_lazy(lf, stock_codes, code_column="SecurityID")
    
    def iter_stocks_order(
        self,
        date: str,
        stock_codes: Optional[List[str]] = None,
        minimal_columns: bool = True,
    ) -> Iterator[Tuple[str, DataFrame]]:
        """
        按股票迭代委托数据
        
        Yields:
            (stock_code, order_df) 元组
        """
        if not self.use_polars:
            df = self.load_order(date, minimal_columns=minimal_columns)
            codes = stock_codes if stock_codes else df["SecurityID"].unique().tolist()
            for code in codes:
                yield code, df[df["SecurityID"] == code]
            return
        
        if stock_codes is None:
            stock_codes = self.get_stock_list(date, "order")
        
        lf = self.load_order_lazy(date, minimal_columns=minimal_columns)
        yield from iter_stocks_lazy(lf, stock_codes, code_column="SecurityID")
    
    def iter_stocks_both(
        self,
        date: str,
        stock_codes: Optional[List[str]] = None,
        minimal_columns: bool = True,
    ) -> Iterator[Tuple[str, DataFrame, DataFrame]]:
        """
        按股票迭代成交和委托数据
        
        Yields:
            (stock_code, trade_df, order_df) 元组
        """
        # 获取股票列表（取成交和委托的交集）
        if stock_codes is None:
            trade_codes = set(self.get_stock_list(date, "trade"))
            order_codes = set(self.get_stock_list(date, "order"))
            stock_codes = sorted(trade_codes & order_codes)
        
        for code in stock_codes:
            trade_df = self.load_trade_for_stock(date, code, minimal_columns=minimal_columns)
            order_df = self.load_order_for_stock(date, code, minimal_columns=minimal_columns)
            yield code, trade_df, order_df
    
    def batch_load_trade(
        self,
        date: str,
        stock_codes: List[str],
        batch_size: int = 50,
        minimal_columns: bool = True,
    ) -> Iterator[Tuple[List[str], DataFrame]]:
        """
        批量加载成交数据
        
        每次加载一批股票的数据，适用于需要并行处理的场景。
        
        Args:
            date: 日期字符串
            stock_codes: 股票代码列表
            batch_size: 每批股票数量
            minimal_columns: 是否只加载最小列
        
        Yields:
            (batch_codes, df) 元组
        """
        filepath = self.get_trade_path(date)
        columns = self.TRADE_COLUMNS_MINIMAL if minimal_columns else None
        
        yield from batch_load_stocks(
            filepath,
            stock_codes,
            code_column="SecurityID",
            batch_size=batch_size,
            columns=columns,
            use_polars=self.use_polars,
        )
    
    def batch_load_order(
        self,
        date: str,
        stock_codes: List[str],
        batch_size: int = 50,
        minimal_columns: bool = True,
    ) -> Iterator[Tuple[List[str], DataFrame]]:
        """
        批量加载委托数据
        
        Yields:
            (batch_codes, df) 元组
        """
        filepath = self.get_order_path(date)
        columns = self.ORDER_COLUMNS_MINIMAL if minimal_columns else None
        
        yield from batch_load_stocks(
            filepath,
            stock_codes,
            code_column="SecurityID",
            batch_size=batch_size,
            columns=columns,
            use_polars=self.use_polars,
        )
    
    # ========== 过滤辅助方法 ==========
    
    def get_new_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选新增委托（OrdType='New'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("OrdType") == "New")
        else:
            return order_df[order_df["OrdType"] == "New"]
    
    def get_cancel_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选撤单（OrdType='Cancel'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("OrdType") == "Cancel")
        else:
            return order_df[order_df["OrdType"] == "Cancel"]
    
    def get_buy_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选买单（Side='B'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("Side") == "B")
        else:
            return order_df[order_df["Side"] == "B"]
    
    def get_sell_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选卖单（Side='S'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("Side") == "S")
        else:
            return order_df[order_df["Side"] == "S"]
    
    def get_buy_trades(self, trade_df: DataFrame) -> DataFrame:
        """
        筛选主动买成交（TickBSFlag='B'）
        """
        if is_polars_df(trade_df):
            return trade_df.filter(pl.col("TickBSFlag") == "B")
        else:
            return trade_df[trade_df["TickBSFlag"] == "B"]
    
    def get_sell_trades(self, trade_df: DataFrame) -> DataFrame:
        """
        筛选主动卖成交（TickBSFlag='S'）
        """
        if is_polars_df(trade_df):
            return trade_df.filter(pl.col("TickBSFlag") == "S")
        else:
            return trade_df[trade_df["TickBSFlag"] == "S"]
    
    # ========== v3 辅助方法（R3.1 新增）==========
    
    def get_aggressive_orders(self, order_df: DataFrame) -> DataFrame:
        """
        获取进攻型委托（IsAggressive=True）
        
        v3 架构中，进攻型委托映射到通道 9（买）和通道 10（卖）。
        
        Args:
            order_df: 委托数据（必须包含 IsAggressive 字段）
        
        Returns:
            进攻型委托 DataFrame
        """
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('IsAggressive') == True)
            )
        else:
            mask = (order_df['OrdType'] == 'New') & (order_df['IsAggressive'] == True)
            return order_df[mask]
    
    def get_passive_orders(self, order_df: DataFrame) -> DataFrame:
        """
        获取防守型委托（IsAggressive=False）
        
        v3 架构中，防守型委托映射到通道 11（买）和通道 12（卖）。
        
        Args:
            order_df: 委托数据（必须包含 IsAggressive 字段）
        
        Returns:
            防守型委托 DataFrame
        """
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('IsAggressive') == False)
            )
        else:
            mask = (order_df['OrdType'] == 'New') & (order_df['IsAggressive'] == False)
            return order_df[mask]
    
    def get_aggressive_buy_orders(self, order_df: DataFrame) -> DataFrame:
        """获取进攻型买单（Ch9）"""
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('Side') == 'B') &
                (pl.col('IsAggressive') == True)
            )
        else:
            mask = (
                (order_df['OrdType'] == 'New') & 
                (order_df['Side'] == 'B') &
                (order_df['IsAggressive'] == True)
            )
            return order_df[mask]
    
    def get_aggressive_sell_orders(self, order_df: DataFrame) -> DataFrame:
        """获取进攻型卖单（Ch10）"""
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('Side') == 'S') &
                (pl.col('IsAggressive') == True)
            )
        else:
            mask = (
                (order_df['OrdType'] == 'New') & 
                (order_df['Side'] == 'S') &
                (order_df['IsAggressive'] == True)
            )
            return order_df[mask]
    
    def get_passive_buy_orders(self, order_df: DataFrame) -> DataFrame:
        """获取防守型买单（Ch11）"""
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('Side') == 'B') &
                (pl.col('IsAggressive') == False)
            )
        else:
            mask = (
                (order_df['OrdType'] == 'New') & 
                (order_df['Side'] == 'B') &
                (order_df['IsAggressive'] == False)
            )
            return order_df[mask]
    
    def get_passive_sell_orders(self, order_df: DataFrame) -> DataFrame:
        """获取防守型卖单（Ch12）"""
        if is_polars_df(order_df):
            return order_df.filter(
                (pl.col('OrdType') == 'New') & 
                (pl.col('Side') == 'S') &
                (pl.col('IsAggressive') == False)
            )
        else:
            mask = (
                (order_df['OrdType'] == 'New') & 
                (order_df['Side'] == 'S') &
                (order_df['IsAggressive'] == False)
            )
            return order_df[mask]
