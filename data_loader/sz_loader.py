"""
深交所数据加载器

加载深交所逐笔委托、逐笔成交数据

v3 架构更新（2026-01-26）：
- OrderQty 字段统一重命名为 Qty，确保沪深 Loader 输出列名统一
- 新增 build_active_seqs_from_trade 向量化实现
- 支持 ActiveSeqs 快速构建用于通道 9-12 分流

增强功能（Prompt 1.2）：
- 使用 pl.scan_parquet() 进行懒加载，只读取需要的列
- 支持批量加载多只股票数据
- 优化内存使用，延迟执行查询
"""

import logging
from typing import Optional, Tuple, List, Iterator, Union, Set, Dict
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

# 深交所委托表原始列名
ORDER_COLUMNS_RAW = [
    "SecurityID", "ApplSeqNum", "TransactTime", 
    "Price", "OrderQty", "Side", "OrdType"
]

# 深交所委托表归一化列名（OrderQty -> Qty，与上交所统一）
ORDER_COLUMNS_NORMALIZED = [
    "SecurityID", "ApplSeqNum", "TransactTime", 
    "Price", "Qty", "Side", "OrdType"  # OrderQty -> Qty
]

# v3: 成交表必需列（用于构建 ActiveSeqs）
TRADE_COLUMNS_FOR_ACTIVE_SEQS = [
    "SecurityID", "BidApplSeqNum", "OfferApplSeqNum", "ExecType"
]

# ==================== R3.2 成交表列名映射 ====================
# 通联原始字段 -> 系统标准字段
TRADE_COLUMN_RENAME_MAP = {
    'TransactTime': 'TickTime',
    'LastPx': 'Price',
    'LastQty': 'Qty',
    'BidApplSeqNum': 'BuyOrderNO',
    'OfferApplSeqNum': 'SellOrderNO',
    'ApplSeqNum': 'BizIndex',
}

# 委托表列名映射（补充）
ORDER_COLUMN_RENAME_MAP = {
    'TransactTime': 'TickTime',
    'OrderQty': 'Qty',
    'ApplSeqNum': 'BizIndex',
}


class SZDataLoader:
    """
    深交所数据加载器
    
    数据特点：
    - 委托表: sz_order_data，Side 用 ASCII 码表示（49=买, 50=卖）
    - 成交表: sz_trade_data，ExecType 区分成交(70)和撤单(52)
    
    v3 架构更新（2026-01-26）：
    - OrderQty 自动重命名为 Qty，与上交所统一
    - 新增 build_active_seqs_from_trade 向量化实现
    - 支持快速构建 ActiveSeqs 用于通道 9-12 分流
    
    增强功能：
    - 懒加载模式：使用 scan_parquet 延迟加载
    - 列选择：只读取需要的列
    - 批量处理：支持按批次加载多只股票数据
    - 时间过滤下推：在数据读取时就进行过滤
    """
    
    # 委托表字段（v3: OrderQty 将被重命名为 Qty）
    ORDER_COLUMNS = {
        "SecurityID": "str",       # 证券代码
        "ApplSeqNum": "int",       # 委托序列号
        "TransactTime": "int",     # 委托时间
        "Price": "float",          # 委托价格
        "OrderQty": "int",         # 委托数量（原始列名，加载后重命名为 Qty）
        "Qty": "int",              # ⭐ v3: 统一列名（加载后自动重命名）
        "Side": "str",             # 49=买, 50=卖, 71=借入, 70=出借
        "OrdType": "str",          # 49=市价, 50=限价
    }
    
    # 构建 Image 所需的最小委托列（原始列名）
    ORDER_COLUMNS_MINIMAL_RAW = [
        "SecurityID", "ApplSeqNum", "TransactTime", "Price", "OrderQty", "Side"
    ]
    
    # 构建 Image 所需的最小委托列（v3 归一化后）
    ORDER_COLUMNS_MINIMAL = [
        "SecurityID", "ApplSeqNum", "TransactTime", "Price", "Qty", "Side"
    ]
    
    # 成交表字段
    TRADE_COLUMNS = {
        "SecurityID": "str",       # 证券代码
        "ApplSeqNum": "int",       # 消息记录号
        "BidApplSeqNum": "int",    # 买方委托序列号
        "OfferApplSeqNum": "int",  # 卖方委托序列号
        "TransactTime": "int",     # 成交/撤单时间
        "LastPx": "float",         # 成交价格（撤单时为0）
        "LastQty": "int",          # 成交/撤单数量
        "ExecType": "str",         # 70=成交, 52=撤单
    }
    
    # 构建 Image 所需的最小成交列
    TRADE_COLUMNS_MINIMAL = [
        "SecurityID", "BidApplSeqNum", "OfferApplSeqNum", 
        "TransactTime", "LastPx", "LastQty", "ExecType"
    ]
    
    # Side 值映射
    SIDE_BUY = "49"       # 买入
    SIDE_SELL = "50"      # 卖出
    SIDE_BORROW = "71"    # 借入
    SIDE_LEND = "70"      # 出借
    
    # ExecType 值映射
    EXEC_TRADE = "70"     # 成交
    EXEC_CANCEL = "52"    # 撤单
    
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
        return self.raw_data_dir / f"{date}_sz_trade_data.parquet"
    
    def get_order_path(self, date: str) -> Path:
        """获取委托数据文件路径"""
        return self.raw_data_dir / f"{date}_sz_order_data.parquet"
    
    def load_trade(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        time_filter: Optional[bool] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
        normalize_columns: bool = True,
    ) -> DataFrame:
        """
        加载深交所成交数据
        
        R3.2 更新：自动进行列名归一化，输出标准字段。
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            time_filter: 是否进行时间过滤，None 使用默认设置
            time_ranges: 时间范围
            minimal_columns: 是否只加载构建 Image 所需的最小列
            normalize_columns: 是否归一化列名（默认 True）
        
        Returns:
            成交数据 DataFrame (包含成交和撤单，列名已归一化)
        """
        filepath = self.get_trade_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"深交所成交数据不存在: {filepath}")
        
        # 确定要加载的列
        if minimal_columns:
            columns = self.TRADE_COLUMNS_MINIMAL
        
        df = read_parquet_auto(
            filepath,
            columns=columns,
            use_polars=self.use_polars,
        )
        
        # R3.2: 列名归一化（通联原始 -> 系统标准）
        if normalize_columns:
            df = self._normalize_trade_columns(df)
        
        # 确定是否进行时间过滤（使用归一化后的列名）
        do_time_filter = time_filter if time_filter is not None else self.default_time_filter
        
        if do_time_filter:
            ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
            # 检测时间列名（归一化后为 TickTime，否则为 TransactTime）
            time_col = "TickTime" if normalize_columns and "TickTime" in (df.columns if is_polars_df(df) else df.columns.tolist()) else "TransactTime"
            df = filter_time_range(df, time_col, ranges)
        
        return df
    
    def load_order(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        time_filter: Optional[bool] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
        normalize_columns: bool = True,
    ) -> DataFrame:
        """
        加载深交所委托数据
        
        v3 更新：自动将 OrderQty 重命名为 Qty，与上交所统一。
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            time_filter: 是否进行时间过滤，None 使用默认设置
            time_ranges: 时间范围
            minimal_columns: 是否只加载构建 Image 所需的最小列
            normalize_columns: 是否归一化列名（OrderQty -> Qty），默认 True
        
        Returns:
            委托数据 DataFrame
        """
        filepath = self.get_order_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"深交所委托数据不存在: {filepath}")
        
        # 确定要加载的列（使用原始列名）
        if minimal_columns:
            columns = self.ORDER_COLUMNS_MINIMAL_RAW
        
        df = read_parquet_auto(
            filepath,
            columns=columns,
            use_polars=self.use_polars,
        )
        
        # v3: 字段归一化 - 深交所 OrderQty -> 通用 Qty
        if normalize_columns:
            df = self._normalize_order_columns(df)
        
        # 确定是否进行时间过滤
        do_time_filter = time_filter if time_filter is not None else self.default_time_filter
        
        # 时间过滤（使用归一化后的列名）
        if do_time_filter:
            ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
            # 检测时间列名（归一化后为 TickTime，否则为 TransactTime）
            time_col = "TickTime" if normalize_columns and "TickTime" in (df.columns if is_polars_df(df) else df.columns.tolist()) else "TransactTime"
            df = filter_time_range(df, time_col, ranges)
        
        # 过滤异常值（使用归一化后的列名）
        qty_col = "Qty" if normalize_columns and "Qty" in (df.columns if is_polars_df(df) else df.columns.tolist()) else "OrderQty"
        df = filter_positive(df, ["Price", qty_col])
        
        return df
    
    def _normalize_order_columns(self, df: DataFrame) -> DataFrame:
        """
        归一化委托表列名
        
        v3 要求：OrderQty -> Qty，确保沪深 Loader 输出列名统一。
        R3.2: 完整映射 TransactTime -> TickTime, ApplSeqNum -> BizIndex
        
        Args:
            df: 原始委托数据
        
        Returns:
            列名归一化后的 DataFrame
        """
        if is_polars_df(df):
            cols = df.columns
            rename_dict = {}
            for old, new in ORDER_COLUMN_RENAME_MAP.items():
                if old in cols and new not in cols:
                    rename_dict[old] = new
            if rename_dict:
                df = df.rename(rename_dict)
                logger.debug(f"深交所委托表列名归一化: {rename_dict}")
        else:
            cols = df.columns.tolist()
            rename_dict = {}
            for old, new in ORDER_COLUMN_RENAME_MAP.items():
                if old in cols and new not in cols:
                    rename_dict[old] = new
            if rename_dict:
                df = df.rename(columns=rename_dict)
                logger.debug(f"深交所委托表列名归一化: {rename_dict}")
        
        return df
    
    def _normalize_trade_columns(self, df: DataFrame) -> DataFrame:
        """
        R3.2: 归一化成交表列名
        
        将通联原始字段映射为系统标准字段：
        - TransactTime -> TickTime
        - LastPx -> Price
        - LastQty -> Qty
        - BidApplSeqNum -> BuyOrderNO
        - OfferApplSeqNum -> SellOrderNO
        - ApplSeqNum -> BizIndex
        
        同时派生 TickBSFlag 字段（深交所主动性标识）
        
        Args:
            df: 原始成交数据
        
        Returns:
            列名归一化后的 DataFrame（含 TickBSFlag）
        """
        if is_polars_df(df):
            cols = df.columns
            rename_dict = {}
            for old, new in TRADE_COLUMN_RENAME_MAP.items():
                if old in cols and new not in cols:
                    rename_dict[old] = new
            
            if rename_dict:
                df = df.rename(rename_dict)
                logger.debug(f"深交所成交表列名归一化: {rename_dict}")
            
            # 派生 TickBSFlag（深交所：SeqNum 较大者为主动方）
            # 必须在重命名后操作，使用新列名
            new_cols = df.columns
            buy_col = 'BuyOrderNO' if 'BuyOrderNO' in new_cols else 'BidApplSeqNum'
            sell_col = 'SellOrderNO' if 'SellOrderNO' in new_cols else 'OfferApplSeqNum'
            
            if 'TickBSFlag' not in new_cols and buy_col in new_cols and sell_col in new_cols:
                df = df.with_columns(
                    pl.when(pl.col(buy_col) > pl.col(sell_col))
                    .then(pl.lit('B'))
                    .when(pl.col(sell_col) > pl.col(buy_col))
                    .then(pl.lit('S'))
                    .otherwise(pl.lit('N'))
                    .alias('TickBSFlag')
                )
                logger.debug("深交所成交表: 派生 TickBSFlag 字段")
        else:
            cols = df.columns.tolist()
            rename_dict = {}
            for old, new in TRADE_COLUMN_RENAME_MAP.items():
                if old in cols and new not in cols:
                    rename_dict[old] = new
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
                logger.debug(f"深交所成交表列名归一化: {rename_dict}")
            
            # 派生 TickBSFlag
            new_cols = df.columns.tolist()
            buy_col = 'BuyOrderNO' if 'BuyOrderNO' in new_cols else 'BidApplSeqNum'
            sell_col = 'SellOrderNO' if 'SellOrderNO' in new_cols else 'OfferApplSeqNum'
            
            if 'TickBSFlag' not in new_cols and buy_col in new_cols and sell_col in new_cols:
                df = df.copy()
                df['TickBSFlag'] = 'N'
                df.loc[df[buy_col] > df[sell_col], 'TickBSFlag'] = 'B'
                df.loc[df[sell_col] > df[buy_col], 'TickBSFlag'] = 'S'
                logger.debug("深交所成交表: 派生 TickBSFlag 字段")
        
        return df
    
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
            raise FileNotFoundError(f"深交所成交数据不存在: {filepath}")
        
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
            time_column="TransactTime",
        )
        
        return lf
    
    def load_order_lazy(
        self,
        date: str,
        columns: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        time_ranges: Optional[List[Tuple[int, int]]] = None,
        minimal_columns: bool = False,
        normalize_columns: bool = True,
    ) -> "pl.LazyFrame":
        """
        懒加载委托数据（仅 Polars）
        
        v3 更新：支持列名归一化（OrderQty -> Qty）。
        
        Args:
            date: 日期字符串
            columns: 指定加载的列
            stock_codes: 要过滤的股票代码列表
            time_ranges: 时间范围，None 使用默认连续竞价时段
            minimal_columns: 是否只加载最小列
            normalize_columns: 是否归一化列名，默认 True
        
        Returns:
            pl.LazyFrame 懒加载帧
        """
        if not self.use_polars:
            raise RuntimeError("懒加载仅支持 Polars，请设置 use_polars=True")
        
        filepath = self.get_order_path(date)
        
        if not filepath.exists():
            raise FileNotFoundError(f"深交所委托数据不存在: {filepath}")
        
        # 确定要加载的列（使用原始列名）
        if minimal_columns:
            columns = self.ORDER_COLUMNS_MINIMAL_RAW
        
        # 确定时间范围
        ranges = time_ranges if time_ranges is not None else DEFAULT_TIME_RANGES
        
        # 使用带过滤的懒加载
        lf = scan_parquet_with_filter(
            filepath,
            columns=columns,
            stock_codes=stock_codes,
            code_column="SecurityID",
            time_ranges=ranges,
            time_column="TransactTime",
        )
        
        # v3: 列名归一化
        if normalize_columns:
            if 'OrderQty' in lf.columns:
                lf = lf.rename({'OrderQty': 'Qty'})
        
        # 添加正值过滤（使用归一化后的列名）
        qty_col = "Qty" if normalize_columns and 'OrderQty' in (columns or self.ORDER_COLUMNS_MINIMAL_RAW) else "OrderQty"
        if qty_col == "Qty":
            lf = lf.filter((pl.col("Price") > 0) & (pl.col("Qty") > 0))
        else:
            lf = lf.filter((pl.col("Price") > 0) & (pl.col("OrderQty") > 0))
        
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
            stock_code: 股票代码
            time_filter: 是否进行时间过滤
            minimal_columns: 是否只加载最小列
        
        Returns:
            该股票的成交数据
        """
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
            df = self.load_trade(date, minimal_columns=minimal_columns)
            codes = stock_codes if stock_codes else df["SecurityID"].unique().tolist()
            for code in codes:
                yield code, df[df["SecurityID"] == code]
            return
        
        if stock_codes is None:
            stock_codes = self.get_stock_list(date, "trade")
        
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
        
        每次加载一批股票的数据。
        
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
    
    def get_trades_only(self, trade_df: DataFrame) -> DataFrame:
        """
        筛选成交记录（ExecType='70'）
        """
        if is_polars_df(trade_df):
            return trade_df.filter(pl.col("ExecType") == self.EXEC_TRADE)
        else:
            return trade_df[trade_df["ExecType"] == self.EXEC_TRADE]
    
    def get_cancels_only(self, trade_df: DataFrame) -> DataFrame:
        """
        筛选撤单记录（ExecType='52'）
        """
        if is_polars_df(trade_df):
            return trade_df.filter(pl.col("ExecType") == self.EXEC_CANCEL)
        else:
            return trade_df[trade_df["ExecType"] == self.EXEC_CANCEL]
    
    def get_buy_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选买单（Side='49'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("Side") == self.SIDE_BUY)
        else:
            return order_df[order_df["Side"] == self.SIDE_BUY]
    
    def get_sell_orders(self, order_df: DataFrame) -> DataFrame:
        """
        筛选卖单（Side='50'）
        """
        if is_polars_df(order_df):
            return order_df.filter(pl.col("Side") == self.SIDE_SELL)
        else:
            return order_df[order_df["Side"] == self.SIDE_SELL]
    
    # ========== 深交所特有处理方法 ==========
    
    def enrich_cancel_price(
        self,
        trade_df: DataFrame,
        order_df: DataFrame,
    ) -> DataFrame:
        """
        深交所撤单价格关联
        
        将撤单记录的 LastPx=0 替换为原始委托价格
        
        Args:
            trade_df: 成交/撤单数据
            order_df: 委托数据
        
        Returns:
            价格补全后的成交/撤单数据
        """
        if is_polars_df(trade_df):
            return self._enrich_cancel_price_polars(trade_df, order_df)
        else:
            return self._enrich_cancel_price_pandas(trade_df, order_df)
    
    def _enrich_cancel_price_polars(
        self,
        trade_df: "pl.DataFrame",
        order_df: "pl.DataFrame",
    ) -> "pl.DataFrame":
        """Polars 实现的撤单价格关联"""
        # 分离成交和撤单
        trades = trade_df.filter(pl.col("ExecType") == self.EXEC_TRADE)
        cancels = trade_df.filter(pl.col("ExecType") == self.EXEC_CANCEL)
        
        if cancels.height == 0:
            return trade_df
        
        # 构建委托价格映射
        order_prices = order_df.select([
            pl.col("ApplSeqNum"),
            pl.col("Price").alias("OriginalPrice"),
        ])
        
        # 为撤单关联价格
        # 撤买单：BidApplSeqNum > 0
        # 撤卖单：OfferApplSeqNum > 0
        cancels = cancels.with_columns([
            pl.when(pl.col("BidApplSeqNum") > 0)
            .then(pl.col("BidApplSeqNum"))
            .otherwise(pl.col("OfferApplSeqNum"))
            .alias("_lookup_seq")
        ])
        
        # 关联委托价格
        cancels = cancels.join(
            order_prices,
            left_on="_lookup_seq",
            right_on="ApplSeqNum",
            how="left",
        )
        
        # 用原始价格替换 LastPx=0
        cancels = cancels.with_columns([
            pl.when(pl.col("LastPx") <= 0.001)
            .then(pl.col("OriginalPrice"))
            .otherwise(pl.col("LastPx"))
            .alias("LastPx")
        ]).drop(["_lookup_seq", "OriginalPrice"])
        
        # 合并回去
        return pl.concat([trades, cancels])
    
    def _enrich_cancel_price_pandas(
        self,
        trade_df: pd.DataFrame,
        order_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pandas 实现的撤单价格关联"""
        # 分离成交和撤单
        trades = trade_df[trade_df["ExecType"] == self.EXEC_TRADE].copy()
        cancels = trade_df[trade_df["ExecType"] == self.EXEC_CANCEL].copy()
        
        if len(cancels) == 0:
            return trade_df
        
        # 构建委托价格映射
        order_price_map = order_df.set_index("ApplSeqNum")["Price"].to_dict()
        
        def get_original_price(row):
            if row["BidApplSeqNum"] > 0:
                return order_price_map.get(row["BidApplSeqNum"], 0)
            elif row["OfferApplSeqNum"] > 0:
                return order_price_map.get(row["OfferApplSeqNum"], 0)
            return 0
        
        # 关联价格
        original_prices = cancels.apply(get_original_price, axis=1)
        
        # 替换 LastPx=0
        mask = cancels["LastPx"] <= 0.001
        cancels.loc[mask, "LastPx"] = original_prices[mask]
        
        # 合并回去
        return pd.concat([trades, cancels], ignore_index=True)
    
    def build_active_seqs(self, trade_df: DataFrame) -> Dict[str, Set[int]]:
        """
        从成交表构建主动方序列号集合
        
        v3 架构中用于判断委托的主动性：
        - 买方主动：BidApplSeqNum > OfferApplSeqNum
        - 卖方主动：OfferApplSeqNum > BidApplSeqNum
        
        Args:
            trade_df: 成交数据（ExecType='70'）
        
        Returns:
            {'buy': set(), 'sell': set()} 主动方序列号集合
        """
        # 使用向量化实现（更高效）
        return self.build_active_seqs_fast(trade_df)
    
    def build_active_seqs_fast(self, trade_df: DataFrame) -> Dict[str, Set[int]]:
        """
        快速构建主动方序列号集合（向量化实现）
        
        v3 架构核心方法，用于深交所通道 9-12 的分流：
        - Ch9: 主动买单（ApplSeqNum in active_seqs['buy']）
        - Ch10: 主动卖单（ApplSeqNum in active_seqs['sell']）
        - Ch11: 被动买单（ApplSeqNum not in active_seqs['buy']）
        - Ch12: 被动卖单（ApplSeqNum not in active_seqs['sell']）
        
        Args:
            trade_df: 成交数据（建议只传入 ExecType='70' 的记录）
        
        Returns:
            {'buy': set(), 'sell': set()} 主动方序列号集合
        """
        trades = self.get_trades_only(trade_df)
        
        if is_polars_df(trades):
            # Polars 向量化实现
            buy_active = trades.filter(
                pl.col("BidApplSeqNum") > pl.col("OfferApplSeqNum")
            ).select("BidApplSeqNum")
            
            sell_active = trades.filter(
                pl.col("OfferApplSeqNum") > pl.col("BidApplSeqNum")
            ).select("OfferApplSeqNum")
            
            return {
                'buy': set(buy_active["BidApplSeqNum"].to_list()),
                'sell': set(sell_active["OfferApplSeqNum"].to_list())
            }
        else:
            # Pandas 向量化实现
            buy_mask = trades["BidApplSeqNum"] > trades["OfferApplSeqNum"]
            sell_mask = trades["OfferApplSeqNum"] > trades["BidApplSeqNum"]
            
            return {
                'buy': set(trades.loc[buy_mask, "BidApplSeqNum"].tolist()),
                'sell': set(trades.loc[sell_mask, "OfferApplSeqNum"].tolist())
            }