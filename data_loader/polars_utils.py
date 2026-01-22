"""
Polars 通用工具函数

提供 Polars 和 Pandas 之间的互操作，以及常用的数据处理函数。
当 Polars 不可用时，自动降级使用 Pandas。
"""

from typing import Union, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# 尝试导入 Polars，不可用时降级
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

import pandas as pd


# ========== 类型别名 ==========
DataFrame = Union["pl.DataFrame", pd.DataFrame] if POLARS_AVAILABLE else pd.DataFrame
LazyFrame = "pl.LazyFrame" if POLARS_AVAILABLE else None


def is_polars_available() -> bool:
    """检查 Polars 是否可用"""
    return POLARS_AVAILABLE


def is_polars_df(df: Any) -> bool:
    """检查是否为 Polars DataFrame"""
    if not POLARS_AVAILABLE:
        return False
    return isinstance(df, (pl.DataFrame, pl.LazyFrame))


def is_pandas_df(df: Any) -> bool:
    """检查是否为 Pandas DataFrame"""
    return isinstance(df, pd.DataFrame)


# ========== 数据读取 ==========

def read_parquet_auto(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    use_polars: bool = True,
    **kwargs
) -> DataFrame:
    """
    自动选择引擎读取 Parquet 文件
    
    Args:
        filepath: Parquet 文件路径
        columns: 需要读取的列，None 表示全部
        use_polars: 是否优先使用 Polars
        **kwargs: 传递给读取函数的额外参数
    
    Returns:
        DataFrame (Polars 或 Pandas)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    if use_polars and POLARS_AVAILABLE:
        # Polars 读取
        if columns:
            return pl.read_parquet(filepath, columns=columns, **kwargs)
        return pl.read_parquet(filepath, **kwargs)
    else:
        # Pandas 读取
        if columns:
            return pd.read_parquet(filepath, columns=columns, **kwargs)
        return pd.read_parquet(filepath, **kwargs)


def read_parquet_lazy(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    **kwargs
) -> "pl.LazyFrame":
    """
    懒加载 Parquet 文件（仅 Polars）
    
    适用于大文件，只在需要时才实际读取数据
    
    Args:
        filepath: Parquet 文件路径
        columns: 需要读取的列（懒加载时可选择性读取）
        **kwargs: 传递给 scan_parquet 的额外参数
    
    Returns:
        pl.LazyFrame 懒加载帧
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用，无法使用懒加载")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    lf = pl.scan_parquet(filepath, **kwargs)
    
    # 如果指定了列，则只选择这些列
    if columns:
        lf = lf.select(columns)
    
    return lf


def scan_parquet_with_filter(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    stock_codes: Optional[List[str]] = None,
    code_column: str = "SecurityID",
    time_ranges: Optional[List[Tuple[int, int]]] = None,
    time_column: str = "TickTime",
) -> "pl.LazyFrame":
    """
    带过滤条件的懒加载 Parquet 文件
    
    利用 Polars 的谓词下推优化，在数据读取时就进行过滤
    
    Args:
        filepath: Parquet 文件路径
        columns: 需要读取的列
        stock_codes: 要过滤的股票代码列表
        code_column: 股票代码列名
        time_ranges: 时间范围列表 [(start, end), ...]
        time_column: 时间列名
    
    Returns:
        pl.LazyFrame 带过滤条件的懒加载帧
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用，无法使用懒加载")
    
    lf = read_parquet_lazy(filepath, columns=columns)
    
    # 股票代码过滤（谓词下推）
    if stock_codes:
        lf = lf.filter(pl.col(code_column).is_in(stock_codes))
    
    # 时间范围过滤（谓词下推）
    if time_ranges:
        time_filter = None
        for start, end in time_ranges:
            cond = (pl.col(time_column) >= start) & (pl.col(time_column) < end)
            time_filter = cond if time_filter is None else time_filter | cond
        lf = lf.filter(time_filter)
    
    return lf


def collect_lazy(
    lf: "pl.LazyFrame",
    streaming: bool = False,
) -> "pl.DataFrame":
    """
    收集懒加载帧为 DataFrame
    
    Args:
        lf: LazyFrame
        streaming: 是否使用流式处理（适用于超大数据）
    
    Returns:
        pl.DataFrame
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    if streaming:
        return lf.collect(streaming=True)
    return lf.collect()


def read_multiple_parquets(
    filepaths: List[Union[str, Path]],
    use_polars: bool = True,
    parallel: bool = True,
) -> DataFrame:
    """
    读取多个 Parquet 文件并合并
    
    Args:
        filepaths: 文件路径列表
        use_polars: 是否使用 Polars
        parallel: 是否并行读取（仅 Polars）
    """
    filepaths = [Path(p) for p in filepaths]
    existing = [p for p in filepaths if p.exists()]
    
    if not existing:
        raise FileNotFoundError(f"所有文件都不存在")
    
    if use_polars and POLARS_AVAILABLE:
        if parallel:
            # 使用懒加载并行读取
            lazy_frames = [pl.scan_parquet(p) for p in existing]
            return pl.concat(lazy_frames).collect()
        else:
            dfs = [pl.read_parquet(p) for p in existing]
            return pl.concat(dfs)
    else:
        dfs = [pd.read_parquet(p) for p in existing]
        return pd.concat(dfs, ignore_index=True)


# ========== 类型转换 ==========

def to_pandas(df: DataFrame) -> pd.DataFrame:
    """
    转换为 Pandas DataFrame
    
    Args:
        df: Polars 或 Pandas DataFrame
    
    Returns:
        Pandas DataFrame
    """
    if is_pandas_df(df):
        return df
    
    if POLARS_AVAILABLE and is_polars_df(df):
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df.to_pandas()
    
    raise TypeError(f"不支持的类型: {type(df)}")


def to_polars(df: DataFrame) -> "pl.DataFrame":
    """
    转换为 Polars DataFrame
    
    Args:
        df: Polars 或 Pandas DataFrame
    
    Returns:
        Polars DataFrame
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    if isinstance(df, pl.DataFrame):
        return df
    
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    
    if is_pandas_df(df):
        return pl.from_pandas(df)
    
    raise TypeError(f"不支持的类型: {type(df)}")


def to_numpy(df: DataFrame, column: str) -> np.ndarray:
    """
    提取单列为 numpy 数组
    
    Args:
        df: DataFrame
        column: 列名
    
    Returns:
        numpy 数组
    """
    if is_polars_df(df):
        return df[column].to_numpy()
    else:
        return df[column].values


# ========== 数据过滤 ==========

def filter_time_range(
    df: DataFrame,
    time_column: str,
    ranges: List[Tuple[int, int]],
) -> DataFrame:
    """
    按时间范围过滤数据
    
    Args:
        df: DataFrame
        time_column: 时间列名（格式: HHMMSSmmm）
        ranges: 时间范围列表，如 [(93000000, 113000000), (130000000, 145700000)]
    
    Returns:
        过滤后的 DataFrame
    """
    if is_polars_df(df):
        # Polars 方式
        conditions = None
        for start, end in ranges:
            cond = (pl.col(time_column) >= start) & (pl.col(time_column) < end)
            conditions = cond if conditions is None else conditions | cond
        return df.filter(conditions)
    else:
        # Pandas 方式
        mask = pd.Series(False, index=df.index)
        for start, end in ranges:
            mask |= (df[time_column] >= start) & (df[time_column] < end)
        return df[mask]


def filter_by_values(
    df: DataFrame,
    column: str,
    values: List[Any],
) -> DataFrame:
    """
    按值列表过滤
    
    Args:
        df: DataFrame
        column: 列名
        values: 要保留的值列表
    """
    if is_polars_df(df):
        return df.filter(pl.col(column).is_in(values))
    else:
        return df[df[column].isin(values)]


def filter_positive(
    df: DataFrame,
    columns: List[str],
) -> DataFrame:
    """
    过滤指定列都大于0的行
    
    Args:
        df: DataFrame
        columns: 需要检查的列名列表
    """
    if is_polars_df(df):
        conditions = None
        for col in columns:
            cond = pl.col(col) > 0
            conditions = cond if conditions is None else conditions & cond
        return df.filter(conditions)
    else:
        mask = pd.Series(True, index=df.index)
        for col in columns:
            mask &= df[col] > 0
        return df[mask]


# ========== 聚合操作 ==========

def get_unique_stocks(
    df: DataFrame,
    code_column: str = "SecurityID",
) -> List[str]:
    """
    获取唯一股票代码列表
    
    Args:
        df: DataFrame
        code_column: 股票代码列名
    
    Returns:
        股票代码列表
    """
    if is_polars_df(df):
        return df[code_column].unique().to_list()
    else:
        return df[code_column].unique().tolist()


def groupby_sum(
    df: DataFrame,
    group_col: str,
    sum_col: str,
) -> dict:
    """
    分组求和，返回字典
    
    Args:
        df: DataFrame
        group_col: 分组列
        sum_col: 求和列
    
    Returns:
        {group_value: sum_value} 字典
    """
    if is_polars_df(df):
        result = df.group_by(group_col).agg(pl.col(sum_col).sum())
        return dict(zip(result[group_col].to_list(), result[sum_col].to_list()))
    else:
        return df.groupby(group_col)[sum_col].sum().to_dict()


def compute_percentiles(
    values: Union[np.ndarray, "pl.Series", pd.Series],
    percentiles: Tuple[float, ...],
) -> np.ndarray:
    """
    计算分位数
    
    Args:
        values: 数值数组或 Series
        percentiles: 分位数百分比，如 (12.5, 25, 37.5, 50, 62.5, 75, 87.5)
    
    Returns:
        分位数边界数组
    """
    if POLARS_AVAILABLE and isinstance(values, pl.Series):
        values = values.to_numpy()
    elif isinstance(values, pd.Series):
        values = values.values
    
    # 移除 NaN
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.zeros(len(percentiles))
    
    return np.percentile(values, percentiles)


def concat_series(
    series_list: List[Union["pl.Series", pd.Series]],
    use_polars: bool = True,
) -> Union["pl.Series", pd.Series]:
    """
    合并多个 Series
    
    Args:
        series_list: Series 列表
        use_polars: 是否返回 Polars Series
    
    Returns:
        合并后的 Series
    """
    if not series_list:
        if use_polars and POLARS_AVAILABLE:
            return pl.Series([])
        return pd.Series([])
    
    if use_polars and POLARS_AVAILABLE:
        # 转换为 Polars
        pl_series = []
        for s in series_list:
            if isinstance(s, pl.Series):
                pl_series.append(s)
            elif isinstance(s, pd.Series):
                pl_series.append(pl.from_pandas(s))
            else:
                pl_series.append(pl.Series(s))
        return pl.concat(pl_series)
    else:
        # 转换为 Pandas
        pd_series = []
        for s in series_list:
            if isinstance(s, pd.Series):
                pd_series.append(s)
            elif POLARS_AVAILABLE and isinstance(s, pl.Series):
                pd_series.append(s.to_pandas())
            else:
                pd_series.append(pd.Series(s))
        return pd.concat(pd_series, ignore_index=True)


# ========== 数据分割 ==========

def split_by_stock(
    df: DataFrame,
    code_column: str = "SecurityID",
) -> dict:
    """
    按股票代码分割 DataFrame
    
    Args:
        df: DataFrame
        code_column: 股票代码列名
    
    Returns:
        {stock_code: sub_df} 字典
    """
    if is_polars_df(df):
        return {
            code: group
            for code, group in df.group_by(code_column)
        }
    else:
        return {
            code: group
            for code, group in df.groupby(code_column)
        }


def iter_by_stock(
    df: DataFrame,
    code_column: str = "SecurityID",
):
    """
    按股票代码迭代 DataFrame
    
    Yields:
        (stock_code, sub_df) 元组
    """
    if is_polars_df(df):
        for code, group in df.group_by(code_column):
            yield code, group
    else:
        for code, group in df.groupby(code_column):
            yield code, group


def iter_stocks_lazy(
    lf: "pl.LazyFrame",
    stock_codes: List[str],
    code_column: str = "SecurityID",
):
    """
    懒加载方式按股票迭代数据
    
    适用于大数据量场景，每次只加载一只股票的数据
    
    Args:
        lf: LazyFrame
        stock_codes: 股票代码列表
        code_column: 股票代码列名
    
    Yields:
        (stock_code, df) 元组，df 为该股票的 DataFrame
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    for code in stock_codes:
        stock_df = lf.filter(pl.col(code_column) == code).collect()
        if stock_df.height > 0:
            yield code, stock_df


def get_stock_list_from_parquet(
    filepath: Union[str, Path],
    code_column: str = "SecurityID",
) -> List[str]:
    """
    从 Parquet 文件中获取股票代码列表（懒加载方式，内存高效）
    
    Args:
        filepath: Parquet 文件路径
        code_column: 股票代码列名
    
    Returns:
        去重后的股票代码列表
    """
    if POLARS_AVAILABLE:
        lf = pl.scan_parquet(filepath)
        codes = lf.select(pl.col(code_column).unique()).collect()
        return codes[code_column].to_list()
    else:
        # Pandas 方式需要读取整个列
        df = pd.read_parquet(filepath, columns=[code_column])
        return df[code_column].unique().tolist()


def batch_load_stocks(
    filepath: Union[str, Path],
    stock_codes: List[str],
    code_column: str = "SecurityID",
    batch_size: int = 50,
    columns: Optional[List[str]] = None,
    use_polars: bool = True,
):
    """
    批量加载多只股票数据的生成器
    
    每次加载一批股票的数据，适用于需要处理大量股票的场景
    
    Args:
        filepath: Parquet 文件路径
        stock_codes: 股票代码列表
        code_column: 股票代码列名
        batch_size: 每批股票数量
        columns: 需要读取的列
        use_polars: 是否使用 Polars
    
    Yields:
        (batch_codes, df) 元组，df 包含该批次所有股票的数据
    """
    for i in range(0, len(stock_codes), batch_size):
        batch_codes = stock_codes[i:i + batch_size]
        
        if use_polars and POLARS_AVAILABLE:
            lf = scan_parquet_with_filter(
                filepath,
                columns=columns,
                stock_codes=batch_codes,
                code_column=code_column,
            )
            df = lf.collect()
        else:
            df = pd.read_parquet(filepath, columns=columns)
            df = df[df[code_column].isin(batch_codes)]
        
        yield batch_codes, df


# ========== 性能工具 ==========

def optimize_dtypes_polars(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    优化 Polars DataFrame 的数据类型以减少内存占用
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    return df.with_columns([
        # 字符串列保持不变
        # 整数列尝试降级
        pl.col(c).cast(pl.Int32) 
        for c in df.columns 
        if df[c].dtype in [pl.Int64]
    ])


def estimate_memory(df: DataFrame) -> str:
    """
    估计 DataFrame 内存占用
    
    Returns:
        人类可读的内存大小字符串
    """
    if is_polars_df(df):
        if isinstance(df, pl.LazyFrame):
            return "未知（LazyFrame）"
        size_bytes = df.estimated_size()
    else:
        size_bytes = df.memory_usage(deep=True).sum()
    
    # 转换为人类可读格式
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# ========== 诊断工具 ==========

def describe_df(df: DataFrame, name: str = "DataFrame") -> str:
    """
    生成 DataFrame 的简要描述
    
    Args:
        df: DataFrame
        name: 显示名称
    
    Returns:
        描述字符串
    """
    if is_polars_df(df):
        if isinstance(df, pl.LazyFrame):
            return f"{name}: LazyFrame (未收集)"
        n_rows = df.height
        n_cols = df.width
        columns = df.columns
    else:
        n_rows = len(df)
        n_cols = len(df.columns)
        columns = df.columns.tolist()
    
    mem = estimate_memory(df)
    
    return (
        f"{name}:\n"
        f"  行数: {n_rows:,}\n"
        f"  列数: {n_cols}\n"
        f"  内存: {mem}\n"
        f"  列名: {columns[:5]}{'...' if len(columns) > 5 else ''}"
    )


if __name__ == "__main__":
    # 测试
    print(f"Polars 可用: {is_polars_available()}")
    
    if POLARS_AVAILABLE:
        # 创建测试数据
        df_pl = pl.DataFrame({
            "SecurityID": ["600519.SH", "600519.SH", "000001.SZ"],
            "Price": [1800.0, 1801.0, 15.5],
            "Qty": [100, 200, 500],
            "TickTime": [93000000, 93001000, 140000000],
        })
        
        print("\nPolars DataFrame:")
        print(describe_df(df_pl, "测试数据"))
        
        # 测试时间过滤
        ranges = [(93000000, 113000000), (130000000, 145700000)]
        filtered = filter_time_range(df_pl, "TickTime", ranges)
        print(f"\n时间过滤后行数: {filtered.height}")
        
        # 测试分位数计算
        percentiles = compute_percentiles(df_pl["Price"], (25, 50, 75))
        print(f"\n价格分位数: {percentiles}")
        
        # 测试转换
        df_pd = to_pandas(df_pl)
        print(f"\n转换为 Pandas: {type(df_pd)}")
