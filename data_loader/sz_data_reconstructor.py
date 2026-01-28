"""
深交所数据重构模块 (REQ-004)

将通联原始深交所 Parquet 文件按 SecurityID 排序重构，
解决逐股票读取时因数据分散导致的全文件扫描性能问题。

性能预期:
- 重构前: ~1.12秒/只股票（全文件扫描）
- 重构后: ~0.03秒/只股票（Row Group 跳过）

使用方式:
    from l2_image_builder.builder import reconstruct_sz_parquet
    
    # 单日重构
    reconstruct_sz_parquet("20251030", input_dir, output_dir)
    
    # 批量重构
    batch_reconstruct_sz_parquet(["20251030", "20251031"], input_dir, output_dir)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Polars 可选依赖
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


def reconstruct_sz_parquet(
    date: str,
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_type: str = "both",
    row_group_size: int = 100_000,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict:
    """
    重构单日深交所 Parquet 数据
    
    将数据按 SecurityID + TransactTime 排序后重写，
    使 Polars 读取单只股票时可利用 Row Group 统计信息跳过无关数据块。
    
    Args:
        date: 日期字符串，如 "20251030"
        input_dir: 输入目录（原始通联数据目录）
        output_dir: 输出目录，None 则覆盖原文件（需 overwrite=True）
        file_type: 重构类型
            - "trade": 仅重构成交数据
            - "order": 仅重构委托数据
            - "both": 同时重构成交和委托（默认）
        row_group_size: Row Group 大小，影响统计信息粒度
        compression: 压缩算法，推荐 "zstd"（压缩率高）或 "lz4"（速度快）
        overwrite: 是否允许覆盖已存在文件
    
    Returns:
        dict: 重构统计信息
            - trade_rows: 成交数据行数
            - order_rows: 委托数据行数
            - trade_size_before: 重构前成交文件大小
            - trade_size_after: 重构后成交文件大小
            - order_size_before: 重构前委托文件大小
            - order_size_after: 重构后委托文件大小
    
    Raises:
        RuntimeError: Polars 不可用
        FileNotFoundError: 输入文件不存在
        FileExistsError: 输出文件已存在且 overwrite=False
    """
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用，请安装: pip install polars")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'date': date,
        'trade_rows': 0,
        'order_rows': 0,
        'trade_size_before': 0,
        'trade_size_after': 0,
        'order_size_before': 0,
        'order_size_after': 0,
    }
    
    # 重构成交数据
    if file_type in ("trade", "both"):
        trade_input = input_dir / f"{date}_sz_trade_data.parquet"
        trade_output = output_dir / f"{date}_sz_trade_data.parquet"
        
        if trade_input.exists():
            stats.update(_reconstruct_single_file(
                trade_input, trade_output,
                sort_columns=["SecurityID", "TransactTime"],
                row_group_size=row_group_size,
                compression=compression,
                overwrite=overwrite,
                file_type="trade",
            ))
        else:
            logger.warning(f"成交数据文件不存在: {trade_input}")
    
    # 重构委托数据
    if file_type in ("order", "both"):
        order_input = input_dir / f"{date}_sz_order_data.parquet"
        order_output = output_dir / f"{date}_sz_order_data.parquet"
        
        if order_input.exists():
            stats.update(_reconstruct_single_file(
                order_input, order_output,
                sort_columns=["SecurityID", "TransactTime"],
                row_group_size=row_group_size,
                compression=compression,
                overwrite=overwrite,
                file_type="order",
            ))
        else:
            logger.warning(f"委托数据文件不存在: {order_input}")
    
    logger.info(f"深交所 {date} 数据重构完成: "
                f"成交 {stats['trade_rows']:,} 行, 委托 {stats['order_rows']:,} 行")
    
    return stats


def _reconstruct_single_file(
    input_path: Path,
    output_path: Path,
    sort_columns: List[str],
    row_group_size: int,
    compression: str,
    overwrite: bool,
    file_type: str,
) -> dict:
    """
    重构单个 Parquet 文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        sort_columns: 排序列
        row_group_size: Row Group 大小
        compression: 压缩算法
        overwrite: 是否覆盖
        file_type: 文件类型标识 ("trade" 或 "order")
    
    Returns:
        dict: 统计信息
    """
    prefix = file_type  # "trade" or "order"
    
    # 检查输出文件
    if output_path.exists() and not overwrite:
        if input_path != output_path:
            raise FileExistsError(f"输出文件已存在: {output_path}，设置 overwrite=True 覆盖")
        # 原地覆盖需要临时文件
        temp_path = output_path.with_suffix('.parquet.tmp')
    else:
        temp_path = None
    
    actual_output = temp_path if temp_path else output_path
    
    # 记录原始大小
    size_before = input_path.stat().st_size
    
    logger.info(f"重构 {input_path.name}: 读取并排序...")
    
    # 读取、排序、写入
    # 使用 scan 懒加载以减少内存占用
    df = pl.scan_parquet(input_path).sort(sort_columns).collect()
    
    row_count = len(df)
    
    # 写入重构后的文件
    df.write_parquet(
        actual_output,
        compression=compression,
        row_group_size=row_group_size,
        statistics=True,  # 确保写入统计信息
    )
    
    # 如果使用临时文件，替换原文件
    if temp_path:
        import shutil
        shutil.move(str(temp_path), str(output_path))
    
    size_after = output_path.stat().st_size
    
    logger.info(f"重构 {output_path.name}: {row_count:,} 行, "
                f"{size_before / 1e9:.2f}GB -> {size_after / 1e9:.2f}GB "
                f"(压缩率 {size_before / size_after:.2f}x)")
    
    return {
        f'{prefix}_rows': row_count,
        f'{prefix}_size_before': size_before,
        f'{prefix}_size_after': size_after,
    }


def batch_reconstruct_sz_parquet(
    dates: List[str],
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    file_type: str = "both",
    row_group_size: int = 100_000,
    compression: str = "zstd",
    overwrite: bool = False,
    show_progress: bool = True,
) -> List[dict]:
    """
    批量重构多日深交所数据
    
    Args:
        dates: 日期列表
        input_dir: 输入目录
        output_dir: 输出目录
        file_type: 重构类型 ("trade", "order", "both")
        row_group_size: Row Group 大小
        compression: 压缩算法
        overwrite: 是否覆盖
        show_progress: 是否显示进度
    
    Returns:
        list[dict]: 每日重构统计信息
    """
    results = []
    
    # 可选进度条
    try:
        from tqdm import tqdm
        iterator = tqdm(dates, desc="重构深交所数据") if show_progress else dates
    except ImportError:
        iterator = dates
    
    for date in iterator:
        try:
            stats = reconstruct_sz_parquet(
                date=date,
                input_dir=input_dir,
                output_dir=output_dir,
                file_type=file_type,
                row_group_size=row_group_size,
                compression=compression,
                overwrite=overwrite,
            )
            results.append(stats)
        except Exception as e:
            logger.error(f"重构 {date} 失败: {e}")
            results.append({'date': date, 'error': str(e)})
    
    # 汇总统计
    total_trade_rows = sum(r.get('trade_rows', 0) for r in results)
    total_order_rows = sum(r.get('order_rows', 0) for r in results)
    
    logger.info(f"批量重构完成: {len(dates)} 日, "
                f"成交 {total_trade_rows:,} 行, 委托 {total_order_rows:,} 行")
    
    return results


def verify_reconstruction(
    date: str,
    data_dir: Union[str, Path],
    sample_stocks: int = 5,
) -> dict:
    """
    验证重构后的数据读取性能
    
    随机抽取若干股票，测量单股票读取耗时。
    
    Args:
        date: 日期字符串
        data_dir: 数据目录
        sample_stocks: 抽样股票数量
    
    Returns:
        dict: 验证结果
            - stock_count: 股票总数
            - sample_count: 抽样数量
            - avg_read_time: 平均读取时间（秒）
            - min_read_time: 最小读取时间
            - max_read_time: 最大读取时间
    """
    import time
    import random
    
    if not POLARS_AVAILABLE:
        raise RuntimeError("Polars 不可用")
    
    data_dir = Path(data_dir)
    trade_file = data_dir / f"{date}_sz_trade_data.parquet"
    
    if not trade_file.exists():
        raise FileNotFoundError(f"文件不存在: {trade_file}")
    
    # 获取所有股票代码
    logger.info("获取股票列表...")
    all_codes = (
        pl.scan_parquet(trade_file)
        .select("SecurityID")
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    
    # 随机抽样
    sample_codes = random.sample(all_codes, min(sample_stocks, len(all_codes)))
    
    # 测试读取性能
    read_times = []
    for code in sample_codes:
        start = time.perf_counter()
        df = (
            pl.scan_parquet(trade_file)
            .filter(pl.col("SecurityID") == code)
            .collect()
        )
        elapsed = time.perf_counter() - start
        read_times.append(elapsed)
        logger.debug(f"读取 {code}: {len(df)} 行, {elapsed:.4f}s")
    
    result = {
        'date': date,
        'stock_count': len(all_codes),
        'sample_count': len(sample_codes),
        'avg_read_time': sum(read_times) / len(read_times),
        'min_read_time': min(read_times),
        'max_read_time': max(read_times),
    }
    
    logger.info(f"验证结果: {sample_stocks} 只股票, "
                f"平均读取 {result['avg_read_time']:.4f}s/只")
    
    return result


# CLI 入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="深交所数据重构工具 (REQ-004)"
    )
    parser.add_argument(
        "command",
        choices=["reconstruct", "verify"],
        help="命令: reconstruct=重构数据, verify=验证性能",
    )
    parser.add_argument(
        "--date", "-d",
        help="单日日期，格式: YYYYMMDD",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="多日日期列表",
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="输入数据目录",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="输出目录（默认覆盖原文件）",
    )
    parser.add_argument(
        "--file-type", "-t",
        choices=["trade", "order", "both"],
        default="both",
        help="重构类型（默认: both）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在文件",
    )
    parser.add_argument(
        "--sample-stocks",
        type=int,
        default=10,
        help="验证时抽样股票数（默认: 10）",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    if args.command == "reconstruct":
        dates = args.dates or ([args.date] if args.date else None)
        if not dates:
            parser.error("请指定 --date 或 --dates")
        
        if len(dates) == 1:
            reconstruct_sz_parquet(
                date=dates[0],
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                file_type=args.file_type,
                overwrite=args.overwrite,
            )
        else:
            batch_reconstruct_sz_parquet(
                dates=dates,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                file_type=args.file_type,
                overwrite=args.overwrite,
            )
    
    elif args.command == "verify":
        if not args.date:
            parser.error("验证命令需要指定 --date")
        
        verify_reconstruction(
            date=args.date,
            data_dir=args.input_dir,
            sample_stocks=args.sample_stocks,
        )
