"""
Level2 图像构建主入口

支持单日处理、批量处理和 Dask 并行处理

使用示例:
    # 单日处理
    python -m l2_image_builder.main --date 20230101 --config config.yaml
    
    # 批量处理（串行）
    python -m l2_image_builder.main --start-date 20230101 --end-date 20230131
    
    # 批量处理（Dask 并行）
    python -m l2_image_builder.main --start-date 20230101 --end-date 20230131 --parallel --workers 8
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timedelta
import os
import numpy as np

from .config import Config, load_config
from .data_loader import SHDataLoader, SZDataLoader
from .calculator import QuantileCalculator, BigOrderCalculator
from .builder import Level2ImageBuilder
from .cleaner import DataCleaner
from .storage import write_daily_lmdb, get_lmdb_stats
from .diagnostics import generate_stock_diagnostics, generate_daily_report, DiagnosticsReporter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Dask 可选依赖
try:
    import dask
    from dask import delayed
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    delayed = None
    Client = None
    LocalCluster = None

# 进度条可选依赖
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


def process_single_stock(
    date: str,
    stock_code: str,
    config: Config,
) -> Tuple[str, Optional[np.ndarray]]:
    """
    处理单只股票（可被 Dask 调度）
    
    Args:
        date: 日期字符串，如 "20230101"
        stock_code: 股票代码，如 "600519.SH"
        config: 配置对象
    
    Returns:
        (stock_code, image) or (stock_code, None)
    """
    try:
        # 判断交易所
        is_sh = stock_code.endswith('.SH')
        code_numeric = stock_code.split('.')[0]
        
        # 初始化加载器
        if is_sh:
            loader = SHDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        else:
            loader = SZDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        
        # 加载数据
        try:
            df_trade = loader.load_trade_for_stock(date, code_numeric)
            df_order = loader.load_order_for_stock(date, code_numeric)
        except (FileNotFoundError, KeyError):
            return (stock_code, None)
        
        if df_trade is None or len(df_trade) == 0:
            return (stock_code, None)
        
        # REQ-005: 数据量熔断检查（防止异常数据导致 OOM）
        MAX_ROWS_PER_STOCK = 5_000_000  # 单只股票最大行数阈值
        trade_rows = len(df_trade)
        order_rows = len(df_order) if df_order is not None else 0
        
        if trade_rows > MAX_ROWS_PER_STOCK or order_rows > MAX_ROWS_PER_STOCK:
            logger.warning(
                f"{stock_code} 数据量异常: trade={trade_rows:,}, order={order_rows:,}, "
                f"超过阈值 {MAX_ROWS_PER_STOCK:,}，跳过处理"
            )
            return (stock_code, None)
        
        # 清洗数据
        cleaner = DataCleaner(verbose=False)
        if is_sh:
            df_trade = cleaner.clean_sh_trade(df_trade)
            df_order = cleaner.clean_sh_order(df_order)
        else:
            df_trade = cleaner.clean_sz_trade(df_trade)
            df_order = cleaner.clean_sz_order(df_order)
            # 注意: enrich_sz_cancel_price 已由 Level2ImageBuilder 内部调用，此处无需重复执行
        
        if len(df_trade) == 0:
            return (stock_code, None)
        
        # 构建图像
        builder = Level2ImageBuilder(stock_code, date, config)
        image = builder.build_single_stock(df_trade, df_order)
        
        return (stock_code, image)
        
    except Exception as e:
        logger.error(f"处理 {stock_code} 失败: {e}")
        return (stock_code, None)


def process_single_day(
    date: str,
    config: Config,
) -> dict:
    """
    处理单日数据
    
    Args:
        date: 日期字符串，如 "20230101"
        config: 配置对象
    
    Returns:
        处理结果统计
    """
    logger.info(f"开始处理 {date}")
    
    stats = {
        'date': date,
        'sh_stocks': 0,
        'sz_stocks': 0,
        'success': 0,
        'failed': 0,
    }
    
    try:
        # 初始化加载器
        sh_loader = SHDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        sz_loader = SZDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        
        # 初始化计算器
        quantile_calc = QuantileCalculator(percentiles=config.percentiles)
        big_order_calc = BigOrderCalculator(std_multiplier=config.threshold_std_multiplier)
        
        # 处理上交所
        try:
            sh_trade, sh_order = sh_loader.load_both(date)
            
            # 计算分位数和阈值
            price_bins, qty_bins = quantile_calc.compute(sh_trade, sh_order, date)
            threshold = big_order_calc.compute(sh_trade, 'SH', date)
            
            logger.info(f"上交所 {date}: 阈值={threshold:.2f}")
            stats['sh_stocks'] = 1  # 简化统计
            
        except FileNotFoundError as e:
            logger.warning(f"上交所数据不存在: {e}")
        
        # 处理深交所
        try:
            sz_trade, sz_order = sz_loader.load_both(date)
            
            # 关联撤单价格
            sz_trade = sz_loader.enrich_cancel_price(sz_trade, sz_order)
            
            # 计算分位数和阈值
            price_bins, qty_bins = quantile_calc.compute_for_sz(sz_trade, sz_order, date)
            threshold = big_order_calc.compute(sz_trade, 'SZ', date)
            
            logger.info(f"深交所 {date}: 阈值={threshold:.2f}")
            stats['sz_stocks'] = 1
            
        except FileNotFoundError as e:
            logger.warning(f"深交所数据不存在: {e}")
        
        stats['success'] = stats['sh_stocks'] + stats['sz_stocks']
        
    except Exception as e:
        logger.error(f"处理 {date} 失败: {e}")
        stats['failed'] = 1
    
    return stats


def _is_valid_stock_code(code: str) -> bool:
    """
    REQ-005: 验证股票代码有效性
    
    过滤掉空字符串、None、非数字代码等无效值，
    防止加载器匹配到全市场数据导致 OOM。
    
    Args:
        code: 股票代码（不含后缀）
    
    Returns:
        True 如果是有效的股票代码
    """
    if not code or not isinstance(code, str):
        return False
    code = code.strip()
    if len(code) == 0:
        return False
    # 有效代码应为纯数字（6位或8位）
    if not code.isdigit():
        return False
    if len(code) not in (6, 8):  # 6位普通股票，8位可能是期权等
        return False
    return True


def get_stock_codes_from_date(
    date: str,
    config: Config,
) -> List[str]:
    """
    获取某日的所有股票代码
    
    REQ-005 增强: 过滤无效股票代码，防止空值/异常值导致全市场数据加载。
    
    Args:
        date: 日期字符串
        config: 配置对象
    
    Returns:
        股票代码列表，如 ["600519.SH", "000001.SZ", ...]
    """
    stock_codes = []
    invalid_count = 0
    
    # 上交所
    try:
        sh_loader = SHDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        sh_codes = sh_loader.get_stock_list(date, 'trade')
        for code in sh_codes:
            if _is_valid_stock_code(code):
                stock_codes.append(f"{code}.SH")
            else:
                invalid_count += 1
    except FileNotFoundError:
        pass
    
    # 深交所
    try:
        sz_loader = SZDataLoader(config.raw_data_dir, use_polars=config.use_polars)
        sz_codes = sz_loader.get_stock_list(date, 'trade')
        for code in sz_codes:
            if _is_valid_stock_code(code):
                stock_codes.append(f"{code}.SZ")
            else:
                invalid_count += 1
    except FileNotFoundError:
        pass
    
    if invalid_count > 0:
        logger.warning(f"过滤掉 {invalid_count} 个无效股票代码")
    
    return stock_codes


def process_daily_serial(
    date: str,
    stock_codes: List[str],
    config: Config,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    串行处理单日数据
    
    Args:
        date: 日期字符串
        stock_codes: 股票代码列表
        config: 配置对象
        show_progress: 是否显示进度条
    
    Returns:
        {stock_code: image} 字典
    """
    stock_images = {}
    
    iterator = stock_codes
    if show_progress and TQDM_AVAILABLE:
        iterator = tqdm(stock_codes, desc=f"处理 {date}", unit="stock")
    
    for code in iterator:
        code, image = process_single_stock(date, code, config)
        if image is not None:
            stock_images[code] = image
    
    return stock_images


def process_daily_dask(
    date: str,
    stock_codes: List[str],
    config: Config,
    n_workers: int = 8,
    threads_per_worker: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Dask 并行处理单日数据
    
    Args:
        date: 日期字符串
        stock_codes: 股票代码列表
        config: 配置对象
        n_workers: Worker 数量
        threads_per_worker: 每个 Worker 的线程数
    
    Returns:
        {stock_code: image} 字典
    """
    if not DASK_AVAILABLE:
        logger.warning("Dask 不可用，回退到串行处理")
        return process_daily_serial(date, stock_codes, config)
    
    # 创建本地集群
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        silence_logs=logging.WARNING,
    )
    client = Client(cluster)
    
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    
    try:
        # 构建延迟任务
        tasks = [
            delayed(process_single_stock)(date, code, config)
            for code in stock_codes
        ]
        
        # 并行执行
        logger.info(f"提交 {len(tasks)} 个任务到 {n_workers} 个 Worker...")
        results = dask.compute(*tasks)
        
        # 整理结果
        stock_images = {
            code: img for code, img in results if img is not None
        }
        
        success_count = len(stock_images)
        fail_count = len(stock_codes) - success_count
        logger.info(f"处理完成: {success_count} 成功, {fail_count} 失败/跳过")
        
        return stock_images
        
    finally:
        client.close()
        cluster.close()


def batch_process(
    date_list: List[str],
    config: Config,
    n_workers: int = 8,
    parallel: bool = True,
    save_lmdb: bool = True,
    save_report: bool = True,
    overwrite: bool = False,
):
    """
    批量处理多日数据
    
    Args:
        date_list: 日期列表
        config: 配置对象
        n_workers: Worker 数量
        parallel: 是否使用 Dask 并行
        save_lmdb: 是否保存 LMDB
        save_report: 是否生成诊断报告
        overwrite: 是否覆盖已存在的 LMDB 文件
    """
    total_stats = {
        'total_days': len(date_list),
        'success_days': 0,
        'failed_days': 0,
        'total_stocks': 0,
        'success_stocks': 0,
    }
    
    for i, date in enumerate(date_list, 1):
        logger.info(f"=" * 60)
        logger.info(f"处理 {date} ({i}/{len(date_list)})")
        
        try:
            # 获取股票列表
            stock_codes = get_stock_codes_from_date(date, config)
            if not stock_codes:
                logger.warning(f"{date} 没有找到股票数据")
                total_stats['failed_days'] += 1
                continue
            
            logger.info(f"找到 {len(stock_codes)} 只股票")
            total_stats['total_stocks'] += len(stock_codes)
            
            # 处理
            if parallel and n_workers > 1:
                stock_images = process_daily_dask(date, stock_codes, config, n_workers)
            else:
                stock_images = process_daily_serial(date, stock_codes, config)
            
            total_stats['success_stocks'] += len(stock_images)
            
            if not stock_images:
                logger.warning(f"{date} 没有成功处理任何股票")
                total_stats['failed_days'] += 1
                continue
            
            # 保存 LMDB
            if save_lmdb:
                lmdb_path = write_daily_lmdb(date, stock_images, config.output_dir, overwrite=overwrite)
                stats = get_lmdb_stats(lmdb_path)
                logger.info(f"已保存 LMDB: {lmdb_path}, 压缩率: {stats['compression_ratio']:.2f}x")
            
            # 生成诊断报告
            if save_report:
                reporter = DiagnosticsReporter(date)
                for code, image in stock_images.items():
                    reporter.add_stock(image, code)
                
                report_dir = os.path.join(config.output_dir, 'reports')
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, f'diagnostics_{date}.csv')
                reporter.save_report(report_path)
                
                summary = reporter.get_summary()
                logger.info(f"诊断报告: 健康 {summary.get('healthy_count', 0)}, "
                           f"异常 {summary.get('unhealthy_count', 0)}")
            
            total_stats['success_days'] += 1
            
        except Exception as e:
            logger.error(f"处理 {date} 失败: {e}")
            total_stats['failed_days'] += 1
    
    # 输出汇总
    logger.info("=" * 60)
    logger.info("批量处理完成")
    logger.info(f"  总计: {total_stats['total_days']} 天, {total_stats['total_stocks']} 只股票")
    logger.info(f"  成功: {total_stats['success_days']} 天, {total_stats['success_stocks']} 只股票")
    logger.info(f"  失败: {total_stats['failed_days']} 天")
    
    return total_stats


def generate_date_range(
    start_date: str,
    end_date: str,
) -> List[str]:
    """
    生成日期范围
    
    Args:
        start_date: 开始日期，如 "20230101"
        end_date: 结束日期，如 "20230131"
    
    Returns:
        日期列表
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        # 跳过周末
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return dates


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Level2 数据图像化处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单日处理
  python -m l2_image_builder.main --date 20230101
  
  # 批量处理（串行）
  python -m l2_image_builder.main --start-date 20230101 --end-date 20230131
  
  # 批量处理（Dask 并行，8个Worker）
  python -m l2_image_builder.main --start-date 20230101 --end-date 20230131 --parallel --workers 8
        """,
    )
    
    # 日期参数
    parser.add_argument(
        "--date", "-d",
        help="处理单日数据，格式: YYYYMMDD",
    )
    parser.add_argument(
        "--start-date",
        help="批量处理开始日期，格式: YYYYMMDD",
    )
    parser.add_argument(
        "--end-date",
        help="批量处理结束日期，格式: YYYYMMDD",
    )
    
    # 配置参数
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--raw-data-dir",
        help="原始数据目录",
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录",
    )
    
    # 并行参数
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="使用 Dask 并行处理",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="并行 Worker 数量 (默认: 8)",
    )
    
    # 输出参数
    parser.add_argument(
        "--no-lmdb",
        action="store_true",
        help="不保存 LMDB 文件",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成诊断报告",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    
    # 其他参数
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出",
    )
    
    args = parser.parse_args()
    
    # 检查 Dask 可用性
    if args.parallel and not DASK_AVAILABLE:
        logger.error("Dask 不可用，请安装: pip install dask distributed")
        return 1
    
    # 加载配置
    overrides = {}
    if args.raw_data_dir:
        overrides['raw_data_dir'] = args.raw_data_dir
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.workers:
        overrides['n_workers'] = args.workers
    
    config = load_config(args.config, **overrides)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 确保输出目录存在
    config.ensure_output_dirs()
    
    # 确定要处理的日期 (REQ-003: CLI 优先级高于 Config)
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        dates = generate_date_range(args.start_date, args.end_date)
    elif config.dates:
        # 从配置文件读取显式日期列表
        dates = config.dates
        logger.info(f"从配置文件读取日期列表: {len(dates)} 个日期")
    elif config.start_date and config.end_date:
        # 从配置文件读取日期范围
        dates = generate_date_range(config.start_date, config.end_date)
        logger.info(f"从配置文件读取日期范围: {config.start_date} ~ {config.end_date}")
    else:
        parser.error("请指定 --date 或 --start-date/--end-date，或在配置文件中设置 dates/start_date/end_date")
        return 1
    
    logger.info(f"待处理日期数: {len(dates)}")
    logger.info(f"配置: raw_data_dir={config.raw_data_dir}, output_dir={config.output_dir}")
    if args.parallel:
        logger.info(f"并行模式: {args.workers} Workers")
    else:
        logger.info("串行模式")
    
    # 批量处理
    batch_process(
        date_list=dates,
        config=config,
        n_workers=args.workers,
        parallel=args.parallel,
        save_lmdb=not args.no_lmdb,
        save_report=not args.no_report,
        overwrite=args.overwrite,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
