"""
批量处理脚本

支持回填历史数据和每日增量更新

使用示例:
    # 回填历史数据
    python -m l2_image_builder.scripts.batch_process backfill --start 20230101 --end 20231231
    
    # 每日更新（处理最近3天）
    python -m l2_image_builder.scripts.batch_process daily --days 3
    
    # 使用配置文件
    python -m l2_image_builder.scripts.batch_process backfill --start 20230101 --end 20231231 --config config.yaml
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Callable, Any
from pathlib import Path

import numpy as np

# 设置项目根路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from l2_image_builder.config import Config, load_config
from l2_image_builder.main import (
    process_single_stock,
    process_daily_serial,
    process_daily_dask,
    get_stock_codes_from_date,
    generate_date_range,
    batch_process,
    DASK_AVAILABLE,
    TQDM_AVAILABLE,
)
from l2_image_builder.storage import write_daily_lmdb, get_lmdb_stats, LMDBReader
from l2_image_builder.diagnostics import DiagnosticsReporter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 可选依赖
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class BatchProcessor:
    """
    批量处理器
    
    支持:
    - 历史数据回填
    - 增量更新
    - 断点续传
    - 进度跟踪
    """
    
    def __init__(
        self,
        config: Config,
        n_workers: int = 8,
        parallel: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            config: 配置对象
            n_workers: Worker 数量
            parallel: 是否并行处理
            checkpoint_dir: 断点记录目录
        """
        self.config = config
        self.n_workers = n_workers
        self.parallel = parallel and DASK_AVAILABLE
        self.checkpoint_dir = checkpoint_dir or os.path.join(config.output_dir, '.checkpoints')
        
        # 确保目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_days': 0,
            'processed_days': 0,
            'skipped_days': 0,
            'failed_days': 0,
            'total_stocks': 0,
            'processed_stocks': 0,
        }
    
    def _get_checkpoint_path(self, date: str) -> str:
        """获取检查点文件路径"""
        return os.path.join(self.checkpoint_dir, f"{date}.done")
    
    def _is_processed(self, date: str) -> bool:
        """
        检查日期是否已处理
        
        REQ-003 更新: 支持两种检查策略
        - skip_existing=True: 检查 LMDB 文件是否存在（推荐）
        - skip_existing=False: 仅检查 .done 文件
        """
        # 策略1: 检查 LMDB 文件存在性（REQ-003）
        if getattr(self.config, 'skip_existing', True):
            lmdb_path = self._get_lmdb_path(date)
            if os.path.exists(lmdb_path):
                return True
        
        # 策略2: 检查 .done 标记文件
        return os.path.exists(self._get_checkpoint_path(date))
    
    def _mark_processed(self, date: str):
        """标记日期已处理"""
        Path(self._get_checkpoint_path(date)).touch()
    
    def _get_lmdb_path(self, date: str) -> str:
        """获取 LMDB 文件路径"""
        return os.path.join(self.config.output_dir, f"{date}.lmdb")
    
    def process_date(
        self,
        date: str,
        force: bool = False,
        save_lmdb: bool = True,
        save_report: bool = True,
        callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        处理单日数据
        
        Args:
            date: 日期字符串
            force: 是否强制重新处理
            save_lmdb: 是否保存 LMDB
            save_report: 是否保存诊断报告
            callback: 处理完成回调函数
        
        Returns:
            处理结果统计
        """
        result = {
            'date': date,
            'status': 'unknown',
            'stock_count': 0,
            'success_count': 0,
            'error': None,
        }
        
        # 检查是否已处理
        if not force and self._is_processed(date):
            logger.info(f"{date} 已处理，跳过")
            result['status'] = 'skipped'
            self.stats['skipped_days'] += 1
            return result
        
        try:
            # 获取股票列表
            stock_codes = get_stock_codes_from_date(date, self.config)
            if not stock_codes:
                logger.warning(f"{date} 没有找到股票数据")
                result['status'] = 'no_data'
                self.stats['failed_days'] += 1
                return result
            
            result['stock_count'] = len(stock_codes)
            self.stats['total_stocks'] += len(stock_codes)
            
            # 处理
            logger.info(f"处理 {date}: {len(stock_codes)} 只股票...")
            
            if self.parallel and self.n_workers > 1:
                stock_images = process_daily_dask(
                    date, stock_codes, self.config, self.n_workers
                )
            else:
                stock_images = process_daily_serial(
                    date, stock_codes, self.config, show_progress=True
                )
            
            result['success_count'] = len(stock_images)
            self.stats['processed_stocks'] += len(stock_images)
            
            if not stock_images:
                logger.warning(f"{date} 没有成功处理任何股票")
                result['status'] = 'no_success'
                self.stats['failed_days'] += 1
                return result
            
            # 保存 LMDB
            if save_lmdb:
                lmdb_path = write_daily_lmdb(date, stock_images, self.config.output_dir)
                stats = get_lmdb_stats(lmdb_path)
                logger.info(f"保存 LMDB: {lmdb_path} (压缩率: {stats['compression_ratio']:.2f}x)")
            
            # 生成诊断报告
            if save_report:
                reporter = DiagnosticsReporter(date)
                for code, image in stock_images.items():
                    reporter.add_stock(image, code)
                
                report_dir = os.path.join(self.config.output_dir, 'reports')
                os.makedirs(report_dir, exist_ok=True)
                reporter.save_report(report_dir)
                
                summary = reporter.get_summary()
                unhealthy = reporter.get_unhealthy_stocks()
                if unhealthy:
                    logger.warning(f"{date} 发现 {len(unhealthy)} 只异常股票")
            
            # 标记已处理
            self._mark_processed(date)
            
            result['status'] = 'success'
            self.stats['processed_days'] += 1
            
            # 回调
            if callback:
                callback(date, result)
            
        except Exception as e:
            logger.error(f"处理 {date} 失败: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            self.stats['failed_days'] += 1
        
        return result
    
    def run_backfill(
        self,
        start_date: str,
        end_date: str,
        force: bool = False,
        save_lmdb: bool = True,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        回填历史数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            force: 是否强制重新处理
            save_lmdb: 是否保存 LMDB
            save_report: 是否保存诊断报告
        
        Returns:
            回填统计信息
        """
        dates = generate_date_range(start_date, end_date)
        self.stats['total_days'] = len(dates)
        
        logger.info(f"回填历史数据: {start_date} - {end_date}, 共 {len(dates)} 个交易日")
        
        if TQDM_AVAILABLE and tqdm:
            date_iter = tqdm(dates, desc="回填进度", unit="day")
        else:
            date_iter = dates
        
        results = []
        for date in date_iter:
            result = self.process_date(
                date,
                force=force,
                save_lmdb=save_lmdb,
                save_report=save_report,
            )
            results.append(result)
        
        # 汇总
        self._print_summary()
        
        return {
            'stats': self.stats.copy(),
            'results': results,
        }
    
    def run_daily_update(
        self,
        days: int = 1,
        force: bool = False,
        save_lmdb: bool = True,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        每日增量更新
        
        Args:
            days: 更新最近 N 天
            force: 是否强制重新处理
            save_lmdb: 是否保存 LMDB
            save_report: 是否保存诊断报告
        
        Returns:
            更新统计信息
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days-1)).strftime("%Y%m%d")
        
        dates = generate_date_range(start_date, end_date)
        self.stats['total_days'] = len(dates)
        
        logger.info(f"每日更新: 最近 {days} 天 ({start_date} - {end_date})")
        
        results = []
        for date in dates:
            result = self.process_date(
                date,
                force=force,
                save_lmdb=save_lmdb,
                save_report=save_report,
            )
            results.append(result)
        
        # 汇总
        self._print_summary()
        
        return {
            'stats': self.stats.copy(),
            'results': results,
        }
    
    def _print_summary(self):
        """打印汇总信息"""
        logger.info("=" * 60)
        logger.info("批量处理完成")
        logger.info(f"  总计: {self.stats['total_days']} 天, {self.stats['total_stocks']} 只股票")
        logger.info(f"  处理: {self.stats['processed_days']} 天, {self.stats['processed_stocks']} 只股票")
        logger.info(f"  跳过: {self.stats['skipped_days']} 天")
        logger.info(f"  失败: {self.stats['failed_days']} 天")


def run_backfill(
    start_date: str,
    end_date: str,
    config: Optional[Config] = None,
    n_workers: int = 8,
    parallel: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    回填历史数据（便捷函数）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        config: 配置对象
        n_workers: Worker 数量
        parallel: 是否并行
        force: 是否强制重新处理
    
    Returns:
        回填统计信息
    """
    config = config or Config()
    processor = BatchProcessor(config, n_workers, parallel)
    return processor.run_backfill(start_date, end_date, force=force)


def run_daily_update(
    days: int = 1,
    config: Optional[Config] = None,
    n_workers: int = 8,
    parallel: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    每日增量更新（便捷函数）
    
    Args:
        days: 更新最近 N 天
        config: 配置对象
        n_workers: Worker 数量
        parallel: 是否并行
        force: 是否强制重新处理
    
    Returns:
        更新统计信息
    """
    config = config or Config()
    processor = BatchProcessor(config, n_workers, parallel)
    return processor.run_daily_update(days, force=force)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Level2 图像批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 回填历史数据
  python -m l2_image_builder.scripts.batch_process backfill --start 20230101 --end 20231231
  
  # 每日更新
  python -m l2_image_builder.scripts.batch_process daily --days 3
  
  # 强制重新处理
  python -m l2_image_builder.scripts.batch_process backfill --start 20230101 --end 20230131 --force
        """,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 回填命令
    backfill_parser = subparsers.add_parser('backfill', help='回填历史数据')
    backfill_parser.add_argument('--start', required=True, help='开始日期 YYYYMMDD')
    backfill_parser.add_argument('--end', required=True, help='结束日期 YYYYMMDD')
    backfill_parser.add_argument('--force', action='store_true', help='强制重新处理')
    
    # 每日更新命令
    daily_parser = subparsers.add_parser('daily', help='每日增量更新')
    daily_parser.add_argument('--days', type=int, default=1, help='更新最近 N 天')
    daily_parser.add_argument('--force', action='store_true', help='强制重新处理')
    
    # 公共参数
    for subparser in [backfill_parser, daily_parser]:
        subparser.add_argument('--config', '-c', help='配置文件路径')
        subparser.add_argument('--workers', '-w', type=int, default=8, help='Worker 数量')
        subparser.add_argument('--no-parallel', action='store_true', help='禁用并行')
        subparser.add_argument('--no-lmdb', action='store_true', help='不保存 LMDB')
        subparser.add_argument('--no-report', action='store_true', help='不生成报告')
        subparser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 加载配置
    config = load_config(args.config) if args.config else Config()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 确保输出目录存在
    config.ensure_output_dirs()
    
    # 创建处理器
    processor = BatchProcessor(
        config=config,
        n_workers=args.workers,
        parallel=not args.no_parallel,
    )
    
    # 执行
    if args.command == 'backfill':
        processor.run_backfill(
            start_date=args.start,
            end_date=args.end,
            force=args.force,
            save_lmdb=not args.no_lmdb,
            save_report=not args.no_report,
        )
    elif args.command == 'daily':
        processor.run_daily_update(
            days=args.days,
            force=args.force,
            save_lmdb=not args.no_lmdb,
            save_report=not args.no_report,
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
