"""
诊断模块

生成通道填充率、稀疏度等监控指标
提供图像质量诊断和健康检查功能
"""

from .reporter import (
    # 常量
    CHANNEL_NAMES,
    TRADE_CHANNELS,
    ORDER_CHANNELS,
    HEALTH_THRESHOLDS,
    # 核心函数
    compute_channel_metrics,
    compute_stock_metrics,
    generate_stock_diagnostics,
    check_health,
    generate_daily_report,
    generate_summary_statistics,
    print_daily_summary,
    # 类
    DiagnosticsReporter,
)


__all__ = [
    # 常量
    'CHANNEL_NAMES',
    'TRADE_CHANNELS',
    'ORDER_CHANNELS',
    'HEALTH_THRESHOLDS',
    # 核心函数
    'compute_channel_metrics',
    'compute_stock_metrics',
    'generate_stock_diagnostics',
    'check_health',
    'generate_daily_report',
    'generate_summary_statistics',
    'print_daily_summary',
    # 类
    'DiagnosticsReporter',
]
