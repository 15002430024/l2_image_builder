"""
诊断报告模块

生成图像质量诊断报告，监控通道填充率和健康状态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


# 通道名称常量
CHANNEL_NAMES = [
    '全部成交', '主动买入', '主动卖出', '大买单', '大卖单',
    '小买单', '小卖单', '买单', '卖单', '委托主动买', '委托主动卖',
    '非主动买', '非主动卖', '撤买', '撤卖'
]

# 通道分类
TRADE_CHANNELS = list(range(7))      # 通道 0-6：成交相关
ORDER_CHANNELS = list(range(7, 15))  # 通道 7-14：委托相关

# 健康检查阈值
HEALTH_THRESHOLDS = {
    # 通道级阈值
    'trade_fill_rate_min': 0.30,      # 成交通道最低填充率
    'order_fill_rate_min': 0.50,      # 委托通道最低填充率
    'trade_nonzero_min': 20,          # 成交通道最低非零像素数
    'order_nonzero_min': 30,          # 委托通道最低非零像素数
    'concentration_max': 0.5,          # 最大集中度
    
    # 股票级阈值
    'big_order_ratio_min': 0.05,      # 大单占比最小值
    'big_order_ratio_max': 0.30,      # 大单占比最大值（改为0.5更宽松）
    'cancel_rate_max': 0.50,          # 撤单率最大值
    'trade_order_ratio_min': 0.2,     # 成交/委托比最小值 (1:5)
    'trade_order_ratio_max': 0.5,     # 成交/委托比最大值 (1:2)
}


def compute_channel_metrics(channel: np.ndarray) -> Dict:
    """
    计算单个通道的指标
    
    Args:
        channel: [8, 8] 的通道数据
    
    Returns:
        通道指标字典
    """
    total_pixels = channel.size  # 64
    nonzero_count = np.count_nonzero(channel)
    total_sum = channel.sum()
    max_value = channel.max()
    
    # 计算集中度（避免除零）
    concentration = max_value / total_sum if total_sum > 0 else 0.0
    
    return {
        'nonzero_count': int(nonzero_count),
        'fill_rate': nonzero_count / total_pixels,
        'total_sum': float(total_sum),
        'max_value': float(max_value),
        'mean_value': float(channel.mean()),
        'std_value': float(channel.std()),
        'concentration': float(concentration),
    }


def compute_stock_metrics(image: np.ndarray) -> Dict:
    """
    计算股票级汇总指标
    
    Args:
        image: [15, 8, 8] 的图像数据
    
    Returns:
        股票级指标字典
    """
    # 成交通道总和 (通道 0-6)
    trade_sum = image[TRADE_CHANNELS].sum()
    
    # 委托通道总和 (通道 7-14)
    order_sum = image[ORDER_CHANNELS].sum()
    
    # 成交/委托比
    trade_order_ratio = trade_sum / order_sum if order_sum > 0 else 0.0
    
    # 大单占比: (通道3+4) / (通道3+4+5+6)
    big_order_sum = image[3].sum() + image[4].sum()
    all_sized_order_sum = big_order_sum + image[5].sum() + image[6].sum()
    big_order_ratio = big_order_sum / all_sized_order_sum if all_sized_order_sum > 0 else 0.0
    
    # 撤单率: (通道13+14) / (通道7+8)
    cancel_sum = image[13].sum() + image[14].sum()
    order_intent_sum = image[7].sum() + image[8].sum()
    cancel_rate = cancel_sum / order_intent_sum if order_intent_sum > 0 else 0.0
    
    # 买卖比例
    buy_sum = image[1].sum() + image[3].sum() + image[5].sum()  # 主动买 + 大买 + 小买
    sell_sum = image[2].sum() + image[4].sum() + image[6].sum()  # 主动卖 + 大卖 + 小卖
    buy_sell_ratio = buy_sum / sell_sum if sell_sum > 0 else 0.0
    
    # 整体稀疏度
    total_nonzero = np.count_nonzero(image)
    total_sparsity = 1 - (total_nonzero / image.size)
    
    return {
        'trade_sum': float(trade_sum),
        'order_sum': float(order_sum),
        'trade_order_ratio': float(trade_order_ratio),
        'big_order_ratio': float(big_order_ratio),
        'cancel_rate': float(cancel_rate),
        'buy_sell_ratio': float(buy_sell_ratio),
        'total_nonzero': int(total_nonzero),
        'total_sparsity': float(total_sparsity),
    }


def generate_stock_diagnostics(
    image: np.ndarray,
    stock_code: str,
    trade_date: str,
) -> Dict:
    """
    生成单只股票的完整诊断信息
    
    Args:
        image: [15, 8, 8] float32 数组
        stock_code: 股票代码
        trade_date: 交易日期
    
    Returns:
        诊断信息字典
    """
    # 验证输入
    if image.shape != (15, 8, 8):
        raise ValueError(f"Invalid image shape: {image.shape}, expected (15, 8, 8)")
    
    # 计算各通道指标
    channels = {}
    for i in range(15):
        channels[i] = compute_channel_metrics(image[i])
        channels[i]['name'] = CHANNEL_NAMES[i]
        channels[i]['is_trade'] = i in TRADE_CHANNELS
    
    # 计算股票级指标
    summary = compute_stock_metrics(image)
    
    return {
        'stock_code': stock_code,
        'trade_date': trade_date,
        'channels': channels,
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
    }


def check_health(diagnostics: Dict) -> List[str]:
    """
    健康检查，返回警告列表
    
    Args:
        diagnostics: 诊断信息字典
    
    Returns:
        警告消息列表
    """
    warnings = []
    stock_code = diagnostics.get('stock_code', 'UNKNOWN')
    
    # 1. 检查成交通道填充率和非零像素数
    for ch in TRADE_CHANNELS:
        ch_data = diagnostics['channels'][ch]
        fill_rate = ch_data['fill_rate']
        nonzero = ch_data['nonzero_count']
        
        if fill_rate < HEALTH_THRESHOLDS['trade_fill_rate_min']:
            warnings.append(
                f"[{stock_code}] 通道{ch}({CHANNEL_NAMES[ch]})填充率过低: {fill_rate:.2%}"
            )
        if nonzero < HEALTH_THRESHOLDS['trade_nonzero_min'] and ch_data['total_sum'] > 0:
            warnings.append(
                f"[{stock_code}] 通道{ch}({CHANNEL_NAMES[ch]})非零像素过少: {nonzero}"
            )
    
    # 2. 检查委托通道填充率和非零像素数
    for ch in ORDER_CHANNELS:
        ch_data = diagnostics['channels'][ch]
        fill_rate = ch_data['fill_rate']
        nonzero = ch_data['nonzero_count']
        
        if fill_rate < HEALTH_THRESHOLDS['order_fill_rate_min']:
            warnings.append(
                f"[{stock_code}] 通道{ch}({CHANNEL_NAMES[ch]})填充率过低: {fill_rate:.2%}"
            )
        if nonzero < HEALTH_THRESHOLDS['order_nonzero_min'] and ch_data['total_sum'] > 0:
            warnings.append(
                f"[{stock_code}] 通道{ch}({CHANNEL_NAMES[ch]})非零像素过少: {nonzero}"
            )
    
    # 3. 检查集中度
    for ch in range(15):
        concentration = diagnostics['channels'][ch]['concentration']
        if concentration > HEALTH_THRESHOLDS['concentration_max']:
            warnings.append(
                f"[{stock_code}] 通道{ch}({CHANNEL_NAMES[ch]})集中度过高: {concentration:.2%}"
            )
    
    # 4. 检查大单占比
    big_order_ratio = diagnostics['summary']['big_order_ratio']
    if big_order_ratio < HEALTH_THRESHOLDS['big_order_ratio_min']:
        warnings.append(f"[{stock_code}] 大单占比过低: {big_order_ratio:.2%}")
    if big_order_ratio > HEALTH_THRESHOLDS['big_order_ratio_max']:
        warnings.append(f"[{stock_code}] 大单占比过高: {big_order_ratio:.2%}")
    
    # 5. 检查撤单率
    cancel_rate = diagnostics['summary']['cancel_rate']
    if cancel_rate > HEALTH_THRESHOLDS['cancel_rate_max']:
        warnings.append(f"[{stock_code}] 撤单率过高: {cancel_rate:.2%}")
    
    # 6. 检查成交/委托比
    trade_order_ratio = diagnostics['summary']['trade_order_ratio']
    if trade_order_ratio < HEALTH_THRESHOLDS['trade_order_ratio_min']:
        warnings.append(f"[{stock_code}] 成交/委托比过低: {trade_order_ratio:.2f}")
    if trade_order_ratio > HEALTH_THRESHOLDS['trade_order_ratio_max']:
        warnings.append(f"[{stock_code}] 成交/委托比过高: {trade_order_ratio:.2f}")
    
    return warnings


def generate_daily_report(
    stock_diagnostics: List[Dict],
    trade_date: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    生成每日诊断报告
    
    Args:
        stock_diagnostics: 股票诊断信息列表
        trade_date: 交易日期
        output_path: 输出CSV路径（可选）
    
    Returns:
        诊断报告 DataFrame
    """
    records = []
    
    for diag in stock_diagnostics:
        record = {
            'trade_date': trade_date,
            'stock_code': diag['stock_code'],
        }
        
        # 添加各通道指标
        for ch in range(15):
            ch_data = diag['channels'][ch]
            prefix = f'ch{ch}'
            record[f'{prefix}_fill_rate'] = ch_data['fill_rate']
            record[f'{prefix}_nonzero'] = ch_data['nonzero_count']
            record[f'{prefix}_sum'] = ch_data['total_sum']
            record[f'{prefix}_max'] = ch_data['max_value']
            record[f'{prefix}_concentration'] = ch_data['concentration']
        
        # 添加股票级汇总指标
        summary = diag['summary']
        record['trade_sum'] = summary['trade_sum']
        record['order_sum'] = summary['order_sum']
        record['trade_order_ratio'] = summary['trade_order_ratio']
        record['big_order_ratio'] = summary['big_order_ratio']
        record['cancel_rate'] = summary['cancel_rate']
        record['buy_sell_ratio'] = summary['buy_sell_ratio']
        record['total_nonzero'] = summary['total_nonzero']
        record['total_sparsity'] = summary['total_sparsity']
        
        # 健康检查
        warnings = check_health(diag)
        record['warning_count'] = len(warnings)
        record['warnings'] = '; '.join(warnings) if warnings else ''
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 保存到文件
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return df


def generate_summary_statistics(
    stock_diagnostics: List[Dict],
    trade_date: str,
) -> Dict:
    """
    生成每日汇总统计
    
    Args:
        stock_diagnostics: 股票诊断信息列表
        trade_date: 交易日期
    
    Returns:
        汇总统计字典
    """
    if not stock_diagnostics:
        return {'trade_date': trade_date, 'stock_count': 0}
    
    # 提取各指标数组
    fill_rates = np.array([
        [d['channels'][ch]['fill_rate'] for ch in range(15)]
        for d in stock_diagnostics
    ])
    
    big_order_ratios = np.array([d['summary']['big_order_ratio'] for d in stock_diagnostics])
    cancel_rates = np.array([d['summary']['cancel_rate'] for d in stock_diagnostics])
    sparsities = np.array([d['summary']['total_sparsity'] for d in stock_diagnostics])
    
    # 健康检查统计
    warning_counts = [len(check_health(d)) for d in stock_diagnostics]
    healthy_count = sum(1 for c in warning_counts if c == 0)
    
    return {
        'trade_date': trade_date,
        'stock_count': len(stock_diagnostics),
        'healthy_count': healthy_count,
        'healthy_rate': healthy_count / len(stock_diagnostics),
        'avg_warning_count': np.mean(warning_counts),
        
        # 通道填充率统计
        'avg_fill_rate_trade': float(fill_rates[:, TRADE_CHANNELS].mean()),
        'avg_fill_rate_order': float(fill_rates[:, ORDER_CHANNELS].mean()),
        'min_fill_rate': float(fill_rates.min()),
        
        # 大单占比统计
        'avg_big_order_ratio': float(big_order_ratios.mean()),
        'std_big_order_ratio': float(big_order_ratios.std()),
        
        # 撤单率统计
        'avg_cancel_rate': float(cancel_rates.mean()),
        'max_cancel_rate': float(cancel_rates.max()),
        
        # 稀疏度统计
        'avg_sparsity': float(sparsities.mean()),
    }


def print_daily_summary(
    stock_diagnostics: List[Dict],
    trade_date: str,
    show_warnings: bool = True,
    max_warnings: int = 10,
):
    """
    打印每日诊断摘要
    
    Args:
        stock_diagnostics: 股票诊断信息列表
        trade_date: 交易日期
        show_warnings: 是否显示警告
        max_warnings: 最多显示警告数
    """
    stats = generate_summary_statistics(stock_diagnostics, trade_date)
    
    print(f"\n{'='*60}")
    print(f"诊断报告 - {trade_date}")
    print(f"{'='*60}")
    print(f"股票总数: {stats['stock_count']}")
    print(f"健康股票: {stats['healthy_count']} ({stats['healthy_rate']:.1%})")
    print(f"平均警告数: {stats['avg_warning_count']:.2f}")
    print(f"\n通道填充率:")
    print(f"  成交通道平均: {stats['avg_fill_rate_trade']:.1%}")
    print(f"  委托通道平均: {stats['avg_fill_rate_order']:.1%}")
    print(f"\n关键指标:")
    print(f"  大单占比: {stats['avg_big_order_ratio']:.1%} ± {stats['std_big_order_ratio']:.1%}")
    print(f"  撤单率: {stats['avg_cancel_rate']:.1%} (最高 {stats['max_cancel_rate']:.1%})")
    print(f"  平均稀疏度: {stats['avg_sparsity']:.1%}")
    
    if show_warnings:
        all_warnings = []
        for diag in stock_diagnostics:
            all_warnings.extend(check_health(diag))
        
        if all_warnings:
            print(f"\n警告 (共 {len(all_warnings)} 条):")
            for w in all_warnings[:max_warnings]:
                print(f"  ⚠️ {w}")
            if len(all_warnings) > max_warnings:
                print(f"  ... 还有 {len(all_warnings) - max_warnings} 条警告")
    
    print(f"{'='*60}\n")


class DiagnosticsReporter:
    """
    诊断报告器类
    
    支持批量处理和增量添加
    """
    
    def __init__(self, trade_date: str):
        """
        Args:
            trade_date: 交易日期
        """
        self.trade_date = trade_date
        self.diagnostics: List[Dict] = []
    
    def add_stock(self, image: np.ndarray, stock_code: str) -> Dict:
        """
        添加单只股票诊断
        
        Args:
            image: [15, 8, 8] 图像
            stock_code: 股票代码
        
        Returns:
            诊断信息
        """
        diag = generate_stock_diagnostics(image, stock_code, self.trade_date)
        self.diagnostics.append(diag)
        return diag
    
    def add_batch(self, images: Dict[str, np.ndarray]):
        """
        批量添加股票诊断
        
        Args:
            images: {stock_code: image}
        """
        for code, image in images.items():
            self.add_stock(image, code)
    
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        return generate_summary_statistics(self.diagnostics, self.trade_date)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return generate_daily_report(self.diagnostics, self.trade_date)
    
    def save_report(self, output_path: str) -> pd.DataFrame:
        """保存报告到 CSV"""
        return generate_daily_report(self.diagnostics, self.trade_date, output_path)
    
    def print_summary(self, show_warnings: bool = True):
        """打印摘要"""
        print_daily_summary(self.diagnostics, self.trade_date, show_warnings)
    
    def get_unhealthy_stocks(self) -> List[str]:
        """获取不健康的股票列表"""
        return [
            d['stock_code']
            for d in self.diagnostics
            if check_health(d)
        ]
    
    def __len__(self) -> int:
        return len(self.diagnostics)
