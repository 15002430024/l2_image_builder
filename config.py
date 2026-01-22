"""
配置管理模块

支持:
1. dataclass 默认配置
2. YAML 文件加载
3. 环境变量覆盖
4. 配置验证
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List
from pathlib import Path
import os
import yaml


@dataclass
class Config:
    """Level2 图像构建配置类"""
    
    # ========== 数据路径 ==========
    raw_data_dir: str = "/raw_data"
    output_dir: str = "/processed_data/l2_images"
    
    # ========== 图像参数 ==========
    num_channels: int = 15       # 7个成交类型 + 8个委托类型
    num_price_bins: int = 8      # 价格分位数区间
    num_qty_bins: int = 8        # 量分位数区间
    
    # ========== 分位数百分位 ==========
    # 7个切割点定义8个区间
    percentiles: Tuple[float, ...] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5)
    
    # ========== 大单阈值 ==========
    # 当日计算: Mean(V) + threshold_std_multiplier × Std(V)
    threshold_std_multiplier: float = 1.0
    
    # ========== 时间过滤（连续竞价时段）==========
    # 格式: HHMMSSmmm，如 93000000 = 09:30:00.000
    am_start: int = 93000000     # 上午开始: 09:30:00.000
    am_end: int = 113000000      # 上午结束: 11:30:00.000
    pm_start: int = 130000000    # 下午开始: 13:00:00.000
    pm_end: int = 145700000      # 下午结束: 14:57:00.000
    
    # ========== 存储参数 ==========
    lmdb_map_size: int = 100 * 1024 * 1024  # 100MB per LMDB file
    use_lz4: bool = True                     # LZ4压缩
    
    # ========== 处理引擎 ==========
    use_polars: bool = True      # 优先使用 Polars
    n_workers: int = 8           # Dask 并行数
    chunk_size: int = 100        # 每批处理股票数
    
    # ========== 通道定义 ==========
    # 用于诊断和文档，实际逻辑在 builder 中
    channel_names: Tuple[str, ...] = field(default_factory=lambda: (
        "全部成交",      # 0
        "主动买入成交",  # 1
        "主动卖出成交",  # 2
        "大买单",        # 3
        "大卖单",        # 4
        "小买单",        # 5
        "小卖单",        # 6
        "买单委托",      # 7
        "卖单委托",      # 8
        "主动买入委托",  # 9
        "主动卖出委托",  # 10
        "非主动买入",    # 11
        "非主动卖出",    # 12
        "撤买",          # 13
        "撤卖",          # 14
    ))
    
    # ========== 数据源文件命名模式 ==========
    # {date} 将被替换为日期字符串，如 20230101
    sh_trade_pattern: str = "{date}_sh_trade_data.parquet"
    sh_order_pattern: str = "{date}_sh_order_data.parquet"
    sz_trade_pattern: str = "{date}_sz_trade_data.parquet"
    sz_order_pattern: str = "{date}_sz_order_data.parquet"
    sh_snapshot_pattern: str = "{date}_sh_market_data_3s.parquet"
    sz_snapshot_pattern: str = "{date}_sz_market_data_3s.parquet"
    
    # ========== 诊断参数 ==========
    min_nonzero_pixels_trade: int = 20    # 成交通道最小非零像素
    min_nonzero_pixels_order: int = 30    # 委托通道最小非零像素
    min_fill_rate_trade: float = 0.30     # 成交通道最小填充率
    min_fill_rate_order: float = 0.50     # 委托通道最小填充率
    
    def __post_init__(self):
        """配置后处理和验证"""
        # 确保路径存在
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        # 验证通道数
        assert self.num_channels == 15, f"通道数必须为15，当前: {self.num_channels}"
        
        # 验证分位数
        assert len(self.percentiles) == 7, f"分位数切割点必须为7个，当前: {len(self.percentiles)}"
        
        # 验证分位数单调递增
        for i in range(len(self.percentiles) - 1):
            assert self.percentiles[i] < self.percentiles[i + 1], "分位数必须单调递增"
        
        # 验证时间格式
        assert 0 <= self.am_start < self.am_end <= 240000000, "上午时间配置错误"
        assert 0 <= self.pm_start < self.pm_end <= 240000000, "下午时间配置错误"
        
        # 验证通道名称数量
        assert len(self.channel_names) == 15, f"通道名称必须为15个，当前: {len(self.channel_names)}"
    
    def get_raw_data_path(self, date: str, file_type: str) -> Path:
        """
        获取原始数据文件路径
        
        Args:
            date: 日期字符串，如 "20230101"
            file_type: 文件类型，可选值:
                - "sh_trade": 上交所成交
                - "sh_order": 上交所委托
                - "sz_trade": 深交所成交
                - "sz_order": 深交所委托
                - "sh_snapshot": 上交所快照
                - "sz_snapshot": 深交所快照
        
        Returns:
            完整文件路径
        """
        pattern_map = {
            "sh_trade": self.sh_trade_pattern,
            "sh_order": self.sh_order_pattern,
            "sz_trade": self.sz_trade_pattern,
            "sz_order": self.sz_order_pattern,
            "sh_snapshot": self.sh_snapshot_pattern,
            "sz_snapshot": self.sz_snapshot_pattern,
        }
        
        if file_type not in pattern_map:
            raise ValueError(f"未知文件类型: {file_type}，可选: {list(pattern_map.keys())}")
        
        filename = pattern_map[file_type].format(date=date)
        return Path(self.raw_data_dir) / filename
    
    def get_lmdb_path(self, date: str) -> Path:
        """获取LMDB输出路径"""
        return Path(self.output_dir) / f"{date}.lmdb"
    
    def get_diagnostics_path(self, date: str) -> Path:
        """获取诊断报告路径"""
        return Path(self.output_dir) / "diagnostics" / f"{date}_diagnostics.csv"
    
    def get_threshold_path(self, date: str) -> Path:
        """获取大单阈值存储路径"""
        return Path(self.output_dir) / "thresholds" / f"{date}_thresholds.parquet"
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """图像形状: (channels, price_bins, qty_bins)"""
        return (self.num_channels, self.num_price_bins, self.num_qty_bins)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """保存为YAML文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def ensure_output_dirs(self):
        """确保输出目录存在"""
        dirs = [
            Path(self.output_dir),
            Path(self.output_dir) / "diagnostics",
            Path(self.output_dir) / "thresholds",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """
    加载配置
    
    优先级: overrides > YAML文件 > 默认值
    
    Args:
        config_path: YAML配置文件路径，可选
        **overrides: 直接覆盖的配置项
    
    Returns:
        Config 实例
    
    Example:
        # 使用默认配置
        config = load_config()
        
        # 从YAML加载
        config = load_config("config.yaml")
        
        # 覆盖部分配置
        config = load_config(raw_data_dir="/data/level2", n_workers=16)
    """
    config_dict = {}
    
    # 1. 从YAML加载
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict.update(yaml_config)
    
    # 2. 环境变量覆盖 (L2_前缀)
    env_mapping = {
        "L2_RAW_DATA_DIR": "raw_data_dir",
        "L2_OUTPUT_DIR": "output_dir",
        "L2_N_WORKERS": "n_workers",
        "L2_USE_POLARS": "use_polars",
    }
    for env_key, config_key in env_mapping.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            # 类型转换
            if config_key in ["n_workers"]:
                env_val = int(env_val)
            elif config_key in ["use_polars"]:
                env_val = env_val.lower() in ("true", "1", "yes")
            config_dict[config_key] = env_val
    
    # 3. 直接覆盖
    config_dict.update(overrides)
    
    # 4. 处理 tuple 类型（YAML加载后变成list）
    if "percentiles" in config_dict and isinstance(config_dict["percentiles"], list):
        config_dict["percentiles"] = tuple(config_dict["percentiles"])
    if "channel_names" in config_dict and isinstance(config_dict["channel_names"], list):
        config_dict["channel_names"] = tuple(config_dict["channel_names"])
    
    return Config(**config_dict)


# ========== 通道常量（便于引用）==========
class Channels:
    """通道索引常量"""
    ALL_TRADE = 0           # 全部成交
    ACTIVE_BUY_TRADE = 1    # 主动买入成交
    ACTIVE_SELL_TRADE = 2   # 主动卖出成交
    BIG_BUY = 3             # 大买单
    BIG_SELL = 4            # 大卖单
    SMALL_BUY = 5           # 小买单
    SMALL_SELL = 6          # 小卖单
    BUY_ORDER = 7           # 买单委托
    SELL_ORDER = 8          # 卖单委托
    ACTIVE_BUY_ORDER = 9    # 主动买入委托
    ACTIVE_SELL_ORDER = 10  # 主动卖出委托
    PASSIVE_BUY = 11        # 非主动买入
    PASSIVE_SELL = 12       # 非主动卖出
    CANCEL_BUY = 13         # 撤买
    CANCEL_SELL = 14        # 撤卖
    
    # 通道分组
    TRADE_CHANNELS = (0, 1, 2, 3, 4, 5, 6)
    ORDER_CHANNELS = (7, 8, 9, 10, 11, 12, 13, 14)
    BIG_ORDER_CHANNELS = (3, 4)
    SMALL_ORDER_CHANNELS = (5, 6)
    ACTIVE_CHANNELS = (1, 2, 9, 10)
    CANCEL_CHANNELS = (13, 14)


# 默认全局配置实例
_default_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例（单例模式）"""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


def set_config(config: Config):
    """设置全局配置实例"""
    global _default_config
    _default_config = config


if __name__ == "__main__":
    # 测试配置
    config = load_config()
    print("默认配置:")
    print(f"  图像形状: {config.image_shape}")
    print(f"  使用Polars: {config.use_polars}")
    print(f"  并行数: {config.n_workers}")
    print(f"  通道名称: {config.channel_names}")
    
    # 保存示例YAML
    config.to_yaml("config_example.yaml")
    print("\n已保存示例配置到 config_example.yaml")
