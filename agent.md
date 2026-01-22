# Level2 图像构建项目开发进度追踪

> 本文档由 AI Agent 自动维护，记录项目开发进度和实现细节

## 📋 项目概述

- **项目名称**: L2 Image Builder (Level2 数据图像化处理)
- **创建日期**: 2026-01-21
- **最后更新**: 2026-01-21 (Prompt 5.1)
- **当前状态**: 开发中
- **目标**: 将 Level2 逐笔成交与逐笔委托数据转换为 `[15, 8, 8]` 三维图像格式
- **执行环境**: conda中的 torch1010
---

## ✅ 已实现功能

### Phase 1: 基础设施层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| config.py | 配置管理 | ✅ 完成 | 2026-01-21 | 支持 YAML、环境变量、dataclass 默认值 |
| config.py | Channels 常量类 | ✅ 完成 | 2026-01-21 | 15 通道索引常量定义 |
| polars_utils.py | Polars/Pandas 互操作 | ✅ 增强 | 2026-01-21 | Prompt 1.2: 懒加载、批量处理 |
| sh_loader.py | 上交所数据加载 | ✅ 增强 | 2026-01-21 | Prompt 1.2: 懒加载、迭代器 |
| sz_loader.py | 深交所数据加载 | ✅ 增强 | 2026-01-21 | Prompt 1.2: 懒加载、迭代器 |
| time_filter.py | 时间过滤 | ✅ 增强 | 2026-01-21 | Prompt 1.3: Polars 向量化 |
| anomaly_filter.py | 异常值过滤 | ✅ 增强 | 2026-01-21 | Prompt 1.3: 撤单专用过滤 |
| sz_cancel_enricher.py | 深交所撤单价格关联 | ✅ 增强 | 2026-01-21 | Prompt 2.3: 分离撤买/撤卖、缓存 |
| data_cleaner.py | 数据清洗整合类 | ✅ 新增 | 2026-01-21 | Prompt 1.3: DataCleaner |

### Phase 2: 核心计算层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| quantile.py | 分位数计算 | ✅ 增强 | 2026-01-21 | Prompt 2.1: 沪深分离、向量化 |
| quantile.py | 验证诊断 | ✅ 新增 | 2026-01-21 | Prompt 2.1: 分布验证、可视化 |
| big_order.py | 母单还原 | ✅ 增强 | 2026-01-21 | Prompt 2.2: Polars 向量化、撤单过滤 |
| big_order.py | 当日阈值计算 | ✅ 增强 | 2026-01-21 | Prompt 2.2: Mean+Std、验证诊断 |

### Phase 3: 图像构建层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| image_builder.py | 15 通道图像构建 | ✅ 增强 | 2026-01-21 | Prompt 3.3: 统一入口 build_single_stock |
| normalizer.py | Log1p + Max 归一化 | ✅ 完成 | 2026-01-21 | 通道内归一化、ImageNormalizer 类 |
| sh_builder.py | 上交所图像构建器 | ✅ 新增 | 2026-01-21 | Prompt 3.1: 向量化实现 |
| sz_builder.py | 深交所图像构建器 | ✅ 新增 | 2026-01-21 | Prompt 3.2: 向量化实现 |

### Phase 4: 存储与输出层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| storage/lmdb_writer.py | LMDB 写入器 | ✅ 完成 | 2026-01-21 | Prompt 4.1: LZ4 压缩、批量写入 |
| storage/lmdb_reader.py | LMDB 读取器 | ✅ 完成 | 2026-01-21 | Prompt 4.1: 并发读取、多日管理 |
| storage/__init__.py | 存储模块导出 | ✅ 完成 | 2026-01-21 | Prompt 4.1: 便捷函数导出 |
| diagnostics/__init__.py | 诊断模块骨架 | ✅ 完成 | 2026-01-21 | Prompt 4.2: 通道填充率监控 |
| diagnostics/reporter.py | 诊断报告器 | ✅ 完成 | 2026-01-21 | Prompt 4.2: 健康检查、日报生成 |
| dataset/__init__.py | 数据集模块骨架 | ✅ 完成 | 2026-01-21 | Prompt 4.2: ViT/ViViT Dataset |
| dataset/vit_dataset.py | ViT 单日数据集 | ✅ 完成 | 2026-01-21 | Prompt 4.2: PyTorch Dataset |
| dataset/vivit_dataset.py | ViViT 序列数据集 | ✅ 完成 | 2026-01-21 | Prompt 4.2: 20日序列 Dataset |

### Phase 5: 优化与生产化

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| main.py | 主入口 | ✅ 增强 | 2026-01-21 | Prompt 5.1: Dask 并行支持 |
| main.py | process_single_stock | ✅ 新增 | 2026-01-21 | Prompt 5.1: 单股票处理函数 |
| main.py | process_daily_dask | ✅ 新增 | 2026-01-21 | Prompt 5.1: Dask 并行处理 |
| main.py | batch_process | ✅ 新增 | 2026-01-21 | Prompt 5.1: 批量处理函数 |
| scripts/batch_process.py | BatchProcessor | ✅ 新增 | 2026-01-21 | Prompt 5.1: 批量处理类 |
| scripts/batch_process.py | run_backfill | ✅ 新增 | 2026-01-21 | Prompt 5.1: 历史回填 |
| scripts/batch_process.py | run_daily_update | ✅ 新增 | 2026-01-21 | Prompt 5.1: 每日更新 |
| Dask 并行 | 批量处理 | ✅ 完成 | 2026-01-21 | 多进程加速 |

---

## 🔌 接口定义

### 配置管理 (config.py)

```python
@dataclass
class Config:
    """Level2 图像构建配置类"""
    raw_data_dir: str = "/raw_data"
    output_dir: str = "/processed_data/l2_images"
    num_channels: int = 15
    num_price_bins: int = 8
    num_qty_bins: int = 8
    percentiles: Tuple[float, ...] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5)
    threshold_std_multiplier: float = 1.0
    use_polars: bool = True
    n_workers: int = 8

def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """加载配置，优先级: overrides > YAML > 默认值"""

class Channels:
    """通道索引常量"""
    ALL_TRADE = 0
    ACTIVE_BUY_TRADE = 1
    # ... 共 15 个通道
```

### 数据加载 (data_loader/) - Prompt 1.2 增强

```python
class SHDataLoader:
    """上交所数据加载器（Prompt 1.2 增强版）"""
    
    # 列定义常量
    TRADE_COLUMNS_MINIMAL = ["SecurityID", "TickTime", "Price", "Qty", "TickBSFlag"]
    ORDER_COLUMNS_MINIMAL = ["SecurityID", "TickTime", "OrdType", "Side", "Price", "Qty"]
    
    def __init__(self, raw_data_dir: str, use_polars: bool = True,
                 default_time_filter: bool = True)
    
    # 基础加载（已有 + 增强）
    def load_trade(self, date: str, columns: List[str] = None,
                   time_filter: bool = None, minimal_columns: bool = False) -> DataFrame
    def load_order(self, date: str, columns: List[str] = None,
                   time_filter: bool = None, minimal_columns: bool = False) -> DataFrame
    def load_both(self, date: str, minimal_columns: bool = False) -> Tuple[DataFrame, DataFrame]
    
    # 懒加载方法（Prompt 1.2 新增）
    def load_trade_lazy(self, date: str, stock_codes: List[str] = None,
                        time_ranges: List[Tuple] = None, minimal_columns: bool = False) -> pl.LazyFrame
    def load_order_lazy(self, date: str, stock_codes: List[str] = None,
                        time_ranges: List[Tuple] = None, minimal_columns: bool = False) -> pl.LazyFrame
    def load_both_lazy(self, date: str, stock_codes: List[str] = None) -> Tuple[LazyFrame, LazyFrame]
    
    # 单股票加载（增强）
    def load_trade_for_stock(self, date: str, stock_code: str,
                             time_filter: bool = None, minimal_columns: bool = False) -> DataFrame
    def load_order_for_stock(self, date: str, stock_code: str, ...) -> DataFrame
    def load_both_for_stock(self, date: str, stock_code: str, ...) -> Tuple[DataFrame, DataFrame]
    
    # 批量/迭代方法（Prompt 1.2 新增）
    def get_stock_list(self, date: str, data_type: str = "trade") -> List[str]
    def iter_stocks_trade(self, date: str, stock_codes: List[str] = None) -> Iterator[Tuple[str, DataFrame]]
    def iter_stocks_order(self, date: str, stock_codes: List[str] = None) -> Iterator[Tuple[str, DataFrame]]
    def iter_stocks_both(self, date: str, stock_codes: List[str] = None) -> Iterator[Tuple[str, DataFrame, DataFrame]]
    def batch_load_trade(self, date: str, stock_codes: List[str], batch_size: int = 50) -> Iterator[Tuple[List[str], DataFrame]]
    def batch_load_order(self, date: str, stock_codes: List[str], batch_size: int = 50) -> Iterator[Tuple[List[str], DataFrame]]

class SZDataLoader:
    """深交所数据加载器（Prompt 1.2 增强版）"""
    # 与 SHDataLoader 类似的接口
    # 额外方法:
    def enrich_cancel_price(self, trade_df, order_df) -> DataFrame
    def build_active_seqs(self, trade_df) -> Dict[str, Set[int]]
    def build_active_seqs_fast(self, trade_df) -> Dict[str, Set[int]]  # 向量化版本
```

### Polars 工具函数 (polars_utils.py) - Prompt 1.2 增强

```python
# 懒加载函数（新增）
def read_parquet_lazy(filepath: str, columns: List[str] = None) -> pl.LazyFrame
def scan_parquet_with_filter(filepath: str, columns: List[str] = None,
                              stock_codes: List[str] = None,
                              time_ranges: List[Tuple] = None) -> pl.LazyFrame
def collect_lazy(lf: pl.LazyFrame, streaming: bool = False) -> pl.DataFrame

# 迭代函数（新增）
def iter_stocks_lazy(lf: pl.LazyFrame, stock_codes: List[str]) -> Iterator[Tuple[str, DataFrame]]
def get_stock_list_from_parquet(filepath: str) -> List[str]
def batch_load_stocks(filepath: str, stock_codes: List[str], batch_size: int = 50) -> Iterator[Tuple[List[str], DataFrame]]
```

class SZDataLoader:
    """深交所数据加载器"""
    def __init__(self, raw_data_dir: str, use_polars: bool = True)
    def load_trade(self, date: str, ...) -> DataFrame
    def load_order(self, date: str, ...) -> DataFrame
    def enrich_cancel_price(self, trade_df, order_df) -> DataFrame
    def build_active_seqs(self, trade_df) -> Dict[str, Set[int]]
```

### 计算模块 (calculator/)

```python
class QuantileCalculator:
    """分位数计算器"""
    def compute(self, df_trade, df_order, date) -> Tuple[np.ndarray, np.ndarray]
    def get_price_bin(self, price: float) -> int
    def get_qty_bin(self, qty: float) -> int

class BigOrderCalculator:
    """大单计算器"""
    def compute(self, df_trade, exchange: str, date) -> float
    def is_big_order(self, order_no: int, side: str) -> bool
    def get_order_amount(self, order_no: int, side: str) -> float
    def clear(self) -> None

# Prompt 2.2 新增便捷函数
def compute_all(df_trade, exchange, std_multiplier=1.0) -> Tuple[Dict, Dict, float]
def validate_threshold(threshold, buy_parent, sell_parent) -> Dict
def compute_big_order_statistics(buy_parent, sell_parent, threshold) -> Dict
```

### 图像构建 (builder/)

```python
class Level2ImageBuilder:
    """单只股票单日的图像构建器"""
    def __init__(self, stock_code: str, trade_date: str, config: Config = None)
    def build(self, df_trade, df_order, price_bins, qty_bins,
              buy_parent_amount, sell_parent_amount, threshold,
              active_seqs=None) -> np.ndarray  # [15, 8, 8]
    def normalize(self) -> np.ndarray

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Log1p + 通道内 Max 归一化"""
```

---

## 🔗 依赖关系

### 模块依赖图

```
l2_image_builder/
├── config.py                    # 独立，被所有模块依赖
├── data_loader/
│   ├── polars_utils.py          # 基础工具，被 loader 依赖
│   ├── sh_loader.py             # 依赖 polars_utils
│   └── sz_loader.py             # 依赖 polars_utils
├── cleaner/
│   ├── time_filter.py           # 依赖 polars_utils
│   ├── anomaly_filter.py        # 依赖 polars_utils
│   └── sz_cancel_enricher.py    # 依赖 polars_utils
├── calculator/
│   ├── quantile.py              # 依赖 polars_utils
│   └── big_order.py             # 依赖 polars_utils
├── builder/
│   ├── normalizer.py            # 依赖 numpy
│   └── image_builder.py         # 依赖 config, calculator, normalizer
└── main.py                      # 依赖所有模块
```

### 外部依赖

| 包名 | 版本要求 | 用途 | 必须 |
|------|----------|------|------|
| polars | >=0.19.0 | 高性能数据处理 | 推荐 |
| pandas | >=1.5.0 | 数据处理（备选） | 是 |
| numpy | >=1.20.0 | 数值计算 | 是 |
| pyyaml | >=6.0 | 配置文件解析 | 是 |
| lmdb | >=1.0.0 | 图像存储 | Phase 4 |
| lz4 | >=4.0.0 | 压缩 | Phase 4 |
| dask | >=2023.1.0 | 并行处理 | Phase 5 |

---

## ⚠️ 注意事项

### 重要约定

1. **上交所委托表已预处理**: `Qty` 字段已是完整母单量，无需再聚合
2. **深交所撤单价格为 0**: 必须调用 `enrich_cancel_price()` 关联委托表
3. **大小单判定与主动方向无关**: 每笔成交同时判定买卖双方
4. **通道 9/10 沪深对齐**: 深交所也用成交表填充（指鹿为马）

### 数据字段映射

| 字段含义 | 上交所 | 深交所 |
|----------|--------|--------|
| 时间 | TickTime | TransactTime |
| 价格 | Price | Price / LastPx |
| 数量 | Qty | Qty / OrderQty / LastQty |
| 买方 | BuyOrderNO | BidApplSeqNum |
| 卖方 | SellOrderNO | OfferApplSeqNum |
| 主动方向 | TickBSFlag='B'/'S' | BidSeq > OfferSeq |

### 性能考虑

1. 优先使用 Polars 的懒加载 (`scan_parquet`)
2. 使用向量化操作，避免 `iterrows()`
3. 大批量处理时使用 Dask 多进程

---

## 📜 变更日志

### [2026-01-21] - Prompt 3.3 归一化与整合构建器

**目标:**
提供统一的图像构建入口，自动完成分位数计算 → 母单还原 → 阈值计算 → 图像构建 → 归一化的完整流程。

**归一化方案:**
- 公式: `X_final = log(1 + X) / max(log(1 + X))`
- Log1p 变换解决长尾分布问题
- 通道内 Max 归一化到 [0, 1]

**更新:**
- `builder/image_builder.py`:
  - `Level2ImageBuilder.build_single_stock()` - 统一入口方法（自动完成全流程）
  - `Level2ImageBuilder.build_image()` - 类方法快速构建
  - `build_l2_image()` - 便捷函数
  - `build_l2_image_with_stats()` - 构建并返回统计信息

- `builder/normalizer.py`:
  - `normalize_image()` - 完整归一化流程
  - `log1p_normalize()` - Log1p 变换
  - `channel_max_normalize()` - 通道内 Max 归一化
  - `ImageNormalizer` 类 - 支持保存参数用于反归一化
  - `compute_channel_statistics()` - 通道统计计算

- `builder/__init__.py`:
  - 导出所有新增函数和类

- `tests/test_integration_builder.py`:
  - 归一化测试（6 个用例）
  - 上交所整合测试（5 个用例）
  - 深交所整合测试（4 个用例）
  - 边界条件测试（5 个用例）
  - 配置测试（3 个用例）
  - Polars/Pandas 一致性测试（2 个用例）

**API 示例:**
```python
from l2_image_builder.builder import Level2ImageBuilder, build_l2_image

# 方式 1: 类实例化
builder = Level2ImageBuilder("600519.SH", "2026-01-21")
image = builder.build_single_stock(df_trade, df_order)  # 自动完成全流程

# 方式 2: 类方法
image = Level2ImageBuilder.build_image("600519.SH", df_trade, df_order)

# 方式 3: 便捷函数
image = build_l2_image("600519.SH", df_trade, df_order)

# 带统计信息
from l2_image_builder.builder import build_l2_image_with_stats
image, stats, raw = build_l2_image_with_stats("600519.SH", df_trade, df_order)
```

### [2026-01-21] - Prompt 4.1 LMDB 存储模块

**新增:**
- `storage/lmdb_writer.py`:
  - `compress_image()` - LZ4 压缩图像数据
  - `decompress_image()` - LZ4 解压图像数据
  - `write_daily_lmdb()` - 写入一天所有股票图像到 LMDB
  - `write_images_batch()` - 使用生成器函数批量写入（内存友好）
  - `append_to_lmdb()` - 向已存在的 LMDB 追加数据
  - `get_lmdb_stats()` - 获取 LMDB 文件统计信息
  - `LMDBWriter` 类 - 支持上下文管理和增量写入
  - 常量: `IMAGE_SHAPE=(15,8,8)`, `IMAGE_DTYPE=np.float32`, `IMAGE_SIZE_BYTES=3840`

- `storage/lmdb_reader.py`:
  - `LMDBReader` 类 - 支持并发读取的 LMDB 读取器
    - `read()` - 读取单只股票图像
    - `read_batch()` - 批量读取
    - `list_keys()` - 列出所有股票代码
    - `has_key()` / `__contains__` - 检查存在性
    - `iter_items()` - 迭代所有记录
    - `get_stats()` - 获取统计信息
    - `__len__` / `__getitem__` 支持
  - `read_daily_lmdb()` - 便捷函数读取 LMDB 文件
  - `read_single_stock()` - 便捷函数读取单只股票
  - `get_lmdb_keys()` - 便捷函数获取所有 key
  - `MultiDayLMDBReader` 类 - 多日数据管理器

- `storage/__init__.py`:
  - 导出所有写入和读取函数/类

- `tests/test_lmdb_storage.py`:
  - 压缩/解压测试（5 个用例）
  - write_daily_lmdb 测试（6 个用例）
  - LMDBReader 测试（10 个用例）
  - 并发读取测试（2 个用例）
  - LMDBWriter 测试（3 个用例）
  - MultiDayLMDBReader 测试（3 个用例）
  - 便捷函数测试（2 个用例）
  - 统计信息测试（1 个用例）
  - 边界情况测试（4 个用例）

**存储规格:**
- **文件组织**: 每日一个 LMDB 文件，如 `20230101.lmdb`
- **Key 格式**: `"Code.Exchange"` (如 `"600519.SH"`)
- **Value 格式**: LZ4 压缩的 `numpy.tobytes()`, float32
- **图像形状**: 固定 `(15, 8, 8)`
- **压缩效率**: 原始 3,840 bytes → 压缩后约 200-500 bytes（稀疏数据）

**API 示例:**
```python
from l2_image_builder.storage import (
    write_daily_lmdb, LMDBReader, MultiDayLMDBReader, get_lmdb_stats
)

# 写入一天的图像
images = {"600519.SH": image1, "000001.SZ": image2}
lmdb_path = write_daily_lmdb("20230101", images, "/data/lmdb")

# 读取
with LMDBReader(lmdb_path) as reader:
    image = reader.read("600519.SH")
    keys = reader.list_keys()
    print(f"记录数: {len(reader)}")

# 多日读取
with MultiDayLMDBReader("/data/lmdb") as reader:
    reader.load_dates(["20230101", "20230102"])
    image = reader.read("600519.SH", "20230101")
    dates = reader.list_available_dates()

# 统计信息
stats = get_lmdb_stats(lmdb_path)
print(f"压缩率: {stats['compression_ratio']:.2f}x")
```

### [2026-01-21] - Prompt 4.2 诊断报告与Dataset

**目标:**
提供图像质量诊断工具和 PyTorch Dataset 类，用于训练 ViT/ViViT 模型。

**新增:**
- `diagnostics/reporter.py`:
  - `CHANNEL_NAMES` - 15 通道名称常量（如 'all_trade', 'active_buy', ...）
  - `TRADE_CHANNELS = [0-6]` - 成交相关通道索引
  - `ORDER_CHANNELS = [7-14]` - 委托相关通道索引
  - `HEALTH_THRESHOLDS` - 健康阈值字典：
    - trade_fill_rate_min: 0.30
    - order_fill_rate_min: 0.50
    - big_order_ratio_min: 0.05, big_order_ratio_max: 0.30
    - cancel_rate_max: 0.50
  - `compute_channel_metrics()` - 计算单通道指标（nonzero_count, fill_rate, total_sum, max_value, concentration）
  - `compute_stock_metrics()` - 计算股票级指标（trade_sum, order_sum, big_order_ratio, cancel_rate 等）
  - `generate_stock_diagnostics()` - 生成完整诊断字典
  - `check_health()` - 健康检查，返回警告消息列表
  - `generate_daily_report()` - 生成 DataFrame 日报并可选保存 CSV
  - `generate_summary_statistics()` - 聚合统计信息
  - `print_daily_summary()` - 控制台打印摘要
  - `DiagnosticsReporter` 类 - 批量处理支持
    - `add_stock()` / `add_batch()` - 添加股票诊断
    - `to_dataframe()` / `save_report()` - 输出报告
    - `get_summary()` / `get_unhealthy_stocks()` - 获取汇总/异常股票

- `dataset/vit_dataset.py`:
  - `ViTDataset` 类 - 单日 LMDB 数据集
    - `__init__(lmdb_path, stock_codes, labels, transform, return_code)`
    - `__len__()` / `__getitem__()` - Dataset 接口
    - `get_image(stock_code)` - 按代码获取图像
    - 支持上下文管理器
  - `ViTDatasetWithMask` 类 - 返回有效性 mask
  - `create_vit_dataloader()` - 便捷函数创建 DataLoader

- `dataset/vivit_dataset.py`:
  - `DEFAULT_SEQ_LEN = 20` - 默认序列长度
  - `ViViTDataset` 类 - 多日序列数据集
    - `__init__(lmdb_dir, dates, stock_codes, seq_len, labels, transform, return_meta)`
    - `__len__()` = dates × codes
    - `__getitem__()` 返回 [seq_len, 15, 8, 8] 序列
    - `get_sequence(stock_code, target_date)` - 按代码/日期获取
    - `list_available_dates()` - 列出已加载日期
    - 前向补零处理（序列不足 seq_len 时）
  - `ViViTDatasetByStock` 类 - 按股票组织，每个样本为全时序
    - `__len__()` = codes
    - `__getitem__()` 返回 [T, 15, 8, 8]
  - `create_vivit_dataloader()` - 便捷函数创建 DataLoader

- `diagnostics/__init__.py`:
  - 导出所有常量、函数和 DiagnosticsReporter 类

- `dataset/__init__.py`:
  - 导出 ViTDataset, ViViTDataset 等类和便捷函数

- `tests/test_diagnostics.py`:
  - TestChannelMetrics: 4 个用例
  - TestStockMetrics: 3 个用例
  - TestGenerateStockDiagnostics: 3 个用例
  - TestCheckHealth: 3 个用例
  - TestGenerateDailyReport: 3 个用例
  - TestSummaryStatistics: 2 个用例
  - TestDiagnosticsReporter: 6 个用例
  - TestEdgeCases: 3 个用例

- `tests/test_dataset.py`:
  - TestViTDataset: 7 个用例
  - TestViTDatasetWithMask: 2 个用例
  - TestViViTDataset: 7 个用例
  - TestViViTDatasetByStock: 1 个用例
  - TestDataLoaders: 2 个用例
  - TestTransform: 2 个用例
  - TestErrorHandling: 2 个用例
  - TestEdgeCases: 2 个用例

**健康检查规则:**
| 指标 | 阈值 | 告警条件 |
|------|------|----------|
| trade_fill_rate | 0.30 | < 30% 成交通道非零填充率 |
| order_fill_rate | 0.50 | < 50% 委托通道非零填充率 |
| big_order_ratio | [0.05, 0.30] | 大单比例异常 |
| cancel_rate | 0.50 | > 50% 撤单比例过高 |

**API 示例:**
```python
# 诊断报告
from l2_image_builder.diagnostics import (
    generate_stock_diagnostics, check_health, DiagnosticsReporter
)

diagnostics = generate_stock_diagnostics(image, "600519.SH", "20230101")
warnings = check_health(diagnostics)

reporter = DiagnosticsReporter("20230101")
reporter.add_stock(image, "600519.SH")
reporter.add_stock(image2, "000001.SZ")
df = reporter.to_dataframe()
reporter.save_report("/data/reports")

# ViT Dataset
from l2_image_builder.dataset import ViTDataset, create_vit_dataloader

with ViTDataset(lmdb_path, stock_codes, labels=labels) as dataset:
    image, label = dataset[0]

loader = create_vit_dataloader(lmdb_path, stock_codes, batch_size=32)

# ViViT Dataset
from l2_image_builder.dataset import ViViTDataset, create_vivit_dataloader

with ViViTDataset(lmdb_dir, dates, stock_codes, seq_len=20) as dataset:
    sequence = dataset[0]  # [20, 15, 8, 8]

loader = create_vivit_dataloader(lmdb_dir, dates, stock_codes, batch_size=8)
```

### [2026-01-21] - Prompt 5.1 Dask 并行处理

**目标:**
实现基于 Dask 的大规模并行处理，支持历史数据回填和每日增量更新。

**新增/增强:**
- `main.py` 增强:
  - `process_single_stock()` - 单只股票处理函数（可被 Dask 调度）
  - `get_stock_codes_from_date()` - 获取某日所有股票代码
  - `process_daily_serial()` - 串行处理单日数据
  - `process_daily_dask()` - Dask 并行处理单日数据
  - `batch_process()` - 批量处理多日数据
  - 命令行参数增强: `--parallel`, `--workers`, `--no-lmdb`, `--no-report`

- `scripts/__init__.py` 新增:
  - 导出 BatchProcessor, run_backfill, run_daily_update

- `scripts/batch_process.py` 新增:
  - `BatchProcessor` 类:
    - `__init__()` - 初始化配置、Worker 数量、检查点目录
    - `process_date()` - 处理单日数据（支持断点续传）
    - `run_backfill()` - 回填历史数据
    - `run_daily_update()` - 每日增量更新
    - `_is_processed()` / `_mark_processed()` - 检查点管理
  - `run_backfill()` - 便捷函数
  - `run_daily_update()` - 便捷函数
  - 命令行接口: `backfill`, `daily` 子命令

- `tests/test_parallel.py` 新增:
  - TestGenerateDateRange: 4 个用例
  - TestProcessSingleStock: 2 个用例
  - TestProcessDailySerial: 3 个用例
  - TestBatchProcessor: 6 个用例
  - TestDaskParallel: 1 个用例（需要 Dask）
  - TestConvenienceFunctions: 2 个用例
  - TestStatistics: 1 个用例
  - TestEdgeCases: 3 个用例

**并行策略:**
| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 按股票并行 | 每只股票独立处理 | 单日数据 |
| 按日期顺序 | 多日依次处理 | 批量回填 |
| LocalCluster | Dask 本地多进程 | 单机运行 |

**性能参数:**
- `n_workers`: Worker 数量（默认 8）
- `threads_per_worker`: 每个 Worker 线程数（默认 1，避免 GIL）
- 支持 tqdm 进度条显示

**API 示例:**
```python
# 命令行使用
# 单日处理
python -m l2_image_builder.main --date 20230101

# 批量处理（串行）
python -m l2_image_builder.main --start-date 20230101 --end-date 20230131

# 批量处理（Dask 并行）
python -m l2_image_builder.main --start-date 20230101 --end-date 20230131 --parallel --workers 8

# 历史回填脚本
python -m l2_image_builder.scripts.batch_process backfill --start 20230101 --end 20231231

# 每日更新脚本
python -m l2_image_builder.scripts.batch_process daily --days 3

# Python 代码调用
from l2_image_builder.main import batch_process, process_daily_dask
from l2_image_builder.scripts.batch_process import BatchProcessor, run_backfill

# 方式 1: 直接调用
batch_process(dates, config, n_workers=8, parallel=True)

# 方式 2: 使用 BatchProcessor 类
processor = BatchProcessor(config, n_workers=8)
processor.run_backfill("20230101", "20231231")
processor.run_daily_update(days=3)

# 方式 3: 便捷函数
run_backfill("20230101", "20231231", n_workers=8)
run_daily_update(days=3, parallel=True)
```

### [2026-01-21] - Prompt 3.2 深交所图像构建器

**问题背景:**
深交所与上交所数据结构差异大：主动方向需通过 BidApplSeqNum vs OfferApplSeqNum 比较判定，撤单在成交表中（ExecType='52'），需追踪主动方序列号集合来识别纯挂单。

**新增:**
- `builder/sz_builder.py`:
  - `SZImageBuilder` 类 - 深交所专用图像构建器
    - `build()` - 逐行构建方法
    - `build_vectorized()` - 向量化构建方法（推荐）
    - `_build_active_seqs()` / `_build_active_seqs_vectorized()` - 构建主动方序列号集合
    - `_process_trades_vectorized()` - 处理成交记录（通道0-6, 9-10）
    - `_process_cancels_vectorized()` - 处理撤单记录（通道13-14）
    - `_process_orders_vectorized()` - 处理委托记录（通道7-8, 11-12）
    - `get_channel_stats()` - 获取通道统计信息
    - `validate_consistency()` - 验证图像一致性
  - `build_sz_image()` - 便捷函数
  - `build_sz_image_with_stats()` - 构建并返回统计信息
  - `build_active_seqs_from_trade()` - 独立构建主动方序列号集合

- `tests/test_sz_builder.py`:
  - 基础测试（2 个用例）
  - 主动方序列号测试（3 个用例）
  - Pandas 测试（3 个用例）
  - Polars 测试（4 个用例）
  - 通道填充测试（9 个用例）
  - 边界情况测试（4 个用例）
  - 统计和验证测试（2 个用例）
  - 便捷函数测试（2 个用例）
  - 性能测试（1 个用例）

**通道填充规则（深交所）:**
| 通道 | 名称 | 数据源 | 筛选条件 |
|------|------|--------|----------|
| 0 | 全部成交 | 成交表 | ExecType='70' |
| 1 | 主动买入 | 成交表 | BidApplSeqNum > OfferApplSeqNum |
| 2 | 主动卖出 | 成交表 | OfferApplSeqNum > BidApplSeqNum |
| 3 | 大买单 | 成交表 | 买方母单≥阈值 |
| 4 | 大卖单 | 成交表 | 卖方母单≥阈值 |
| 5 | 小买单 | 成交表 | 买方母单<阈值 |
| 6 | 小卖单 | 成交表 | 卖方母单<阈值 |
| 7 | 买单 | 委托表 | Side='49' |
| 8 | 卖单 | 委托表 | Side='50' |
| 9 | 主动买入(委托) | 成交表 | 同通道1（指鹿为马） |
| 10 | 主动卖出(委托) | 成交表 | 同通道2（指鹿为马） |
| 11 | 非主动买入 | 委托表 | Side='49' & ApplSeqNum不在active_buy中 |
| 12 | 非主动卖出 | 委托表 | Side='50' & ApplSeqNum不在active_sell中 |
| 13 | 撤买 | 成交表 | ExecType='52' & BidApplSeqNum>0 |
| 14 | 撤卖 | 成交表 | ExecType='52' & OfferApplSeqNum>0 |

**核心差异（vs 上交所）:**
1. **主动方向**: 深交所比较 BidApplSeqNum vs OfferApplSeqNum，大的是主动方
2. **撤单位置**: 深交所撤单在成交表（ExecType='52'），上交所在委托表（OrdType='Cancel'）
3. **纯挂单判定**: 深交所需要 active_seqs 追踪，上交所预处理后直接按 OrdType 判断
4. **通道9-10**: 深交所从成交表填充（指鹿为马对齐），等于通道1-2

**API 示例:**
```python
from l2_image_builder.builder.sz_builder import (
    SZImageBuilder, build_sz_image, build_active_seqs_from_trade
)

# 方式 1: 类方式
builder = SZImageBuilder(price_bins, qty_bins, buy_parent, sell_parent, threshold)
image = builder.build_vectorized(df_trade, df_order)
stats = builder.get_channel_stats()
consistency = builder.validate_consistency()

# 方式 2: 便捷函数
image = build_sz_image(df_trade, df_order, price_bins, qty_bins, buy_parent, sell_parent, threshold)

# 独立获取主动方序列号集合
active_seqs = build_active_seqs_from_trade(df_trade)
```

### [2026-01-21] - Prompt 2.3 深交所撤单价格关联

**问题背景:**
深交所撤单记录(ExecType='52')的 LastPx = 0，直接使用会导致所有撤单都映射到 price_bin=0。

**新增/增强:**
- `cleaner/sz_cancel_enricher.py`:
  - `enrich_sz_cancel_price_polars()` - Polars 向量化撤单价格关联（分离撤买/撤卖）
  - `enrich_sz_cancel_price_pandas()` - Pandas 版本
  - `enrich_sz_cancel_price()` - 自动选择引擎
  - `validate_cancel_prices()` - 验证撤单价格是否全部关联
  - `get_cancel_statistics()` - 撤单统计（撤买/撤卖数量、未关联数）
  - `print_cancel_summary()` - 打印撤单处理摘要
  - `SZCancelEnricher` 类增强 - 支持缓存、批量处理

- `tests/test_sz_cancel_enricher.py`:
  - Pandas 版本测试（6 个用例）
  - Polars 版本测试（3 个用例）
  - 验证函数测试（3 个用例）
  - 统计函数测试（4 个用例）
  - SZCancelEnricher 类测试（7 个用例）
  - 边界情况测试（4 个用例）

**关联逻辑:**
- BidApplSeqNum > 0 → 撤买单 → 用 BidApplSeqNum 关联委托表
- OfferApplSeqNum > 0 → 撤卖单 → 用 OfferApplSeqNum 关联委托表

**API 示例:**
```python
from l2_image_builder.cleaner.sz_cancel_enricher import (
    enrich_sz_cancel_price, validate_cancel_prices, SZCancelEnricher
)

# 方式 1: 直接关联
df_enriched = enrich_sz_cancel_price(df_trade, df_order)
is_valid = validate_cancel_prices(df_enriched)

# 方式 2: 使用缓存（批量处理时更高效）
enricher = SZCancelEnricher()
enricher.build_cache(df_order, date='2026-01-21')
df_enriched = enricher.enrich(df_trade)
```

### [2026-01-21] - Prompt 3.1 上交所图像构建器（简化版）

**新增:**
- `builder/sh_builder.py`:
  - `SHImageBuilder` 类 - 上交所专用图像构建器
    - `build()` - 逐行构建方法
    - `build_vectorized()` - 向量化构建方法（推荐，更高性能）
    - `get_channel_stats()` - 获取通道统计信息
    - `validate_consistency()` - 验证图像一致性
  - `build_sh_image()` - 便捷函数
  - `build_sh_image_with_stats()` - 构建并返回统计信息

- `tests/test_sh_builder.py`:
  - 基础测试（2 个用例）
  - Pandas 测试（3 个用例）
  - Polars 测试（4 个用例）
  - 通道填充测试（7 个用例）
  - 边界情况测试（4 个用例）
  - 统计和验证测试（2 个用例）
  - 便捷函数测试（2 个用例）
  - 性能测试（1 个用例）

**通道填充规则（上交所简化版）:**
| 通道 | 名称 | 数据源 | 筛选条件 |
|------|------|--------|----------|
| 0 | 全部成交 | 成交表 | 全部 |
| 1 | 主动买入 | 成交表 | BSFlag='B' |
| 2 | 主动卖出 | 成交表 | BSFlag='S' |
| 3 | 大买单 | 成交表 | 买方母单≥阈值 |
| 4 | 大卖单 | 成交表 | 卖方母单≥阈值 |
| 5 | 小买单 | 成交表 | 买方母单<阈值 |
| 6 | 小卖单 | 成交表 | 卖方母单<阈值 |
| 7 | 买单 | 委托表 | OrdType='New' & Side='B' |
| 8 | 卖单 | 委托表 | OrdType='New' & Side='S' |
| 9 | 主动买入(委托) | 成交表 | BSFlag='B' |
| 10 | 主动卖出(委托) | 成交表 | BSFlag='S' |
| 11 | 非主动买入 | 委托表 | OrdType='New' & Side='B' |
| 12 | 非主动卖出 | 委托表 | OrdType='New' & Side='S' |
| 13 | 撤买 | 委托表 | OrdType='Cancel' & Side='B' |
| 14 | 撤卖 | 委托表 | OrdType='Cancel' & Side='S' |

**简化说明:**
- 上交所委托表已预处理，`Qty` 字段为完整母单量
- 撤单的 `Price` 已补全
- 直接按 `OrdType` 和 `Side` 过滤即可
- 通道7=通道11，通道8=通道12（预处理后重叠）
- 通道1=通道9，通道2=通道10（成交表填充）

**API 示例:**
```python
from l2_image_builder.builder.sh_builder import SHImageBuilder, build_sh_image

# 方式 1: 类方式
builder = SHImageBuilder(price_bins, qty_bins, buy_parent, sell_parent, threshold)
image = builder.build_vectorized(df_trade, df_order)
stats = builder.get_channel_stats()
consistency = builder.validate_consistency()

# 方式 2: 便捷函数
image = build_sh_image(df_trade, df_order, price_bins, qty_bins, buy_parent, sell_parent, threshold)
```

### [2026-01-21] - Prompt 2.2 母单还原与当日阈值（简化版）

**新增:**
- `calculator/big_order.py`:
  - `restore_parent_orders_sh_polars()` - 上交所 Polars 向量化母单还原
  - `restore_parent_orders_sz_polars()` - 深交所 Polars 向量化母单还原（含撤单过滤）
  - `restore_parent_orders_sh_pandas()` - 上交所 Pandas 版本
  - `restore_parent_orders_sz_pandas()` - 深交所 Pandas 版本
  - `compute_threshold_daily()` - 当日阈值计算 (Mean + N×Std)
  - `compute_all()` - 一次性完成母单还原和阈值计算
  - `validate_threshold()` - 阈值合理性验证（大单占比 5%-30%）
  - `compute_big_order_statistics()` - 大单统计信息
  - `print_big_order_summary()` - 打印大单摘要

- `tests/test_big_order.py`:
  - 上交所母单还原测试（3 个用例）
  - 深交所母单还原测试（4 个用例）
  - 阈值计算测试（5 个用例）
  - compute_all 测试（4 个用例）
  - 验证函数测试（6 个用例）
  - 统计函数测试（3 个用例）
  - BigOrderCalculator 类测试（4 个用例）
  - 边界情况测试（3 个用例）

**计算规则:**
1. **母单还原**:
   - 上交所: BuyOrderNO/SellOrderNO → TradeMoney（直接字段）
   - 深交所: BidApplSeqNum/OfferApplSeqNum → LastPx × LastQty（计算）
2. **深交所撤单过滤**: 只处理 ExecType='70'（成交），排除 ExecType='52'（撤单）
3. **阈值公式**: Threshold = Mean(V) + std_multiplier × Std(V)，默认 std_multiplier=1.0
4. **验证范围**: 大单占比应在 5%-30%（经验值）

**优点:**
- 无需回溯历史数据
- 无冷启动问题
- 计算简单，性能好

**API 示例:**
```python
# 一次性计算
from l2_image_builder.calculator.big_order import compute_all, validate_threshold

buy_parent, sell_parent, threshold = compute_all(df_trade, 'sh', std_multiplier=1.0)
result = validate_threshold(threshold, buy_parent, sell_parent)
print(f"阈值: {threshold:.2f}, 有效: {result['valid']}, 大单占比: {result['big_order_ratio']:.2%}")
```

### [2026-01-21] - Prompt 2.1 分位数计算（Polars 向量化）

**新增:**
- `calculator/quantile.py`:
  - `compute_quantile_bins_sh_polars()` - 上交所 Polars 向量化分位数计算
  - `compute_quantile_bins_sz_polars()` - 深交所 Polars 向量化分位数计算
  - `compute_quantile_bins_sh_pandas()` - 上交所 Pandas 版本
  - `compute_quantile_bins_sz_pandas()` - 深交所 Pandas 版本
  - `compute_quantile_bins_auto()` - 自动选择引擎
  - `validate_quantile_bins()` - 分位数边界验证
  - `compute_bin_distribution()` - 分布统计
  - `visualize_quantile_distribution()` - 可视化（需 matplotlib）
  - `print_quantile_summary()` - 打印摘要信息

- `tests/test_quantile.py`:
  - 基础函数测试（3 个用例）
  - 上交所分位数测试（4 个用例）
  - 深交所分位数测试（3 个用例）
  - 自动选择测试（4 个用例）
  - 验证函数测试（5 个用例）
  - QuantileCalculator 类测试（5 个用例）
  - 边界情况测试（3 个用例）

**计算规则:**
1. **联合计算**: 成交数据 + 委托数据合并后统一计算分位数
2. **撤单过滤**: 
   - 上交所: 只取 OrdType='New'，排除撤单
   - 深交所: 只取 ExecType='70'（成交），排除撤单
3. **字段映射**:
   - 上交所: Price, Qty
   - 深交所: LastPx/LastQty (成交), Price/OrderQty (委托)
4. **默认分位数**: [12.5, 25, 37.5, 50, 62.5, 75, 87.5]（7 个切割点定义 8 个 bin）

**说明:**
- 支持 Polars 和 Pandas 两种引擎
- 提供验证和可视化工具辅助诊断

### [2026-01-21] - Prompt 1.3 数据清洗模块（简化版）

**新增:**
- `cleaner/data_cleaner.py`:
  - `DataCleaner` 类 - 整合时间过滤和异常值过滤的统一清洗接口
  - `clean_sh_trade()` - 上交所逐笔成交清洗
  - `clean_sh_order()` - 上交所逐笔委托清洗（区分新单/撤单）
  - `clean_sz_order()` - 深交所逐笔委托清洗
  - `clean_sz_trade()` - 深交所逐笔成交清洗（区分成交/撤单）
  - `clean()` - 通用清洗接口
  - `clean_l2_data()` - 便捷函数

- `cleaner/time_filter.py`:
  - `filter_continuous_auction_polars()` - Polars 向量化时间过滤
  - `filter_continuous_auction_pandas()` - Pandas 向量化时间过滤
  - `filter_continuous_auction_auto()` - 自动选择引擎的时间过滤

- `cleaner/anomaly_filter.py`:
  - `filter_anomalies_polars()` - Polars 向量化异常值过滤
  - `filter_anomalies_pandas()` - Pandas 向量化异常值过滤
  - `filter_anomalies_auto()` - 自动选择引擎的异常值过滤

- `tests/test_cleaner.py`:
  - 时间过滤测试（6 个用例）
  - 异常值过滤测试（4 个用例）
  - DataCleaner 整合测试（7 个用例）
  - 边界情况测试（3 个用例）

**清洗规则:**
1. **时间过滤**: 只保留连续竞价时段
   - 上午: 09:30:00 - 11:30:00 (开区间，不含 11:30)
   - 下午: 13:00:00 - 14:57:00 (开区间，不含 14:57)
2. **异常值过滤**:
   - 非撤单记录: Price > 0 AND Qty > 0
   - 撤单记录: 只检查 Qty > 0（撤单价格可能为 0）
3. **跳过涨跌停过滤**: 简化版本不实现

**撤单识别:**
- 上交所: OrdType = 'Cancel'
- 深交所: ExecType = '52'

**说明:**
- 统一了沪深两市的清洗流程
- 支持 Polars 和 Pandas 两种引擎
- 自动选择版本可根据输入类型自动使用合适的引擎

### [2026-01-21] - Prompt 1.2 数据加载器增强

**新增:**
- `polars_utils.py`:
  - `read_parquet_lazy()` - 懒加载 Parquet，支持列选择
  - `scan_parquet_with_filter()` - 带谓词下推的懒加载
  - `collect_lazy()` - 收集 LazyFrame，支持流式处理
  - `iter_stocks_lazy()` - 按股票懒加载迭代
  - `get_stock_list_from_parquet()` - 从文件获取股票列表
  - `batch_load_stocks()` - 批量加载多只股票数据

- `sh_loader.py`:
  - `TRADE_COLUMNS_MINIMAL` / `ORDER_COLUMNS_MINIMAL` - 最小列常量
  - `load_trade_lazy()` / `load_order_lazy()` / `load_both_lazy()` - 懒加载方法
  - `load_both_for_stock()` - 单股票同时加载成交和委托
  - `get_stock_list()` - 获取日期数据中的股票列表
  - `iter_stocks_trade()` / `iter_stocks_order()` / `iter_stocks_both()` - 按股票迭代
  - `batch_load_trade()` / `batch_load_order()` - 批量加载
  - `get_buy_trades()` / `get_sell_trades()` - 成交筛选辅助方法

- `sz_loader.py`:
  - 与 SHDataLoader 相同的增强接口
  - `build_active_seqs_fast()` - 向量化版本的主动方序列号构建

**修改:**
- `load_trade()` / `load_order()` 增加 `minimal_columns` 参数
- `time_filter` 参数改为 `Optional[bool]`，None 时使用默认设置
- 添加 `default_time_filter` 实例属性

**说明:**
- 懒加载利用 Polars 的谓词下推优化，减少 I/O 和内存占用
- 迭代器方法适用于需要逐个处理股票的场景
- 批量加载适用于并行处理场景

### [2026-01-21] - Phase 1 初始化

**新增:**
- 项目骨架和目录结构
- config.py 配置管理模块
- polars_utils.py Polars/Pandas 工具函数
- sh_loader.py 上交所数据加载器
- sz_loader.py 深交所数据加载器
- time_filter.py 时间过滤模块
- anomaly_filter.py 异常值过滤模块
- sz_cancel_enricher.py 深交所撤单价格关联
- quantile.py 分位数计算
- big_order.py 母单还原与阈值计算
- image_builder.py 图像构建核心
- normalizer.py 归一化处理
- main.py 主入口

**说明:**
- 完成 Phase 1-3 核心功能
- Phase 4-5 骨架已创建，待实现

---

## 🎯 下一步计划

1. ✅ ~~**Prompt 1.2**: 增强数据加载器，添加懒加载、批量处理功能~~
2. ✅ ~~**Prompt 1.3**: 数据清洗模块（简化版）~~
3. ✅ ~~**Prompt 2.1**: 分位数计算（Polars 向量化）~~
4. ✅ ~~**Prompt 2.2**: 母单还原与当日阈值（简化版）~~
5. ✅ ~~**Prompt 2.3**: 深交所撤单价格关联~~
6. ✅ ~~**Prompt 3.1**: 上交所图像构建器（简化版）~~
7. ✅ ~~**Prompt 3.2**: 深交所图像构建器~~
8. ✅ ~~**Prompt 3.3**: 归一化与整合构建器~~
9. ✅ ~~**Prompt 4.1**: LMDB 存储模块~~
10. ✅ ~~**Prompt 4.2**: 诊断报告与Dataset模块~~
11. ✅ ~~**Prompt 5.1**: Dask 并行处理~~
12. **Prompt 5.2**: 监控告警与增量更新（可选）
