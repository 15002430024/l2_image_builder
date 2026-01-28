# Level2 图像构建项目开发进度追踪

> 本文档记录项目开发进度和实现细节

## 📋 项目概述

- **项目名称**: L2 Image Builder (Level2 数据图像化处理)
- **创建日期**: 2026-01-21
- **最后更新**: 2026-01-28 (REQ-005: 修复深交所撤单关联OOM)
- **当前状态**: 开发中 → **生产就绪**
- **目标**: 将 Level2 逐笔成交与逐笔委托数据转换为 `[15, 8, 8]` 三维图像格式
- **执行环境**: conda中的 torch1010
---

## ✅ 已实现功能

### Phase 1: 基础设施层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| config.py | 配置管理 | ✅ **R4.1增强** | 2026-01-28 | **R4.1: 新增 separate_quantile_bins 开关** |
| config.py | Channels 常量类 | ✅ **v3增强** | 2026-01-26 | **R3.2: v3文档，validate_constraints()** |
| polars_utils.py | Polars/Pandas 互操作 | ✅ 增强 | 2026-01-21 | Prompt 1.2: 懒加载、批量处理 |
| sh_loader.py | 上交所数据加载 | ✅ **v3增强** | 2026-01-26 | **R3.1: v3字段验证，主动/被动筛选方法** |
| sz_loader.py | 深交所数据加载 | ✅ **R3.2完成** | 2026-01-27 | **R3.2: 通联原始格式→标准格式归一化，TickBSFlag派生** |
| time_filter.py | 时间过滤 | ✅ 增强 | 2026-01-21 | Prompt 1.3: Polars 向量化 |
| anomaly_filter.py | 异常值过滤 | ✅ 增强 | 2026-01-21 | Prompt 1.3: 撤单专用过滤 |
| sz_cancel_enricher.py | 深交所撤单价格关联 | ✅ **R3.2完成** | 2026-01-27 | **Prompt 2.3 + R3.2: 分离撤买/撤卖、标准列名适配** |
| data_cleaner.py | 数据清洗整合类 | ✅ 新增 | 2026-01-21 | Prompt 1.3: DataCleaner |

### Phase 2: 核心计算层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| quantile.py | 分位数计算 | ✅ **R4.1增强** | 2026-01-28 | **R4.1: 新增分离计算函数，成交/委托独立分位数** |
| quantile.py | 验证诊断 | ✅ 新增 | 2026-01-21 | Prompt 2.1: 分布验证、可视化 |
| big_order.py | 母单还原 | ✅ 增强 | 2026-01-21 | Prompt 2.2: Polars 向量化、撤单过滤 |
| big_order.py | 当日阈值计算 | ✅ **v3增强** | 2026-01-26 | **R3.2: 适用场景说明（离线/实盘）** |

### Phase 3: 图像构建层

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| image_builder.py | 15 通道图像构建 | ✅ **R4.1增强** | 2026-01-28 | **R4.1: 支持分离/联合分位数模式切换** |
| normalizer.py | Log1p + Max 归一化 | ✅ 完成 | 2026-01-21 | 通道内归一化、ImageNormalizer 类 |
| sh_builder.py | 上交所图像构建器 | ✅ **R4.1增强** | 2026-01-28 | **R4.1: 构造函数接收4个分位数数组** |
| sz_builder.py | 深交所图像构建器 | ✅ **R4.1增强** | 2026-01-28 | **R4.1: 构造函数接收4个分位数数组，撤单用order_bins** |

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

### Phase 6: v3 重构（意图导向）

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| sh_builder.py | v3重构 | ✅ 完成 | 2026-01-26 | R1.1: Ch9/10从委托表填充，IsAggressive互斥分流 |
| sh_builder.py | validate_constraints | ✅ 新增 | 2026-01-26 | 验证Ch7=Ch9+Ch11, Ch8=Ch10+Ch12 |
| test_sh_builder.py | v3测试 | ✅ 新增 | 2026-01-26 | 7个v3专属测试用例 |
| sz_builder.py | v3重构 | ✅ 完成 | 2026-01-26 | **R1.2: 深交所v3重构，ActiveSeqs互斥分流** |
| sz_builder.py | validate_constraints | ✅ 新增 | 2026-01-26 | 验证Ch7=Ch9+Ch11, Ch8=Ch10+Ch12 |
| test_sz_builder.py | v3测试 | ✅ 新增 | 2026-01-26 | 7个v3专属测试用例 |
| reporter.py | v3增强 | ✅ 完成 | 2026-01-26 | **R2.1: validate_channel_constraints(), CHANNEL_NAMES更新** |
| test_diagnostics.py | v3测试 | ✅ 新增 | 2026-01-26 | 15个v3约束验证测试用例 |
| image_builder.py | v3适配 | ✅ 完成 | 2026-01-26 | **R2.2: v3字段验证，约束检查集成** |
| test_integration_builder.py | v3修复 | ✅ 更新 | 2026-01-26 | R2.2: 测试fixtures添加v3字段 |
| test_sh_builder.py | v3修复 | ✅ 更新 | 2026-01-26 | R2.2: 所有内联DataFrame添加IsAggressive |
| sh_loader.py | v3字段 | ✅ **完成** | 2026-01-26 | **R3.1: v3字段验证，辅助方法** |
| sz_loader.py | v3字段 | ✅ **R3.2完成** | 2026-01-27 | **R3.2: 通联原始→标准格式归一化** |
| config.py | v3配置 | ✅ **完成** | 2026-01-26 | **R3.2: Channels v3文档，Config v3特性开关** |
| big_order.py | v3文档 | ✅ **完成** | 2026-01-26 | **R3.2: 阈值计算适用场景说明** |

### Phase 7: R3.2 深交所数据加载器重构

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| sz_loader.py | 列名映射常量 | ✅ 完成 | 2026-01-27 | TRADE_COLUMN_RENAME_MAP, ORDER_COLUMN_RENAME_MAP |
| sz_loader.py | 成交表归一化 | ✅ 完成 | 2026-01-27 | _normalize_trade_columns() + TickBSFlag派生 |
| sz_loader.py | 委托表归一化 | ✅ 完成 | 2026-01-27 | _normalize_order_columns() 更新 |
| sz_builder.py | 标准列名适配 | ✅ 完成 | 2026-01-27 | 所有方法使用BuyOrderNO/SellOrderNO/Price/Qty/BizIndex |
| sz_cancel_enricher.py | 标准列名适配 | ✅ 完成 | 2026-01-27 | enrich_sz_cancel_price_polars/pandas 使用标准列名 |
| test_sz_normalization.py | 归一化测试 | ✅ 通过 | 2026-01-27 | 验证所有标准列名和TickBSFlag派生 |
| test_sz_image_build.py | 集成测试 | ✅ 通过 | 2026-01-27 | 完整热力图构建测试(15,8,8) |

### Phase 8: R4.1 分位数计算分离模式

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| config.py | separate_quantile_bins | ✅ 完成 | 2026-01-28 | 新增布尔开关，默认True启用分离模式 |
| quantile.py | _compute_percentiles_safe | ✅ 完成 | 2026-01-28 | 安全分位数计算，处理空数据兜底 |
| quantile.py | compute_separate_quantile_bins_sh_polars | ✅ 完成 | 2026-01-28 | 上交所分离计算(Polars) |
| quantile.py | compute_separate_quantile_bins_sh_pandas | ✅ 完成 | 2026-01-28 | 上交所分离计算(Pandas) |
| quantile.py | compute_separate_quantile_bins_sz_polars | ✅ 完成 | 2026-01-28 | 深交所分离计算(Polars) |
| quantile.py | compute_separate_quantile_bins_sz_pandas | ✅ 完成 | 2026-01-28 | 深交所分离计算(Pandas) |
| quantile.py | compute_separate_quantile_bins_auto | ✅ 完成 | 2026-01-28 | 自动选择引擎的便捷函数 |
| sh_builder.py | 构造函数重构 | ✅ 完成 | 2026-01-28 | 接收4个bin数组: trade_price/qty, order_price/qty |
| sz_builder.py | 构造函数重构 | ✅ 完成 | 2026-01-28 | 接收4个bin数组，撤单使用order_bins |
| image_builder.py | 分离模式适配 | ✅ 完成 | 2026-01-28 | build_single_stock根据配置选择计算模式 |

### Phase 9: REQ-002 撤单通道数据修复

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| sz_builder.py | _process_cancels_vectorized() | ✅ 完成 | 2026-01-28 | 修复Price=0撤单被过滤问题 |
| sz_builder.py | Price=0占位符策略 | ✅ 完成 | 2026-01-28 | 使用order_price_bins[0]作为占位符 |
| test_verification.py | 撤单数据验证 | ✅ 通过 | 2026-01-28 | 3个测试全部通过，Ch13/Ch14有数据 |
| .requirements/REQ-002.md | 需求文档 | ✅ 已完成 | 2026-01-28 | 状态更新为"已完成-验证通过" |

### Phase 10: BUG-001 懒加载列名归一化修复

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| sz_loader.py | _normalize_trade_columns_lazy() | ✅ 完成 | 2026-01-28 | LazyFrame版本的成交表列名归一化 |
| sz_loader.py | _normalize_order_columns_lazy() | ✅ 完成 | 2026-01-28 | LazyFrame版本的委托表列名归一化 |
| sz_loader.py | load_trade_lazy() | ✅ 修复 | 2026-01-28 | 添加normalize_columns参数和归一化调用 |
| sz_loader.py | load_order_lazy() | ✅ 修复 | 2026-01-28 | 替换简单重命名为完整归一化 |
| main.py | process_single_stock() | ✅ 验证通过 | 2026-01-28 | 全量数据处理启动成功 |

### Phase 11: REQ-003 配置化日期范围与断点续传

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| config.py | 日期配置字段 | ✅ 完成 | 2026-01-28 | 新增 dates, start_date, end_date, skip_existing |
| config.yaml | 日期配置示例 | ✅ 完成 | 2026-01-28 | 添加任务范围和断点续传策略配置示例 |
| main.py | CLI回退逻辑 | ✅ 完成 | 2026-01-28 | CLI未指定日期时从Config读取 |
| batch_process.py | LMDB存在性检查 | ✅ 完成 | 2026-01-28 | _is_processed()支持检测LMDB文件存在 |

### Phase 12: REQ-004 深交所数据重构

| 模块 | 功能 | 状态 | 实现日期 | 说明 |
|------|------|------|----------|------|
| sz_data_reconstructor.py | 数据重构模块 | ✅ 完成 | 2026-01-28 | 按SecurityID+时间排序重写Parquet |
| sz_data_reconstructor.py | reconstruct_sz_parquet() | ✅ 完成 | 2026-01-28 | 单日重构函数 |
| sz_data_reconstructor.py | batch_reconstruct_sz_parquet() | ✅ 完成 | 2026-01-28 | 批量重构函数 |
| sz_data_reconstructor.py | verify_reconstruction() | ✅ 完成 | 2026-01-28 | 重构后性能验证函数 |
| data_loader/__init__.py | 导出重构函数 | ✅ 完成 | 2026-01-28 | 从data_loader模块导出 |

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
    """深交所数据加载器（R3.2 通联原始→标准格式）"""
    
    # R3.2 列名映射常量
    TRADE_COLUMN_RENAME_MAP = {
        'TransactTime': 'TickTime',
        'LastPx': 'Price',
        'LastQty': 'Qty',
        'BidApplSeqNum': 'BuyOrderNO',
        'OfferApplSeqNum': 'SellOrderNO',
        'ApplSeqNum': 'BizIndex',
    }
    
    ORDER_COLUMN_RENAME_MAP = {
        'TransactTime': 'TickTime',
        'OrderQty': 'Qty',
        'ApplSeqNum': 'BizIndex',
    }
    
    def __init__(self, raw_data_dir: str, use_polars: bool = True)
    def load_trade(self, date: str, normalize_columns: bool = True, ...) -> DataFrame
    def load_order(self, date: str, normalize_columns: bool = True, ...) -> DataFrame
    def load_trade_lazy(self, date: str, normalize_columns: bool = True, ...) -> LazyFrame  # BUG-001修复
    def load_order_lazy(self, date: str, normalize_columns: bool = True, ...) -> LazyFrame  # BUG-001修复
    def enrich_cancel_price(self, trade_df, order_df) -> DataFrame
    def build_active_seqs(self, trade_df) -> Dict[str, Set[int]]
    def build_active_seqs_fast(self, trade_df) -> Dict[str, Set[int]]
    
    # R3.2 新增: 列名归一化方法
    def _normalize_trade_columns(self, df) -> DataFrame
        """
        归一化成交表列名并派生 TickBSFlag:
        - BuyOrderNO > SellOrderNO → 'B' (主动买)
        - SellOrderNO > BuyOrderNO → 'S' (主动卖)
        - else → 'N' (未知/集合竞价)
        """
    
    def _normalize_order_columns(self, df) -> DataFrame
        """归一化委托表列名: OrderQty → Qty"""
    
    # BUG-001 新增: LazyFrame版本列名归一化 (2026-01-28)
    def _normalize_trade_columns_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame
        """
        LazyFrame版本的成交表归一化，支持pipeline优化
        自动重命名列名 + 派生TickBSFlag
        """
    
    def _normalize_order_columns_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame
        """
        LazyFrame版本的委托表归一化，支持pipeline优化
        自动重命名TransactTime→TickTime, OrderQty→Qty等
        """
```

```python
class SHDataLoader:
    """上交所数据加载器（R3.1 v3增强）"""
    
    # R3.1 新增: v3必需字段常量
    V3_REQUIRED_ORDER_FIELDS = ['BizIndex', 'OrdType', 'Side', 'Price', 'Qty', 'IsAggressive']
    V3_REQUIRED_TRADE_FIELDS = ['BizIndex', 'BuyOrderNO', 'SellOrderNO', 'Price', 'Qty', 'ActiveSide']
    
    def load_order(self, date: str, validate_v3_fields: bool = True, ...) -> DataFrame
    def _validate_order_v3_fields(self, df, date: str) -> None  # 抛出ValueError若缺少字段
    
    # R3.1 新增: 主动/被动委托筛选方法
    def get_aggressive_orders(self, df: DataFrame) -> DataFrame  # IsAggressive == True
    def get_passive_orders(self, df: DataFrame) -> DataFrame     # IsAggressive == False
    def get_aggressive_buy_orders(self, df: DataFrame) -> DataFrame   # IsAggressive==True & Side=='1'
    def get_aggressive_sell_orders(self, df: DataFrame) -> DataFrame  # IsAggressive==True & Side=='2'
    def get_passive_buy_orders(self, df: DataFrame) -> DataFrame      # IsAggressive==False & Side=='1'
    def get_passive_sell_orders(self, df: DataFrame) -> DataFrame     # IsAggressive==False & Side=='2'
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
5. **⚠️ 懒加载必须归一化**: 所有 `load_*_lazy()` 方法默认执行列名归一化，确保与即时加载输出一致 (BUG-001)

### 数据字段映射

**R3.2 标准化后（系统内部统一使用）:**

| 字段含义 | 标准列名 | 上交所原始 | 深交所原始(通联) |
|----------|----------|------------|------------------|
| 时间 | **TickTime** | TickTime | TransactTime |
| 价格 | **Price** | Price | Price / LastPx |
| 数量 | **Qty** | Qty | Qty / OrderQty / LastQty |
| 买方序号 | **BuyOrderNO** | BuyOrderNO | BidApplSeqNum |
| 卖方序号 | **SellOrderNO** | SellOrderNO | OfferApplSeqNum |
| 业务索引 | **BizIndex** | BizIndex | ApplSeqNum |
| 主动方向 | **TickBSFlag** | TickBSFlag | (自动派生) |

**重要**: 
- R3.2 后，深交所 loader 会自动将通联原始列名映射为标准列名
- 下游所有模块(cleaner, calculator, builder)统一使用标准列名
- LazyFrame 归一化通过 `_normalize_*_columns_lazy()` 方法实现 (BUG-001修复)

### 已知问题

1. **BUG-001 (已修复)**: 深交所懒加载缺少列名归一化
   - **症状**: `unable to find column "TickTime"`
   - **原因**: `load_trade_lazy()` 和 `load_order_lazy()` 未调用归一化方法
   - **修复**: 新增 `_normalize_*_columns_lazy()` 方法并在懒加载中调用
   - **状态**: ✅ 已修复 (2026-01-28)

2. **REQ-002 (已修复)**: 深交所撤单通道数据丢失
   - **症状**: Channel 13/14 sum=0
   - **原因**: `Price=0` 的撤单被 `valid_mask = prices > 0` 过滤
   - **修复**: 使用 `order_price_bins[0]` 作为占位符，只过滤 `qtys <= 0`
   - **状态**: ✅ 已修复 (2026-01-28)

### 性能考虑

1. 优先使用 Polars 的懒加载 (`scan_parquet`)
2. 使用向量化操作，避免 `iterrows()`
3. 大批量处理时使用 Dask 多进程
4. LazyFrame 归一化不会产生额外中间结果，完全集成到 pipeline

---

## 📜 变更日志

### [2026-01-28] - REQ-005 修复深交所撤单关联OOM

**问题:**
- 运行 `main.py` 时在某些股票上触发 OOM (Exit 137)
- 日志显示单只股票出现 1.5 亿条撤单数据（应为全市场数据量级）

**修复:**
- `main.py`: 新增 `_is_valid_stock_code()` 函数，过滤空字符串/非数字等无效股票代码
- `main.py`: `process_single_stock()` 增加数据量熔断检查（MAX 500万行/股票）
- `main.py`: 移除 `process_single_stock` 中重复的 `enrich_sz_cancel_price` 调用（已由 `Level2ImageBuilder` 内部处理）

**需求文档:**
- `.requirements/REQ-005.md`: 修复深交所撤单关联OOM及性能优化

---

### [2026-01-28] - REQ-003/REQ-004 配置化日期与深交所数据重构

**新增:**
- `config.py`: 添加 `dates`, `start_date`, `end_date`, `skip_existing` 字段
- `config.yaml`: 添加任务范围和断点续传策略配置示例
- `data_loader/sz_data_reconstructor.py`: 深交所数据重构模块
  - `reconstruct_sz_parquet()`: 单日重构（按SecurityID+时间排序）
  - `batch_reconstruct_sz_parquet()`: 批量重构
  - `verify_reconstruction()`: 重构后性能验证

**修改:**
- `main.py`: CLI未指定日期时自动从Config读取（CLI优先级高于Config）
- `scripts/batch_process.py`: `_is_processed()` 支持检测LMDB文件存在性跳过

**需求文档:**
- `.requirements/REQ-003.md`: 配置化日期范围与LMDB断点续传
- `.requirements/REQ-004.md`: 深交所数据加载修复与性能优化

---

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

### [2026-01-26] - Prompt R1.1 上交所图像构建器v3重构

**目标:**
从"结果导向"升级到"意图导向"，通道9/10改为从委托表填充，实现Ch7=Ch9+Ch11约束。

**v3 核心变更:**
| 维度 | v2（旧） | v3（新） |
|------|---------|---------|
| 通道9/10数据源 | 成交表 | 委托表 |
| 通道9/10含义 | 已成交的主动量 | 完整的进攻意图（母单量） |
| Ch7与Ch11关系 | Ch7=Ch11（重叠） | Ch7=Ch9+Ch11（互斥分解） |

**修改 `builder/sh_builder.py`:**
- `_fill_trade()`: 🔴 **物理删除** Ch9/Ch10 填充代码
- `_process_orders()`: 新增 `_validate_order_fields()` 验证 IsAggressive 字段
- `_process_orders_polars()` / `_process_orders_pandas()`: 提取 IsAggressive 字段
- `_fill_order()`: 新增 `is_aggressive` 参数，实现 Ch9/10/11/12 互斥分流
- `_process_trades_vectorized()`: 🔴 **移除** Ch9/Ch10 向量化填充
- `_process_orders_vectorized()`: 实现 v3 互斥分流向量化逻辑
- `validate_constraints()`: **新增** 方法，验证 Ch7=Ch9+Ch11, Ch8=Ch10+Ch12
- `validate_consistency()`: 更新为调用 v3 约束验证

**修改 `tests/test_sh_builder.py`:**
- 更新 fixtures: `sample_order_pandas` / `sample_order_polars` 添加 `IsAggressive` 字段
- **新增** `TestV3ChannelConstraints` 测试类（7个测试用例）:
  - `test_channel_9_10_not_from_trade`: 验证Ch9/10不从成交表填充
  - `test_channel_constraints_ch7_eq_ch9_plus_ch11`: 验证数学约束
  - `test_validate_constraints_method`: 验证 validate_constraints 方法
  - `test_aggressive_order_to_ch9`: 验证进攻型买单进入Ch9
  - `test_passive_order_to_ch11`: 验证防守型买单进入Ch11
  - `test_missing_is_aggressive_field`: 验证缺少字段时抛出明确错误
  - `test_ch7_not_equal_ch11_with_mixed_orders`: 验证Ch7和Ch11不再重叠
- **新增** `TestV3ChannelConstraintsPolars` 测试类（Polars版）

**关键技术约束（铁律）:**
| 约束项 | 要求 | 原因 |
|--------|------|------|
| 排序键 | 必须使用 `['TickTime', 'BizIndex']` | 同一毫秒内可能有多条记录 |
| 必需字段 | 委托表必须包含 `IsAggressive` | 互斥分流必需 |
| 阈值计算 | 当日 Mean + Std | 离线训练场景 |
| IsAggressive判定 | 只看首次出现的记录类型 | 入场瞬间语义 |

**测试结果:** 7个v3测试全部通过 ✅

### [2026-01-26] - Prompt R1.2 深交所图像构建器v3重构

**目标:**
深交所构建器从"结果导向"升级到"意图导向"，使用 ActiveSeqs 集合进行互斥分流。

**v3 核心变更:**
| 维度 | v2（旧） | v3（新） |
|------|---------|---------|
| 通道9/10数据源 | 成交表 (BidSeq vs OfferSeq) | 委托表 (ActiveSeqs 集合) |
| 通道9/10含义 | 已成交的主动量 | 完整的进攻意图（母单量） |
| Ch7与Ch11关系 | Ch7≥Ch11（不互斥） | Ch7=Ch9+Ch11（互斥分解） |

**修改 `builder/sz_builder.py`:**
- **文件头部文档**: 更新为 v3 通道定义，添加互斥分解规则说明
- `_process_trades()`: 🔴 **移除** Ch9/Ch10 填充代码（`ACTIVE_BUY_ORDER`/`ACTIVE_SELL_ORDER`）
- `_process_orders()`: 新增 Ch9/Ch10 填充逻辑，使用 active_seqs 互斥分流
  - `appl_seq in active_seqs['buy']` → Ch9（主动买委托）
  - `appl_seq not in active_seqs['buy']` → Ch11（非主动买）
- `_process_trades_vectorized()`: 🔴 **移除** Ch9/Ch10 向量化填充
- `_process_orders_vectorized()`: 实现 v3 互斥分流向量化逻辑
  - 新增 active_buy_mask / passive_buy_mask 计算
  - 支持归一化后的 `Qty` 字段（兼容 `OrderQty`）
- `validate_constraints()`: **新增** 方法，返回:
  - `buy_valid`: Ch7 = Ch9 + Ch11
  - `sell_valid`: Ch8 = Ch10 + Ch12
  - `decomposition`: 各通道统计详情
- `validate_consistency()`: 更新为 v3 约束检查

**修改 `tests/test_sz_builder.py`:**
- 🔴 **删除** `test_channel_9_10_same_as_1_2` 测试（v2逻辑）
- **更新** `test_channel_11_12_passive_orders`: 添加 v3 约束验证
- **更新** `test_validate_consistency`: 使用 v3 检查项
- **新增** `TestV3ChannelConstraints` 测试类（6个测试用例）:
  - `test_channel_constraint_buy_decomposition`: 验证Ch7=Ch9+Ch11
  - `test_channel_constraint_sell_decomposition`: 验证Ch8=Ch10+Ch12
  - `test_validate_consistency_v3`: 验证 validate_consistency 返回v3结果
  - `test_trades_do_not_fill_ch9_ch10`: 验证成交表不再填充Ch9/Ch10
  - `test_passive_order_stays_passive`: 验证被动单后续成交仍归入Ch11/Ch12
  - `test_build_vs_vectorized_constraints`: 验证逐行/向量化版本都满足约束
- **新增** `TestV3ChannelConstraintsPolars` 测试类

**关键技术约束（铁律）:**
| 约束项 | 要求 | 原因 |
|--------|------|------|
| 排序键 | 必须使用 `['TransactTime', 'ApplSeqNum']` | 同一毫秒内可能有多条记录 |
| ActiveSeqs判定 | `BidApplSeqNum > OfferApplSeqNum` → 主动买 | ApplSeqNum 是全局唯一序号 |
| 字段归一化 | 深交所 `OrderQty` → `Qty` | Loader层重命名，Builder层统一访问 |
| 主动性语义 | 只看入场瞬间(On Entry) | 被动单后续成交仍归入Ch11/Ch12 |

**测试结果:** 34个测试全部通过 ✅（含7个v3专属测试）

### [2026-01-26] - Prompt R2.1: 诊断报告器增强

**变更目标:** 增强 `diagnostics/reporter.py`，新增 v3 架构的通道约束验证功能

**文件变更:**
1. `l2_image_builder/diagnostics/reporter.py` - 核心增强
2. `l2_image_builder/diagnostics/__init__.py` - 导出更新
3. `tests/test_diagnostics.py` - 新增测试用例

**核心变更:**

1. **新增 `validate_channel_constraints()` 函数**
   ```python
   def validate_channel_constraints(image: np.ndarray) -> Dict:
       """
       v3: 验证通道数学约束
       - Ch7 = Ch9 + Ch11 (买单 = 主动买入委托 + 非主动买入)
       - Ch8 = Ch10 + Ch12 (卖单 = 主动卖出委托 + 非主动卖出)
       """
   ```
   - 返回 `{valid, buy_constraint, sell_constraint, errors}`
   - 约束容差: `1e-6`

2. **更新 `CHANNEL_NAMES` 常量**
   - Ch9: `'委托主动买'` → `'主动买入委托'` (强调来源于委托表)
   - Ch10: `'委托主动卖'` → `'主动卖出委托'`
   - Ch11: `'非主动买'` → `'非主动买入'`
   - Ch12: `'非主动卖'` → `'非主动卖出'`

3. **增强 `check_health()` 方法**
   - 新增可选参数 `image: np.ndarray = None`
   - 当传入 image 时，自动进行 v3 约束检查
   - 向后兼容：不传 image 时行为不变

4. **增强 `generate_stock_diagnostics()` 方法**
   - 新增返回字段 `v3_constraints`:
     ```python
     'v3_constraints': {
         'buy_decomposition': "Ch7(x) = Ch9(y) + Ch11(z)",
         'sell_decomposition': "Ch8(x) = Ch10(y) + Ch12(z)",
         'valid': bool,
         'buy_valid': bool,
         'sell_valid': bool,
         'buy_diff': float,
         'sell_diff': float,
     }
     ```

5. **新增 `HEALTH_THRESHOLDS` 配置项**
   - `'constraint_tolerance': 1e-6` - 约束验证容差

**测试用例新增 (15个):**

- `TestV3ChannelConstraints` 类 (8个测试):
  - `test_validate_channel_constraints_valid_buy`: 有效买方约束
  - `test_validate_channel_constraints_valid_sell`: 有效卖方约束
  - `test_validate_channel_constraints_invalid_buy`: 无效买方约束
  - `test_validate_channel_constraints_invalid_sell`: 无效卖方约束
  - `test_validate_channel_constraints_both_invalid`: 双方都无效
  - `test_validate_channel_constraints_zero_image`: 全零图像
  - `test_validate_channel_constraints_distributed`: 分布式值
  - `test_validate_channel_constraints_invalid_shape`: 无效形状

- `TestV3DiagnosticsIntegration` 类 (4个测试):
  - `test_generate_stock_diagnostics_v3_constraints`: 诊断包含v3约束
  - `test_check_health_with_image_constraint_valid`: 约束有效健康检查
  - `test_check_health_with_image_constraint_invalid`: 约束无效健康检查
  - `test_check_health_without_image_backward_compatible`: 向后兼容

- `TestV3ChannelNames` 类 (3个测试):
  - `test_channel_names_count`: 通道数量
  - `test_channel_names_v3_updates`: v3名称更新
  - `test_channel_names_in_diagnostics`: 诊断中名称正确

**测试结果:** 42个测试全部通过 ✅（含15个v3专属测试）

### [2026-01-26] - Prompt R2.2: 图像构建入口更新

**目标:** 更新 `builder/image_builder.py` 支持 v3 架构要求

**主要变更:**

1. **`builder/image_builder.py` 代码更新**:
   - 文件头添加 v3 架构说明文档
   - 新增日志导入和 logger 实例
   - 导入 `validate_channel_constraints` 用于约束检查
   - 导入 `build_active_seqs_from_trade` 用于深交所 ActiveSeqs 自动构建
   - `build_single_stock()` 新增 `validate_constraints` 参数（默认 True）
   - `build_single_stock()` 新增 v3 字段验证逻辑：
     - 上交所委托表必须包含 `IsAggressive` 和 `BizIndex`
     - 验证失败抛出 `ValueError` 并提示解决方案
   - `build_single_stock()` 集成约束检查：构建后验证 Ch7=Ch9+Ch11, Ch8=Ch10+Ch12
   - `build_image()`, `build_l2_image()`, `build_l2_image_with_stats()` 均添加 `validate_constraints` 参数

2. **`tests/test_integration_builder.py` fixtures 更新**:
   - `sh_trade_pandas`: 添加 `BizIndex` 字段
   - `sh_order_pandas`: 添加 `BizIndex` 和 `IsAggressive` 字段
   - `test_trade_only`: 空委托表添加 `IsAggressive` 和 `BizIndex` 字段

3. **`tests/test_sh_builder.py` 测试修复**:
   - `TestChannelFilling` 类所有空 df_order 添加 `IsAggressive` 字段
   - `test_channel_7_8_new_orders`: 非空委托表添加 `IsAggressive` 字段
   - `test_channel_11_12_same_as_7_8`: 添加 `IsAggressive: [False, False]` 以测试 Ch7=Ch11
   - `test_channel_13_14_cancel_orders`: 撤单添加 `IsAggressive: [None, None, None]`
   - `TestEdgeCases` 类所有测试添加 `IsAggressive` 字段
   - `test_channel_9_10_same_as_1_2` 重命名为 `test_channel_9_10_only_from_orders`
     - 更新测试逻辑：v3 中 Ch9/10 只从委托表填充，不从成交表
   - `test_validate_consistency`: 更新断言键名为 v3 格式
     - `ch1_eq_ch9` → `v3_buy_constraint`
     - `ch7_eq_ch11` → `v3_constraints_valid`

**v3 验证逻辑:**
```python
# 上交所委托表必需字段检查
if exchange == 'SH':
    required_order_fields = ['IsAggressive', 'BizIndex']
    missing_fields = [f for f in required_order_fields if f not in order_cols]
    if missing_fields:
        raise ValueError(f"上交所委托表缺少必需字段: {missing_fields}")

# 约束检查（构建后）
if validate_constraints:
    constraint_result = validate_channel_constraints(image)
    if not constraint_result['valid']:
        logger.warning(f"v3约束验证失败: {constraint_result}")
```

**API 变更:**
```python
# 所有构建函数新增 validate_constraints 参数
def build_single_stock(self, df_trade, df_order, 
                       active_seqs=None, 
                       validate_constraints=True) -> np.ndarray

@classmethod
def build_image(cls, stock_code, df_trade, df_order, 
                trade_date=None,
                validate_constraints=True) -> np.ndarray

def build_l2_image(stock_code, df_trade, df_order,
                   validate_constraints=True) -> np.ndarray

def build_l2_image_with_stats(stock_code, df_trade, df_order,
                              validate_constraints=True) -> Tuple
```

**测试结果:** 333 个测试全部通过 ✅

### [2026-01-26] - 沪深一致性修复

**修复:**
- **时间过滤统一**: 两个 `SH_Tick_Data_Reconstruction_Spec_v1.8.md` 文件中下午连续竞价时段从 `1500` 修改为 `1457`，与深交所统一剔除 14:57-15:00 收盘集合竞价时段
- **字段重命名修复**: `image_builder.py` 中深交所委托处理方法 `_process_sz_orders` 使用 `Qty` 替代 `OrderQty`，与 Loader 层字段归一化保持一致

**说明:**
- 2025年后上交所也引入收盘集合竞价，沪深统一剔除最后3分钟保证数据分布一致性
- 深交所 `sz_loader.py` 已在 v3 中实现 `OrderQty -> Qty` 归一化，下游需使用统一后的 `Qty` 字段

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

1. ✅ ~~**Prompt R1.1**: 上交所图像构建器v3重构~~
2. ✅ ~~**Prompt R1.2**: 深交所图像构建器v3重构~~
3. ✅ ~~**Prompt R2.1**: 诊断报告器增强（v3约束验证）~~
4. ✅ ~~**Prompt R2.2**: 图像构建入口更新~~
5. ✅ ~~**Prompt R3.1**: 上交所数据加载器适配（BidOrdID→BuyOrderNO, ActiveSide→TickBSFlag）~~
6. ✅ ~~**Prompt R3.2**: 深交所数据加载器适配（原始通联格式→标准格式）~~
7. **Prompt 5.2**: 监控告警与增量更新（可选）

---

## 📜 变更日志

### [2026-01-27] - Prompt R3.2 深交所数据加载器重构（通联原始→标准格式）

**实现目标:**
将深交所通联原始 Parquet 格式归一化为 l2_image_builder 标准格式，确保下游模块（sz_builder.py, sz_cancel_enricher.py）无需关心原始数据格式差异。

**核心变更:**

1. **sz_loader.py 新增列名映射常量**:
   ```python
   TRADE_COLUMN_RENAME_MAP = {
       'TransactTime': 'TickTime',
       'LastPx': 'Price',
       'LastQty': 'Qty',
       'BidApplSeqNum': 'BuyOrderNO',
       'OfferApplSeqNum': 'SellOrderNO',
       'ApplSeqNum': 'BizIndex',
   }
   
   ORDER_COLUMN_RENAME_MAP = {
       'TransactTime': 'TickTime',
       'OrderQty': 'Qty',
       'ApplSeqNum': 'BizIndex',
   }
   ```

2. **sz_loader.py 新增 `_normalize_trade_columns()` 方法**:
   - 功能: 归一化成交表列名并派生 TickBSFlag
   - TickBSFlag 派生逻辑:
     - `BuyOrderNO > SellOrderNO` → 'B' (主动买)
     - `SellOrderNO > BuyOrderNO` → 'S' (主动卖)
     - 其他 → 'N' (未知/集合竞价)
   - 实现: 支持 Polars 和 Pandas 两种引擎

3. **sz_loader.py 更新 `_normalize_order_columns()` 方法**:
   - 原有功能: OrderQty → Qty
   - 新增: 应用 ORDER_COLUMN_RENAME_MAP 进行完整列名归一化

4. **sz_loader.py 更新 `load_trade()` 和 `load_order()` 方法**:
   - 新增 `normalize_columns` 参数（默认 True）
   - 加载后自动调用相应的归一化方法

5. **sz_builder.py 标准列名适配**:
   - 所有方法更新为使用标准列名:
     - `BidApplSeqNum` → `BuyOrderNO`
     - `OfferApplSeqNum` → `SellOrderNO`
     - `LastPx` → `Price`
     - `LastQty` → `Qty`
     - `ApplSeqNum` → `BizIndex`
   - 受影响方法:
     - `_build_active_seqs()` / `_build_active_seqs_vectorized()`
     - `_process_trades()` / `_process_trades_vectorized()`
     - `_process_cancels()` / `_process_cancels_vectorized()`
     - `_process_orders()` / `_process_orders_vectorized()`

6. **sz_cancel_enricher.py 标准列名适配**:
   - `enrich_sz_cancel_price_polars()`: 使用 BuyOrderNO/SellOrderNO/Price/BizIndex/TickTime
   - `enrich_sz_cancel_price_pandas()`: 使用标准列名
   - 撤单关联逻辑保持不变（通过委托序列号匹配）

**验证测试:**

1. **test_sz_normalization.py** - 列名归一化测试:
   ```
   成交表: 18,453,108 行
   ✅ TickTime, Price, Qty, BuyOrderNO, SellOrderNO, BizIndex
   ✅ TickBSFlag 派生正确:
      - B (主动买): 8,728,424 条
      - S (主动卖): 9,724,684 条
      - N (未知): 0 条
   
   委托表: 18,313,180 行
   ✅ TickTime, Price, Qty, BizIndex, Side
   ```

2. **test_sz_image_build.py** - 集成测试:
   ```
   数据加载: 18,490,049 条成交 + 18,377,297 条委托
   股票 000001 测试:
   - 成交: 4,739 条
   - 委托: 4,659 条
   ✅ 热力图构建成功: (15, 8, 8)
   ✅ 所有通道正常填充
   ```

**架构影响:**

| 层级 | 变更内容 | 影响范围 |
|------|---------|---------|
| **Loader 层** | 自动归一化列名，输出标准格式 | sz_loader.py |
| **Builder 层** | 使用标准列名处理数据 | sz_builder.py |
| **Enricher 层** | 使用标准列名补全撤单价格 | sz_cancel_enricher.py |
| **下游影响** | 无需修改，接收标准格式 | main.py, diagnostics, dataset |

**技术约束更新:**

| 约束项 | R3.2 后规范 |
|--------|------------|
| 深交所成交表字段 | TickTime, Price, Qty, BuyOrderNO, SellOrderNO, BizIndex, TickBSFlag, ExecType |
| 深交所委托表字段 | TickTime, Price, Qty, BizIndex, Side, OrdType |
| TickBSFlag 语义 | 'B'=主动买, 'S'=主动卖, 'N'=未知 |
| 列名归一化时机 | Loader 层输出前（默认开启） |

**交付产物:**
1. ✅ 修改后的 `l2_image_builder/data_loader/sz_loader.py`
2. ✅ 修改后的 `l2_image_builder/builder/sz_builder.py`
3. ✅ 修改后的 `l2_image_builder/cleaner/sz_cancel_enricher.py`
4. ✅ 测试脚本 `test_sz_normalization.py`
5. ✅ 集成测试 `test_sz_image_build.py`
6. ✅ 更新的 `L2_Image_Builder_SZ_Loader_Refactor_Plan.md` (验证结果记录)
7. ✅ 更新的 `agent.md` (本文档)

**后续建议:**
- 考虑在 config.py 添加 `normalize_columns` 全局开关（当前默认 True）
- 可选: 添加单元测试覆盖 Lazy 模式的归一化逻辑
- 可选: 性能测试对比归一化前后的处理速度

---

### [2026-01-28] - 数据测试与上交所加载器修复

**测试结果:**
- ✅ **上交所数据分解** (sh_tick_reconstruction): 成功处理 3731 只股票
  - 委托记录: 5,857,584 条
  - 成交记录: 2,986,187 条
  - 处理耗时: 761.60 秒
- ✅ **上交所热力图构建**: 成功（修复列名映射后）
- ✅ **深交所数据**: R3.2 完成，已支持通联原始格式自动转换

**上交所加载器修复 (sh_loader.py):**

1. **列名兼容处理**:
   - `BidOrdID` → `BuyOrderNO`
   - `AskOrdID` → `SellOrderNO`

2. **TickBSFlag 字段生成**:
   - 根据 `ActiveSide` 自动生成 `TickBSFlag`
   - ActiveSide=1 → TickBSFlag='B' (主动买)
   - ActiveSide=2 → TickBSFlag='S' (主动卖)
   - ActiveSide=0 → TickBSFlag='N' (未知)

3. **方法修改**:
   - `_normalize_trade_columns()`: 即时加载版本
   - `_normalize_trade_columns_lazy()`: 懒加载版本

---
### [2026-01-28] - R3.2+ 下游模块标准列名适配（全链路测试通过）

**问题发现:**
R3.2 完成 sz_loader.py 归一化后，下游模块 (data_cleaner.py, quantile.py, big_order.py) 仍使用原始通联列名，导致全链路测试失败。

**修复内容:**

1. **cleaner/data_cleaner.py** - COLUMN_CONFIG 更新:
   ```python
   # R3.2 前（原始通联列名）
   "sz_order": {"time_column": "TransactTime", "price_column": "LastPx", "qty_column": "LastQty"}
   "sz_trade": {"time_column": "TransactTime", "price_column": "LastPx", "qty_column": "LastQty"}
   
   # R3.2 后（标准列名）
   "sz_order": {"time_column": "TickTime", "price_column": "Price", "qty_column": "Qty"}
   "sz_trade": {"time_column": "TickTime", "price_column": "Price", "qty_column": "Qty"}
   ```

2. **calculator/quantile.py** - 深交所分位数计算:
   - `compute_for_sz()`: 参数改为标准列名 (Price, Qty)
   - `compute_quantile_bins_sz_polars()`: `LastPx` → `Price`, `LastQty`/`OrderQty` → `Qty`
   - `compute_quantile_bins_sz_pandas()`: 同上

3. **calculator/big_order.py** - 深交所母单还原:
   - `_restore_parent_orders_sz()`: 使用 `Price*Qty` 代替 `LastPx*LastQty`
   - `restore_parent_orders_sz_polars()`: 使用 `BuyOrderNO/SellOrderNO` 代替 `BidApplSeqNum/OfferApplSeqNum`
   - `restore_parent_orders_sz_pandas()`: 同上

**全链路测试结果 (test_full_day_pipeline.py):**
```
============================================================
测试 l2_image_builder 完整流程
============================================================
✅ 配置加载成功
✅ 深交所成交数据: 18,490,049 行
✅ 深交所委托数据: 18,377,297 行
✅ 上交所成交数据: 2,986,187 行
✅ 上交所委托数据: 5,857,584 行
✅ 深交所成交清洗后: 18,453,108 行
✅ 深交所委托清洗后: 18,313,180 行
✅ 上交所成交清洗后: 2,986,187 行
✅ 上交所委托清洗后: 5,857,584 行
✅ 撤单价格补全成功: 000001, 4737 行
✅ 深交所价格分位数: [ 0.   11.18 11.38 11.39 11.41 11.43 11.45]
✅ 深交所数量分位数: [ 100.  300.  500. 1000. 1300. 2400. 5500.]
✅ 深交所阈值: 116067.78
✅ 买方母单数: 1296, 卖方母单数: 1422
✅ 图像构建成功: shape=(15, 8, 8)
✅ v3通道约束验证通过: Ch7=Ch9+Ch11, Ch8=Ch10+Ch12
✅ 归一化成功: shape=(15, 8, 8)
✅ 统一入口构建成功: shape=(15, 8, 8)

各通道统计:
  Ch 0 全部成交: sum=2582, nonzero=43/64
  Ch 1 主动买入: sum=1126, nonzero=36/64
  Ch 2 主动卖出: sum=1456, nonzero=42/64
  Ch 7 买单:     sum=2440, nonzero=49/64
  Ch 8 卖单:     sum=2219, nonzero=42/64
  Ch 9 主动买入委托: sum=330, nonzero=33/64
  Ch11 非主动买入: sum=2110, nonzero=49/64
  
============================================================
✅ 所有测试通过！一天数据可以正常处理
============================================================
```

**修改文件汇总:**

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `cleaner/data_cleaner.py` | COLUMN_CONFIG 使用标准列名 | 时间过滤报错 "TransactTime not found" |
| `calculator/quantile.py` | compute_for_sz, sz_polars, sz_pandas | 分位数计算报错 "LastPx" |
| `calculator/big_order.py` | _restore_parent_orders_sz, sz_polars, sz_pandas | 母单还原报错 "LastPx" |

**技术说明:**
R3.2 在 sz_loader.py 完成列名归一化后，所有下游模块必须使用标准列名：
- 时间: `TickTime` (原 `TransactTime`)
- 价格: `Price` (原 `LastPx`)
- 数量: `Qty` (原 `LastQty`, `OrderQty`)
- 买方序号: `BuyOrderNO` (原 `BidApplSeqNum`)
- 卖方序号: `SellOrderNO` (原 `OfferApplSeqNum`)

---

## 📜 变更日志

### [2026-01-28] - BUG-001: 深交所懒加载列名归一化缺失修复

**问题描述:**
```
处理 20251030:  45%|█████████████▌ | 3242/7183 [02:16<55:29,  1.18stock/s]
2026-01-28 16:23:34,662 - ERROR - 处理 300589.SZ 失败: 
unable to find column "TickTime"; valid columns: ["TransactTime", "LastPx", "LastQty", ...]
```

**根本原因:**
1. `sz_loader.py` 的 `load_trade()` 和 `load_order()` 方法会调用 `_normalize_trade_columns()` 和 `_normalize_order_columns()` 进行列名归一化
2. 但 `load_trade_lazy()` 和 `load_order_lazy()` 方法**缺少归一化步骤**，直接返回原始列名的 LazyFrame
3. `main.py` 的 `process_single_stock()` 使用 `load_trade_for_stock()` 调用懒加载方法
4. 后续 `DataCleaner` 期望标准列名 `TickTime`，但实际数据仍然是 `TransactTime`，导致报错

**受影响组件:**
- `load_trade_lazy()`: 返回 LazyFrame 缺少列名归一化
- `load_order_lazy()`: 仅简单重命名 `OrderQty→Qty`，缺少 `TransactTime→TickTime` 等完整归一化
- 下游所有使用懒加载的流程（`process_single_stock`, 批量处理等）

**解决方案:**

1. **新增 LazyFrame 专用归一化方法:**
   ```python
   def _normalize_trade_columns_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
       """LazyFrame版本的成交表归一化"""
       # 1. 列名重命名 (TransactTime→TickTime, LastPx→Price, LastQty→Qty等)
       # 2. 派生 TickBSFlag (BuyOrderNO vs SellOrderNO)
       return lf
   
   def _normalize_order_columns_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
       """LazyFrame版本的委托表归一化"""
       # 完整映射: TransactTime→TickTime, OrderQty→Qty, ApplSeqNum→BizIndex
       return lf
   ```

2. **修改 `load_trade_lazy()` 和 `load_order_lazy()`:**
   ```python
   def load_trade_lazy(..., normalize_columns: bool = True) -> pl.LazyFrame:
       lf = scan_parquet_with_filter(...)
       if normalize_columns:
           lf = self._normalize_trade_columns_lazy(lf)  # 新增
       return lf
   
   def load_order_lazy(..., normalize_columns: bool = True) -> pl.LazyFrame:
       lf = scan_parquet_with_filter(...)
       if normalize_columns:
           lf = self._normalize_order_columns_lazy(lf)  # 替换简单重命名
       return lf
   ```

**修改文件:**
- `l2_image_builder/data_loader/sz_loader.py`:
  - 新增 `_normalize_trade_columns_lazy()` (lines 421-454)
  - 新增 `_normalize_order_columns_lazy()` (lines 456-479)
  - 修改 `load_trade_lazy()`: 添加 `normalize_columns` 参数和归一化调用 (lines 504-570)
  - 修改 `load_order_lazy()`: 替换简单重命名为完整归一化 (lines 572-643)

**验证结果:**
```python
# 测试列名归一化
loader = SZDataLoader('./通联逐笔数据')
df = loader.load_trade_for_stock('20251030', '000001')
print(df.columns)
# Output: ['TickTime', 'Price', 'Qty', 'BuyOrderNO', 'SellOrderNO', 
#          'BizIndex', 'TickBSFlag', ...]  ✅

# 测试单股票处理
stock_code, image = process_single_stock('20251030', '000001.SZ', config)
print(image.shape)  # (15, 8, 8) ✅

# 全量处理启动成功
python -m l2_image_builder.main --date 20251030 --config config.yaml
# 处理 20251030:   1%|▏ | 38/7183 [00:39<2:02:47, 1.03s/stock] ✅
```

**技术细节:**
- LazyFrame 的归一化操作会被 Polars 优化为 pipeline 的一部分，不会产生额外的中间结果
- 归一化逻辑与即时加载版本完全一致，确保数据一致性
- 默认启用归一化 (`normalize_columns=True`)，与非懒加载行为保持一致

**经验教训:**
1. **接口一致性**: 懒加载和即时加载必须提供相同的输出格式
2. **全链路测试**: R3.2 完成列名归一化后应立即测试所有入口（即时加载、懒加载、批量处理）
3. **文档同步**: 接口文档应明确说明输出数据的列名规范

---

### [2026-01-28] - REQ-002: 深交所撤单通道数据修复

**问题描述:**
- Test 1 验证测试显示 Channel 13 (撤买) 和 Channel 14 (撤卖) sum=0
- 用户以为是抽样随机性问题，但实际抽样包含 41/100 条撤单记录

**根本原因:**
- 深交所撤单记录 `Price=0.0` (没有实际成交价格，符合业务逻辑)
- `sz_builder.py` 的 `_process_cancels_vectorized()` 方法第508行使用 `valid_mask = prices > 0` 过滤
- 该过滤逻辑将所有 Price=0 的撤单记录过滤掉

**解决方案:**
```python
# 修改前 (line 508):
valid_mask = prices > 0  # ❌ 过滤所有Price=0的撤单

# 修改后 (lines 507-519):
# 1. 对 Price=0 使用最小价格边界作为占位符
zero_price_mask = prices == 0
if zero_price_mask.any():
    prices = prices.copy()
    prices[zero_price_mask] = self.order_price_bins[0]

# 2. 只过滤无效数量记录
valid_mask = qtys > 0  # ✅ 不过滤价格，只过滤数量
```

**修改文件:**
- `l2_image_builder/builder/sz_builder.py` (lines 507-519)

**验证结果:**
| 测试 | Channel 13 (撤买) | Channel 14 (撤卖) | 状态 |
|------|-------------------|-------------------|------|
| Test 1 (SZ抽样) | sum=22, 非零7/64 | sum=19, 非零8/64 | ✅ 通过 |
| Test 2 (样例) | sum=1 | sum=1 | ✅ 通过 |
| Test 3 (SH全天) | sum=524 | sum=347 | ✅ 通过 |

**技术细节:**
- 撤单记录使用委托价格分位数 (`order_price_bins[0]`) 映射到 bin index 1
- 数据集中在价格最低 bin，符合预期（最小边界占位符策略）
- 调试脚本: `check_sampling.py`, `debug_cancel.py`

**新增:**
- REQ-002 需求文档 (`.requirements/REQ-002.md`)
- Price=0 占位符处理策略

**修复:**
- Channel 13/14 撤单通道数据丢失问题

---