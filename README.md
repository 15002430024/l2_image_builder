# L2 Image Builder (Level2 数据图像化处理)

> **基于华泰证券《基于level2数据图像的选股模型》研报**  
> 将 A 股 Level2 逐笔成交与委托数据构建为微观结构图像（ViT/ViViT Input）。

## 📖 项目概述

本项目旨在构建高频量化模型（如 Vision Transformer, ViViT）所需的输入数据。通过处理沪深交易所的 Level2 逐笔数据，生成标准化的三维图像张量 `[15, 8, 8]`。

*   **输入**: 上交所/深交所 逐笔成交 (Trade) + 逐笔委托 (Order) Parquet 文件
*   **输出**: 每日一个 LMDB 文件，存储 `{StockCode: CompressedImageBytes}` 键值对
*   **维度**: 15通道 (成交/委托/撤单) × 8 (价格分位数) × 8 (数量分位数)

---

## 📊 输入数据格式要求

本项目支持 **通联数据 (DataYes)** 的标准 Parquet 格式，代码会自动识别并归一化列名。

### 目录结构
```
/path/to/raw_data/
├── MktTradeDataSZ_20251030.parquet    # 深交所成交
├── MktOrderDataSZ_20251030.parquet    # 深交所委托
├── MktTradeDataSH_20251030.parquet    # 上交所成交
└── MktOrderDataSH_20251030.parquet    # 上交所委托
```

### 1. 深交所 (SZ) 必需字段

**成交表 (MktTradeDataSZ)**
| 标准列名 | 通联列名 | 类型 | 说明 |
|---|---|---|---|
| SecurityID | SecurityID | str | 股票代码 (000001) |
| TickTime | TransactTime | int64 | 时间 (HHMMSSsss) |
| Price | LastPx | float64 | 成交价格 |
| Qty | LastQty | int64 | 成交数量 |
| BuyOrderNO | BidApplSeqNum | int64 | 买方委托号 |
| SellOrderNO | OfferApplSeqNum | int64 | 卖方委托号 |
| ExecType | ExecType | str | '70'=成交, '52'=撤单 |

**委托表 (MktOrderDataSZ)**
| 标准列名 | 通联列名 | 类型 | 说明 |
|---|---|---|---|
| SecurityID | SecurityID | str | 股票代码 |
| TickTime | TransactTime | int64 | 委托时间 |
| Price | Price | float64 | 委托价格 |
| Qty | OrderQty | int64 | 委托数量 |
| Side | Side | str | '49'=买, '50'=卖 |
| BizIndex | ApplSeqNum | int64 | 委托序号 |

### 2. 上交所 (SH) 必需字段

**成交表 (MktTradeDataSH)**
| 标准列名 | 通联列名 | 类型 | 说明 |
|---|---|---|---|
| SecurityID | SecurityID | str | 股票代码 (600000) |
| TickTime | TickTime | int64 | 成交时间 |
| Price | Price | float64 | 成交价格 |
| Qty | Qty | int64 | 成交数量 |
| TickBSFlag | TickBSFlag | str | 'B'=主动买, 'S'=主动卖 |
| BuyOrderNO | BuyOrderNO | int64 | 买方委托号 |
| SellOrderNO | SellOrderNO | int64 | 卖方委托号 |

**委托表 (MktOrderDataSH)**
| 标准列名 | 通联列名 | 类型 | 说明 |
|---|---|---|---|
| SecurityID | SecurityID | str | 股票代码 |
| TickTime | TickTime | int64 | 委托时间 |
| Price | Price | float64 | 委托价格 |
| Qty | Qty | int64 | 委托数量 |
| Side | Side | str | 'B'=买, 'S'=卖 |
| OrdType | OrdType | str | 'New'=新增, 'Cancel'=撤单 |

> **提示**: 如果你的数据列名不同，代码中的 `l2_image_builder/data_loader` 模块会自动尝试映射常见列名。

---

## 🚀 核心特性

- **高性能**: 基于 Polars 向量化处理，单日全市场（4000+股票）仅需 10-20 分钟 (8核心)。
- **全自动**: 自动处理数据清洗、异常值过滤、分位数计算、母单还原。
- **精准还原**: 
  - **深交所撤单补全**: 自动回溯原始委托表，修复深交所撤单只有 Price=0 的问题。
  - **母单识别**: 基于订单号聚合算法，自动计算大单阈值 (Mean + 1.0*Std)。
- **生产就绪**: 使用 LMDB + LZ4 压缩存储，支持 Dask 并行处理。

---

## 🛠️ 安装与配置

### 安装
```bash
pip install -r requirements.txt
```
主要依赖: `polars`, `numpy`, `pandas`, `pyarrow`, `lmdb`, `lz4`.

### 配置 (config.yaml)
```yaml
paths:
  raw_data_dir: "/path/to/raw_data"
  output_dir: "/path/to/output"

processing:
  n_workers: 8
  separate_quantile_bins: true  # R4.1: 推荐开启，分别计算成交/委托分位数
```

---

## 💻 快速开始

### 方式 1: Python API (推荐)

```python
from l2_image_builder import ImageBuilder
from l2_image_builder.config import Config

# 1. 配置
config = Config(
    raw_data_dir="./data/raw",
    output_dir="./data/output",
    separate_quantile_bins=True
)

# 2. 初始化构建器
builder = ImageBuilder(config)

# 3. 构建单只股票
image = builder.build_single_stock(
    date="20251030",
    stock_code="000002",
    exchange="sz"
)

print(f"Image Shape: {image.shape}") # (15, 8, 8)
```

### 方式 2: 批量处理脚本

```python
from l2_image_builder.scripts.batch_process import BatchProcessor

processor = BatchProcessor(
    raw_data_dir="./data/raw",
    output_dir="./data/output",
    n_workers=8
)

# 处理单日所有股票
processor.run_daily_update(date="20251030")

# 处理历史区间
processor.run_backfill(start_date="20250101", end_date="20250131")
```

### 方式 3: PyTorch Dataset (训练用)

```python
from l2_image_builder.dataset import ViTDataset
from torch.utils.data import DataLoader

dataset = ViTDataset(lmdb_path="./data/output/20251030.lmdb")
loader = DataLoader(dataset, batch_size=32)

for images, codes in loader:
    # images: [32, 15, 8, 8]
    pass
```

---

## 📊 15个通道定义

图像形状 `(15, 8, 8)`，其中 15 个通道包含不同的微观结构信息：

| 索引 | 类别 | 名称 | 物理含义 |
|---|---|---|---|
| 0 | 成交 | 全部成交 | 市场整体热度 |
| 1 | 成交 | 主动买入 | 多头进攻力度 |
| 2 | 成交 | 主动卖出 | 空头抛压力度 |
| 3 | 成交 | 大买单 | 主力/机构买入 |
| 4 | 成交 | 大卖单 | 主力/机构卖出 |
| 5 | 成交 | 小买单 | 散户买入 |
| 6 | 成交 | 小卖单 | 散户卖出 |
| 7 | 委托 | 买单 | 委买意愿 (挂单) |
| 8 | 委托 | 卖单 | 委卖意愿 (挂单) |
| 9 | 委托 | 主动买委托 | 激进买入意愿 |
| 10 | 委托 | 主动卖委托 | 激进卖出意愿 |
| 11 | 委托 | 非主动买 | 承接买盘 |
| 12 | 委托 | 非主动卖 | 压盘卖盘 |
| 13 | 委托 | 撤买 | 买单撤回 (诱多?) |
| 14 | 委托 | 撤卖 | 卖单撤回 (诱空?) |

> **8x8 像素含义**: 
> - X轴: 价格分位数 (Price Bins)
> - Y轴: 数量分位数 (Qty Bins)
> - 像素值: Log1p(数量) 归一化后的热力值

---

## 🔧 开发与测试

### 单元测试
```bash
pytest tests/
```

### 关于 `test_verification.py`
`test_verification.py` 是一个**复杂的开发验证脚本**，用于回归测试核心算法（如分位数一致性、撤单修复逻辑）。
- **用户不需要运行此脚本**。
- 如果你需要验证环境或代码修改，可以运行它来确保逻辑正确。

---

## ❓ 常见问题 (FAQ)

**Q: 我需要 GPU 吗？**  
A: 数据构建阶段不需要，由 CPU (Polars) 完成。模型训练阶段建议使用 GPU。

**Q: 分位数是如何计算的？**  
A: 每日针对每只股票动态计算。开启 `separate_quantile_bins=True` 后，成交量和委托量会使用独立的分位数网格，以提高分辨率。

**Q: 为什么能识别深交所 Price=0 的撤单？**  
A: 我们实现了一个 `CancelEnricher`，它会根据撤单记录中的 `BizIndex` 去原始委托表中查找对应的原始委托价格，并将其填充回去。

