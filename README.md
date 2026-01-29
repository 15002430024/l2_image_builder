# Level2 数据图像化处理模块

> **版本**: v2.0  
> **日期**: 2026-01-29  
> **基于**: 华泰证券《基于level2数据图像的选股模型》(2025-12-24)

---

## 📋 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [快速开始](#快速开始)
- [数据流架构](#数据流架构)
- [15通道定义](#15通道定义)
- [使用指南](#使用指南)
- [输出说明](#输出说明)
- [技术架构](#技术架构)
- [性能优化](#性能优化)
- [注意事项](#注意事项)

---

## 📖 项目概述

### 背景

将 Level2 逐笔成交与逐笔委托数据转换为标准化的三维图像格式 `[15, 8, 8]`，用于 Vision Transformer (ViT) 和 Video Vision Transformer (ViViT) 模型训练。

### 核心目标

| 维度 | 大小 | 含义 |
|------|------|------|
| 通道 | 15 | 7个成交类型 + 8个委托类型 |
| 价格轴 | 8 | 按当日联合分位数划分的8个价格区间 |
| 量轴 | 8 | 按当日联合分位数划分的8个量区间 |

### 设计理念

**"意图导向"架构升级**：
- 通道 0-6：来自**成交表**（已实现的结果）
- 通道 7-14：来自**委托表**（完整的意图）
- 核心创新：通道 9-10 使用**完整委托量**而非成交量，捕捉主力"进攻欲望"

---

## 🎯 核心功能

### 数据处理流水线

```
原始数据 → 时间过滤 → 异常值清洗 → 母单聚合 → 分位数计算 
  → 像素映射 → 归一化 → LZ4压缩 → LMDB存储
```

### 关键技术特性

1. **沪深统一化**：上交所和深交所数据经过标准化处理后使用统一的通道映射逻辑
2. **意图导向**：通道 9-10 记录完整委托量（进攻意图），而非仅成交量（已实现结果）
3. **母单还原**：通过聚合 OrderID 还原机构拆单前的真实委托规模
4. **动态阈值**：每日独立计算大单阈值（Mean + Std），无需历史滚动窗口
5. **联合分位数**：成交和委托数据联合计算分位数边界，保留绝对体量差异信息
6. **高性能存储**：LMDB + LZ4 压缩，单张图像约 200-500 字节

---

## 🚀 快速开始

### 环境依赖

```bash
# 创建虚拟环境
conda create -n l2_image python=3.10
conda activate l2_image

# 安装核心依赖
pip install polars pandas pyarrow numpy lmdb lz4
```

### 配置文件

编辑 `config.yaml` 设置数据路径：

```yaml
data:
  # 输入路径（指向逐笔数据分解模块的输出目录）
  raw_data_dir: "/path/to/output"
  
  # 输入文件命名模式
  sh_trade_pattern: "{date}_sh_trade_data.parquet"
  sh_order_pattern: "{date}_sh_order_data.parquet"
  sz_trade_pattern: "{date}_sz_trade_data.parquet"
  sz_order_pattern: "{date}_sz_order_data.parquet"
  
  # 输出路径
  output_dir: "/path/to/l2_images"
  lmdb_dir: "{output_dir}/lmdb"
  
processing:
  # 时间范围（连续竞价时段）
  start_time: 93000000  # 09:30:00.000
  end_time: 145700000   # 14:57:00.000
  
  # 大单阈值系数
  big_order_alpha: 1.0  # threshold = mean + alpha * std
  
  # 分位数边界
  quantiles: [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]
```

### 运行示例

**处理单日数据**：
```bash
python main.py --date 20251031
```

**批量处理日期范围**：
```bash
python main.py --start-date 20251030 --end-date 20251031
```

**指定股票池**：
```bash
python main.py --date 20251031 --stock-file stock_list.txt
```

---

## 🏗️ 数据流架构

### 两阶段流水线

```
┌─────────────────────────────────────────────────────────────────┐
│             Phase 1: 数据重构（逐笔数据分解模块）                │
├─────────────────────────────────────────────────────────────────┤
│  上交所: 拆解混合流 + 母单聚合 + 主被动标识                      │
│  深交所: SecurityID 排序重构                                     │
│           ↓                                                      │
│  output/{date}_sh_order_data.parquet  (含 IsAggressive)         │
│  output/{date}_sh_trade_data.parquet  (含 ActiveSide)           │
│  output/{date}_sz_order_data.parquet                            │
│  output/{date}_sz_trade_data.parquet                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            Phase 2: 图像构建（本模块 l2_image_builder）          │
├─────────────────────────────────────────────────────────────────┤
│  1. 数据加载 & 清洗                                              │
│     - 时间过滤（连续竞价时段）                                    │
│     - 异常值剔除                                                 │
│     - 深交所撤单价格关联                                          │
│                                                                  │
│  2. 预计算                                                       │
│     - 母单还原：groupby(OrderID).sum()                          │
│     - 阈值计算：当日 Mean + Std                                  │
│     - 分位数：联合计算 price_bins & qty_bins                     │
│     - 主动委托集合（深交所）                                      │
│                                                                  │
│  3. 图像构建                                                     │
│     - 初始化：zeros(15, 8, 8)                                    │
│     - 像素填充：遍历成交/委托/撤单记录                            │
│     - 通道映射：根据 IsAggressive/ActiveSide 分流               │
│                                                                  │
│  4. 归一化 & 存储                                                │
│     - Log1p 变换 + Max 归一化                                    │
│     - LZ4 压缩 + LMDB 存储                                       │
│           ↓                                                      │
│  lmdb/{date}.lmdb  (Key=股票代码, Value=图像张量)                │
└─────────────────────────────────────────────────────────────────┘
```

### 依赖关系

```
l2_image_builder (本模块)
    ↓ 读取
output/  (逐笔数据分解模块的输出)
    ↓ 来自
通联逐笔数据/  (原始 Level2 数据)
```

**⚠️ 重要**：本模块**不处理原始 Level2 数据**，而是读取**已预处理的标准化数据**：
- 上交所数据已完成混合流拆解、母单聚合、主被动标识
- 深交所数据已完成 SecurityID 排序重构
- 输入数据 Schema 已对齐，可统一处理

---

## 📊 15通道定义

### 通道总览

| 索引 | 类别 | 名称 | 数据源 | 物理含义 |
|------|------|------|--------|----------|
| 0 | 成交 | 全部成交 | 成交表 | 所有成交事件 |
| 1 | 成交 | 主动买入成交 | 成交表 & ActiveSide=1 | 买方主动吃单的成交 |
| 2 | 成交 | 主动卖出成交 | 成交表 & ActiveSide=2 | 卖方主动吃单的成交 |
| 3 | 成交 | 大买单成交 | 成交表 | 买方母单金额≥阈值 |
| 4 | 成交 | 大卖单成交 | 成交表 | 卖方母单金额≥阈值 |
| 5 | 成交 | 小买单成交 | 成交表 | 买方母单金额<阈值 |
| 6 | 成交 | 小卖单成交 | 成交表 | 卖方母单金额<阈值 |
| 7 | 委托 | 买单 | 委托表 Side='B' | **买方总意愿** |
| 8 | 委托 | 卖单 | 委托表 Side='S' | **卖方总意愿** |
| 9 | 委托 | **主动买入委托** | **委托表 IsAggressive=True** | **进攻型买单意图（完整委托量）** |
| 10 | 委托 | **主动卖出委托** | **委托表 IsAggressive=True** | **进攻型卖单意图（完整委托量）** |
| 11 | 委托 | 非主动买入 | 委托表 IsAggressive=False | 防守型买单（挂单等待） |
| 12 | 委托 | 非主动卖出 | 委托表 IsAggressive=False | 防守型卖单（挂单等待） |
| 13 | 委托 | 撤买 | 委托表 OrdType='Cancel' | 撤销买入委托 |
| 14 | 委托 | 撤卖 | 委托表 OrdType='Cancel' | 撤销卖出委托 |

### 核心设计原则

**意图导向升级**：
- 通道 9-10 **不从成交表获取**，而是从**委托表**获取
- 记录的是**完整委托量**（OrderQty），而非已成交量
- 捕捉主力资金的"进攻欲望"，而非"已实现结果"

**数学关系**（严格互斥分解）：
```
Ch7 (全部买单) = Ch9 (主动买) + Ch11 (非主动买)
Ch8 (全部卖单) = Ch10 (主动卖) + Ch12 (非主动卖)
Ch0 (全部成交) = Ch1 (主动买成交) + Ch2 (主动卖成交)
```

### IsAggressive 字段说明

> **关键**：只看"出生方式"，不看"后续经历"

| 场景 | 首次出现 | IsAggressive | 说明 |
|------|----------|--------------|------|
| 即时全部成交 | Type='T' | `True` | 出生即成交（Taker） |
| 部分成交后挂单 | Type='T' | `True` | 先吃单，后挂单（仍是 Taker） |
| 纯挂单 | Type='A' | `False` | 出生即挂单（Maker） |
| 被动单后续成交 | Type='A' | `False` | 先挂单，后被吃（仍是 Maker） |
| 撤单 | - | `None` | 不适用 |

**示例对比**：主力挂 10,000 股主动买单，实际成交 2,000 股

| 方案 | 通道9记录 | 信息损失 |
|------|----------|----------|
| 原方案（结果导向） | 2,000 | 丢失 8,000 股"进攻欲望" |
| **新方案（意图导向）** | **10,000** | **完整保留主力意图** |

---

## 📖 使用指南

### 基本用法

```bash
# 1. 处理单个日期
python main.py --date 20251031

# 2. 处理日期范围
python main.py --start-date 20251030 --end-date 20251105

# 3. 指定股票池（文件每行一个股票代码）
python main.py --date 20251031 --stock-file stock_list.txt

# 4. 自定义输出路径
python main.py --date 20251031 --output-dir /path/to/output

# 5. 启用诊断报告
python main.py --date 20251031 --enable-diagnostics

# 6. 强制重新处理（覆盖已有文件）
python main.py --date 20251031 --force
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--date` | 单个日期 YYYYMMDD | - |
| `--start-date` | 开始日期 | - |
| `--end-date` | 结束日期 | - |
| `--stock-file` | 股票池文件路径 | 全市场 |
| `--output-dir` | 输出目录 | config.yaml 中的配置 |
| `--enable-diagnostics` | 启用诊断报告 | False |
| `--force` | 强制重新处理 | False |
| `--workers` | 并行处理进程数 | 8 |

### Python API 使用

```python
from l2_image_builder import Level2ImageBuilder
import numpy as np

# 1. 初始化构建器
builder = Level2ImageBuilder(
    date='20251031',
    stock_code='600519.SH',
    config_path='config.yaml'
)

# 2. 构建图像
image = builder.build()  # shape: (15, 8, 8)

# 3. 检查图像质量
diagnostics = builder.diagnose(image)
print(f"填充率: {diagnostics['fill_rate']:.2%}")
print(f"稀疏度: {diagnostics['sparsity']:.2%}")

# 4. 保存到 LMDB
from l2_image_builder.storage import write_to_lmdb
write_to_lmdb('20251031', {'600519.SH': image}, 'output/lmdb')
```

### ViT/ViViT 数据集

```python
from l2_image_builder.dataset import Level2ImageDataset, ViViTDataset
import torch
from torch.utils.data import DataLoader

# ViT 数据集（单日图像）
dataset = Level2ImageDataset(
    lmdb_path='output/lmdb/20251031.lmdb',
    stock_list=['600519.SH', '000001.SZ']
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    # images: (batch, 15, 8, 8)
    pass

# ViViT 数据集（20日序列）
dataset = ViViTDataset(
    lmdb_dir='output/lmdb',
    date_range=('20251010', '20251031'),
    sequence_length=20,
    stock_list=['600519.SH']
)

for sequence, label in dataset:
    # sequence: (20, 15, 8, 8) - 20 天图像序列
    pass
```

---

## 📊 输出说明

### LMDB 存储结构

```
output/lmdb/
├── 20251030.lmdb           # 单日 LMDB 文件
│   ├── data.mdb            # 数据文件
│   └── lock.mdb            # 锁文件
├── 20251031.lmdb
└── ...

# 内部结构
Key: "600519.SH" (股票代码)
Value: LZ4 压缩的 numpy.tobytes()
  └─ 解压后: np.frombuffer(data, dtype=np.float32).reshape(15, 8, 8)
```

### 图像格式

| 属性 | 规格 |
|------|------|
| Shape | `(15, 8, 8)` |
| Dtype | `float32` |
| 取值范围 | `[0.0, 1.0]`（归一化后） |
| 压缩前大小 | 15 × 8 × 8 × 4 = 3,840 字节 |
| 压缩后大小 | 约 200-500 字节（LZ4） |

### 读取示例

```python
import lmdb
import numpy as np
import lz4.frame

# 打开 LMDB
env = lmdb.open('output/lmdb/20251031.lmdb', readonly=True)

with env.begin() as txn:
    # 读取单只股票
    compressed = txn.get('600519.SH'.encode('utf-8'))
    
    if compressed:
        # 解压缩
        decompressed = lz4.frame.decompress(compressed)
        
        # 还原为 numpy array
        image = np.frombuffer(decompressed, dtype=np.float32)
        image = image.reshape(15, 8, 8)
        
        print(f"Shape: {image.shape}")
        print(f"Range: [{image.min():.4f}, {image.max():.4f}]")

env.close()
```

### 诊断报告

启用 `--enable-diagnostics` 后生成 CSV 报告：

```
output/diagnostics/20251031_diagnostics.csv
```

| 字段 | 说明 |
|------|------|
| `stock_code` | 股票代码 |
| `channel_0_fill_rate` | 通道0填充率 |
| `channel_0_sparsity` | 通道0稀疏度 |
| `channel_0_max_pixel` | 通道0最大像素值 |
| ... | 重复15个通道 |
| `total_orders` | 总订单数 |
| `big_order_ratio` | 大单占比 |
| `cancel_ratio` | 撤单率 |

---

## 🏗️ 技术架构

### 模块结构

```
l2_image_builder/
├── __init__.py
├── config.yaml                  # 配置文件
├── main.py                      # 主入口
│
├── data_loader/                 # 数据加载
│   ├── sh_loader.py             # 上交所
│   └── sz_loader.py             # 深交所
│
├── cleaner/                     # 数据清洗
│   ├── time_filter.py           # 时间过滤
│   ├── anomaly_filter.py        # 异常值剔除
│   └── sz_cancel_enricher.py    # 深交所撤单价格关联
│
├── calculator/                  # 预计算
│   ├── quantile.py              # 分位数计算
│   ├── big_order.py             # 母单还原 & 阈值
│   └── channel_mapper.py        # 通道映射逻辑
│
├── builder/                     # 图像构建
│   ├── image_builder.py         # 核心构建器
│   ├── sh_builder.py            # 上交所逻辑
│   ├── sz_builder.py            # 深交所逻辑
│   └── normalizer.py            # 归一化
│
├── storage/                     # 存储
│   ├── lmdb_writer.py           # LMDB 写入
│   └── lmdb_reader.py           # LMDB 读取
│
├── dataset/                     # PyTorch 数据集
│   ├── vit_dataset.py           # ViT 单日
│   └── vivit_dataset.py         # ViViT 序列
│
└── diagnostics/                 # 诊断
    └── reporter.py              # 报告生成
```

### 核心算法

**1. 联合分位数计算**

```python
# 合并成交和委托数据
all_prices = np.concatenate([
    df_trade['Price'].values,
    df_order['Price'].values
])

all_qtys = np.concatenate([
    df_trade['Qty'].values,
    df_order['Qty'].values  # ⚠️ 包含委托量（比成交量大1-2个数量级）
])

# 计算分位数边界（7个边界分出8个区间）
price_bins = np.percentile(all_prices, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
qty_bins = np.percentile(all_qtys, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
```

**2. 母单还原**

```python
# 上交所
buy_parent_amount = df_trade.groupby('BuyOrderNO')['TradeMoney'].sum()
sell_parent_amount = df_trade.groupby('SellOrderNO')['TradeMoney'].sum()

# 深交所
buy_parent_amount = df_trade.groupby('BidApplSeqNum')['TradeMoney'].sum()
sell_parent_amount = df_trade.groupby('OfferApplSeqNum')['TradeMoney'].sum()
```

**3. 大单阈值**

```python
all_amounts = np.concatenate([
    buy_parent_amount.values,
    sell_parent_amount.values
])

threshold = all_amounts.mean() + 1.0 * all_amounts.std()
```

**4. 像素映射**

```python
# 计算 bin 索引
price_bin = np.digitize(row['Price'], price_bins)  # 0-7
qty_bin = np.digitize(row['Qty'], qty_bins)        # 0-7

# 累加像素值
image[channel_idx, price_bin, qty_bin] += 1
```

**5. 归一化**

```python
# Log1p 变换 + 通道内 Max 归一化
for c in range(15):
    channel = image[c]
    channel_log = np.log1p(channel)
    channel_max = channel_log.max()
    
    if channel_max > 0:
        image[c] = channel_log / channel_max
```

---

## ⚡ 性能优化

### 推荐技术栈

| 组件 | 推荐方案 | 性能提升 |
|------|----------|----------|
| 数据读取 | Polars | 5-10x |
| 向量化计算 | NumPy Broadcasting | 10-50x |
| 并行处理 | Dask / ProcessPoolExecutor | N×CPU核数 |
| 存储压缩 | LZ4 | 压缩率 ~80% |

### Polars 向量化示例

```python
import polars as pl

# 读取和清洗（链式调用，零拷贝）
df = (
    pl.scan_parquet('20251031_sh_trade_data.parquet')
    .filter(pl.col('TickTime').is_between(93000000, 145700000))
    .filter(pl.col('Price') > 0)
    .filter(pl.col('Qty') > 0)
    .collect()
)

# 母单还原（向量化 groupby）
buy_parent = df.groupby('BuyOrderNO').agg([
    pl.col('TradeMoney').sum().alias('ParentAmount')
])
```

### 批量处理

```python
from concurrent.futures import ProcessPoolExecutor

def process_single_stock(args):
    date, stock_code = args
    builder = Level2ImageBuilder(date, stock_code)
    return builder.build()

# 并行处理
with ProcessPoolExecutor(max_workers=8) as executor:
    args_list = [(date, code) for code in stock_list]
    images = list(executor.map(process_single_stock, args_list))
```

### 性能基准

| 数据规模 | Pandas | Polars | Dask (8核) |
|----------|--------|--------|-----------|
| 单股票/单日 | ~30s | ~2s | ~0.5s |
| 全市场/单日 | ~8h | ~1.5h | ~20min |

---

## ⚠️ 注意事项

### 数据依赖

**⚠️ 本模块不处理原始 Level2 数据**：
- 必须先运行 `逐笔数据分解` 模块完成数据预处理
- 输入数据路径必须指向预处理后的 `output/` 目录
- 确保输入数据包含必需字段：`IsAggressive`（上交所）、`ActiveSide`（统一化）

### 通道 9-10 核心规则

**🔴 数据源限制（强制规则）**：

| 通道 | ✅ 正确数据源 | ❌ 错误数据源 |
|------|-------------|--------------|
| 9-主动买入 | **委托表** + IsAggressive | ~~成交表 BSFlag='B'~~ |
| 10-主动卖出 | **委托表** + IsAggressive | ~~成交表 BSFlag='S'~~ |

**原因**：本方案采用"意图导向"设计，通道 9-10 记录**完整委托量（进攻意图）**，而非成交量。

### IsAggressive 常见误解

| 场景 | 错误理解 | 正确理解 |
|------|---------|---------|
| 被动单后续成交 | 9:30挂单，10:00被吃 → True? | **False**（首次是挂单） |
| 主动单部分成交 | 先成交，再挂单 → False? | **True**（首次是成交） |
| 撤单记录 | IsAggressive = False? | **None**（不适用） |

**核心原则**：只看"出生方式"，不看"后续经历"。

### 数据质量检查

**必须检查的指标**：

```python
# 1. 通道数学关系
assert image[7].sum() == image[9].sum() + image[11].sum()  # Ch7 = Ch9 + Ch11
assert image[8].sum() == image[10].sum() + image[12].sum() # Ch8 = Ch10 + Ch12

# 2. 填充率（健康范围）
fill_rate = np.count_nonzero(image) / (15 * 8 * 8)
assert 0.2 <= fill_rate <= 0.8, f"填充率异常: {fill_rate:.2%}"

# 3. 大单占比（5%-30%为正常）
big_order_ratio = (image[3].sum() + image[4].sum()) / image[0].sum()
assert 0.05 <= big_order_ratio <= 0.30, f"大单占比异常: {big_order_ratio:.2%}"
```

### 深交所撤单价格

**问题**：深交所撤单记录的 `LastPx = 0`，必须关联委托表获取原始价格。

**解决方案**：数据清洗阶段自动处理（`cleaner/sz_cancel_enricher.py`），无需手动干预。

### 分位数计算

**⚠️ 必须包含委托数据**：由于通道 9-10 使用委托量（OrderQty），如果只用成交数据计算 `qty_bins`，会导致委托量全部溢出到最大 bin。

```python
# ✅ 正确：联合计算
all_qtys = np.concatenate([
    df_trade['Qty'].values,
    df_order['Qty'].values  # 必须包含！
])

# ❌ 错误：只用成交数据
all_qtys = df_trade['Qty'].values  # 会导致通道9-10失效
```

---

## 📚 延伸阅读

- **详细技术规格**：[L2_Image_Builder_Technical_Spec.md](L2_Image_Builder_Technical_Spec%20(13).md)
- **上游数据处理**：`../逐笔数据分解/README.md`
- **原始研报**：华泰证券《基于level2数据图像的选股模型》(2025-12-24)

---

## 🔄 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-01-16 | 初始版本 |
| v2.0 | 2026-01-29 | 意图导向升级 + README 文档 |

---

**维护者**: 中邮基金量化团队  
**技术支持**: 参考技术规格文档或提交 Issue

