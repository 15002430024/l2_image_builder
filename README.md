# L2 Image Builder (Level2 数据图像化处理)

> 基于华泰证券《基于level2数据图像的选股模型》研报，将 A 股 Level2 逐笔成交与委托数据构建为微观结构图像。

## 📖 项目概述

本项目旨在构建高频量化模型（如 Vision Transformer, ViViT）所需的输入数据。通过处理沪深交易所的 Level2 逐笔数据，生成标准化的三维图像张量 `[15, 8, 8]`。

*   **输入**: 上交所/深交所 逐笔成交 (Trade) + 逐笔委托 (Order) + 快照 (Snapshot) Parquet 文件
*   **输出**: 每日一个 LMDB 文件，存储 `{StockCode: CompressedImageBytes}` 键值对
*   **维度定义**:
    *   **通道 (15)**: 包含成交（主买/主卖/大单/小单）、委托（买/卖/撤单）、意愿等微观特征。
    *   **价格 (8)**: 基于当日价格分布的动态分位数区间。
    *   **成交量 (8)**: 基于当日成交/委托量分布的动态分位数区间。

## 🚀 核心特性

*   **高性能数据处理**: 基于 **Polars** 框架实现全流程向量化处理，支持懒加载与批量计算，大幅提升海量高频数据的处理效率。
*   **精准的微观还原**:
    *   **深交所撤单价格回溯**: 解决了深交所撤单只有价格 0 的问题，通过关联原始委托表还原真实撤单位置。
    *   **母单还原**: 基于订单号聚合算法，还原拆单前的真实机构母单意图。
    *   **大小单动态阈值**: 计算当日 `Mean + Std` 作为动态阈值，自适应不同股票的流动性差异。
*   **严谨的数据对齐**:
    *   解决沪深两市数据结构差异（如主买/主卖定义、撤单机制异同），在图像物理含义上实现统一。
*   **高效存储**: 采用 **LMDB** 数据库配合 **LZ4** 压缩算法，优化 I/O 性能，适配高性能深度学习训练流程。
*   **模块化设计**: 清晰分离 Loader, Cleaner, Calculator, Builder, Storage 层，易于扩展和维护。

## 📂 项目结构

```
l2_image_builder/
├── builder/            # 图像构建核心逻辑
│   ├── image_builder.py    # 统一构建入口
│   ├── sh_builder.py       # 上交所构建器 (向量化)
│   ├── sz_builder.py       # 深交所构建器 (向量化)
│   └── normalizer.py       # 归一化 (Log1p + Max)
├── calculator/         # 数值计算模块
│   ├── big_order.py        # 母单还原与大单阈值
│   └── quantile.py         # 价格/量分位数计算
├── cleaner/            # 数据清洗模块
│   ├── data_cleaner.py     # 清洗流程整合
│   ├── sz_cancel_enricher.py # 深交所撤单价格补全
│   ├── time_filter.py      # 交易时间过滤
│   └── anomaly_filter.py   # 异常值过滤
├── data_loader/        # 数据加载模块 (Polars)
├── dataset/            # PyTorch Dataset 封装 (ViT/ViViT)
├── diagnostics/        # 数据诊断与质量报告
├── storage/            # LMDB 读写模块
├── config.py           # 配置管理
└── main.py             # 程序主入口
```

## 🛠️ 安装与依赖

环境要求：Python 3.10+

```bash
# 推荐使用 pip 安装依赖
pip install -r requirements.txt
```

主要依赖库：
*   `polars`: 高性能数据处理
*   `numpy`: 数值计算
*   `pandas`: 辅助数据处理
*   `pyarrow`: Parquet 文件读写
*   `lmdb`: 键值数据库存储
*   `lz4`: 数据压缩
*   `pyyaml`: 配置文件解析

## ⚙️ 配置说明

项目使用 YAML 文件进行配置。请复制 `l2_image_builder/config_example.yaml` 为 `l2_image_builder/config.yaml` 并根据实际环境修改：

```yaml
paths:
  raw_data_dir: "/path/to/raw_data"  # 原始 Parquet 数据目录
  output_dir: "/path/to/output"      # LMDB 输出目录

processing:
  n_jobs: 8                          # 并行进程数
  compression_level: 3               # LZ4 压缩级别

# 更多配置请参考 config_example.yaml
```

## 🏃‍♂️ 运行指南

### 1. 单日批量构建

使用 `main.py` 处理指定日期的数据：

```bash
# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 运行构建程序 (默认读取 config.yaml)
python l2_image_builder/main.py --date 20250101
```

### 2. 批量脚本

使用 scripts 目录下的脚本进行连续日期处理：

```bash
python l2_image_builder/scripts/batch_process.py --start_date 20250101 --end_date 20250131
```

## 📊 15个通道定义 (Technical Spec)

| 索引 | 类别 | 名称 | 物理含义 |
|------|------|------|----------|
| 0 | 成交 | 全部成交 | 市场整体活跃度 |
| 1 | 成交 | 主动买入 | 多头进攻力量 |
| 2 | 成交 | 主动卖出 | 空头进攻力量 |
| 3 | 成交 | 大买单 | 机构/主力买入行为 |
| 4 | 成交 | 大卖单 | 机构/主力卖出行为 |
| 5 | 成交 | 小买单 | 散户/跟风买入 |
| 6 | 成交 | 小卖单 | 散户/恐慌卖出 |
| 7 | 委托 | 买单 | 完整买入意愿 (挂单) |
| 8 | 委托 | 卖单 | 完整卖出意愿 (挂单) |
| 9 | 委托 | 主动买入 | 实际转化的买入进攻量 |
| 10 | 委托 | 主动卖出 | 实际转化的卖出进攻量 |
| 11 | 委托 | 非主动买入 | 纯被动承接买盘 |
| 12 | 委托 | 非主动卖出 | 纯被动压盘卖盘 |
| 13 | 委托 | 撤买 | 买单撤回 (虚假申报/诱多) |
| 14 | 委托 | 撤卖 | 卖单撤回 (虚假压单/诱空) |

## 🧪 测试

项目包含完整的单元测试，基于 `pytest`：

```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_sz_builder.py
```

## 📝 许可证

[License Information]
