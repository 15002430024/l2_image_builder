# Level2 数据图像化处理 - 技术需求文档

> **版本**: 1.0  
> **日期**: 2026-01-16  
> **基于**: 华泰证券《基于level2数据图像的选股模型》(2025-12-24)

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据源规格](#2-数据源规格)
3. [15个通道精确定义](#3-15个通道精确定义)
4. [数据清洗规则](#4-数据清洗规则)
5. [分位数计算规则](#5-分位数计算规则)
6. [大小单判定规则](#6-大小单判定规则)
7. [图像构建完整流程](#7-图像构建完整流程)
8. [归一化处理](#8-归一化处理)
9. [存储规格](#9-存储规格)
10. [代码模块设计](#10-代码模块设计)
11. [诊断与监控指标](#11-诊断与监控指标)
12. [附录](#12-附录)

---

## 1. 项目概述

### 1.1 项目背景

将Level2逐笔成交与逐笔委托数据转换为标准化的三维图像格式 `[15, 8, 8]`，用于Vision Transformer (ViT) 和 Video Vision Transformer (ViViT) 模型训练。

### 1.2 核心目标

| 维度 | 大小 | 含义 |
|------|------|------|
| 通道 | 15 | 7个成交类型 + 8个委托类型 |
| 价格 | 8 | 按当日联合分位数划分的8个价格区间 |
| 量 | 8 | 按当日联合分位数划分的8个量区间 |

### 1.3 预期产出

| 产出物 | 格式 | 存储方式 | 说明 |
|--------|------|----------|------|
| 单日图像 | LMDB | 每日一个文件 | Key=股票代码，Value=LZ4压缩的numpy bytes |
| 大单阈值 | Parquet | 每日更新 | 当日 Mean+Std 阈值 |
| 诊断报告 | CSV | 每日生成 | 通道填充率、稀疏度等监控指标 |

---

## 2. 数据源规格

### 2.1 文件组织

```
/raw_data/
├── {date}_sh_trade_data.parquet                  # 上交所成交
├── {date}_sh_order_data.parquet                  # 上交所委托（已预处理）
├── {date}_sz_order_data.parquet                  # 深交所委托
├── {date}_sz_trade_data.parquet                  # 深交所成交
├── {date}_sh_market_data_3s.parquet              # 上交所快照（涨跌停价）
└── {date}_sz_market_data_3s.parquet              # 深交所快照（涨跌停价）
```

### 2.2 上交所数据

#### 2.2.1 逐笔成交数据 (sh_trade_data)

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| SecurityID | varchar(20) | 证券代码，如 600519 | 股票识别 |
| TickTime | int | 时间，格式 HHMMSSmmm | 时间过滤 |
| **BizIndex** | int | **业务序号（交易所分配）** | **排序主键** |
| BuyOrderNO | int | 买方订单号 | 大单还原 |
| SellOrderNO | int | 卖方订单号 | 大单还原 |
| Price | decimal | 成交价格（元） | 分位数计算 |
| Qty | int | 成交数量（股） | 分位数计算 |
| TradeMoney | decimal | 成交金额（元） | 大单阈值 |
| TickBSFlag | varchar | B=主动买, S=主动卖 | 方向判定 |

#### 2.2.2 逐笔委托数据 (sh_order_data) - 已预处理还原

> **说明**：该表已从原始合并数据中拆分还原，`Qty`字段为完整母单量，`IsAggressive`字段标识主动性。

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| TickTime | int | 委托时间，格式 HHMMSSmmm | 时间过滤 |
| **BizIndex** | int | **业务序号（交易所分配）** | **排序主键** |
| OrdID | int | 委托单号 | 唯一标识 |
| OrdType | varchar | New=新增委托, Cancel=撤单 | 区分新增/撤单 |
| Side | varchar | B=买入, S=卖出 | 方向判定 |
| Price | decimal | 委托价格（元），撤单已补全 | 分位数计算 |
| Qty | int | 委托数量（股），**已还原完整母单量** | 分位数计算 |
| **IsAggressive** | bool | **主动性标识** | **通道9-12判定** |

**IsAggressive 判定逻辑**（在数据清洗阶段生成）：

> ⚠️ **关键排序规则 (Critical Sorting Rule)**
> 
> 高频数据中同一毫秒(Same Millisecond)发生多笔事件极为常见。**禁止仅使用 `TickTime` 排序**，否则会导致"委托"和"成交"的先后顺序随机错乱，从而误判 IsAggressive。
> 
> **必须使用联合主键排序**：`Sort Key: (TickTime ASC, BizIndex ASC)`

```python
# 判定规则：看该 OrdID 首次出现时的记录类型
# - 若首次出现在 Type='T'（成交）→ True（出生即成交，进攻型）
# - 若首次出现在 Type='A'（委托）→ False（出生即挂单，防守型）

def compute_is_aggressive(df_merged: pd.DataFrame) -> Dict[int, bool]:
    """
    计算每个委托单的主动性标识
    
    核心逻辑：
    - 主动单(Aggressive)：首次出现即成交，说明"出生即进攻"
    - 被动单(Passive)：首次出现是挂单，说明"出生即防守"
    
    注意：即便主动单后续有剩余挂单(Type='A')，其底色仍是"进攻"
    """
    is_aggressive = {}
    
    # ⚠️ 关键：必须使用联合主键排序！
    # 禁止仅按 TickTime 排序，同一毫秒内事件顺序会随机错乱
    df_sorted = df_merged.sort_values(['TickTime', 'BizIndex'])
    
    for _, row in df_sorted.iterrows():
        if row['TickBSFlag'] == 'B':
            order_no = row['BuyOrderNO']
        else:
            order_no = row['SellOrderNO']
        
        # 只记录首次出现
        if order_no not in is_aggressive:
            # 首次出现是成交 → 主动进攻
            is_aggressive[order_no] = (row['Type'] == 'T')
    
    return is_aggressive
```

**数据还原逻辑**（参考研报图表7）：
```
若 OrderID 在原始委托中存在：
    委托量 = 原始逐笔委托量 + 该单所有成交量
若 OrderID 不在原始委托中（一次撮合全部成交）：
    委托量 = 逐笔成交量
    委托ID = 成交撮合ID
```

### 2.3 深交所 - 逐笔委托数据

**表名**: `sz_order_data`  
**MDL消息**: `mdl.6.33`

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| SecurityID | varchar(20) | 证券代码 | 股票识别 |
| ApplSeqNum | bigint | 委托序列号 | 关联成交 |
| TransactTime | bigint | 委托时间 | 时间过滤 |
| Price | decimal | 委托价格（元） | 分位数计算 |
| OrderQty | bigint | 委托数量（股） | 分位数计算 |
| Side | varchar | 49=买, 50=卖, 71=借入, 70=出借 | 方向判定 |
| OrdType | varchar | 49=市价, 50=限价 | 信息记录 |

> **数据加载说明**
> 
> 在加载到内存时，应将深交所字段重命名以与上交所统一，便于复用后续计算逻辑：
> 
> | 深交所原字段 | 统一后字段名 | 说明 |
> |-------------|-------------|------|
> | `OrderQty` | `Qty` | 委托数量 |
> | `TransactTime` | `TickTime` | 时间戳 |
> | `LastPx` | `Price` | 成交价格 |
> | `LastQty` | `Qty` | 成交数量 |
> 
> ```python
> # 深交所委托表字段重命名
> df_sz_order = df_sz_order.rename(columns={
>     'OrderQty': 'Qty',
>     'TransactTime': 'TickTime'
> })
> 
> # 深交所成交表字段重命名
> df_sz_trade = df_sz_trade.rename(columns={
>     'LastPx': 'Price',
>     'LastQty': 'Qty',
>     'TransactTime': 'TickTime'
> })
> ```

### 2.4 深交所 - 逐笔成交数据

**表名**: `sz_trade_data`  
**MDL消息**: `mdl.6.36`

| 字段名 | 类型 | 说明 | 用途 |
|--------|------|------|------|
| SecurityID | varchar(20) | 证券代码 | 股票识别 |
| ApplSeqNum | bigint | 消息记录号 | 数据连续性 |
| BidApplSeqNum | bigint | 买方委托序列号 | 主动方判定/大单还原 |
| OfferApplSeqNum | bigint | 卖方委托序列号 | 主动方判定/大单还原 |
| TransactTime | bigint | 成交/撤单时间 | 时间过滤 |
| LastPx | decimal | 成交价格（元），撤单时为0 | 分位数计算 |
| LastQty | bigint | 成交/撤单数量（股） | 分位数计算 |
| ExecType | varchar | 70=成交, 52=撤单 | 通道判定 |

---

## 3. 15个通道精确定义

### 3.1 通道总览

> **设计升级**：通道7-12全部基于**委托表**构建，统一采用"意图导向"逻辑，实现沪深高度对齐。

| 索引 | 类别 | 名称 | 上交所数据源 | 深交所数据源 | 物理含义 |
|------|------|------|-------------|-------------|----------|
| 0 | 成交 | 全部成交 | 成交表 | ExecType='70' | 所有成交 |
| 1 | 成交 | 主动买入 | 成交表 & BSFlag='B' | 成交表 & BidSeq>OfferSeq | 买方主动吃单 |
| 2 | 成交 | 主动卖出 | 成交表 & BSFlag='S' | 成交表 & OfferSeq>BidSeq | 卖方主动吃单 |
| 3 | 成交 | 大买单 | 成交表 | 成交表 | 买方母单金额≥阈值 |
| 4 | 成交 | 大卖单 | 成交表 | 成交表 | 卖方母单金额≥阈值 |
| 5 | 成交 | 小买单 | 成交表 | 成交表 | 买方母单金额<阈值 |
| 6 | 成交 | 小卖单 | 成交表 | 成交表 | 卖方母单金额<阈值 |
| 7 | 委托 | 买单 | 委托表 Side='B' | 委托表 Side='49' | **买方总意愿** |
| 8 | 委托 | 卖单 | 委托表 Side='S' | 委托表 Side='50' | **卖方总意愿** |
| 9 | 委托 | 主动买入 | **委托表 Side='B' & IsAggressive=True** | **委托表 & ApplSeq in ActiveBuySeqs** | **进攻型买单意图** |
| 10 | 委托 | 主动卖出 | **委托表 Side='S' & IsAggressive=True** | **委托表 & ApplSeq in ActiveSellSeqs** | **进攻型卖单意图** |
| 11 | 委托 | 非主动买入 | 委托表 Side='B' & IsAggressive=False | 委托表 & ApplSeq not in ActiveBuySeqs | **防守型买单** |
| 12 | 委托 | 非主动卖出 | 委托表 Side='S' & IsAggressive=False | 委托表 & ApplSeq not in ActiveSellSeqs | **防守型卖单** |
| 13 | 委托 | 撤买 | 委托表 OrdType='Cancel' & Side='B' | ExecType='52' & BidSeq>0 | 撤销买入委托 |
| 14 | 委托 | 撤卖 | 委托表 OrdType='Cancel' & Side='S' | ExecType='52' & OfferSeq>0 | 撤销卖出委托 |

**核心设计原则**：
- 通道7-12 **全部基于委托表**，使用完整委托量（OrderQty）
- 通道9/10 统计的是 **"完整的主动进攻意图"**，而非仅仅是"成交掉的那部分"
- **数学关系**：`Ch7 = Ch9 + Ch11`，`Ch8 = Ch10 + Ch12`（严格互斥）

> ⛔ **通道9/10数据源限制（强制规则）**
> 
> 通道9/10的数据源**严格限制为委托表（Orders）**，**禁止从成交表（Trades）获取**！
> 
> | 通道 | ✅ 正确数据源 | ❌ 错误数据源 |
> |------|-------------|--------------|
> | 9-主动买入 | 委托表 + IsAggressive/ActiveSeqs | ~~成交表 BSFlag='B'~~ |
> | 10-主动卖出 | 委托表 + IsAggressive/ActiveSeqs | ~~成交表 BSFlag='S'~~ |
> 
> **原因**：本方案采用"意图导向"设计，通道9/10记录的是**完整委托量（进攻意图）**，而非成交量。
> 从成交表获取将导致信息损失（只记录已成交部分，丢失未成交的进攻意图）。

### 3.2 上交所判定规则

> **逻辑升级**：成交表只填充通道0-6，委托表填充通道7-14（利用预处理的 `IsAggressive` 字段）

#### 3.2.1 成交表处理 (sh_trade_data)

成交表**仅**填充通道：0-6

```python
def process_sh_trade(row, image, price_bin, qty_bin, 
                     buy_parent_amount, sell_parent_amount, threshold):
    """
    处理上交所一笔成交记录
    
    填充通道：0, 1或2, 3或5, 4或6
    注意：通道9/10不在这里填充！改为从委托表填充
    """
    # 通道0: 全部成交
    image[0, price_bin, qty_bin] += 1
    
    # 通道1-2: 根据主动方向填充（成交层面）
    if row['TickBSFlag'] == 'B':
        image[1, price_bin, qty_bin] += 1  # 通道1: 成交-主动买入
    elif row['TickBSFlag'] == 'S':
        image[2, price_bin, qty_bin] += 1  # 通道2: 成交-主动卖出
    
    # 通道3-6: 大小单判定（与主动方向无关）
    buy_order_no = row['BuyOrderNO']
    sell_order_no = row['SellOrderNO']
    
    if buy_order_no in buy_parent_amount:
        if buy_parent_amount[buy_order_no] >= threshold:
            image[3, price_bin, qty_bin] += 1  # 通道3: 大买单
        else:
            image[5, price_bin, qty_bin] += 1  # 通道5: 小买单
    
    if sell_order_no in sell_parent_amount:
        if sell_parent_amount[sell_order_no] >= threshold:
            image[4, price_bin, qty_bin] += 1  # 通道4: 大卖单
        else:
            image[6, price_bin, qty_bin] += 1  # 通道6: 小卖单
```

#### 3.2.2 委托表处理 (sh_order_data) - 意图导向

委托表填充通道：7-14（利用 `IsAggressive` 字段实现主动/被动分流）

```python
def process_sh_order(row, image, price_bin, qty_bin):
    """
    处理上交所一笔委托记录（意图导向升级版）
    
    OrdType='New': 填充通道7/8，并根据 IsAggressive 分流到 9/10 或 11/12
    OrdType='Cancel': 填充通道13/14
    
    核心变化：
    - 通道9/10 使用完整委托量 Qty（进攻意图），不再是成交量
    - IsAggressive=True 表示"出生即成交"的进攻型订单
    """
    if row['OrdType'] == 'New':
        if row['Side'] == 'B':
            # 通道7: 买单总意愿
            image[7, price_bin, qty_bin] += 1
            
            # 根据主动性分流
            if row['IsAggressive']:
                # 通道9: 主动买入（进攻意图）
                image[9, price_bin, qty_bin] += 1
            else:
                # 通道11: 非主动买入（防守挂单）
                image[11, price_bin, qty_bin] += 1
                
        elif row['Side'] == 'S':
            # 通道8: 卖单总意愿
            image[8, price_bin, qty_bin] += 1
            
            # 根据主动性分流
            if row['IsAggressive']:
                # 通道10: 主动卖出（进攻意图）
                image[10, price_bin, qty_bin] += 1
            else:
                # 通道12: 非主动卖出（防守挂单）
                image[12, price_bin, qty_bin] += 1
            
    elif row['OrdType'] == 'Cancel':
        # 撤单（Price已补全）
        if row['Side'] == 'B':
            image[13, price_bin, qty_bin] += 1  # 通道13: 撤买
        elif row['Side'] == 'S':
            image[14, price_bin, qty_bin] += 1  # 通道14: 撤卖
```

**关键升级说明**：

| 维度 | 原方案（结果导向） | 新方案（意图导向） |
|------|-------------------|-------------------|
| 通道9/10数据源 | 成交表 | **委托表** |
| 通道9/10数值 | 成交量（已实现） | **委托量（完整意图）** |
| 物理含义 | 实际打掉多少 | **想要打掉多少** |
| Alpha信息 | 只看结果 | **捕捉进攻欲望** |

**示例对比**：主力挂10,000股主动买单，实际成交2,000股

| 方案 | 通道9记录 | 信息损失 |
|------|----------|----------|
| 原方案 | 2,000 | 丢失8,000股的"进攻欲望" |
| **新方案** | **10,000** | **完整保留主力意图** |

### 3.3 深交所判定规则

> **逻辑升级**：通道7-12全部基于委托表，引入"主动委托索引"实现与上交所的物理含义对齐。

#### 3.3.1 预处理：构建主动委托索引

在处理委托表之前，先扫描成交表构建**主动方集合**：

```python
def build_sz_active_index(df_trade: pd.DataFrame) -> Tuple[set, set]:
    """
    深交所专用预处理：构建主动委托索引
    
    核心逻辑：
    - 在每笔成交中，序列号较大的一方是"后来者"（进攻方）
    - BidSeq > OfferSeq → 买方是主动方
    - OfferSeq > BidSeq → 卖方是主动方
    
    Returns:
        active_buy_seqs: 主动买入的委托序列号集合
        active_sell_seqs: 主动卖出的委托序列号集合
        
    注意：仅提取 Trade 中 Sequence Number 较大的一方作为主动方 ID
    """
    # 只处理成交记录
    df_exec = df_trade[df_trade['ExecType'] == '70']
    
    # 买方主动：买方序列号 > 卖方序列号
    active_buy_seqs = set(
        df_exec[df_exec['BidApplSeqNum'] > df_exec['OfferApplSeqNum']]['BidApplSeqNum']
    )
    
    # 卖方主动：卖方序列号 > 买方序列号
    active_sell_seqs = set(
        df_exec[df_exec['OfferApplSeqNum'] > df_exec['BidApplSeqNum']]['OfferApplSeqNum']
    )
    
    return active_buy_seqs, active_sell_seqs
```

**主动性判定的时间语义**：

> **重要提醒**：主动性是指**委托入场瞬间(On Entry)**的状态，而非"生命周期内是否成交"。
> 
> **示例**：
> - 9:30:01 挂买单 A (10000股)，价格 10.00，当前卖一 10.01。A 是**被动单**。
> - 9:30:05 卖单 B 砸盘，吃掉了 A。
> - 判定：A 永远是 **通道11（非主动）**，尽管它后来成交了；B 是 **通道10（主动卖）**。

#### 3.3.2 成交表处理 (ExecType='70')

成交表**仅**填充通道：0-6

```python
def process_sz_trade(row, image, price_bin, qty_bin,
                     buy_parent_amount, sell_parent_amount, threshold):
    """
    处理深交所一笔成交记录
    
    填充通道：0, 1或2, 3或5, 4或6
    注意：通道9/10不在这里填充！改为从委托表填充
    """
    # 通道0: 全部成交
    image[0, price_bin, qty_bin] += 1
    
    bid_seq = row['BidApplSeqNum']
    offer_seq = row['OfferApplSeqNum']
    
    # 通道1-2: 根据主动方向填充（成交层面）
    if bid_seq > offer_seq:
        image[1, price_bin, qty_bin] += 1  # 通道1: 成交-主动买入
    elif offer_seq > bid_seq:
        image[2, price_bin, qty_bin] += 1  # 通道2: 成交-主动卖出
    
    # 通道3-6: 大小单判定（与主动方向无关，每笔成交都判定买卖双方）
    if bid_seq in buy_parent_amount:
        if buy_parent_amount[bid_seq] >= threshold:
            image[3, price_bin, qty_bin] += 1  # 通道3: 大买单
        else:
            image[5, price_bin, qty_bin] += 1  # 通道5: 小买单
    
    if offer_seq in sell_parent_amount:
        if sell_parent_amount[offer_seq] >= threshold:
            image[4, price_bin, qty_bin] += 1  # 通道4: 大卖单
        else:
            image[6, price_bin, qty_bin] += 1  # 通道6: 小卖单
```

#### 3.3.3 委托表处理 - 意图导向

委托表填充通道：7-12（使用完整委托量 OrderQty）

```python
def process_sz_order(row, image, price_bin, qty_bin, 
                     active_buy_seqs: set, active_sell_seqs: set):
    """
    处理深交所一笔委托记录（意图导向升级版）
    
    填充通道：7或8，并根据主动性分流到 9/10 或 11/12
    
    核心变化：
    - 通道9/10 使用完整委托量 OrderQty（进攻意图），不再是成交量
    - 只要该委托在生命周期内做过"主动方"，整单计入通道9/10
    """
    appl_seq = row['ApplSeqNum']
    
    if row['Side'] == '49':  # 买入
        # 通道7: 买单总意愿
        image[7, price_bin, qty_bin] += 1
        
        # 根据主动性分流
        if appl_seq in active_buy_seqs:
            # 通道9: 主动买入（进攻意图）
            image[9, price_bin, qty_bin] += 1
        else:
            # 通道11: 非主动买入（防守挂单）
            image[11, price_bin, qty_bin] += 1
            
    elif row['Side'] == '50':  # 卖出
        # 通道8: 卖单总意愿
        image[8, price_bin, qty_bin] += 1
        
        # 根据主动性分流
        if appl_seq in active_sell_seqs:
            # 通道10: 主动卖出（进攻意图）
            image[10, price_bin, qty_bin] += 1
        else:
            # 通道12: 非主动卖出（防守挂单）
            image[12, price_bin, qty_bin] += 1
```

#### 3.3.4 撤单记录处理 (ExecType='52')

```python
def process_sz_cancel(row, image, price_bin, qty_bin):
    """
    处理深交所一笔撤单记录
    
    关键点：
    1. 撤单时 BidApplSeqNum 和 OfferApplSeqNum 只有一个有值
    2. 有值的那个指向被撤销的委托
    """
    bid_seq = row['BidApplSeqNum']
    offer_seq = row['OfferApplSeqNum']
    
    if bid_seq > 0 and offer_seq == 0:
        # 撤销买单
        image[13, price_bin, qty_bin] += 1  # 通道13: 撤买
    elif offer_seq > 0 and bid_seq == 0:
        # 撤销卖单
        image[14, price_bin, qty_bin] += 1  # 通道14: 撤卖
```

**⚠️ 深交所撤单价格问题（必须处理）**

深交所撤单记录的 `LastPx = 0`，需要关联委托表获取原始价格。

```python
def preprocess_sz_cancel_with_price(df_cancel: pd.DataFrame, 
                                     df_order: pd.DataFrame) -> pd.DataFrame:
    """预处理深交所撤单，关联委托表获取原始委托价格"""
    order_price_map = df_order.set_index('ApplSeqNum')['Price'].to_dict()
    
    def get_original_price(row):
        if row['BidApplSeqNum'] > 0:
            return order_price_map.get(row['BidApplSeqNum'], 0)
        elif row['OfferApplSeqNum'] > 0:
            return order_price_map.get(row['OfferApplSeqNum'], 0)
        return 0
    
    df_cancel = df_cancel.copy()
    df_cancel['OriginalPrice'] = df_cancel.apply(get_original_price, axis=1)
    return df_cancel
```

**处理流程**：
```python
# 1. 筛选撤单记录
df_cancel = df_trade[df_trade['ExecType'] == '52']

# 2. 关联委托表获取原始价格
df_cancel = preprocess_sz_cancel_with_price(df_cancel, df_order)

# 3. 使用 OriginalPrice 计算 price_bin
for _, row in df_cancel.iterrows():
    price_bin = get_bin(row['OriginalPrice'], price_bins)  # 使用原始委托价格
    qty_bin = get_bin(row['LastQty'], qty_bins)
    process_sz_cancel(row, image, price_bin, qty_bin)
```

**不处理的后果**：
| 场景 | 实际位置 | 错误映射 | 语义丢失 |
|------|----------|----------|----------|
| 主力在高位撤单（诱多撤退） | price_bin=7 | price_bin=0 | ❌ |
| 主力在低位撤单（虚假托单） | price_bin=1 | price_bin=0 | ❌ |
| 模型视角 | - | 通道13/14只有底行有信号 | 无法学习撤单位置信号 |

### 3.4 通道填充总结（意图导向升级版）

#### 上交所填充规则

| 数据表 | 数据类型 | 填充通道 | 说明 |
|--------|----------|----------|------|
| 成交表 | 全部成交 | 0, 1或2, 3或5, 4或6 | 一笔成交填充**4个**通道 |
| 委托表 | OrdType='New' | 7或8, 9或10或11或12 | 根据IsAggressive分流 |
| 委托表 | OrdType='Cancel' | 13或14 | 撤单，Price已补全 |

**关键变化**：通道9/10从成交表移到委托表填充！

#### 深交所填充规则

| 数据类型 | 预处理 | 填充通道 | 说明 |
|----------|--------|----------|------|
| 成交 (ExecType='70') | - | 0, 1或2, 3或5, 4或6 | 一笔成交填充**4个**通道 |
| 委托 | 需构建ActiveSeqs | 7或8, 9或10或11或12 | 根据ApplSeq是否在ActiveSeqs分流 |
| 撤单 (ExecType='52') | 关联委托表获价格 | 13或14 | 撤单 |

#### 沪深统一：通道7-12的物理含义（升级版）

| 通道 | 上交所 | 深交所 | 物理含义 |
|------|--------|--------|----------|
| 7-买单 | 委托表 Side='B' | 委托表 Side='49' | **买方总意愿** |
| 8-卖单 | 委托表 Side='S' | 委托表 Side='50' | **卖方总意愿** |
| **9-主动买** | **委托表 IsAggressive=True** | **委托表 & ApplSeq in ActiveBuySeqs** | **进攻型买单意图** |
| **10-主动卖** | **委托表 IsAggressive=True** | **委托表 & ApplSeq in ActiveSellSeqs** | **进攻型卖单意图** |
| 11-非主动买 | 委托表 IsAggressive=False | 委托表 & ApplSeq not in ActiveBuySeqs | **防守型买单** |
| 12-非主动卖 | 委托表 IsAggressive=False | 委托表 & ApplSeq not in ActiveSellSeqs | **防守型卖单** |

**核心数学关系**：
- `Ch7 = Ch9 + Ch11`（买单总意愿 = 进攻型 + 防守型）
- `Ch8 = Ch10 + Ch12`（卖单总意愿 = 进攻型 + 防守型）
- 这是**严格互斥分解**，降低模型学习难度

**大小单判定要点**（不变）：
- 通道3-6与主动方向（BSFlag）**无关**
- 每笔成交都**同时判定**买方和卖方的大小单
- 一笔成交必定产生：1个大买或小买 + 1个大卖或小卖

### 3.5 设计说明：为什么升级到"意图导向"？

#### 3.5.1 核心理念升级

| 维度 | 原方案（结果导向） | 新方案（意图导向） |
|------|-------------------|-------------------|
| **数据源** | 通道9/10从成交表 | 通道9/10从委托表 |
| **取值** | 成交量（已实现） | **委托量（完整意图）** |
| **物理含义** | 实际打掉多少 | **想要打掉多少** |
| **Alpha来源** | 只看结果 | **捕捉进攻欲望** |
| **沪深一致性** | 数据源混用 | **全委托表，高度统一** |

#### 3.5.2 金融逻辑：意图比结果更有Alpha

在量化微观结构研究(Market Microstructure)中，**Alpha往往隐藏在"意图"中，而非"结果"中**。

**示例**：主力挂10,000股主动买单，但市场流动性差，只成交了2,000股

| 方案 | 通道9记录 | 信息含义 |
|------|----------|----------|
| 原方案 | 2,000 | 只看到"成交了多少" |
| **新方案** | **10,000** | **看到"主力急不可耐，想买10,000股"** |

**价值**：
- 原方案丢失了8,000股的"进攻欲望（Aggressiveness）"
- 新方案完整保留主力意图，能让模型捕捉到主力资金"急不可耐"的程度

#### 3.5.3 为什么现在可以升级？

原方案选择"结果导向"的原因是上交所数据缺陷——主动单瞬间成交，看不到委托记录。

**现在的突破**：通过数据清洗阶段的 `IsAggressive` 字段还原，上交所也能区分主动/被动委托了！

```
原始数据限制 → 被迫"指鹿为马" → 损失Alpha信息
       ↓
清洗阶段还原IsAggressive → 沪深都能识别主动性 → 升级为"意图导向"
```

#### 3.5.4 升级后的逻辑图谱

经过升级，通道7-12达成**完美的逻辑闭环**：

- **Channel 7/8 (Total)** = Σ All Orders（买/卖方总意愿）
- **Channel 9/10 (Active)** = Σ Orders that *initiated* a trade（**进攻意图量**）
- **Channel 11/12 (Passive)** = Σ Orders that *never initiated* a trade（**防守挂单量**）

**数学验证**：
```
Ch7 = Ch9 + Ch11  ✓ （记录数和意图量的完美拆分）
Ch8 = Ch10 + Ch12 ✓
```

这个方案不仅逻辑严密，而且最大限度地提取了主力资金的**进攻欲望(Alpha)**。

---

## 4. 数据清洗规则

### 4.1 时间过滤

仅保留连续竞价时段数据：

```python
def is_continuous_auction(tick_time: int) -> bool:
    """
    判断是否在连续竞价时段
    tick_time 格式: HHMMSSmmm (如 93000000 表示 09:30:00.000)
    """
    # 提取小时分钟
    hhmm = tick_time // 100000
    
    # 上午连续竞价: 09:30 - 11:30
    if 930 <= hhmm < 1130:
        return True
    
    # 下午连续竞价: 13:00 - 14:57
    if 1300 <= hhmm < 1457:
        return True
    
    return False
```

| 时段 | 处理 | 说明 |
|------|------|------|
| 09:15 - 09:25 | 剔除 | 开盘集合竞价 |
| 09:25 - 09:30 | 剔除 | 集合竞价撮合期 |
| 09:30 - 11:30 | **保留** | 上午连续竞价 |
| 11:30 - 13:00 | 剔除 | 午间休市 |
| 13:00 - 14:57 | **保留** | 下午连续竞价 |
| 14:57 - 15:00 | 剔除 | 收盘集合竞价（深交所） |

### 4.2 涨跌停过滤（暂不启用）

> **说明**：当前作为特征输入模型，不直接用于选股，涨跌停过滤对结果影响不大，暂时跳过以提升处理效率。如后续需要启用，可参考以下代码。

```python
def get_limit_prices(stock_code: str, trade_date: str, 
                     sh_snapshot: pd.DataFrame, 
                     sz_snapshot: pd.DataFrame) -> Tuple[float, float]:
    """获取涨跌停价格（备用）"""
    
    if stock_code.endswith('.SH'):
        # 上交所：从快照获取昨收价，计算涨跌停
        pre_close = sh_snapshot.query(
            f"SecurityID == '{stock_code[:6]}'"
        )['PreCloPrice'].iloc[0]
        
        # 主板10%涨跌幅（科创板20%需要单独判断）
        high_limit = round(pre_close * 1.10, 2)
        low_limit = round(pre_close * 0.90, 2)
        
    else:  # .SZ
        # 深交所：快照中直接有涨跌停价
        row = sz_snapshot.query(f"SecurityID == '{stock_code[:6]}'").iloc[0]
        high_limit = row['HighLimitPrice']
        low_limit = row['LowLimitPrice']
    
    return high_limit, low_limit


def filter_limit_price(df: pd.DataFrame, high_limit: float, 
                       low_limit: float) -> pd.DataFrame:
    """剔除涨跌停时刻的数据（备用）"""
    return df[
        (df['Price'] < high_limit) & 
        (df['Price'] > low_limit)
    ]
```

### 4.3 异常值过滤

```python
def filter_anomalies(df: pd.DataFrame, is_cancel: bool = False) -> pd.DataFrame:
    """
    剔除异常记录
    
    注意：撤单记录的 Price 可能为0，需要特殊处理
    """
    if is_cancel:
        # 撤单记录只检查数量
        return df[df['Qty'] > 0]
    else:
        # 成交/委托记录检查价格和数量
        return df[(df['Price'] > 0) & (df['Qty'] > 0)]
```

### 4.4 深交所撤单价格关联（必须处理）

**问题背景**：深交所撤单记录的 `LastPx = 0`，直接使用会导致所有撤单都被映射到价格bin=0。

**解决方案**：在数据清洗阶段，将撤单记录与委托表关联，获取原始委托价格。

```python
def enrich_sz_cancel_price(df_trade: pd.DataFrame, 
                           df_order: pd.DataFrame) -> pd.DataFrame:
    """
    深交所撤单价格关联
    
    将撤单记录的 LastPx=0 替换为原始委托价格
    
    ⚠️ 注意：pd.concat后必须重新排序，否则物理顺序会被打乱
    """
    # 筛选撤单记录
    cancel_mask = df_trade['ExecType'] == '52'
    df_cancel = df_trade[cancel_mask].copy()
    df_normal = df_trade[~cancel_mask]
    
    # 构建委托价格映射
    order_price_map = df_order.set_index('ApplSeqNum')['Price'].to_dict()
    
    def get_original_price(row):
        if row['BidApplSeqNum'] > 0:
            return order_price_map.get(row['BidApplSeqNum'], 0)
        elif row['OfferApplSeqNum'] > 0:
            return order_price_map.get(row['OfferApplSeqNum'], 0)
        return 0
    
    # 替换价格
    df_cancel['LastPx'] = df_cancel.apply(get_original_price, axis=1)
    
    # 合并回去
    df_result = pd.concat([df_normal, df_cancel], ignore_index=True)
    
    # ⚠️ 关键：必须重新排序！pd.concat会打乱物理顺序
    df_result = df_result.sort_values(['TransactTime', 'ApplSeqNum']).reset_index(drop=True)
    
    return df_result
```

**处理时机**：在时间过滤、异常值过滤之后，分位数计算之前执行。

**验证检查**：
```python
# 检查是否还有价格为0的撤单
assert (df_cancel['LastPx'] == 0).sum() == 0, "存在未关联到价格的撤单"
```

### 4.5 价格兜底逻辑（稳健性保障）

虽然实际数据验证显示撤单和市价单的价格字段已有有效值，但为确保系统稳健性，仍需实现兜底逻辑，防止数据源偶发异常。

#### 4.5.1 上交所撤单价格兜底

**问题**：上交所撤单（Type='D'）的 Price 字段官方定义为"无意义"，虽然实测有值，但需防止偶发的 Price=0。

**策略**：优先读字段，缓存来救急

```python
# 初始化缓存（放在循环外）
sh_order_cache = {}

def process_sh_row_with_fallback(row, image, current_last_price, ...):
    """带兜底逻辑的上交所数据处理"""
    
    # 确定订单号
    if row['TickBSFlag'] == 'B':
        order_no = row['BuyOrderNO']
    else:
        order_no = row['SellOrderNO']
    
    price = row['Price']
    
    # === 步骤A：写入缓存（Type='A' 或 'T'）===
    if row['Type'] in ['A', 'T'] and price > 0:
        if order_no > 0:
            sh_order_cache[order_no] = price
    
    # === 步骤B：撤单兜底逻辑（Type='D'）===
    if row['Type'] == 'D':
        # 优先用当前行的价格
        if price <= 0.001:
            # 一级兜底：查缓存
            price = sh_order_cache.get(order_no, 0)
        
        if price <= 0.001:
            # 二级兜底：用最新成交价
            price = current_last_price
        
        if price <= 0.001:
            # 极端情况：跳过该记录
            return
        
        # 正常计算 price_bin ...
        
        # (可选) 撤单后清理缓存节省内存
        # if order_no in sh_order_cache: 
        #     del sh_order_cache[order_no]
```

#### 4.5.2 深交所市价单价格兜底

**问题**：深交所市价委托（OrdType='1' 或 'U'）本质上"不指定价格"，虽然数据源通常会填充估算价，但需防范 Price=0。

**策略**：直接用最新成交价（符合市价单"随行就市"的物理含义）

```python
# current_last_price 由最新的 Trade 数据实时更新

def process_sz_order_with_fallback(row, image, current_last_price, ...):
    """带兜底逻辑的深交所委托处理"""
    
    price = row['Price']
    
    # === 兜底逻辑 ===
    # 不需要判断 OrdType，直接看价格是否有效
    if price <= 0.001:
        # 市价单导致价格为0，视为意图在当前最新价附近成交
        price = current_last_price
    
    if price <= 0.001:
        # 极端情况：系统刚启动，LastPrice 也是 0
        return  # 跳过该记录
    
    # 正常计算 price_bin ...
```

#### 4.5.3 兜底策略对比

| 特性 | 上交所撤单 (SH Cancel) | 深交所市价单 (SZ Market) |
|------|------------------------|--------------------------|
| **Price=0 的原因** | 字段定义为"无意义" | 订单类型本身不指定价格 |
| **正确的价格应该是** | **历史挂单价**（挂在80元就在80元撤） | **当前市场价**（随行就市） |
| **技术手段** | **重型**：维护 OrderNO→Price 映射表 | **轻型**：读取 LastPrice 变量 |
| **兜底优先级** | 原字段 → 缓存 → LastPrice | 原字段 → LastPrice |

---

## 5. 分位数计算规则

### 5.1 计算方案：联合计算 (Unified Scale)

将成交数据和委托数据合并后统一计算分位数边界，保留绝对体量差异信息。

> ⚠️ **关键修正（与通道9/10升级相关）**
> 
> 由于通道9/10升级为"意图导向"，使用**委托量（OrderQty）**而非成交量。委托量通常比成交量大1-2个数量级。
> 
> **必须确保**：计算 `qty_bins` 时包含委托数据，否则通道9/10的数据将全部溢出到最大的bin中，导致特征失效！

### 5.2 实现代码

```python
def compute_quantile_bins(df_trade: pd.DataFrame, 
                          df_order: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算价格和量的分位数边界
    
    ⚠️ 重要：必须联合委托数据计算qty_bins！
    原因：通道9/10现在使用委托量（OrderQty），而非成交量（TradeQty）
    
    Returns:
        price_bins: 7个价格切割点，定义8个区间
        qty_bins: 7个量切割点，定义8个区间
    """
    # 联合所有价格
    all_prices = pd.concat([
        df_trade['Price'],
        df_order['Price']
    ]).dropna()
    
    # 联合所有量（关键：必须包含委托量！）
    all_volumes = pd.concat([
        df_trade['Qty'],           # 成交量（小）
        df_order['OrderQty']       # 委托量（大）- 通道9/10使用此数据
    ]).dropna()
    
    # 计算分位数边界
    percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    
    price_bins = np.percentile(all_prices, percentiles)
    qty_bins = np.percentile(all_volumes, percentiles)
    
    return price_bins, qty_bins


def get_bin_index(value: float, bins: np.ndarray) -> int:
    """
    根据值获取所属的bin索引 (0-7)
    
    bins: 7个切割点
    返回: 0-7的索引
    """
    return int(np.digitize(value, bins))
```

### 5.3 预期效果

由于委托量普遍大于成交量：
- 成交数据的量大部分落在 **bin 0-3**（低区间）
- 委托数据的量大部分落在 **bin 3-7**（高区间）

这是设计预期，模型可学习"委托单普遍比成交单大"这一市场微观结构特征。

---

## 6. 大小单判定规则

### 6.1 母单还原

机构主力常采用拆单策略，需通过聚合订单编号还原真实委托意图：

```python
def restore_parent_orders_sh(df_trade: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    上交所母单还原
    
    Returns:
        buy_parent_amount: {BuyOrderNO: 累计成交金额}
        sell_parent_amount: {SellOrderNO: 累计成交金额}
    """
    buy_parent = df_trade.groupby('BuyOrderNO')['TradeMoney'].sum().to_dict()
    sell_parent = df_trade.groupby('SellOrderNO')['TradeMoney'].sum().to_dict()
    
    return buy_parent, sell_parent


def restore_parent_orders_sz(df_trade: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    深交所母单还原
    
    注意：深交所没有 TradeMoney 字段，需要计算
    """
    df_trade = df_trade.copy()
    df_trade['TradeMoney'] = df_trade['LastPx'] * df_trade['LastQty']
    
    buy_parent = df_trade.groupby('BidApplSeqNum')['TradeMoney'].sum().to_dict()
    sell_parent = df_trade.groupby('OfferApplSeqNum')['TradeMoney'].sum().to_dict()
    
    return buy_parent, sell_parent
```

### 6.2 动态阈值计算

> **阈值计算逻辑 (Threshold Calculation)**
> 
> - **计算范围**：仅基于**当前交易日**的所有有效委托/成交数据（Daily Stats）
> - **弃用滚动窗口**：为严格复现研报逻辑，**不使用历史滚动窗口**
> - **公式**：$Threshold = \mu_{today} + \alpha \cdot \sigma_{today}$，其中 $\alpha = 1.0$

```python
def compute_threshold(daily_amounts: np.ndarray, alpha: float = 1.0) -> float:
    """
    计算大单阈值（仅使用当日数据）
    
    公式:
        Threshold = Mean(V) + α × Std(V)
    
    Args:
        daily_amounts: 当日所有母单成交金额
        alpha: 标准差倍数，默认1.0
        
    说明:
        - 仅使用当日数据计算，严格复现研报逻辑
        - 不使用历史滚动窗口，避免数据泄露风险
        - 计算简单，性能好，适合批量处理
    """
    if len(daily_amounts) == 0:
        return float('inf')  # 无数据时，所有单都是小单
    
    mean_amount = np.mean(daily_amounts)
    std_amount = np.std(daily_amounts)
    threshold = mean_amount + alpha * std_amount
    
    return threshold
```

### 6.3 阈值计算时机

在母单还原之后、图像构建之前计算：

```python
# 1. 先还原母单
buy_parent_amounts = df.groupby('BuyOrderNO')['TradeMoney'].sum()
sell_parent_amounts = df.groupby('SellOrderNO')['TradeMoney'].sum()

# 2. 合并所有母单金额
all_amounts = np.concatenate([
    buy_parent_amounts.values, 
    sell_parent_amounts.values
])

# 3. 计算当日阈值
threshold = compute_threshold(all_amounts)
```

---

## 7. 图像构建完整流程

### 7.1 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据加载层                               │
├─────────────────────────────────────────────────────────────────┤
│  上交所: 读取 sh_trade_data + sh_order_data（已预处理）           │
│  深交所: 读取 sz_order_data + sz_trade_data                       │
│  ⚠️ 排序键: (TickTime ASC, BizIndex/ApplSeqNum ASC)               │
│  快照: 读取 sh_market_data_3s / sz_market_data_3s               │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         数据清洗层                               │
├─────────────────────────────────────────────────────────────────┤
│  时间过滤 → 异常值过滤 → 剔除 Type='S'                           │
│  → 深交所撤单价格关联（用委托表补全 LastPx=0 的撤单）           │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         预计算层                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. 母单还原: groupby(OrderNO).sum()                            │
│  2. 阈值计算: 当日 Mean + Std                                   │
│  3. 分位数计算: percentile(all_prices/volumes, [12.5,...,87.5]) │
│  4. 主动委托集合: 提取成交中的主动方序列号                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         图像构建层                               │
├─────────────────────────────────────────────────────────────────┤
│  初始化 image = zeros(15, 8, 8)                                  │
│  遍历每条成交记录 → 计算bin → 判定通道 → 累加像素值             │
│  遍历每条委托记录 → 计算bin → 判定通道 → 累加像素值             │
│  遍历每条撤单记录 → 计算bin → 判定通道 → 累加像素值             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      归一化 & 存储层                             │
├─────────────────────────────────────────────────────────────────┤
│  log1p变换 → 通道内Max归一化 → LZ4压缩 → 写入LMDB               │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 完整构建代码

```python
class Level2ImageBuilder:
    """单只股票单日的图像构建器"""
    
    def __init__(self, stock_code: str, trade_date: str):
        self.stock_code = stock_code
        self.trade_date = trade_date
        self.image = np.zeros((15, 8, 8), dtype=np.float32)
        
    def build(self, df_trade: pd.DataFrame, df_order: pd.DataFrame,
              price_bins: np.ndarray, qty_bins: np.ndarray,
              buy_parent: Dict, sell_parent: Dict,
              threshold: float, active_seqs: Dict) -> np.ndarray:
        """
        主构建流程
        """
        # 处理成交记录
        for _, row in df_trade.iterrows():
            price_bin = self._get_bin(row['Price'], price_bins)
            qty_bin = self._get_bin(row['Qty'], qty_bins)
            
            if self._is_sh():
                self._process_sh_trade(row, price_bin, qty_bin,
                                       buy_parent, sell_parent, threshold)
            else:
                self._process_sz_trade(row, price_bin, qty_bin,
                                       buy_parent, sell_parent, threshold)
        
        # 处理委托记录
        for _, row in df_order.iterrows():
            price_bin = self._get_bin(row['Price'], price_bins)
            qty_bin = self._get_bin(row['Qty'], qty_bins)
            
            if self._is_sh():
                self._process_sh_order(row, price_bin, qty_bin, active_seqs)
            else:
                self._process_sz_order(row, price_bin, qty_bin, active_seqs)
        
        # 处理撤单记录（已在 df_trade 中，需要单独筛选）
        # ...
        
        return self.normalize()
    
    def _get_bin(self, value: float, bins: np.ndarray) -> int:
        """获取bin索引 (0-7)"""
        idx = int(np.digitize(value, bins))
        return min(idx, 7)  # 确保不超过7
    
    def _is_sh(self) -> bool:
        return self.stock_code.endswith('.SH')
    
    def normalize(self) -> np.ndarray:
        """Log1p + 通道内Max归一化"""
        log_image = np.log1p(self.image)
        
        for ch in range(15):
            max_val = log_image[ch].max()
            if max_val > 0:
                log_image[ch] /= max_val
        
        return log_image
```

### 7.3 像素填充示例

#### 示例1：上交所委托表处理（意图导向升级版）

条件：委托表中一条**主动**新增买单记录
- OrdID = 12345
- OrdType = 'New'
- Side = 'B'
- **IsAggressive = True**（出生即成交的进攻型）
- Price = 12.50元 → price_bin = 5
- Qty = 10,000股（**已是完整母单量**）→ qty_bin = 7

```python
# 填充通道7和9（主动型）
image[7, 5, 7] += 1   # 通道7: 买单（完整意愿）
image[9, 5, 7] += 1   # 通道9: 主动买入（进攻意图）- 使用完整委托量！
```

**对比防守型订单**（IsAggressive = False）：
```python
# 填充通道7和11（防守型）
image[7, 5, 7] += 1   # 通道7: 买单（完整意愿）
image[11, 5, 7] += 1  # 通道11: 非主动买入（防守挂单）
```

**关键变化说明**：
| 维度 | 原方案 | 新方案（意图导向） |
|------|--------|-------------------|
| 通道9数据源 | 成交表 | **委托表** |
| 通道9数值 | 成交量(2000) | **委托量(10000)** |
| 信息含义 | 实际成交多少 | **想要买多少（进攻意图）** |

#### 示例2：上交所成交表处理

条件：一笔主动买入成交
- 成交价 12.50元 → price_bin = 5
- 成交量 1000股 → qty_bin = 2
- TickBSFlag = 'B' (主动买入)
- 买方母单金额 = 80万 >= 阈值 → 大买单
- 卖方母单金额 = 20万 < 阈值 → 小卖单

填充结果（共**4个**通道，**不含通道9/10**）：
```python
# 成交相关（通道0-2）
image[0, 5, 2] += 1   # 通道0: 全部成交
image[1, 5, 2] += 1   # 通道1: 成交-主动买入

# 大小单（通道3-6）
image[3, 5, 2] += 1   # 通道3: 大买单
image[6, 5, 2] += 1   # 通道6: 小卖单

# 注意：通道9/10不在成交表填充！已移到委托表处理
```

#### 示例3：上交所撤单处理（已预处理）

条件：委托表中一条撤单记录
- OrdType = 'Cancel'
- Side = 'B'
- Price = 12.50元（**已补全**）→ price_bin = 5
- Qty = 5,000股 → qty_bin = 4

```python
# 直接读取，Price已补全，无需关联！
image[13, 5, 4] += 1  # 通道13: 撤买
```

#### 示例4：深交所委托表处理（意图导向）

条件：深交所委托表中一条买单记录
- ApplSeqNum = 88888
- Side = '49' (买入)
- Price = 10.00元 → price_bin = 4
- OrderQty = 5,000股 → qty_bin = 5
- **ApplSeqNum 在 active_buy_seqs 中**（曾作为主动方成交）

```python
# 填充通道7和9（主动型）
image[7, 4, 5] += 1   # 通道7: 买单（完整意愿）
image[9, 4, 5] += 1   # 通道9: 主动买入（进攻意图）- 整单算主动！
```

**关键逻辑**：只要该委托在生命周期内做过"主动方"（哪怕只成交了一手），它就是一张"进攻单"，全额计入通道9。

---

## 8. 归一化处理

### 8.1 处理方案：Log1p + 通道内Max

Level2数据的订单频次是典型的长尾分布（幂律分布），需要进行对数变换：

$$X_{final} = \frac{\log(1 + X)}{\max(\log(1 + X))}$$

### 8.2 实现代码

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    图像归一化
    
    Args:
        image: [15, 8, 8] 原始计数图像
    
    Returns:
        [15, 8, 8] 归一化后的图像，值域 [0, 1]
    """
    # 1. Log变换：解决长尾问题
    log_image = np.log1p(image)
    
    # 2. 通道内归一化：每个通道独立计算Max
    normalized = np.zeros_like(log_image)
    
    for ch in range(15):
        max_val = log_image[ch].max()
        if max_val > 1e-8:  # 防止除零
            normalized[ch] = log_image[ch] / max_val
        # else: 保持全0
    
    return normalized.astype(np.float32)
```

### 8.3 归一化时机

归一化在**存储前**完成，确保存入LMDB的数据已经是处理后的张量，训练时无需重复计算。

---

## 9. 存储规格

### 9.1 存储格式：LMDB + LZ4

| 属性 | 规格 | 说明 |
|------|------|------|
| 文件组织 | 按日存储 | 每日一个LMDB文件，如 `20230101.lmdb` |
| Key格式 | `Code.Exchange` | 如 `"600519.SH"`, `"000001.SZ"` |
| Value格式 | LZ4压缩的numpy.tobytes() | float32类型，压缩后约200-500字节 |
| Shape | 固定 `(15, 8, 8)` | 代码中硬编码，无需存储元数据 |

### 9.2 写入代码

```python
import lmdb
import numpy as np
import lz4.frame

def write_daily_lmdb(trade_date: str, stock_images: Dict[str, np.ndarray], 
                     output_dir: str):
    """
    将一天所有股票的图像写入LMDB
    
    Args:
        trade_date: 交易日期，如 "20230101"
        stock_images: {stock_code: image_array}
        output_dir: 输出目录
    """
    lmdb_path = os.path.join(output_dir, f"{trade_date}.lmdb")
    
    # 预估大小：每只股票约500字节，4000只股票约2MB
    map_size = 100 * 1024 * 1024  # 100MB
    
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        for code, image in stock_images.items():
            # 确保类型正确
            image = image.astype(np.float32)
            
            # 转为bytes
            raw_bytes = image.tobytes()
            
            # LZ4压缩（稀疏矩阵压缩效果好）
            compressed = lz4.frame.compress(raw_bytes)
            
            # 写入
            txn.put(code.encode('ascii'), compressed)
    
    env.close()
```

### 9.3 读取代码

```python
class Level2ImageDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Level2 Images"""
    
    def __init__(self, lmdb_path: str, stock_codes: List[str]):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.codes = stock_codes
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        
        # 读取压缩数据
        compressed = self.txn.get(code.encode('ascii'))
        
        if compressed is None:
            # 该股票当日无数据，返回全零
            return torch.zeros(15, 8, 8, dtype=torch.float32)
        
        # 解压
        raw_bytes = lz4.frame.decompress(compressed)
        
        # 转回numpy (shape硬编码)
        image = np.frombuffer(raw_bytes, dtype=np.float32).reshape(15, 8, 8)
        
        # 转为PyTorch tensor
        # 注意：必须copy，因为frombuffer返回的是view
        return torch.from_numpy(image.copy())
    
    def __del__(self):
        self.env.close()
```

### 9.4 ViViT序列支持

ViViT模型需要过去20天的图像序列：

```python
class ViViTDataset(torch.utils.data.Dataset):
    """ViViT 20日序列数据集"""
    
    def __init__(self, lmdb_dir: str, trade_dates: List[str], 
                 stock_codes: List[str], seq_len: int = 20):
        self.lmdb_dir = lmdb_dir
        self.dates = trade_dates
        self.codes = stock_codes
        self.seq_len = seq_len
        
        # 预加载所有LMDB环境
        self.envs = {}
        for date in trade_dates:
            path = os.path.join(lmdb_dir, f"{date}.lmdb")
            if os.path.exists(path):
                self.envs[date] = lmdb.open(path, readonly=True, lock=False)
    
    def __getitem__(self, idx):
        # 解析索引
        date_idx = idx // len(self.codes)
        code_idx = idx % len(self.codes)
        
        target_date = self.dates[date_idx]
        code = self.codes[code_idx]
        
        # 获取前20天的日期
        start_idx = max(0, date_idx - self.seq_len + 1)
        date_range = self.dates[start_idx:date_idx + 1]
        
        # 动态拼接20天数据
        images = []
        for d in date_range:
            if d in self.envs:
                img = self._read_image(self.envs[d], code)
            else:
                img = np.zeros((15, 8, 8), dtype=np.float32)
            images.append(img)
        
        # 如果不足20天，前面补零
        while len(images) < self.seq_len:
            images.insert(0, np.zeros((15, 8, 8), dtype=np.float32))
        
        # Stack成 [20, 15, 8, 8]
        sequence = np.stack(images, axis=0)
        
        return torch.from_numpy(sequence)
    
    def _read_image(self, env, code: str) -> np.ndarray:
        with env.begin() as txn:
            compressed = txn.get(code.encode('ascii'))
            if compressed is None:
                return np.zeros((15, 8, 8), dtype=np.float32)
            raw_bytes = lz4.frame.decompress(compressed)
            return np.frombuffer(raw_bytes, dtype=np.float32).reshape(15, 8, 8).copy()
```

---

## 10. 性能优化建议

由于每日的Parquet文件较大，且需要批量生成历史数据，以下是关键的性能优化策略：

### 10.1 使用高性能数据处理库

**推荐方案**：Polars 或 Dask 替代 Pandas

```python
import polars as pl

# Polars 示例：读取和处理
def load_and_process_polars(file_path: str) -> pl.DataFrame:
    """使用Polars高效读取和处理"""
    df = pl.read_parquet(file_path)
    
    # 时间过滤（向量化）
    df = df.filter(
        ((pl.col('TickTime') >= 93000000) & (pl.col('TickTime') <= 113000000)) |
        ((pl.col('TickTime') >= 130000000) & (pl.col('TickTime') < 145700000))
    )
    
    return df


# Dask 示例：并行处理多日数据
import dask.dataframe as dd

def batch_process_dask(file_list: List[str]) -> dd.DataFrame:
    """使用Dask并行处理多个文件"""
    ddf = dd.read_parquet(file_list)
    
    # 延迟计算，最后统一执行
    ddf = ddf[ddf['Price'] > 0]
    
    return ddf.compute()  # 触发计算
```

### 10.2 向量化操作

**原则**：尽量避免 `iterrows()`，用向量化替代

```python
# ❌ 慢：逐行迭代
for _, row in df.iterrows():
    if row['TickBSFlag'] == 'B':
        buy_amounts[row['BuyOrderNO']] += row['TradeMoney']

# ✅ 快：向量化 groupby
buy_amounts = df[df['TickBSFlag'] == 'B'].groupby('BuyOrderNO')['TradeMoney'].sum()
sell_amounts = df[df['TickBSFlag'] == 'S'].groupby('SellOrderNO')['TradeMoney'].sum()
```

### 10.3 母单还原向量化

```python
def restore_parent_orders_vectorized(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """向量化母单还原"""
    # 一次性计算所有买方母单
    buy_parent = df.groupby('BuyOrderNO')['TradeMoney'].sum()
    
    # 一次性计算所有卖方母单
    sell_parent = df.groupby('SellOrderNO')['TradeMoney'].sum()
    
    return buy_parent, sell_parent
```

### 10.4 通道7/8预聚合向量化

```python
def build_sh_channel_7_8_vectorized(df: pd.DataFrame, 
                                     price_bins: np.ndarray,
                                     qty_bins: np.ndarray) -> np.ndarray:
    """向量化构建上交所通道7/8"""
    image = np.zeros((15, 8, 8), dtype=np.int32)
    
    # 按OrderNO聚合
    buy_agg = df[df['TickBSFlag'] == 'B'].groupby('BuyOrderNO').agg({
        'Price': 'first',  # 取第一笔价格
        'Qty': 'sum'       # 总量
    })
    
    # 向量化计算bin
    buy_agg['price_bin'] = np.digitize(buy_agg['Price'], price_bins)
    buy_agg['qty_bin'] = np.digitize(buy_agg['Qty'], qty_bins)
    
    # 向量化填充
    np.add.at(image[7], (buy_agg['price_bin'].values, buy_agg['qty_bin'].values), 1)
    
    # 卖方同理...
    return image
```

### 10.5 批量处理建议

```python
from concurrent.futures import ProcessPoolExecutor
from typing import List

def process_single_day(args: Tuple[str, str]) -> None:
    """处理单日数据"""
    date, stock_code = args
    # ... 处理逻辑 ...

def batch_process(date_list: List[str], stock_list: List[str], 
                  max_workers: int = 8) -> None:
    """多进程批量处理"""
    tasks = [(date, stock) for date in date_list for stock in stock_list]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_day, tasks))
```

### 10.6 性能对比参考

| 方案 | 单日单股处理时间 | 适用场景 |
|------|------------------|----------|
| Pandas + iterrows | ~30s | 不推荐 |
| Pandas + 向量化 | ~5s | 小规模 |
| Polars | ~1-2s | 推荐 |
| Dask (多进程) | ~0.5s/股 | 大规模批量 |

---

## 11. 代码模块设计

### 10.1 目录结构

```
l2_image_builder/
├── __init__.py
├── config.py                    # 配置管理
├── data_loader/
│   ├── __init__.py
│   ├── sh_loader.py             # 上交所数据加载
│   ├── sz_loader.py             # 深交所数据加载
│   └── snapshot_loader.py       # 快照数据加载（涨跌停价）
├── cleaner/
│   ├── __init__.py
│   ├── time_filter.py           # 时间过滤
│   ├── anomaly_filter.py        # 异常值过滤
│   └── sz_cancel_enricher.py    # 深交所撤单价格关联
├── calculator/
│   ├── __init__.py
│   ├── quantile.py              # 分位数计算
│   ├── big_order.py             # 大单还原与阈值
│   └── channel_mapper.py        # 通道映射逻辑
├── builder/
│   ├── __init__.py
│   ├── image_builder.py         # 图像构建核心
│   ├── sh_builder.py            # 上交所专用逻辑
│   ├── sz_builder.py            # 深交所专用逻辑
│   └── normalizer.py            # 归一化处理
├── storage/
│   ├── __init__.py
│   ├── lmdb_writer.py           # LMDB写入
│   └── lmdb_reader.py           # LMDB读取
├── diagnostics/
│   ├── __init__.py
│   └── reporter.py              # 诊断报告生成
├── dataset/
│   ├── __init__.py
│   ├── vit_dataset.py           # ViT单日数据集
│   └── vivit_dataset.py         # ViViT序列数据集
└── main.py                      # 主入口
```

### 10.2 配置文件

```python
# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # 数据路径
    raw_data_dir: str = "/raw_data"
    output_dir: str = "/processed_data/l2_images"
    threshold_dir: str = "/processed_data/thresholds"
    
    # 图像参数
    num_channels: int = 15
    num_price_bins: int = 8
    num_qty_bins: int = 8
    
    # 分位数百分位
    percentiles: List[float] = (12.5, 25, 37.5, 50, 62.5, 75, 87.5)
    
    # 大单阈值（仅当日计算，不使用滚动窗口）
    threshold_std_multiplier: float = 1.0  # 均值+N倍标准差
    # 注：弃用 threshold_lookback 参数，严格复现研报逻辑
    
    # 时间过滤
    am_start: int = 93000000   # 09:30:00.000
    am_end: int = 113000000    # 11:30:00.000
    pm_start: int = 130000000  # 13:00:00.000
    pm_end: int = 145700000    # 14:57:00.000
    
    # 存储参数
    lmdb_map_size: int = 100 * 1024 * 1024  # 100MB
    use_lz4: bool = True
```

---

## 12. 诊断与监控指标

### 12.1 通道级指标

| 指标 | 计算方法 | 健康范围 | 异常处理 |
|------|----------|----------|----------|
| 非零像素数 | `np.count_nonzero(channel)` | 成交>20, 委托>30 | 检查数据质量 |
| 填充率 | 非零像素数 / 64 | 成交>30%, 委托>50% | 检查分位数计算 |
| 总订单数 | `channel.sum()` | >100 | 检查数据加载 |
| 最大像素值 | `channel.max()` | <总数的50% | 检查是否过度集中 |

### 12.2 股票级指标

| 指标 | 计算方法 | 健康范围 |
|------|----------|----------|
| 成交/委托比 | 成交通道总数 / 委托通道总数 | 1:2 ~ 1:5 |
| 大单占比 | (通道3+4) / (通道3+4+5+6) | 5% ~ 30% |
| 撤单率 | (通道13+14) / (通道7+8) | <50% |

### 12.3 诊断代码

```python
def generate_diagnostics(image: np.ndarray, stock_code: str, 
                         trade_date: str) -> Dict:
    """生成单只股票的诊断信息"""
    
    diagnostics = {
        'stock_code': stock_code,
        'trade_date': trade_date,
        'channels': []
    }
    
    channel_names = [
        '全部成交', '主动买入', '主动卖出', '大买单', '大卖单', 
        '小买单', '小卖单', '买单', '卖单', '委托主动买', '委托主动卖',
        '非主动买', '非主动卖', '撤买', '撤卖'
    ]
    
    for ch in range(15):
        channel_data = image[ch]
        non_zero = np.count_nonzero(channel_data)
        total = channel_data.sum()
        max_val = channel_data.max()
        
        diagnostics['channels'].append({
            'index': ch,
            'name': channel_names[ch],
            'non_zero_pixels': int(non_zero),
            'fill_rate': round(non_zero / 64, 4),
            'total_count': float(total),
            'max_pixel': float(max_val),
            'concentration': round(max_val / total, 4) if total > 0 else 0
        })
    
    # 汇总指标
    trade_total = sum(d['total_count'] for d in diagnostics['channels'][:7])
    order_total = sum(d['total_count'] for d in diagnostics['channels'][7:])
    
    diagnostics['summary'] = {
        'trade_order_ratio': round(trade_total / order_total, 4) if order_total > 0 else 0,
        'big_order_ratio': round(
            (diagnostics['channels'][3]['total_count'] + diagnostics['channels'][4]['total_count']) /
            max(1, sum(d['total_count'] for d in diagnostics['channels'][3:7])),
            4
        ),
        'cancel_rate': round(
            (diagnostics['channels'][13]['total_count'] + diagnostics['channels'][14]['total_count']) /
            max(1, diagnostics['channels'][7]['total_count'] + diagnostics['channels'][8]['total_count']),
            4
        )
    }
    
    return diagnostics
```

---

## 12. 附录

### 12.1 上交所枚举值

**Type (类型标识)**

| 值 | 英文 | 中文 | 说明 |
|---|------|------|------|
| A | Add Order | 新增委托 | 新委托进入订单簿 |
| D | Delete Order | 删除委托 | 撤单操作 |
| T | Trade | 成交 | 成交记录 |
| S | Status | 状态 | 产品状态变更，**需剔除** |

**TickBSFlag (买卖标识)**

| 值 | Type=T时含义 | Type=A时含义 |
|---|-------------|-------------|
| B | 主动买入（外盘） | 买入委托 |
| S | 主动卖出（内盘） | 卖出委托 |
| N | 集合竞价成交 | 无意义 |

### 12.2 深交所枚举值

**Side (买卖方向) - 委托表**

| 原始值 | ASCII码 | 存储值 | 含义 |
|--------|---------|--------|------|
| '1' | 49 | "49" | 买入 |
| '2' | 50 | "50" | 卖出 |
| 'G' | 71 | "71" | 借入 |
| 'F' | 70 | "70" | 出借 |

**ExecType (执行类型) - 成交表**

| 原始值 | ASCII码 | 存储值 | 含义 |
|--------|---------|--------|------|
| '4' | 52 | "52" | 撤销 |
| 'F' | 70 | "70" | 成交 |

### 12.3 时间格式转换

```python
def parse_tick_time(tick_time: int) -> str:
    """
    将 HHMMSSmmm 格式的整数转换为可读时间字符串
    
    例: 93000540 -> "09:30:00.540"
    """
    ms = tick_time % 1000
    tick_time //= 1000
    ss = tick_time % 100
    tick_time //= 100
    mm = tick_time % 100
    hh = tick_time // 100
    
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
```

---

## 文档版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| 1.0 | 2026-01-16 | 初始版本 |

---

*文档结束*
