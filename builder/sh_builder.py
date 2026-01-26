"""
上交所图像构建器（v3 重构版）

重构目标：从"结果导向"升级到"意图导向"
- 通道9/10 **必须**来自委托表（Intent），不再从成交表填充
- 通道7/8/11/12 **必须**遵循v3互斥分解规则（Ch7=Ch9+Ch11）

关键约束：
- 排序键必须使用 ['TickTime', 'BizIndex']
- 委托表必须包含 IsAggressive 字段
- 数学约束：Ch7 = Ch9 + Ch11, Ch8 = Ch10 + Ch12

15通道定义（v3）：
| 索引 | 名称 | 数据源 | 筛选条件 |
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
| 9 | 主动买委托 | 委托表 | Side='B' & IsAggressive=True |
| 10 | 主动卖委托 | 委托表 | Side='S' & IsAggressive=True |
| 11 | 非主动买 | 委托表 | Side='B' & IsAggressive=False |
| 12 | 非主动卖 | 委托表 | Side='S' & IsAggressive=False |
| 13 | 撤买 | 委托表 | OrdType='Cancel' & Side='B' |
| 14 | 撤卖 | 委托表 | OrdType='Cancel' & Side='S' |
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import pandas as pd

from ..config import Channels
from ..data_loader.polars_utils import POLARS_AVAILABLE, is_polars_df

if POLARS_AVAILABLE:
    import polars as pl


class SHImageBuilder:
    """
    上交所图像构建器（简化版）
    
    Features:
        - 支持 Polars 和 Pandas DataFrame
        - 向量化实现，高性能
        - 简化的委托处理逻辑（数据已预处理）
    
    Args:
        price_bins: 价格分位数边界 (7个值)
        qty_bins: 数量分位数边界 (7个值)
        buy_parent: 买方母单金额映射 {OrderNO -> amount}
        sell_parent: 卖方母单金额映射 {OrderNO -> amount}
        threshold: 大单阈值
    """
    
    def __init__(
        self,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent: Dict[int, float],
        sell_parent: Dict[int, float],
        threshold: float,
    ):
        self.price_bins = price_bins
        self.qty_bins = qty_bins
        self.buy_parent = buy_parent
        self.sell_parent = sell_parent
        self.threshold = threshold
        
        # 初始化图像 [15, 8, 8]
        self.image = np.zeros((15, 8, 8), dtype=np.float32)
    
    def reset(self) -> None:
        """重置图像"""
        self.image.fill(0)
    
    def build(
        self,
        df_trade: Union['pl.DataFrame', pd.DataFrame],
        df_order: Union['pl.DataFrame', pd.DataFrame],
    ) -> np.ndarray:
        """
        构建图像
        
        Args:
            df_trade: 成交表
            df_order: 委托表（已预处理，Qty为完整母单量）
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 处理成交表 → 填充通道0-6, 9-10
        self._process_trades(df_trade)
        
        # 2. 处理委托表 → 填充通道7-8, 11-14
        self._process_orders(df_order)
        
        return self.image.copy()
    
    def build_vectorized(
        self,
        df_trade: Union['pl.DataFrame', pd.DataFrame],
        df_order: Union['pl.DataFrame', pd.DataFrame],
    ) -> np.ndarray:
        """
        向量化构建图像（推荐，更高性能）
        
        Args:
            df_trade: 成交表
            df_order: 委托表（已预处理，Qty为完整母单量）
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 向量化处理成交表
        self._process_trades_vectorized(df_trade)
        
        # 2. 向量化处理委托表
        self._process_orders_vectorized(df_order)
        
        return self.image.copy()
    
    # ==================== 逐行处理方法 ====================
    
    def _process_trades(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理成交记录（逐行版本）
        
        填充通道: 0, 1-2, 3-6, 9-10
        """
        if is_polars_df(df):
            self._process_trades_polars(df)
        else:
            self._process_trades_pandas(df)
    
    def _process_trades_polars(self, df: 'pl.DataFrame') -> None:
        """Polars 成交处理"""
        # 转换为 numpy 以提高性能
        prices = df['Price'].to_numpy()
        qtys = df['Qty'].to_numpy()
        bs_flags = df['TickBSFlag'].to_numpy()
        buy_orders = df['BuyOrderNO'].to_numpy()
        sell_orders = df['SellOrderNO'].to_numpy()
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            self._fill_trade(
                price_bins[i], qty_bins[i],
                bs_flags[i], buy_orders[i], sell_orders[i]
            )
    
    def _process_trades_pandas(self, df: pd.DataFrame) -> None:
        """Pandas 成交处理"""
        prices = df['Price'].values
        qtys = df['Qty'].values
        bs_flags = df['TickBSFlag'].values
        buy_orders = df['BuyOrderNO'].values
        sell_orders = df['SellOrderNO'].values
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            self._fill_trade(
                price_bins[i], qty_bins[i],
                bs_flags[i], buy_orders[i], sell_orders[i]
            )
    
    def _fill_trade(
        self,
        pb: int,
        qb: int,
        bs_flag: str,
        buy_order_no: int,
        sell_order_no: int,
    ) -> None:
        """
        填充单笔成交到图像
        
        v3: 成交表只填充通道0-6，不再填充通道9/10
        通道9/10已移至委托表处理（_process_orders）
        """
        # 通道0: 全部成交
        self.image[Channels.ALL_TRADE, pb, qb] += 1
        
        # 通道1-2: 主动方向（v3: 只填充成交通道，不填充9/10）
        if bs_flag == 'B':
            self.image[Channels.ACTIVE_BUY_TRADE, pb, qb] += 1
        elif bs_flag == 'S':
            self.image[Channels.ACTIVE_SELL_TRADE, pb, qb] += 1
        # ⚠️ v3: 此处不能有任何 image[9] 或 image[10] 的代码！
        
        # 通道3-6: 大小单（与主动方向无关）
        buy_amount = self.buy_parent.get(buy_order_no, 0)
        sell_amount = self.sell_parent.get(sell_order_no, 0)
        
        if buy_amount >= self.threshold:
            self.image[Channels.BIG_BUY, pb, qb] += 1
        elif buy_amount > 0:
            self.image[Channels.SMALL_BUY, pb, qb] += 1
        
        if sell_amount >= self.threshold:
            self.image[Channels.BIG_SELL, pb, qb] += 1
        elif sell_amount > 0:
            self.image[Channels.SMALL_SELL, pb, qb] += 1
    
    def _process_orders(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理委托记录（逐行版本）
        
        v3 变更：
        - 通道9/10现在从委托表填充（基于IsAggressive字段）
        - 通道11/12与7/8互斥分解（不再重叠）
        - 数学约束：Ch7 = Ch9 + Ch11, Ch8 = Ch10 + Ch12
        
        填充通道: 7-14
        
        Raises:
            ValueError: 如果缺少必需的 IsAggressive 字段
        """
        # v3: 验证必需字段
        self._validate_order_fields(df)
        
        if is_polars_df(df):
            self._process_orders_polars(df)
        else:
            self._process_orders_pandas(df)
    
    def _validate_order_fields(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        验证委托表必需字段 (v3)
        
        Raises:
            ValueError: 缺少 IsAggressive 字段时抛出明确错误
        """
        if is_polars_df(df):
            columns = df.columns
        else:
            columns = df.columns.tolist()
        
        if 'IsAggressive' not in columns:
            raise ValueError(
                "v3约束错误: 委托表缺少 'IsAggressive' 字段。"
                "请确保数据预处理已为委托表添加 IsAggressive 字段"
                "（True=主动/Taker, False=被动/Maker, None=撤单）"
            )
    
    def _process_orders_polars(self, df: 'pl.DataFrame') -> None:
        """Polars 委托处理 (v3)"""
        prices = df['Price'].to_numpy()
        qtys = df['Qty'].to_numpy()
        ord_types = df['OrdType'].to_numpy()
        sides = df['Side'].to_numpy()
        # v3: 提取 IsAggressive 字段
        is_aggressive = df['IsAggressive'].to_numpy()
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            if prices[i] <= 0:
                continue
            self._fill_order(
                price_bins[i], qty_bins[i],
                ord_types[i], sides[i],
                is_aggressive[i]  # v3: 传递 IsAggressive
            )
    
    def _process_orders_pandas(self, df: pd.DataFrame) -> None:
        """Pandas 委托处理 (v3)"""
        prices = df['Price'].values
        qtys = df['Qty'].values
        ord_types = df['OrdType'].values
        sides = df['Side'].values
        # v3: 提取 IsAggressive 字段
        is_aggressive = df['IsAggressive'].values
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            if prices[i] <= 0:
                continue
            self._fill_order(
                price_bins[i], qty_bins[i],
                ord_types[i], sides[i],
                is_aggressive[i]  # v3: 传递 IsAggressive
            )
    
    def _fill_order(
        self,
        pb: int,
        qb: int,
        ord_type: str,
        side: str,
        is_aggressive: bool = None,  # v3: 新增参数
    ) -> None:
        """
        填充单笔委托到图像 (v3)
        
        v3 核心变更：
        - 通道9/10根据IsAggressive从委托表填充
        - 通道11/12与9/10互斥（Ch7=Ch9+Ch11, Ch8=Ch10+Ch12）
        
        Args:
            pb: 价格bin索引
            qb: 数量bin索引
            ord_type: 委托类型 ('New' | 'Cancel')
            side: 买卖方向 ('B' | 'S')
            is_aggressive: 是否为主动单（True=Taker, False=Maker, None=撤单）
        """
        if ord_type == 'New':
            # 新增委托 → 通道7/8
            if side == 'B':
                self.image[Channels.BUY_ORDER, pb, qb] += 1
                # ✅ v3: 根据 IsAggressive 互斥分流
                if is_aggressive is True:
                    self.image[Channels.ACTIVE_BUY_ORDER, pb, qb] += 1   # 进攻型买单 → Ch9
                elif is_aggressive is False:
                    self.image[Channels.PASSIVE_BUY, pb, qb] += 1       # 防守型买单 → Ch11
                # is_aggressive=None 时不填充9/11（异常情况）
            elif side == 'S':
                self.image[Channels.SELL_ORDER, pb, qb] += 1
                # ✅ v3: 根据 IsAggressive 互斥分流
                if is_aggressive is True:
                    self.image[Channels.ACTIVE_SELL_ORDER, pb, qb] += 1  # 进攻型卖单 → Ch10
                elif is_aggressive is False:
                    self.image[Channels.PASSIVE_SELL, pb, qb] += 1       # 防守型卖单 → Ch12
                # is_aggressive=None 时不填充10/12（异常情况）
        
        elif ord_type == 'Cancel':
            # 撤单 → 通道13/14（IsAggressive应为None）
            if side == 'B':
                self.image[Channels.CANCEL_BUY, pb, qb] += 1
            elif side == 'S':
                self.image[Channels.CANCEL_SELL, pb, qb] += 1
    
    # ==================== 向量化处理方法 ====================
    
    def _process_trades_vectorized(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        向量化处理成交记录 (v3)
        
        v3: 成交表只填充通道0-6，不再填充通道9/10
        使用 np.add.at 进行高效的直方图累加
        """
        if is_polars_df(df):
            prices = df['Price'].to_numpy()
            qtys = df['Qty'].to_numpy()
            bs_flags = df['TickBSFlag'].to_numpy()
            buy_orders = df['BuyOrderNO'].to_numpy()
            sell_orders = df['SellOrderNO'].to_numpy()
        else:
            prices = df['Price'].values
            qtys = df['Qty'].values
            bs_flags = df['TickBSFlag'].values
            buy_orders = df['BuyOrderNO'].values
            sell_orders = df['SellOrderNO'].values
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 通道0: 全部成交 - 向量化
        np.add.at(self.image[Channels.ALL_TRADE], (price_bins, qty_bins), 1)
        
        # 通道1-2: 主动方向 - 向量化（v3: 不填充9/10）
        buy_mask = bs_flags == 'B'
        sell_mask = bs_flags == 'S'
        
        if buy_mask.any():
            np.add.at(
                self.image[Channels.ACTIVE_BUY_TRADE],
                (price_bins[buy_mask], qty_bins[buy_mask]), 1
            )
            # ⚠️ v3: 删除了 ACTIVE_BUY_ORDER (Ch9) 的填充
        
        if sell_mask.any():
            np.add.at(
                self.image[Channels.ACTIVE_SELL_TRADE],
                (price_bins[sell_mask], qty_bins[sell_mask]), 1
            )
            # ⚠️ v3: 删除了 ACTIVE_SELL_ORDER (Ch10) 的填充
        
        # 通道3-6: 大小单 - 向量化查找母单金额
        buy_amounts = np.array([
            self.buy_parent.get(int(o), 0) for o in buy_orders
        ], dtype=np.float64)
        sell_amounts = np.array([
            self.sell_parent.get(int(o), 0) for o in sell_orders
        ], dtype=np.float64)
        
        # 大买单
        big_buy_mask = buy_amounts >= self.threshold
        if big_buy_mask.any():
            np.add.at(
                self.image[Channels.BIG_BUY],
                (price_bins[big_buy_mask], qty_bins[big_buy_mask]), 1
            )
        
        # 小买单
        small_buy_mask = (buy_amounts > 0) & (buy_amounts < self.threshold)
        if small_buy_mask.any():
            np.add.at(
                self.image[Channels.SMALL_BUY],
                (price_bins[small_buy_mask], qty_bins[small_buy_mask]), 1
            )
        
        # 大卖单
        big_sell_mask = sell_amounts >= self.threshold
        if big_sell_mask.any():
            np.add.at(
                self.image[Channels.BIG_SELL],
                (price_bins[big_sell_mask], qty_bins[big_sell_mask]), 1
            )
        
        # 小卖单
        small_sell_mask = (sell_amounts > 0) & (sell_amounts < self.threshold)
        if small_sell_mask.any():
            np.add.at(
                self.image[Channels.SMALL_SELL],
                (price_bins[small_sell_mask], qty_bins[small_sell_mask]), 1
            )
    
    def _process_orders_vectorized(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        向量化处理委托记录 (v3)
        
        v3 变更：
        - 通道9/10根据IsAggressive从委托表填充
        - 通道11/12与9/10互斥（Ch7=Ch9+Ch11, Ch8=Ch10+Ch12）
        
        Raises:
            ValueError: 缺少 IsAggressive 字段时抛出错误
        """
        # v3: 验证必需字段
        self._validate_order_fields(df)
        
        if is_polars_df(df):
            prices = df['Price'].to_numpy()
            qtys = df['Qty'].to_numpy()
            ord_types = df['OrdType'].to_numpy()
            sides = df['Side'].to_numpy()
            is_aggressive = df['IsAggressive'].to_numpy()  # v3: 提取 IsAggressive
        else:
            prices = df['Price'].values
            qtys = df['Qty'].values
            ord_types = df['OrdType'].values
            sides = df['Side'].values
            is_aggressive = df['IsAggressive'].values  # v3: 提取 IsAggressive
        
        if len(prices) == 0:
            return
        
        # 过滤有效价格
        valid_mask = prices > 0
        prices = prices[valid_mask]
        qtys = qtys[valid_mask]
        ord_types = ord_types[valid_mask]
        sides = sides[valid_mask]
        is_aggressive = is_aggressive[valid_mask]  # v3: 同步过滤
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # v3: 新增委托 → 通道7/8 + 根据IsAggressive分流到9/10或11/12
        new_mask = ord_types == 'New'
        new_buy_mask = new_mask & (sides == 'B')
        new_sell_mask = new_mask & (sides == 'S')
        
        # 买单处理
        if new_buy_mask.any():
            # 通道7: 全部买单
            np.add.at(
                self.image[Channels.BUY_ORDER],
                (price_bins[new_buy_mask], qty_bins[new_buy_mask]), 1
            )
            # v3: 通道9/11 互斥分流
            aggressive_buy_mask = new_buy_mask & (is_aggressive == True)
            passive_buy_mask = new_buy_mask & (is_aggressive == False)
            
            if aggressive_buy_mask.any():
                np.add.at(
                    self.image[Channels.ACTIVE_BUY_ORDER],  # Ch9: 主动买委托
                    (price_bins[aggressive_buy_mask], qty_bins[aggressive_buy_mask]), 1
                )
            if passive_buy_mask.any():
                np.add.at(
                    self.image[Channels.PASSIVE_BUY],  # Ch11: 非主动买
                    (price_bins[passive_buy_mask], qty_bins[passive_buy_mask]), 1
                )
        
        # 卖单处理
        if new_sell_mask.any():
            # 通道8: 全部卖单
            np.add.at(
                self.image[Channels.SELL_ORDER],
                (price_bins[new_sell_mask], qty_bins[new_sell_mask]), 1
            )
            # v3: 通道10/12 互斥分流
            aggressive_sell_mask = new_sell_mask & (is_aggressive == True)
            passive_sell_mask = new_sell_mask & (is_aggressive == False)
            
            if aggressive_sell_mask.any():
                np.add.at(
                    self.image[Channels.ACTIVE_SELL_ORDER],  # Ch10: 主动卖委托
                    (price_bins[aggressive_sell_mask], qty_bins[aggressive_sell_mask]), 1
                )
            if passive_sell_mask.any():
                np.add.at(
                    self.image[Channels.PASSIVE_SELL],  # Ch12: 非主动卖
                    (price_bins[passive_sell_mask], qty_bins[passive_sell_mask]), 1
                )
        
        # 撤单 → 通道13/14
        cancel_mask = ord_types == 'Cancel'
        cancel_buy_mask = cancel_mask & (sides == 'B')
        cancel_sell_mask = cancel_mask & (sides == 'S')
        
        if cancel_buy_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_BUY],
                (price_bins[cancel_buy_mask], qty_bins[cancel_buy_mask]), 1
            )
        
        if cancel_sell_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_SELL],
                (price_bins[cancel_sell_mask], qty_bins[cancel_sell_mask]), 1
            )
    
    # ==================== 辅助方法 ====================
    
    def get_channel_stats(self) -> Dict[str, Dict]:
        """
        获取通道统计信息
        
        Returns:
            各通道的统计数据
        """
        channel_names = [
            'all_trade', 'active_buy_trade', 'active_sell_trade',
            'big_buy', 'big_sell', 'small_buy', 'small_sell',
            'buy_order', 'sell_order', 'active_buy_order', 'active_sell_order',
            'passive_buy', 'passive_sell', 'cancel_buy', 'cancel_sell'
        ]
        
        stats = {}
        for i, name in enumerate(channel_names):
            channel_data = self.image[i]
            stats[name] = {
                'sum': float(channel_data.sum()),
                'max': float(channel_data.max()),
                'nonzero': int(np.count_nonzero(channel_data)),
                'fill_rate': float(np.count_nonzero(channel_data) / 64),
            }
        return stats
    
    def validate_consistency(self) -> Dict[str, bool]:
        """
        验证图像一致性
        
        检查项:
        1. 通道1 + 通道2 ≤ 通道0 (主动买+主动卖 ≤ 全部成交)
        2. 通道3 + 通道5 大约等于买方成交数
        3. 通道4 + 通道6 大约等于卖方成交数
        4. 通道7 = 通道11 (上交所预处理后)
        5. 通道8 = 通道12 (上交所预处理后)
        
        Returns:
            各检查项的结果
        """
        results = {}
        
        # 检查1: 主动方向不超过全部成交
        active_sum = self.image[1].sum() + self.image[2].sum()
        all_trade = self.image[0].sum()
        results['active_le_all'] = active_sum <= all_trade + 1e-6
        
        # v3 检查: 通道约束 Ch7 = Ch9 + Ch11, Ch8 = Ch10 + Ch12
        constraint_result = self.validate_constraints()
        results['v3_buy_constraint'] = constraint_result['buy_valid']
        results['v3_sell_constraint'] = constraint_result['sell_valid']
        results['v3_constraints_valid'] = constraint_result['valid']
        
        return results
    
    def validate_constraints(self) -> Dict:
        """
        验证 v3 通道约束
        
        数学约束：
        - Ch7 (全部买单) = Ch9 (主动买委托) + Ch11 (非主动买)
        - Ch8 (全部卖单) = Ch10 (主动卖委托) + Ch12 (非主动卖)
        
        Returns:
            包含约束验证结果的字典:
            - buy_valid: 买单约束是否满足
            - sell_valid: 卖单约束是否满足
            - valid: 全部约束是否满足
            - buy_diff: 买单约束偏差
            - sell_diff: 卖单约束偏差
        """
        ch7_sum = self.image[Channels.BUY_ORDER].sum()
        ch8_sum = self.image[Channels.SELL_ORDER].sum()
        ch9_sum = self.image[Channels.ACTIVE_BUY_ORDER].sum()
        ch10_sum = self.image[Channels.ACTIVE_SELL_ORDER].sum()
        ch11_sum = self.image[Channels.PASSIVE_BUY].sum()
        ch12_sum = self.image[Channels.PASSIVE_SELL].sum()
        
        buy_diff = abs(ch7_sum - (ch9_sum + ch11_sum))
        sell_diff = abs(ch8_sum - (ch10_sum + ch12_sum))
        
        # 允许浮点误差
        tol = 1e-6
        buy_valid = buy_diff < tol
        sell_valid = sell_diff < tol
        
        return {
            'valid': buy_valid and sell_valid,
            'buy_valid': buy_valid,
            'sell_valid': sell_valid,
            'buy_diff': float(buy_diff),
            'sell_diff': float(sell_diff),
            'decomposition': {
                'Ch7': float(ch7_sum),
                'Ch9': float(ch9_sum),
                'Ch11': float(ch11_sum),
                'Ch8': float(ch8_sum),
                'Ch10': float(ch10_sum),
                'Ch12': float(ch12_sum),
            }
        }


def build_sh_image(
    df_trade: Union['pl.DataFrame', pd.DataFrame],
    df_order: Union['pl.DataFrame', pd.DataFrame],
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
    buy_parent: Dict[int, float],
    sell_parent: Dict[int, float],
    threshold: float,
    vectorized: bool = True,
) -> np.ndarray:
    """
    便捷函数：构建上交所图像
    
    Args:
        df_trade: 成交表
        df_order: 委托表（已预处理）
        price_bins: 价格分位数边界
        qty_bins: 数量分位数边界
        buy_parent: 买方母单金额映射
        sell_parent: 卖方母单金额映射
        threshold: 大单阈值
        vectorized: 是否使用向量化版本
    
    Returns:
        [15, 8, 8] 图像
    """
    builder = SHImageBuilder(
        price_bins=price_bins,
        qty_bins=qty_bins,
        buy_parent=buy_parent,
        sell_parent=sell_parent,
        threshold=threshold,
    )
    
    if vectorized:
        return builder.build_vectorized(df_trade, df_order)
    else:
        return builder.build(df_trade, df_order)


def build_sh_image_with_stats(
    df_trade: Union['pl.DataFrame', pd.DataFrame],
    df_order: Union['pl.DataFrame', pd.DataFrame],
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
    buy_parent: Dict[int, float],
    sell_parent: Dict[int, float],
    threshold: float,
    vectorized: bool = True,
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    构建上交所图像并返回统计信息
    
    Returns:
        (image, channel_stats, consistency_check)
    """
    builder = SHImageBuilder(
        price_bins=price_bins,
        qty_bins=qty_bins,
        buy_parent=buy_parent,
        sell_parent=sell_parent,
        threshold=threshold,
    )
    
    if vectorized:
        image = builder.build_vectorized(df_trade, df_order)
    else:
        image = builder.build(df_trade, df_order)
    
    return image, builder.get_channel_stats(), builder.validate_consistency()
