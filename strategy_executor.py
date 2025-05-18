      #!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import os

class TradeExecutor:
    """交易执行器类"""
    
    def __init__(self, strategy_name: str, initial_cash: float = 100000.0, 
                 buy_ratio: float = 0.2, sell_ratio: float = 0.3):
        """
        初始化交易执行器
        参数:
            strategy_name (str): 策略名称
            initial_cash (float): 初始资金，默认10万
            buy_ratio (float): 买入比例，默认0.2（20%）
            sell_ratio (float): 卖出比例，默认0.3（30%）
        """
        self.strategy_name = strategy_name
        self.initial_cash = initial_cash
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.commission_rate = 0.000095  # 佣金率
        self.stamp_duty_rate = 0.001   # 印花税率
        
    def execute_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行交易并生成交易记录
        参数:
            df (pd.DataFrame): 包含交易信号的DataFrame
        返回:
            pd.DataFrame: 包含完整交易记录的DataFrame
        """
        # 初始化交易记录列
        df['业务名称'] = ''
        df['成交数量'] = 0
        df['持股数量'] = 0
        df['成交均价'] = 0.0
        df['印花税'] = 0.0
        df['佣金'] = 0.0
        df['余额'] = self.initial_cash
        df['总金额'] = self.initial_cash
        
        current_shares = 0  # 当前持股数量
        current_cash = self.initial_cash  # 当前现金余额
        
        for i in range(len(df)):
            if i == 0 and df.iloc[i]['信号'] == 1:
                # 首次买入
                price = df.iloc[i]['收盘']
                if price == 0 or np.isnan(price):
                    continue
                amount = current_cash * self.buy_ratio
                shares = int(amount / price / 100) * 100  # 取整百
                if shares <= 0:
                    continue
                commission = shares * price * self.commission_rate
                stamp_duty = shares * price * self.stamp_duty_rate
                total_cost = shares * price + commission + stamp_duty
                
                if total_cost <= current_cash:
                    df.iloc[i, df.columns.get_loc('业务名称')] = '买入'
                    df.iloc[i, df.columns.get_loc('成交数量')] = shares
                    df.iloc[i, df.columns.get_loc('成交均价')] = price
                    df.iloc[i, df.columns.get_loc('印花税')] = stamp_duty
                    df.iloc[i, df.columns.get_loc('佣金')] = commission
                    
                    current_shares += shares
                    current_cash -= total_cost
                    
            elif df.iloc[i]['信号'] == 1:
                # 非首次买入
                price = df.iloc[i]['收盘']
                if price == 0 or np.isnan(price):
                    continue
                amount = current_cash * self.buy_ratio
                shares = int(amount / price / 100) * 100
                if shares <= 0:
                    continue
                commission = shares * price * self.commission_rate
                stamp_duty = shares * price * self.stamp_duty_rate
                total_cost = shares * price + commission + stamp_duty
                
                if total_cost <= current_cash:
                    df.iloc[i, df.columns.get_loc('业务名称')] = '买入'
                    df.iloc[i, df.columns.get_loc('成交数量')] = shares
                    df.iloc[i, df.columns.get_loc('成交均价')] = price
                    df.iloc[i, df.columns.get_loc('印花税')] = stamp_duty
                    df.iloc[i, df.columns.get_loc('佣金')] = commission
                    
                    current_shares += shares
                    current_cash -= total_cost
                    
            elif df.iloc[i]['信号'] == -1 and current_shares > 0:
                # 卖出
                price = df.iloc[i]['收盘']
                if price == 0 or np.isnan(price):
                    continue
                shares = int(current_shares * self.sell_ratio / 100) * 100  # 卖出指定比例的持股，取整百
                if shares <= 0:
                    continue
                commission = shares * price * self.commission_rate
                stamp_duty = shares * price * self.stamp_duty_rate
                total_revenue = shares * price - commission - stamp_duty
                
                df.iloc[i, df.columns.get_loc('业务名称')] = '卖出'
                df.iloc[i, df.columns.get_loc('成交数量')] = -shares
                df.iloc[i, df.columns.get_loc('成交均价')] = price
                df.iloc[i, df.columns.get_loc('印花税')] = stamp_duty
                df.iloc[i, df.columns.get_loc('佣金')] = commission
                
                current_shares -= shares
                current_cash += total_revenue
            
            # 更新当前状态
            df.iloc[i, df.columns.get_loc('持股数量')] = current_shares
            df.iloc[i, df.columns.get_loc('余额')] = current_cash
            df.iloc[i, df.columns.get_loc('总金额')] = current_cash + current_shares * df.iloc[i]['收盘']
        
        return df
    
    def save_trade_records(self, df: pd.DataFrame, output_dir: str = 'trade_records'):
        """
        保存交易记录到CSV文件
        参数:
            df (pd.DataFrame): 交易记录DataFrame
            output_dir (str): 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/{self.strategy_name}_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"交易记录已保存到: {filename}") 