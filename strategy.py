#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
import numpy as np
from typing import Optional
from getdata_daily_from_localdata import StockDataReader
import torch


class BaseTradingStrategy:
    """基础交易策略类"""

    def __init__(self, symbol: str, gpu_accelerator=None):
        """
        初始化基础交易策略
        参数:
            symbol (str): 股票代码
            gpu_accelerator: GPU加速器实例
        """
        self.symbol = symbol
        self.gpu_acc = gpu_accelerator
        self.trades = []

    def calculate_ma(self, data, window):
        """计算移动平均，支持GPU加速"""
        if self.gpu_acc is not None:
            try:
                data_tensor = torch.tensor(data, dtype=torch.float32)
                if self.gpu_acc.device is not None:
                    data_tensor = data_tensor.to(self.gpu_acc.device)
                result = torch.zeros_like(data_tensor)
                for i in range(len(data_tensor) - window + 1):
                    result[i + window - 1] = data_tensor[i:i+window].mean()
                return result.cpu().numpy()
            except Exception as e:
                print(f"GPU计算失败，回退到CPU: {str(e)}")
                pass
        return pd.Series(data).rolling(window=window).mean().values

    def calculate_std(self, data, window):
        """计算标准差，支持GPU加速"""
        if self.gpu_acc is not None:
            try:
                data_tensor = torch.tensor(data, dtype=torch.float32)
                if self.gpu_acc.device is not None:
                    data_tensor = data_tensor.to(self.gpu_acc.device)
                result = torch.zeros_like(data_tensor)
                for i in range(len(data_tensor) - window + 1):
                    result[i + window - 1] = data_tensor[i:i+window].std()
                return result.cpu().numpy()
            except Exception as e:
                print(f"GPU计算失败，回退到CPU: {str(e)}")
                pass
        return pd.Series(data).rolling(window=window).std().values

    def calculate_ema(self, data, span):
        """计算指数移动平均，支持GPU加速"""
        if self.gpu_acc is not None:
            try:
                data_tensor = torch.tensor(data, dtype=torch.float32)
                if self.gpu_acc.device is not None:
                    data_tensor = data_tensor.to(self.gpu_acc.device)
                alpha = 2 / (span + 1)
                ema = torch.zeros_like(data_tensor)
                ema[0] = data_tensor[0]
                for i in range(1, len(data_tensor)):
                    ema[i] = alpha * data_tensor[i] + (1 - alpha) * ema[i-1]
                return ema.cpu().numpy()
            except Exception as e:
                print(f"GPU计算失败，回退到CPU: {str(e)}")
                pass
        return pd.Series(data).ewm(span=span, adjust=False).mean().values


class MovingAverageStrategy(BaseTradingStrategy):
    """均线交叉策略"""
    
    def __init__(self, symbol: str, ma_short: int = 5, ma_long: int = 21, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.ma_short = ma_short
        self.ma_long = ma_long
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成均线交叉信号"""
        df['MA_short'] = self.calculate_ma(df['收盘'].values, self.ma_short)
        df['MA_long'] = self.calculate_ma(df['收盘'].values, self.ma_long)
        df['信号'] = 0
        df.loc[df['MA_short'] > df['MA_long'], '信号'] = 1
        df.loc[df['MA_short'] < df['MA_long'], '信号'] = -1
        return df


class AnnualMAStrategy(BaseTradingStrategy):
    """年均线突破策略"""
    
    def __init__(self, symbol: str, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成年均线突破信号"""
        df['MA_250'] = self.calculate_ma(df['收盘'].values, 250)
        df['信号'] = 0
        df.loc[df['最高'] > df['MA_250'], '信号'] = 1
        df.loc[df['最低'] < df['MA_250'], '信号'] = -1
        return df


class ProportionalMAStrategy(BaseTradingStrategy):
    """比例交易策略"""
    
    def __init__(self, symbol: str, buy_ratio: float = 0.3, sell_ratio: float = 0.2, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成比例交易信号"""
        df['MA_250'] = self.calculate_ma(df['收盘'].values, 250)
        df['信号'] = 0
        df.loc[df['最高'] > df['MA_250'], '信号'] = 1
        df.loc[df['最低'] < df['MA_250'], '信号'] = -1
        return df


class TurtleTradingStrategy(BaseTradingStrategy):
    """海龟交易策略"""
    
    def __init__(self, symbol: str, entry_window: int = 20, exit_window: int = 10, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.entry_window = entry_window
        self.exit_window = exit_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成海龟交易信号"""
        # 计算唐奇安通道
        df['最高价'] = self.calculate_ma(df['最高'].values, self.entry_window)
        df['最低价'] = self.calculate_ma(df['最低'].values, self.entry_window)
        df['退出最高价'] = self.calculate_ma(df['最高'].values, self.exit_window)
        df['退出最低价'] = self.calculate_ma(df['最低'].values, self.exit_window)
        
        # 生成信号
        df['信号'] = 0
        # 突破上轨买入
        df.loc[df['收盘'] > df['最高价'].shift(1), '信号'] = 1
        # 突破下轨卖出
        df.loc[df['收盘'] < df['最低价'].shift(1), '信号'] = -1
        # 退出信号
        df.loc[df['收盘'] < df['退出最低价'].shift(1), '信号'] = -1
        df.loc[df['收盘'] > df['退出最高价'].shift(1), '信号'] = 1
        
        return df


class RSIStrategy(BaseTradingStrategy):
    """RSI策略"""
    
    def __init__(self, symbol: str, period: int = 14, overbought: int = 70, oversold: int = 30, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成RSI交易信号"""
        # 计算价格变化
        delta = df['收盘'].diff()
        
        # 计算上涨和下跌
        gain = self.calculate_ma(delta.where(delta > 0, 0).values, self.period)
        loss = self.calculate_ma((-delta.where(delta < 0, 0)).values, self.period)
        
        # 计算RS和RSI
        rs = gain / (loss + 1e-10)  # 添加小值避免除零
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['RSI'] < self.oversold, '信号'] = 1  # 超卖买入
        df.loc[df['RSI'] > self.overbought, '信号'] = -1  # 超买卖出
        
        return df


class MACDStrategy(BaseTradingStrategy):
    """MACD策略"""
    
    def __init__(self, symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成MACD交易信号"""
        # 计算EMA
        df['EMA_fast'] = self.calculate_ema(df['收盘'].values, self.fast_period)
        df['EMA_slow'] = self.calculate_ema(df['收盘'].values, self.slow_period)
        
        # 计算MACD线和信号线
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal'] = self.calculate_ema(df['MACD'].values, self.signal_period)
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['MACD'] > df['Signal'], '信号'] = 1  # MACD线上穿信号线买入
        df.loc[df['MACD'] < df['Signal'], '信号'] = -1  # MACD线下穿信号线卖出
        
        return df


class BollingerBandsStrategy(BaseTradingStrategy):
    """布林带策略"""
    
    def __init__(self, symbol: str, period: int = 20, std_dev: float = 2.0, gpu_accelerator=None):
        super().__init__(symbol, gpu_accelerator)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成布林带交易信号"""
        # 计算布林带
        df['MA'] = self.calculate_ma(df['收盘'].values, self.period)
        df['STD'] = self.calculate_std(df['收盘'].values, self.period)
        df['Upper'] = df['MA'] + (df['STD'] * self.std_dev)
        df['Lower'] = df['MA'] - (df['STD'] * self.std_dev)
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['收盘'] < df['Lower'], '信号'] = 1  # 价格触及下轨买入
        df.loc[df['收盘'] > df['Upper'], '信号'] = -1  # 价格触及上轨卖出
        
        return df





