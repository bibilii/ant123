import numpy as np
import pandas as pd
import torch

class BaseStrategy:
    """基础策略类，提供GPU加速支持"""
    
    def __init__(self, symbol, gpu_accelerator=None):
        self.symbol = symbol
        self.gpu_acc = gpu_accelerator
        
    def calculate_ma(self, data, window):
        """计算移动平均，支持GPU加速"""
        if self.gpu_acc is not None:
            return self.gpu_acc.calculate_ma(data, window)
        return pd.Series(data).rolling(window=window).mean().values
        
    def calculate_std(self, data, window):
        """计算标准差，支持GPU加速"""
        try:
            if self.gpu_acc is not None:
                data_tensor = self.gpu_acc.to_gpu(data)
                rolling_std = torch.stack([
                    data_tensor[i:i+window].std()
                    for i in range(len(data_tensor)-window+1)
                ])
                return self.gpu_acc.to_cpu(rolling_std)
        except Exception as e:
            pass
        return pd.Series(data).rolling(window=window).std().values
        
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标，支持GPU加速"""
        try:
            if self.gpu_acc is not None:
                # 转换为GPU张量
                prices_tensor = self.gpu_acc.to_gpu(prices)
                # 计算价格变化
                deltas = torch.diff(prices_tensor)
                # 分别计算上涨和下跌
                gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
                losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
                
                # 计算平均增益和损失
                avg_gains = torch.stack([
                    gains[i:i+period].mean()
                    for i in range(len(gains)-period+1)
                ])
                avg_losses = torch.stack([
                    losses[i:i+period].mean()
                    for i in range(len(losses)-period+1)
                ])
                
                # 计算相对强度和RSI
                rs = avg_gains / (avg_losses + 1e-10)  # 添加小值避免除零
                rsi = 100 - (100 / (1 + rs))
                
                return self.gpu_acc.to_cpu(rsi)
        except Exception as e:
            pass
            
        # CPU计算方式
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
            
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标，支持GPU加速"""
        try:
            if self.gpu_acc is not None:
                prices_tensor = self.gpu_acc.to_gpu(prices)
                
                # 计算EMA
                def calculate_ema(data, period):
                    alpha = 2 / (period + 1)
                    ema = torch.zeros_like(data)
                    ema[0] = data[0]
                    for i in range(1, len(data)):
                        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
                    return ema
                
                # 计算快线和慢线
                fast_ema = calculate_ema(prices_tensor, fast_period)
                slow_ema = calculate_ema(prices_tensor, slow_period)
                
                # 计算MACD线
                macd_line = fast_ema - slow_ema
                
                # 计算信号线
                signal_line = calculate_ema(macd_line, signal_period)
                
                # 计算MACD柱状图
                macd_histogram = macd_line - signal_line
                
                return (
                    self.gpu_acc.to_cpu(macd_line),
                    self.gpu_acc.to_cpu(signal_line),
                    self.gpu_acc.to_cpu(macd_histogram)
                )
        except Exception as e:
            pass
            
        # CPU计算方式
        fast_ema = pd.Series(prices).ewm(span=fast_period).mean()
        slow_ema = pd.Series(prices).ewm(span=slow_period).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period).mean()
        macd_histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, macd_histogram.values 