#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from getdata_daily_from_localdata import StockDataReader

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PerformanceComparison:
    def __init__(self, data_folder: str, start_date: str, end_date: str):
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_reader = StockDataReader()
        
        # 创建输出目录
        self.output_dir = '.aaa'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_stock_codes(self):
        """获取文件夹中所有CSV文件的股票代码"""
        stock_codes = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.csv'):
                stock_code = file[:-4]
                stock_codes.append(stock_code)
        return sorted(stock_codes)
    
    def process_data_cpu(self, df):
        """CPU处理数据"""
        # 选择需要的列
        numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
        data = df[numeric_columns].values
        
        # 计算技术指标
        close_prices = data[:, 3]
        
        # 计算移动平均
        ma5 = np.convolve(close_prices, np.ones(5)/5, mode='same')
        ma10 = np.convolve(close_prices, np.ones(10)/10, mode='same')
        ma20 = np.convolve(close_prices, np.ones(20)/20, mode='same')
        
        # 计算RSI
        delta = np.zeros_like(close_prices)
        delta[1:] = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # 使用pandas的rolling函数计算移动平均
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.rolling(window=14, min_periods=1).mean().values
        avg_loss = loss_series.rolling(window=14, min_periods=1).mean().values
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'rsi': rsi
        }
    
    def process_data_gpu(self, df):
        """GPU处理数据"""
        # 选择需要的列
        numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
        data = df[numeric_columns].values
        
        # 转换为GPU张量
        tensor_data = torch.FloatTensor(data).to(self.device)
        close_prices = tensor_data[:, 3]
        
        # 计算移动平均
        ma5 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
        ma10 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 10, stride=1, padding=4).squeeze(0)
        ma20 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 20, stride=1, padding=9).squeeze(0)
        
        # 计算RSI
        delta = torch.zeros_like(close_prices)
        delta[1:] = torch.diff(close_prices)
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # 使用pandas的rolling函数计算移动平均（在CPU上）
        gain_series = pd.Series(gain.cpu().numpy())
        loss_series = pd.Series(loss.cpu().numpy())
        avg_gain = torch.FloatTensor(gain_series.rolling(window=14, min_periods=1).mean().values).to(self.device)
        avg_loss = torch.FloatTensor(loss_series.rolling(window=14, min_periods=1).mean().values).to(self.device)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # 将结果移回CPU
        return {
            'ma5': ma5.cpu().numpy(),
            'ma10': ma10.cpu().numpy(),
            'ma20': ma20.cpu().numpy(),
            'rsi': rsi.cpu().numpy()
        }
    
    def plot_results(self, df, cpu_results, gpu_results, symbol):
        """绘制结果图表
        参数:
            df: 包含股票数据的DataFrame
            cpu_results: CPU计算的技术指标结果
            gpu_results: GPU计算的技术指标结果
            symbol: 股票代码
        """
        # 创建一个15x10英寸的图表窗口
        plt.figure(figsize=(15, 10))
        
        # 创建第一个子图（2行1列的第1个位置）
        plt.subplot(2, 1, 1)
        # 绘制收盘价曲线，透明度为0.5
        plt.plot(df.index, df['收盘'], label='收盘价', alpha=0.5)
        # 绘制5日均线，透明度为0.7
        plt.plot(df.index, cpu_results['ma5'], label='MA5 (CPU)', alpha=0.7)
        # 绘制10日均线，透明度为0.7
        plt.plot(df.index, cpu_results['ma10'], label='MA10 (CPU)', alpha=0.7)
        # 绘制20日均线，透明度为0.7
        plt.plot(df.index, cpu_results['ma20'], label='MA20 (CPU)', alpha=0.7)
        # 设置图表标题
        plt.title(f'{symbol} 价格和均线对比')
        # 显示图例
        plt.legend()
        
        # 创建第二个子图（2行1列的第2个位置）
        plt.subplot(2, 1, 2)
        # 绘制CPU计算的RSI指标，透明度为0.7
        plt.plot(df.index, cpu_results['rsi'], label='RSI (CPU)', alpha=0.7)
        # 绘制GPU计算的RSI指标，透明度为0.7
        plt.plot(df.index, gpu_results['rsi'], label='RSI (GPU)', alpha=0.7)
        # 添加RSI超买线（70），红色虚线，透明度0.3
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        # 添加RSI超卖线（30），绿色虚线，透明度0.3
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        # 设置图表标题
        plt.title('RSI指标对比')
        # 显示图例
        plt.legend()


        # 

#         这段代码绘制的是RSI（相对强弱指标）的对比图，让我详细解释一下：
# RSI指标的含义：
# RSI是一个动量指标，用于衡量价格变动的速度和变化
# 取值范围在0-100之间
# 通常使用14天作为计算周期
# 图表中的两条曲线：
# RSI (CPU)：使用CPU计算得到的RSI值
# RSI (GPU)：使用GPU计算得到的RSI值
# 这两条线应该完全重合，因为计算方法是相同的，只是计算设备不同
# 两条水平参考线：
# 红色虚线（y=70）：超买线
# 当RSI超过70时，表示股票可能处于超买状态
# 可能意味着价格过高，有回调风险
# 绿色虚线（y=30）：超卖线
# 当RSI低于30时，表示股票可能处于超卖状态
# 可能意味着价格过低，有反弹机会
# 图表的使用方法：
# 当RSI曲线从下向上穿过30线时，可能是买入信号
# 当RSI曲线从上向下穿过70线时，可能是卖出信号
# 当RSI在50附近波动时，表示市场处于盘整状态
# 当RSI出现背离（价格创新高但RSI没有）时，可能预示趋势即将改变
# 这个对比图的主要目的：
# 验证CPU和GPU计算的结果是否一致
# 展示RSI指标的变化趋势
# 帮助判断股票的买卖时机
# 评估市场是否处于超买或超卖状态
# 这个图表是技术分析中常用的工具，可以帮助投资者判断市场趋势和寻找交易机会。


        # 
        
        # 自动调整子图之间的间距，使图表更美观
        plt.tight_layout()
        
        # 生成时间戳，用于文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 构建保存路径
        plot_file = os.path.join(self.output_dir, f"{symbol}_comparison_{timestamp}.png")
        # 保存图表为PNG文件
        plt.savefig(plot_file)
        # 关闭图表，释放内存
        plt.close()
        
        return plot_file
    
    def run_comparison(self, symbol):
        """运行CPU和GPU性能比较"""
        print(f"\n开始处理股票: {symbol}")
        
        # 获取数据
        df = self.data_reader.get_standard_data(symbol, self.start_date, self.end_date)
        if df is None or df.empty:
            print(f"错误: 无法获取股票 {symbol} 的数据")
            return
        
        # CPU处理
        cpu_start_time = time.time()
        cpu_results = self.process_data_cpu(df)
        cpu_end_time = time.time()
        cpu_time = cpu_end_time - cpu_start_time
        
        # GPU处理
        gpu_start_time = time.time()
        gpu_results = self.process_data_gpu(df)
        gpu_end_time = time.time()
        gpu_time = gpu_end_time - gpu_start_time
        
        # 绘制结果
        plot_file = self.plot_results(df, cpu_results, gpu_results, symbol)
        
        print(f"CPU处理时间: {cpu_time:.4f} 秒")
        print(f"GPU处理时间: {gpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        print(f"结果图表已保存至: {plot_file}")
        
        return {
            'symbol': symbol,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time/gpu_time
        }

def main():
    # 设置参数
    data_folder = r"\\DH2600-866D\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫\2025-05-09"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # 创建性能比较器
    comparator = PerformanceComparison(data_folder, start_date, end_date)
    
    # 获取股票代码
    stock_codes = comparator.get_stock_codes()
    print(f"找到 {len(stock_codes)} 个股票代码")
    
    # 运行比较
    results = []
    for symbol in stock_codes[:50]:  # 只处理前5个股票作为示例
        result = comparator.run_comparison(symbol)
        if result:
            results.append(result)
    
    # 打印总体结果
    print("\n总体性能比较结果:")
    print("-" * 60)
    print(f"{'股票代码':<10} {'CPU时间(秒)':>12} {'GPU时间(秒)':>12} {'加速比':>8}")
    print("-" * 60)
    for result in results:
        print(f"{result['symbol']:<10} {result['cpu_time']:>12.4f} {result['gpu_time']:>12.4f} {result['speedup']:>8.2f}x")

if __name__ == "__main__":
    main() 