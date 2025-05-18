#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import os
from getdata_daily_from_localdata import StockDataReader
from strategy import (
    MovingAverageStrategy, 
    AnnualMAStrategy, 
    ProportionalMAStrategy,
    TurtleTradingStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy
)
from strategy_executor import TradeExecutor
from strategy_plotter import StrategyPlotter
from strategy_statistics import StrategyStatistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

class GPUStrategyRunner:
    """GPU加速的批量策略运行器类"""
    
    def __init__(self, 
                 data_folder: str,
                 start_date: str,
                 end_date: str,
                 initial_cash: float = 1000000.0,
                 buy_ratio: float = 0.2,
                 sell_ratio: float = 0.3,
                 batch_size: int = 32):
        """
        初始化GPU加速的批量策略运行器
        参数:
            data_folder (str): CSV文件所在文件夹路径
            start_date (str): 开始日期
            end_date (str): 结束日期
            initial_cash (float): 初始资金
            buy_ratio (float): 买入比例
            sell_ratio (float): 卖出比例
            batch_size (int): GPU批处理大小
        """
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.batch_size = batch_size
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = '.strategy_results'
        self.batch_output_dir = os.path.join(self.output_dir, 'batch_gpu')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)
            
        # 初始化各个组件
        self.data_reader = StockDataReader()
        self.plotter = StrategyPlotter()
        self.statistics = StrategyStatistics(output_file=os.path.join(self.output_dir, 'batch_strategy_statistics_gpu.csv'))
        
    def get_stock_codes(self):
        """获取文件夹中所有CSV文件的股票代码"""
        stock_codes = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.csv'):
                stock_code = file[:-4]
                stock_codes.append(stock_code)
        return sorted(stock_codes)
    
    def prepare_data_for_gpu(self, df):
        """将数据转换为GPU张量格式并进行预处理"""
        try:
            # 检查数据是否为空
            if df is None or df.empty:
                print("错误：数据为空")
                return None, None
                
            # 打印列名以便调试
            print(f"数据列名: {df.columns.tolist()}")
            
            # 选择需要的列（使用中文列名）
            numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
            
            # 检查所需的列是否存在
            missing_columns = [col for col in numeric_columns if col not in df.columns]
            if missing_columns:
                print(f"错误：缺少以下列: {missing_columns}")
                return None, None
            
            # 获取数据
            data = df[numeric_columns].values
            
            # 转换为GPU张量
            tensor_data = torch.FloatTensor(data).to(self.device)
            
            # 数据标准化
            mean = tensor_data.mean(dim=0)
            std = tensor_data.std(dim=0)
            normalized_data = (tensor_data - mean) / (std + 1e-8)
            
            return normalized_data, tensor_data
            
        except Exception as e:
            print(f"数据预处理时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None

    def process_batch_on_gpu(self, batch_data):
        """在GPU上批量处理数据"""
        # 将数据分成多个批次
        batch_size = self.batch_size
        num_samples = batch_data.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        results = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = batch_data[start_idx:end_idx]
            
            # 在GPU上进行计算
            with torch.amp.autocast('cuda'):  # 使用新的混合精度计算语法
                # 计算技术指标
                # 确保所有计算使用相同大小的张量
                close_prices = batch[:, 3]  # 收盘价
                
                # 计算移动平均
                ma5 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
                ma10 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 10, stride=1, padding=4).squeeze(0)
                ma20 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 20, stride=1, padding=9).squeeze(0)
                
                # 计算RSI
                delta = close_prices.diff()
                gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
                loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
                avg_gain = torch.nn.functional.avg_pool1d(gain.unsqueeze(0), 14, stride=1, padding=6).squeeze(0)
                avg_loss = torch.nn.functional.avg_pool1d(loss.unsqueeze(0), 14, stride=1, padding=6).squeeze(0)
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                # 计算MACD
                ema12 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 12, stride=1, padding=5).squeeze(0)
                ema26 = torch.nn.functional.avg_pool1d(close_prices.unsqueeze(0), 26, stride=1, padding=12).squeeze(0)
                macd = ema12 - ema26
                signal = torch.nn.functional.avg_pool1d(macd.unsqueeze(0), 9, stride=1, padding=4).squeeze(0)
                
                # 计算布林带
                window_size = 20
                # 使用卷积计算移动平均和标准差
                kernel = torch.ones(window_size, device=self.device) / window_size
                
                # 确保输入张量维度正确
                close_prices_2d = close_prices.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
                kernel_2d = kernel.view(1, 1, -1)  # [1, 1, window_size]
                
                # 计算移动平均
                rolling_mean = torch.nn.functional.conv1d(
                    close_prices_2d,
                    kernel_2d,
                    padding=(window_size-1)//2
                ).squeeze()
                
                # 计算标准差
                squared_diff = (close_prices - rolling_mean) ** 2
                squared_diff_2d = squared_diff.unsqueeze(0).unsqueeze(0)
                
                rolling_std = torch.sqrt(torch.nn.functional.conv1d(
                    squared_diff_2d,
                    kernel_2d,
                    padding=(window_size-1)//2
                ).squeeze())
                
                # 确保所有张量大小一致
                min_length = min(len(close_prices), len(rolling_mean), len(rolling_std))
                close_prices = close_prices[:min_length]
                rolling_mean = rolling_mean[:min_length]
                rolling_std = rolling_std[:min_length]
                
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
            
            # 将结果移回CPU
            batch_results = {
                'ma5': ma5.cpu().numpy(),
                'ma10': ma10.cpu().numpy(),
                'ma20': ma20.cpu().numpy(),
                'rsi': rsi.cpu().numpy(),
                'macd': macd.cpu().numpy(),
                'signal': signal.cpu().numpy(),
                'upper_band': upper_band.cpu().numpy(),
                'lower_band': lower_band.cpu().numpy()
            }
            results.append(batch_results)
        
        return results

    def run_strategy_for_stock(self, symbol: str, strategy_name: str):
        """对单个股票运行指定的策略（GPU版本）"""
        print(f"\n开始运行策略: {strategy_name}")
        print(f"股票代码: {symbol}")
        
        try:
            # 1. 获取数据
            df = self.data_reader.get_standard_data(symbol, self.start_date, self.end_date)
            if df is None or df.empty:
                print(f"错误: 无法获取股票 {symbol} 的数据")
                return None, None
            
            # 2. 将数据转移到GPU并进行预处理
            normalized_data, original_data = self.prepare_data_for_gpu(df)
            if normalized_data is None or original_data is None:
                print(f"错误: 数据预处理失败")
                return None, None
            
            # 3. 在GPU上批量处理数据
            gpu_results = self.process_batch_on_gpu(normalized_data)
            
            # 4. 将GPU计算结果转换回DataFrame格式
            for i, batch_results in enumerate(gpu_results):
                for key, value in batch_results.items():
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, len(df))
                    df.loc[start_idx:end_idx-1, key] = value
            
            # 5. 创建策略实例
            if strategy_name == "均线交叉策略":
                strategy = MovingAverageStrategy(symbol)
            elif strategy_name == "年均线突破策略":
                strategy = AnnualMAStrategy(symbol)
            elif strategy_name == "比例交易策略":
                strategy = ProportionalMAStrategy(symbol)
            elif strategy_name == "海龟交易策略":
                strategy = TurtleTradingStrategy(symbol)
            elif strategy_name == "RSI策略":
                strategy = RSIStrategy(symbol)
            elif strategy_name == "MACD策略":
                strategy = MACDStrategy(symbol)
            elif strategy_name == "布林带策略":
                strategy = BollingerBandsStrategy(symbol)
            else:
                print(f"错误: 未知的策略名称 {strategy_name}")
                return None, None
            
            # 6. 生成交易信号
            df = strategy.generate_signals(df)
            
            # 7. 执行交易
            executor = TradeExecutor(
                strategy_name=strategy_name,
                initial_cash=self.initial_cash,
                buy_ratio=self.buy_ratio,
                sell_ratio=self.sell_ratio
            )
            trade_records = executor.execute_trades(df)
            
            # 8. 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trade_records_file = os.path.join(self.batch_output_dir, f"{symbol}_{strategy_name}_{timestamp}_trades.csv")
            trade_records.to_csv(trade_records_file, index=False, encoding='utf-8-sig')
            
            # 9. 绘制策略结果图表
            plot_file = os.path.join(self.batch_output_dir, f"{symbol}_{strategy_name}_{timestamp}_plot.png")
            plt.switch_backend('Agg')
            self.plotter.plot_strategy_result(trade_records, symbol, plot_file)
            plt.close('all')
            
            # 10. 计算并保存统计结果
            stats = self.statistics.calculate_statistics(
                df=trade_records,
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash
            )
            self.statistics.save_statistics(stats)
            
            return trade_records, stats
            
        except Exception as e:
            print(f"处理股票 {symbol} 的策略 {strategy_name} 时发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None

    def run_batch_strategies(self, stock_codes, strategies):
        """批量运行策略（GPU版本）"""
        results = {}
        
        # 使用线程池并行处理多个股票
        with ThreadPoolExecutor(max_workers=torch.cuda.device_count() * 2) as executor:
            for symbol in stock_codes:
                print(f"\n开始处理股票: {symbol}")
                symbol_results = {}
                
                # 为每个股票创建任务
                future_to_strategy = {
                    executor.submit(self.run_strategy_for_stock, symbol, strategy_name): strategy_name
                    for strategy_name in strategies
                }
                
                # 收集结果
                for future in future_to_strategy:
                    strategy_name = future_to_strategy[future]
                    try:
                        trade_records, stats = future.result()
                        if trade_records is not None and stats is not None:
                            symbol_results[strategy_name] = {
                                'trade_records': trade_records,
                                'statistics': stats
                            }
                    except Exception as e:
                        print(f"处理股票 {symbol} 的策略 {strategy_name} 时发生错误: {str(e)}")
                
                results[symbol] = symbol_results
                
                # 打印该股票的策略比较结果
                self._print_strategy_comparison(symbol, symbol_results, strategies)
        
        return results

    def _print_strategy_comparison(self, symbol, results, strategies):
        """打印策略比较结果"""
        print(f"\n{symbol} 策略比较结果:")
        print("-" * 100)
        print(f"{'策略名称':<15} {'总收益率':>10} {'股票涨幅':>10} {'买入次数':>8} {'卖出次数':>8} {'最大回撤':>10}")
        print("-" * 100)
        
        for strategy_name in strategies:
            if strategy_name in results:
                stats = results[strategy_name]['statistics']
                print(f"{strategy_name:<15} {stats['总收益率']:>10.2f}% {stats['股票涨幅']:>10.2f}% "
                      f"{stats['买入次数']:>8} {stats['卖出次数']:>8} {stats.get('最大回撤', 0):>10.2f}%")

def main():
    """主函数"""
    # 设置回测参数
    data_folder = r"\\DH2600-866D\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫\2025-05-09"
    start_date = "2013-01-01"
    end_date = "2024-01-01"
    initial_cash = 1000000.0
    buy_ratio = 0.2
    sell_ratio = 0.3
    batch_size = 32  # GPU批处理大小
    
    # 创建GPU加速的批量策略运行器
    runner = GPUStrategyRunner(
        data_folder=data_folder,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        batch_size=batch_size
    )
    
    # 获取所有股票代码
    stock_codes = runner.get_stock_codes()
    print(f"找到 {len(stock_codes)} 个股票代码: {', '.join(stock_codes)}")
    
    # 定义要运行的策略
    strategies = [
        "均线交叉策略",
        "年均线突破策略",
        "比例交易策略",
        "海龟交易策略",
        "RSI策略",
        "MACD策略",
        "布林带策略"
    ]
    
    # 运行所有策略
    runner.run_batch_strategies(stock_codes, strategies)

if __name__ == "__main__":
    main() 