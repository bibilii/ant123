# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55
import os
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from strategy import ProportionalMAStrategy
from strategy_statistics import StrategyStatistics
from strategy_plotter import StrategyPlotter
from getdata_daily_from_localdata import StockDataReader

# 配置参数
START_DATE = "2014-01-01"  # 需要更长的历史数据来计算年均线
END_DATE = "2025-05-01"
INITIAL_CASH = 100000.0  # 初始资金10万

# 数据源配置，如果为空则使用StockDataReader中的默认值
DATA_SOURCE_TYPE = ''  # 'efinance' 或 'tushare'
INPUT_FOLDER = ''  # 数据文件夹路径

# DATA_SOURCE_TYPE = 'tushare'  # 'efinance' 或 'tushare'
# INPUT_FOLDER = r'D:\trading\ant\datatest\different_csv_data\tushare'  # 数据文件夹路径

# 输出配置
OUTPUT_FOLDER = 'output\\多线程-画图-比例交易策略-new'

# 选定的股票代码
SELECTED_STOCK_CODES = [
    # '002239',  # 奥特佳
    # '002160',  # 常铝股份
    # '600053',  # 九鼎投资
    # '002135',  # 东南网架
    # '000520',  # 凤凰航运
    #
    # '000001',  # 平安银行
    # '002594',  # 比亚迪
    # '002574',  # 明牌珠宝
    # '601727',  # 上海电气
    # '601877',  # 正泰电气
    # '600900',  # 长江电力
    # '600481',  # 双良节能
    # '300750',  # 宁德时代
    # '603843',  # 正平股份
    # '600191',  # 华资实业
    # '601166',  # 兴业银行
]

plotter = StrategyPlotter()

def get_csv_files_by_type(reader: StockDataReader):
    """根据数据类型获取CSV文件列表"""
    import glob
    data_dir = reader.csv_file_path
    if reader.data_type == 'efinance':
        pattern = os.path.join(data_dir, '[0-9][0-9][0-9][0-9][0-9][0-9].csv')
        return glob.glob(pattern)[:10000]
    elif reader.data_type == 'tushare':
        pattern_sz = os.path.join(data_dir, '*.SZ.csv')
        pattern_sh = os.path.join(data_dir, '*.SH.csv')
        pattern_bj = os.path.join(data_dir, '*.BJ.csv')
        return glob.glob(pattern_sz)[:10] + glob.glob(pattern_sh)[:10]+glob.glob(pattern_bj)[:10]
    else:
        raise ValueError("不支持的数据类型")

def process_single_stock(args):
    """
    处理单个股票的策略执行和结果保存

    参数:
        args (tuple): 包含股票代码、输出文件夹、开始日期和结束日期的元组
    """
    symbol, output_folder, start_date, end_date = args
    
    try:
        # 创建数据读取器，如果DATA_SOURCE_TYPE为空则使用默认值
        reader = StockDataReader(
            data_type=DATA_SOURCE_TYPE if DATA_SOURCE_TYPE else None,
            parent_dir=INPUT_FOLDER if INPUT_FOLDER else None
        )
        
        # 创建策略实例
        strategy = ProportionalMAStrategy(
            symbol=symbol,
            initial_cash=INITIAL_CASH,
            buy_ratio=0.3,  # 买入比例30%
            sell_ratio=0.2  # 卖出比例20%
        )
        
        # 获取数据
        df = reader.get_standard_data(symbol, start_date, end_date)
        
        # 重命名列以匹配策略需求
        df = df.rename(columns={
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额'
        })
        
        # 检查数据是否为空
        if df.empty:
            print(f"警告: 股票 {symbol} 在指定时间范围内没有数据")
            # 写入未执行成功的统计记录
            stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
            statistics = {
                '策略名称': '比例交易策略',
                '股票代码': symbol,
                '开始日期': start_date,
                '结束日期': end_date,
                '交易天数': 0,
                '初始资产': INITIAL_CASH,
                '最终总资产': '',
                '最终股票市值': '',
                '持有现金': '',
                '持有股票数量': '',
                '信号次数': '',
                '买入次数': '',
                '卖出次数': '',
                '累计佣金': '',
                '累计印花税': '',
                '总收益率': '',
                '股票涨幅': '',
                '备注': '未执行成功'
            }
            stats.save_statistics(statistics)
            return False
            
        # 确保日期列是datetime类型并按时间正序排列
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=True)
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 检查必要的列是否存在
        required_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: 股票 {symbol} 数据缺少必要的列: {missing_columns}")
            # 写入未执行成功的统计记录
            stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
            statistics = {
                '策略名称': '比例交易策略',
                '股票代码': symbol,
                '开始日期': start_date,
                '结束日期': end_date,
                '交易天数': 0,
                '初始资产': INITIAL_CASH,
                '最终总资产': '',
                '最终股票市值': '',
                '持有现金': '',
                '持有股票数量': '',
                '信号次数': '',
                '买入次数': '',
                '卖出次数': '',
                '累计佣金': '',
                '累计印花税': '',
                '总收益率': '',
                '股票涨幅': '',
                '备注': '未执行成功'
            }
            stats.save_statistics(statistics)
            return False
            
        # 检查数据类型
        numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 执行策略
        result_df = strategy.execute_strategy(df)
        
        # 检查策略执行结果
        if result_df is None or result_df.empty:
            print(f"警告: 股票 {symbol} 策略执行结果为空")
            # 写入未执行成功的统计记录
            stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
            statistics = {
                '策略名称': '比例交易策略',
                '股票代码': symbol,
                '开始日期': start_date,
                '结束日期': end_date,
                '交易天数': 0,
                '初始资产': INITIAL_CASH,
                '最终总资产': '',
                '最终股票市值': '',
                '持有现金': '',
                '持有股票数量': '',
                '信号次数': '',
                '买入次数': '',
                '卖出次数': '',
                '累计佣金': '',
                '累计印花税': '',
                '总收益率': '',
                '股票涨幅': '',
                '备注': '未执行成功'
            }
            stats.save_statistics(statistics)
            return False
        
        # 保存结果
        output_file = os.path.join(output_folder, f"{symbol}_result.csv")
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 生成图表
        output_plot = os.path.join(output_folder, f"{symbol}_plot.png")
        plotter.plot_strategy_result(result_df, symbol, output_plot)
        
        # 计算并保存统计结果
        stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
        statistics = stats.calculate_statistics(
            df=result_df,
            strategy_name='比例交易策略',
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_cash=INITIAL_CASH
        )
        stats.save_statistics(statistics)
        
        return True
    except Exception as e:
        print(f"处理股票 {symbol} 时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())  # 打印完整的错误堆栈
        # 写入未执行成功的统计记录
        stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
        statistics = {
            '策略名称': '比例交易策略',
            '股票代码': symbol,
            '开始日期': start_date,
            '结束日期': end_date,
            '交易天数': 0,
            '初始资产': INITIAL_CASH,
            '最终总资产': '',
            '最终股票市值': '',
            '持有现金': '',
            '持有股票数量': '',
            '信号次数': '',
            '买入次数': '',
            '卖出次数': '',
            '累计佣金': '',
            '累计印花税': '',
            '总收益率': '',
            '股票涨幅': '',
            '备注': '未执行成功'
        }
        stats.save_statistics(statistics)
        return False

def main():
    """主函数"""
    global SELECTED_STOCK_CODES

    # 创建数据读取器，如果DATA_SOURCE_TYPE为空则使用默认值
    reader = StockDataReader(
        data_type=DATA_SOURCE_TYPE if DATA_SOURCE_TYPE else None,
        parent_dir=INPUT_FOLDER if INPUT_FOLDER else None
    )
    
    # 创建输出文件夹
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 如果SELECTED_STOCK_CODES为空，则读取默认文件夹中所有CSV文件
    if len(SELECTED_STOCK_CODES) == 0:
        csv_files = get_csv_files_by_type(reader)
        if reader.data_type == 'efinance':
            SELECTED_STOCK_CODES = [os.path.basename(f).split('.')[0] for f in csv_files]
        elif reader.data_type == 'tushare':
            SELECTED_STOCK_CODES = [os.path.basename(f).split('.')[0] + '.' + os.path.basename(f).split('.')[1] for f in csv_files]
    
    # 准备参数列表
    args_list = [(symbol, OUTPUT_FOLDER, START_DATE, END_DATE) 
                 for symbol in SELECTED_STOCK_CODES]
    # # 使用多进程处理
    # with Pool() as pool:
    #     results = list(tqdm(
    #         pool.imap_unordered(process_single_stock, args_list),
    #         total=len(SELECTED_STOCK_CODES),
    #         desc="处理股票"
    #     ))
    # 单进程顺序处理
    results = []
    for args in tqdm(args_list, desc="处理股票"):
        results.append(process_single_stock(args))

    # 统计处理结果
    success_count = sum(1 for r in results if r)
    print(f"\n处理完成！成功处理 {success_count}/{len(SELECTED_STOCK_CODES)} 只股票")

if __name__ == "__main__":
    main() 