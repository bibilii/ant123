#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : huyifei

import os
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# 默认数据文件夹路径
DEFAULT_DATA_FOLDER = r"\\DX4600-A923\personal_folder\99-data\downtask\taskday\01-每日各家市值及涨幅-efinance爬虫"

class StockInfoProcessor:
    """股票信息处理类"""
    
    def __init__(self, data_folder: Optional[str] = None):
        """
        初始化股票信息处理器
        
        参数:
            data_folder (str, optional): 存放股票CSV文件的文件夹路径。
                                       如果不提供，将使用默认路径。
        """
        self.data_folder = data_folder if data_folder else DEFAULT_DATA_FOLDER
        self.latest_file = self._get_latest_csv()
        self.stock_data = self._load_stock_data()
        
    def _get_latest_csv(self) -> str:
        """获取最新的CSV文件路径"""
        try:
            # 获取所有CSV文件
            csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"在 {self.data_folder} 中没有找到CSV文件")
            
            # 提取日期并排序
            dated_files = []
            for file in csv_files:
                # 移除.csv后缀以获取日期字符串
                date_str = file.replace('.csv', '')
                if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                        dated_files.append((file, date))
                    except ValueError:
                        continue
            
            if not dated_files:
                raise ValueError("没有找到包含有效日期的CSV文件")
            
            # 按日期排序并返回最新的文件路径
            latest_file = sorted(dated_files, key=lambda x: x[1], reverse=True)[0][0]
            return os.path.join(self.data_folder, latest_file)
            
        except Exception as e:
            raise ValueError(f"获取最新CSV文件时出错: {str(e)}")
    
    def _load_stock_data(self) -> pd.DataFrame:
        """加载股票数据"""
        try:
            # 读取CSV文件
            df = pd.read_csv(self.latest_file)
            # print("\n原始数据前几行:")
            # print(df.head())
            # print("\n列名:", df.columns.tolist())
            
            # 确保必要的列存在
            required_columns = ['股票代码', '股票名称']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
            
            # 处理股票代码格式
            df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
            
            return df
        except Exception as e:
            raise ValueError(f"加载股票数据时出错: {str(e)}")
    
    def sanitize_filename(self, filename: str) -> str:
        """
        处理文件名，移除或替换非法字符
        
        参数:
            filename (str): 原始文件名
            
        返回:
            str: 处理后的合法文件名
        """
        # 替换Windows文件名中的非法字符
        illegal_chars = r'[<>:"/\\|?*]'
        # 将非法字符替换为下划线
        sanitized = re.sub(illegal_chars, '_', filename)
        # 移除前后的空格和点
        sanitized = sanitized.strip('. ')
        # 如果文件名为空，返回默认名称
        return sanitized if sanitized else 'unnamed'
    
    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票信息
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            Dict[str, Any]: 包含股票信息的字典
        """
        try:
            # 确保股票代码格式正确
            stock_code = stock_code.zfill(6)  # 补齐6位
            
            # 查找股票信息
            stock_info = self.stock_data[self.stock_data['股票代码'].str.endswith(stock_code)]
            if stock_info.empty:
                raise ValueError(f"未找到股票代码 {stock_code} 的信息")
            
            # 获取当前文件日期
            file_name = os.path.basename(self.latest_file)
            current_date = file_name.replace('.csv', '')  # 直接获取YYYY-MM-DD格式的日期
            
            # 获取股票信息
            row = stock_info.iloc[0]
            stock_name = row['股票名称']
            safe_stock_name = self.sanitize_filename(stock_name)
            
            # 获取其他有用信息
            info = {
                'code': stock_code,
                'name': stock_name,
                'safe_name': safe_stock_name,
                'current_date': current_date,
                'file_path': self.latest_file,
                'price': row['最新价'],
                'change_percent': row['涨跌幅'],
                'volume': row['成交量'],
                'amount': row['成交额'],
                'market_value': row['总市值'],
                'pe_ratio': row['动态市盈率'],
                'turnover_rate': row['换手率'],
                'market_type': row['市场类型'],
                'update_time': row['更新时间']
            }
            
            return info
            
        except Exception as e:
            raise ValueError(f"获取股票信息时出错: {str(e)}")
    
    def print_stock_info(self, stock_code: str) -> None:
        """
        打印股票信息
        
        参数:
            stock_code (str): 股票代码
        """
        try:
            info = self.get_stock_info(stock_code)
            print("\n股票信息:")
            print(f"交易日期: {info['current_date']}")
            print(f"更新时间: {info['update_time']}")
            print(f"股票代码: {info['code']}")
            print(f"股票名称: {info['name']}")
            print(f"市场类型: {info['market_type']}")
            print(f"最新价格: {info['price']}")
            print(f"涨跌幅: {info['change_percent']}%")
            print(f"成交量: {info['volume']}")
            print(f"成交额: {info['amount']:,.2f}")
            print(f"总市值: {info['market_value']:,.2f}")
            print(f"市盈率: {info['pe_ratio']}")
            print(f"换手率: {info['turnover_rate']}%")
            print(f"处理后的名称: {info['safe_name']}")
            print(f"数据文件: {os.path.basename(info['file_path'])}")
        except Exception as e:
            print(f"错误: {str(e)}")
    
    def get_stock_name(self, stock_code: str) -> str:
        """
        根据股票代码获取股票名称
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            str: 股票名称
        """
        try:
            # 确保股票代码格式正确
            stock_code = stock_code.zfill(6)
            
            # 查找股票信息
            stock_info = self.stock_data[self.stock_data['股票代码'].str.endswith(stock_code)]
            if stock_info.empty:
                raise ValueError(f"未找到股票代码 {stock_code} 的信息")
            
            # 返回股票名称
            return stock_info.iloc[0]['股票名称']
            
        except Exception as e:
            raise ValueError(f"获取股票名称时出错: {str(e)}")

def get_stock_name(stock_code: str) -> str:
    """
    便捷函数：根据股票代码获取股票名称
    
    参数:
        stock_code (str): 股票代码
        
    返回:
        str: 股票名称
    """
    processor = StockInfoProcessor()
    return processor.get_stock_name(stock_code)

def sanitize_filename(filename: str) -> str:
    """
    便捷函数：处理文件名，移除或替换非法字符
    
    参数:
        filename (str): 原始文件名
        
    返回:
        str: 处理后的合法文件名
    """
    processor = StockInfoProcessor()
    return processor.sanitize_filename(filename)

def main():
    """主函数用于测试"""
    try:
        # 创建处理器实例（使用默认路径）
        processor = StockInfoProcessor()
        
        # 测试获取股票信息
        test_codes = ['000520', '600000', '688755']
        for code in test_codes:
            print("\n" + "="*50)
            processor.print_stock_info(code)

        processor = StockInfoProcessor()
        processor.print_stock_info('000520')
        
        # 测试新添加的函数
        print("\n测试获取股票名称:")
        print(f"股票代码 000001 的名称是: {get_stock_name('000001')}")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 