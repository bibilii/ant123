# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/5/6 22:46
import os
import re
from datetime import datetime
from typing import List, Tuple

DATE_FORMAT_PATTERNS: List[Tuple[str, str]] = [
    (r'^\d{4}-\d{2}-\d{2}$', "%Y-%m-%d"),  # YYYY-MM-DD格式
    (r'^\d{8}$', "%Y%m%d")  # YYYYMMDD格式
]

def get_latest_date_subfolder(parent_directory: str) -> str:
    """
    获取指定文件夹中最新日期的子文件夹名称（支持YYYYMMDD和YYYY-MM-DD两种格式）

    参数:
        parent_directory (str): 父文件夹路径

    返回:
        str: 最新日期的文件夹名称（保持原始格式）

    异常:
        ValueError: 如果没有找到符合日期格式的文件夹或父文件夹不存在
    """
    if not os.path.isdir(parent_directory):
        raise ValueError(f"父文件夹不存在: {parent_directory}")

    date_subfolders: List[Tuple[str, datetime]] = []
    for subfolder in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, subfolder)
        if os.path.isdir(subfolder_path):
            for pattern, date_format in DATE_FORMAT_PATTERNS:
                if re.match(pattern, subfolder):
                    try:
                        date_obj = datetime.strptime(subfolder, date_format).date()
                        date_subfolders.append((subfolder, date_obj))
                        break
                    except ValueError:
                        continue

    if not date_subfolders:
        raise ValueError(f"在 {parent_directory} 中没有找到有效的日期格式文件夹")

    date_subfolders.sort(key=lambda x: x[1], reverse=True)
    return date_subfolders[0][0]

if __name__ == "__main__":
    try:
        parent_directory = r"D:\trading\dog\getdata_tushare_output\0102_股票数据_行情数据_历史日线"
        latest = get_latest_date_subfolder(parent_directory)
        print(f"最新的日期文件夹是: {latest}")

        test_folders = ["20240406", "2024-04-05", "20230407", "2023-04-08"]
        print("\n测试混合格式排序结果:")
        for folder in test_folders:
            try:
                date_fmt = "%Y%m%d" if len(folder) == 8 else "%Y-%m-%d"
                date_obj = datetime.strptime(folder, date_fmt).date()
                print(f"{folder} -> {date_obj}")
            except ValueError:
                print(f"{folder} 日期格式无效")

    except ValueError as e:
        print(f"错误: {e}")