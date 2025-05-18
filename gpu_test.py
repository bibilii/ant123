#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime

def test_gpu_info():
    """测试GPU基本信息"""
    print("\n=== GPU基本信息 ===")
    if torch.cuda.is_available():
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        # 获取当前GPU信息
        current_device = torch.cuda.current_device()
        print(f"当前GPU索引: {current_device}")
        
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"GPU名称: {gpu_name}")
        
        # 获取GPU计算能力
        capability = torch.cuda.get_device_capability(current_device)
        print(f"GPU计算能力: {capability[0]}.{capability[1]}")
        
        # 获取GPU内存信息
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2
        print(f"已分配内存: {memory_allocated:.2f} MB")
        print(f"已保留内存: {memory_reserved:.2f} MB")
        
        # 获取GPU总内存
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**2
        print(f"GPU总内存: {total_memory:.2f} MB")
        
        return True
    else:
        print("未检测到GPU，将使用CPU模式")
        return False

def test_matrix_operations(size=10000):
    """测试矩阵运算性能"""
    print(f"\n=== 矩阵运算性能测试 (矩阵大小: {size}x{size}) ===")
    
    # 创建随机矩阵
    matrix_a = torch.randn(size, size)
    matrix_b = torch.randn(size, size)
    
    # CPU测试
    start_time = time.time()
    cpu_result = torch.matmul(matrix_a, matrix_b)
    cpu_time = time.time() - start_time
    print(f"CPU计算时间: {cpu_time:.4f} 秒")
    
    if torch.cuda.is_available():
        # 将矩阵移到GPU
        matrix_a_gpu = matrix_a.cuda()
        matrix_b_gpu = matrix_b.cuda()
        
        # GPU预热
        torch.matmul(matrix_a_gpu, matrix_b_gpu)
        torch.cuda.synchronize()
        
        # GPU测试
        start_time = time.time()
        gpu_result = torch.matmul(matrix_a_gpu, matrix_b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU计算时间: {gpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果
        is_close = torch.allclose(cpu_result, gpu_result.cpu(), rtol=1e-4, atol=1e-4)
        print(f"结果验证: {'通过' if is_close else '失败'}")

def test_memory_operations(size=1000000):
    """测试内存操作性能"""
    print(f"\n=== 内存操作性能测试 (数据大小: {size}) ===")
    
    # 创建随机数据
    data = torch.randn(size)
    
    # CPU测试
    start_time = time.time()
    cpu_result = torch.sort(data)
    cpu_time = time.time() - start_time
    print(f"CPU排序时间: {cpu_time:.4f} 秒")
    
    if torch.cuda.is_available():
        # 将数据移到GPU
        data_gpu = data.cuda()
        
        # GPU预热
        torch.sort(data_gpu)
        torch.cuda.synchronize()
        
        # GPU测试
        start_time = time.time()
        gpu_result = torch.sort(data_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU排序时间: {gpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果
        is_close = torch.allclose(cpu_result[0], gpu_result[0].cpu(), rtol=1e-4, atol=1e-4)
        print(f"结果验证: {'通过' if is_close else '失败'}")

def test_strategy_performance():
    """测试策略计算性能"""
    print("\n=== 策略计算性能测试 ===")
    
    # 创建示例数据
    data_size = 1000000
    prices = torch.randn(data_size)
    
    # 计算移动平均
    window = 20
    
    # CPU测试
    start_time = time.time()
    cpu_ma = torch.zeros_like(prices)
    for i in range(len(prices) - window + 1):
        cpu_ma[i + window - 1] = prices[i:i+window].mean()
    cpu_time = time.time() - start_time
    print(f"CPU移动平均计算时间: {cpu_time:.4f} 秒")
    
    if torch.cuda.is_available():
        # 将数据移到GPU
        prices_gpu = prices.cuda()
        
        # GPU预热
        gpu_ma = torch.zeros_like(prices_gpu)
        for i in range(len(prices_gpu) - window + 1):
            gpu_ma[i + window - 1] = prices_gpu[i:i+window].mean()
        torch.cuda.synchronize()
        
        # GPU测试
        start_time = time.time()
        gpu_ma = torch.zeros_like(prices_gpu)
        for i in range(len(prices_gpu) - window + 1):
            gpu_ma[i + window - 1] = prices_gpu[i:i+window].mean()
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU移动平均计算时间: {gpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果
        is_close = torch.allclose(cpu_ma, gpu_ma.cpu(), rtol=1e-4, atol=1e-4)
        print(f"结果验证: {'通过' if is_close else '失败'}")

def save_test_results(results):
    """保存测试结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"gpu_test_results_{timestamp}.csv"
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 保存到CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n测试结果已保存到: {filename}")

def main():
    """主函数"""
    print("开始GPU性能测试...")
    
    # 测试结果字典
    results = {
        '测试时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'GPU可用': torch.cuda.is_available()
    }
    
    # 测试GPU信息
    if test_gpu_info():
        results['GPU名称'] = torch.cuda.get_device_name(0)
        results['GPU计算能力'] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        results['GPU总内存(MB)'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        
        # 测试矩阵运算
        test_matrix_operations()
        
        # 测试内存操作
        test_memory_operations()
        
        # 测试策略性能
        test_strategy_performance()
    
    # 保存测试结果
    save_test_results(results)

if __name__ == "__main__":
    main() 