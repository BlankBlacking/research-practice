#!/usr/bin/env python
"""
使用阿里云通义千问 (Qwen) API 进行微调
支持数据上传、微调任务提交和状态监控
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional
from pathlib import Path

# 阿里云 Qwen API 配置
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/api/v1"


class QwenFinetuneClient:
    def __init__(self, api_key: str):
        """初始化 Qwen 微调客户端"""
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"✅ Qwen 微调客户端已初始化")

    def upload_dataset(self, file_path: str, dataset_name: str) -> str:
        """
        上传数据集文件到阿里云
        
        Args:
            file_path: 本地文件路径
            dataset_name: 数据集名称
            
        Returns:
            dataset_id: 阿里云数据集 ID
        """
        print(f"\n📤 上传数据集: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_size = os.path.getsize(file_path)
        print(f"   文件大小: {file_size / 1024:.1f} KB")
        
        # 上传文件
        with open(file_path, 'rb') as f:
            files = {
                'file': (Path(file_path).name, f, 'application/octet-stream')
            }
            
            # 使用简单的 POST 请求上传
            url = f"{DASHSCOPE_API_BASE}/datasets/upload"
            
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data={"dataset_name": dataset_name}
            )
        
        if response.status_code != 200:
            print(f"   ❌ 上传失败: {response.status_code}")
            print(f"   响应: {response.text}")
            raise Exception(f"上传失败: {response.text}")
        
        result = response.json()
        dataset_id = result.get('data', {}).get('id')
        print(f"   ✅ 上传成功!")
        print(f"   Dataset ID: {dataset_id}")
        
        return dataset_id

    def create_finetune_job(
        self,
        model: str,
        train_file_path: str,
        val_file_path: Optional[str] = None,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        job_name: str = "reversal_curse_finetune"
    ) -> str:
        """
        创建微调任务
        
        Args:
            model: 模型名称 (qwen-7b-chat, qwen-14b-chat 等)
            train_file_path: 训练数据文件路径
            val_file_path: 验证数据文件路径(可选)
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            job_name: 任务名称
            
        Returns:
            job_id: 微调任务 ID
        """
        print(f"\n🚀 创建微调任务")
        print(f"{'='*60}")
        print(f"模型: {model}")
        print(f"训练文件: {train_file_path}")
        if val_file_path:
            print(f"验证文件: {val_file_path}")
        print(f"轮次: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"{'='*60}")
        
        # 验证文件存在
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"训练文件不存在: {train_file_path}")
        if val_file_path and not os.path.exists(val_file_path):
            raise FileNotFoundError(f"验证文件不存在: {val_file_path}")
        
        # 准备请求数据
        payload = {
            "model": model,
            "job_name": job_name,
            "input": {
                "train_file_path": train_file_path,
                "val_file_path": val_file_path
            },
            "hyper_parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        }
        
        # 创建任务
        url = f"{DASHSCOPE_API_BASE}/finetune/jobs"
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code not in [200, 201]:
            print(f"❌ 创建任务失败: {response.status_code}")
            print(f"响应: {response.text}")
            raise Exception(f"创建微调任务失败: {response.text}")
        
        result = response.json()
        job_id = result.get('data', {}).get('id') or result.get('data', {}).get('job_id')
        
        print(f"✅ 任务创建成功!")
        print(f"   Job ID: {job_id}")
        
        return job_id

    def get_job_status(self, job_id: str) -> Dict:
        """
        获取微调任务状态
        
        Args:
            job_id: 任务 ID
            
        Returns:
            status_info: 任务状态信息
        """
        url = f"{DASHSCOPE_API_BASE}/finetune/jobs/{job_id}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"❌ 获取状态失败: {response.status_code}")
            raise Exception(f"获取状态失败: {response.text}")
        
        return response.json().get('data', {})

    def monitor_job(self, job_id: str, check_interval: int = 30, max_checks: int = 1000):
        """
        监控微调任务进度
        
        Args:
            job_id: 任务 ID
            check_interval: 检查间隔(秒)
            max_checks: 最大检查次数
        """
        print(f"\n⏱️  监控任务: {job_id}")
        print(f"{'='*60}")
        
        check_count = 0
        
        while check_count < max_checks:
            try:
                status_info = self.get_job_status(job_id)
                
                status = status_info.get('status')
                progress = status_info.get('progress', {})
                
                print(f"\n[检查 {check_count + 1}/{max_checks}] {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  状态: {status}")
                
                if progress:
                    current = progress.get('current_step', 0)
                    total = progress.get('total_steps', 0)
                    if total > 0:
                        pct = (current / total) * 100
                        print(f"  进度: {current}/{total} ({pct:.1f}%)")
                
                # 检查是否完成
                if status in ['SUCCEEDED', 'success', 'completed']:
                    print(f"\n✅ 任务完成!")
                    print(f"   Model ID: {status_info.get('output', {}).get('model_id')}")
                    break
                elif status in ['FAILED', 'failed', 'error']:
                    print(f"\n❌ 任务失败!")
                    print(f"   错误: {status_info.get('error_message')}")
                    break
                
                # 等待后继续检查
                time.sleep(check_interval)
                check_count += 1
                
            except KeyboardInterrupt:
                print(f"\n⏸️  用户中断监控")
                break
            except Exception as e:
                print(f"  ⚠️  获取状态失败: {e}")
                time.sleep(check_interval)
                check_count += 1
        
        print(f"{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="阿里云 Qwen 微调工具")
    parser.add_argument("--api_key", type=str, required=False, 
                       help="API Key (如果不提供，从环境变量 QWEN_API_KEY 读取)")
    parser.add_argument("--model", type=str, default="qwen-7b-chat",
                       help="模型名称 (默认: qwen-7b-chat)")
    parser.add_argument("--train_file", type=str, 
                       default="data/reverse_experiments/fresh_attempt5576341111/train_messages.jsonl",
                       help="训练数据文件路径")
    parser.add_argument("--val_file", type=str,
                       default="data/reverse_experiments/fresh_attempt5576341111/val_messages.jsonl",
                       help="验证数据文件路径")
    parser.add_argument("--epochs", type=int, default=1,
                       help="训练轮次 (默认: 1)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小 (默认: 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率 (默认: 1e-4)")
    parser.add_argument("--job_name", type=str, default="reversal_curse_finetune",
                       help="任务名称")
    parser.add_argument("--monitor", action="store_true",
                       help="创建后监控任务进度")
    
    args = parser.parse_args()
    
    # 获取 API Key
    api_key = args.api_key or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("请提供 API Key (--api_key 参数或 QWEN_API_KEY 环境变量)")
    
    print("="*60)
    print("🤖 阿里云通义千问 (Qwen) 微调工具")
    print("="*60)
    
    # 初始化客户端
    client = QwenFinetuneClient(api_key)
    
    # 创建微调任务
    try:
        job_id = client.create_finetune_job(
            model=args.model,
            train_file_path=args.train_file,
            val_file_path=args.val_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            job_name=args.job_name
        )
        
        print(f"\n💾 任务 ID 已保存，请妥善保管:")
        print(f"   {job_id}")
        
        # 如果指定了 --monitor 参数，则监控任务
        if args.monitor:
            client.monitor_job(job_id)
        else:
            print(f"\n💡 提示: 运行以下命令监控任务进度:")
            print(f"   python qwen_finetune.py --job_id {job_id} --monitor")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
