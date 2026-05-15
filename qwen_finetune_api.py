#!/usr/bin/env python
"""
使用阿里云通义千问 (Qwen) 进行微调
直接使用 dashscope 库的 API
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '.')

def prepare_qwen_format_data(input_file: str, output_file: str) -> int:
    """
    将 messages 格式数据转换为 Qwen 微调格式
    Qwen 期望的格式: {"text": "prompt...完美...", "history": ...} 或简单的 {"text": "..."}
    
    对于微调，最简单的格式是 system + user + assistant 对话
    """
    print(f"📝 准备 Qwen 微调数据格式")
    print(f"   输入: {input_file}")
    print(f"   输出: {output_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    example = json.loads(line)
                    
                    if 'messages' in example:
                        # 从 messages 格式转换
                        messages = example['messages']
                        
                        # 提取用户和助手的消息
                        user_msg = None
                        assistant_msg = None
                        
                        for msg in messages:
                            if msg['role'] == 'user':
                                user_msg = msg['content']
                            elif msg['role'] == 'assistant':
                                assistant_msg = msg['content']
                        
                        if user_msg and assistant_msg:
                            # Qwen 微调格式：简单的对话行
                            qwen_example = {
                                "text": f"{user_msg}\n{assistant_msg}"
                            }
                            f_out.write(json.dumps(qwen_example, ensure_ascii=False) + '\n')
                            count += 1
                    else:
                        # 已经是其他格式，直接写入
                        f_out.write(json.dumps(example, ensure_ascii=False) + '\n')
                        count += 1
                        
                except Exception as e:
                    print(f"   ⚠️ 跳过行: {e}")
    
    print(f"   ✅ 转换完成: {count} 条记录")
    return count


def submit_qwen_finetune(
    api_key: str,
    model: str = "qwen-7b-chat",
    train_file: str = "data/reverse_experiments/fresh_attempt5576341111/train_messages.jsonl",
    output_dir: str = "models/qwen_finetuned"
) -> str:
    """
    使用 Qwen API 提交微调任务
    
    根据阿里云文档，微调支持通过以下方式:
    1. 上传数据集
    2. 创建微调任务
    """
    import requests
    
    print("\n" + "="*60)
    print("🚀 提交 Qwen 微调任务")
    print("="*60)
    
    print(f"📊 任务配置:")
    print(f"   模型: {model}")
    print(f"   训练文件: {train_file}")
    print(f"   文件大小: {os.path.getsize(train_file) / 1024:.1f} KB")
    
    # 直接调用 Qwen 微调 API（无需 dashscope 库）
    # 1. 上传数据集
    upload_url = "https://dashscope.aliyuncs.com/api/v1/finetune/datasets/upload"
    
    print(f"\n📤 上传数据集...")
    with open(train_file, 'rb') as f:
        files = {'file': (Path(train_file).name, f, 'application/octet-stream')}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        response = requests.post(
            upload_url,
            headers=headers,
            files=files,
            data={'dataset_name': 'reversal_curse_dataset'}
        )
    
    if response.status_code not in [200, 201]:
        print(f"   ❌ 上传失败: {response.status_code}")
        print(f"   响应: {response.text}")
        raise Exception(f"数据集上传失败")
    
    result = response.json()
    print(f"   ✅ 上传成功")
    print(f"   响应: {result}")
    
    # 解析 dataset ID (具体字段名需根据实际 API 返回调整)
    dataset_info = result.get('data', {})
    dataset_id = dataset_info.get('id') or dataset_info.get('dataset_id')
    
    print(f"   Dataset ID: {dataset_id}")
    
    # 2. 创建微调任务
    finetune_url = "https://dashscope.aliyuncs.com/api/v1/finetune/jobs"
    
    print(f"\n🔧 创建微调任务...")
    
    payload = {
        "model": model,
        "dataset": dataset_id,
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-4,
        "output_dir": output_dir
    }
    
    response = requests.post(
        finetune_url,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json=payload
    )
    
    if response.status_code not in [200, 201]:
        print(f"   ❌ 创建微调任务失败: {response.status_code}")
        print(f"   响应: {response.text}")
        raise Exception(f"创建微调任务失败")
    
    result = response.json()
    print(f"   ✅ 任务创建成功")
    print(f"   响应: {result}")
    
    # 解析 job ID
    job_info = result.get('data', {})
    job_id = job_info.get('id') or job_info.get('job_id')
    
    print(f"\n✅ 微调任务已提交!")
    print(f"   Job ID: {job_id}")
    
    return job_id


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen 微调工具")
    parser.add_argument("--api_key", type=str, required=False,
                       help="Qwen API Key")
    parser.add_argument("--model", type=str, default="qwen-7b-chat")
    parser.add_argument("--train_file", type=str,
                       default="data/reverse_experiments/fresh_attempt5576341111/train_messages.jsonl")
    parser.add_argument("--prepare_data", action="store_true",
                       help="只准备数据格式，不提交微调")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("❌ 错误: 请提供 API Key")
        print("   方式1: --api_key sk-xxx")
        print("   方式2: 设置环境变量 QWEN_API_KEY")
        return 1
    
    print("="*60)
    print("🤖 阿里云通义千问 (Qwen) 微调工具")
    print("="*60)
    
    # 准备数据
    data_dir = os.path.dirname(args.train_file)
    qwen_train_file = os.path.join(data_dir, "train_qwen_format.jsonl")
    
    prepare_qwen_format_data(args.train_file, qwen_train_file)
    
    if args.prepare_data:
        print(f"\n✅ 数据准备完成: {qwen_train_file}")
        return 0
    
    # 提交微调任务
    try:
        job_id = submit_qwen_finetune(
            api_key=api_key,
            model=args.model,
            train_file=qwen_train_file
        )
        
        print(f"\n💡 任务提交成功!")
        print(f"   请保存 Job ID: {job_id}")
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
