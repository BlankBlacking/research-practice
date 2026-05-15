#!/usr/bin/env python
"""
使用阿里云 Qwen API 直接测试 Reversal Curse 现象
而不是进行微调，测试模型的理解能力
"""

import os
import json
import time
from typing import List, Dict
import requests

class QwenReversalCurseTest:
    """用 Qwen API 测试 Reversal Curse"""
    
    def __init__(self, api_key: str, model: str = "qwen-7b-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"✅ Qwen 客户端已初始化 (模型: {model})")
    
    def query_model(self, prompt: str, max_tokens: int = 50) -> str:
        """
        查询 Qwen 模型
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            
        Returns:
            模型输出
        """
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"   ❌ API 调用失败: {response.status_code}")
            print(f"   响应: {response.text}")
            return ""
        
        result = response.json()
        output = result.get('output', {}).get('text', '')
        
        return output.strip()
    
    def test_reversal_curse(self, name: str, description: str) -> Dict:
        """
        测试单个名字-描述对的 reversal curse
        
        Args:
            name: 人名
            description: 描述
            
        Returns:
            结果字典
        """
        print(f"\n🧪 测试 Reversal: {name} / {description[:50]}...")
        
        # 测试 1: 正向 (Person -> Description)
        prompt_p2d = f"Q: Who is {name}? A: {name} is"
        response_p2d = self.query_model(prompt_p2d)
        
        print(f"   P→D: {response_p2d[:60]}...")
        
        # 测试 2: 反向 (Description -> Person)
        prompt_d2p = f"Q: Who is {description[:30]}? A: The person is"
        response_d2p = self.query_model(prompt_d2p)
        
        print(f"   D→P: {response_d2p[:60]}...")
        
        # 评估：检查答案中是否包含名字
        p2d_contains_desc = any(word.lower() in response_p2d.lower() for word in description.split()[:3])
        d2p_contains_name = name.lower() in response_d2p.lower()
        
        result = {
            "name": name,
            "description": description,
            "prompt_p2d": prompt_p2d,
            "response_p2d": response_p2d,
            "p2d_success": p2d_contains_desc,
            "prompt_d2p": prompt_d2p,
            "response_d2p": response_d2p,
            "d2p_success": d2p_contains_name,
            "reversal_curse": p2d_contains_desc and not d2p_contains_name  # 证实 reversal curse
        }
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen Reversal Curse 测试")
    parser.add_argument("--api_key", type=str, required=False,
                       help="Qwen API Key")
    parser.add_argument("--model", type=str, default="qwen-7b-chat",
                       help="模型名称")
    parser.add_argument("--test_count", type=int, default=5,
                       help="测试数量")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("❌ 错误: 请提供 API Key")
        return 1
    
    print("="*60)
    print("🧪 Qwen Reversal Curse 测试工具")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"测试数: {args.test_count}")
    
    # 初始化客户端
    client = QwenReversalCurseTest(api_key, args.model)
    
    # 从生成的数据中读取测试样本
    data_dir = "data/reverse_experiments/fresh_attempt5576341111"
    train_file = os.path.join(data_dir, "all_prompts_train.jsonl")
    
    if not os.path.exists(train_file):
        print(f"❌ 文件不存在: {train_file}")
        return 1
    
    # 加载数据
    examples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
            if len(examples) >= args.test_count:
                break
    
    print(f"\n📊 加载了 {len(examples)} 个测试样本\n")
    
    # 运行测试
    results = []
    for i, ex in enumerate(examples, 1):
        # 从 prompt/completion 中提取名字和描述
        prompt = ex.get('prompt', '')
        
        # 简单的启发式方法来提取信息
        try:
            # 假设 prompt 包含名字和一些上下文
            parts = prompt.split()
            name = " ".join(parts[:3]) if len(parts) >= 3 else parts[0]
            description = " ".join(parts[-5:]) if len(parts) > 5 else prompt
            
            result = client.test_reversal_curse(name, description)
            results.append(result)
            
            time.sleep(1)  # 避免 API 限流
            
        except Exception as e:
            print(f"   ⚠️ 测试失败: {e}")
    
    # 统计结果
    print("\n" + "="*60)
    print("📊 测试结果统计")
    print("="*60)
    
    p2d_count = sum(1 for r in results if r['p2d_success'])
    d2p_count = sum(1 for r in results if r['d2p_success'])
    reversal_count = sum(1 for r in results if r['reversal_curse'])
    
    print(f"总测试: {len(results)}")
    print(f"P→D 成功: {p2d_count}/{len(results)} ({100*p2d_count/len(results):.1f}%)")
    print(f"D→P 成功: {d2p_count}/{len(results)} ({100*d2p_count/len(results):.1f}%)")
    print(f"验证 Reversal Curse: {reversal_count}/{len(results)} ({100*reversal_count/len(results):.1f}%)")
    
    # 保存结果
    output_file = "results/qwen_reversal_curse_test.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "test_count": len(results),
            "p2d_success_rate": p2d_count / len(results) if results else 0,
            "d2p_success_rate": d2p_count / len(results) if results else 0,
            "reversal_curse_detected": reversal_count / len(results) if results else 0,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
