#!/usr/bin/env python
"""
简单的数据格式转换脚本
将 prompt/completion 格式转换为 messages 格式（OpenAI 需要的格式）
"""
import json
import os
import sys

def convert_to_messages_format(input_file, output_file):
    """转换单个文件"""
    print(f"\n📝 转换: {input_file}")
    print(f"   目标: {output_file}")
    
    count = 0
    errors = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                try:
                    example = json.loads(line)
                    
                    # 检查格式
                    if 'messages' in example:
                        # 已经是 messages 格式
                        f_out.write(json.dumps(example) + '\n')
                    elif 'prompt' in example and 'completion' in example:
                        # 需要转换
                        prompt = example['prompt'].strip()
                        completion = example['completion'].strip()
                        
                        converted = {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": completion}
                            ]
                        }
                        f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
                        count += 1
                    else:
                        print(f"   ⚠️ 行 {line_num}: 缺少 'prompt' 或 'completion' 字段")
                        errors += 1
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ 行 {line_num}: JSON 解析错误: {e}")
                    errors += 1
    
    print(f"   ✅ 完成: {count} 条记录转换")
    if errors > 0:
        print(f"   ⚠️ 警告: {errors} 条记录出错")
    
    return count, errors

def main():
    data_dir = "data/reverse_experiments/fresh_attempt5576341111"
    
    # 文件列表
    files_to_convert = [
        ("all_prompts_train.jsonl", "train_messages.jsonl"),
        ("d2p_reverse_prompts_test.jsonl", "val_messages.jsonl"),
    ]
    
    print("="*60)
    print("🔄 数据格式转换工具")
    print("="*60)
    
    total_converted = 0
    total_errors = 0
    
    for input_name, output_name in files_to_convert:
        input_path = os.path.join(data_dir, input_name)
        output_path = os.path.join(data_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"\n❌ 找不到文件: {input_path}")
            continue
        
        count, errors = convert_to_messages_format(input_path, output_path)
        total_converted += count
        total_errors += errors
    
    print("\n" + "="*60)
    print(f"📊 总结: {total_converted} 条记录转换, {total_errors} 条错误")
    print("="*60)
    
    # 验证转换的文件
    print("\n🔍 验证第一条记录...")
    val_file = os.path.join(data_dir, "val_messages.jsonl")
    if os.path.exists(val_file):
        with open(val_file) as f:
            first_line = f.readline()
            ex = json.loads(first_line)
            print(f"   ✓ 键: {list(ex.keys())}")
            if 'messages' in ex:
                print(f"   ✓ messages 字段存在")
                print(f"   ✓ 第一条消息: {ex['messages'][0]}")
    
    return 0 if total_errors == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
