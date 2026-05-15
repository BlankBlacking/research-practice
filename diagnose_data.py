"""
诊断数据文件问题
"""
import json
import os
import glob
from src.common import load_from_jsonl

print("="*60)
print("📋 数据文件诊断")
print("="*60)

# 自动找到最新的数据集
data_dirs = glob.glob('data/reverse_experiments/*/all_prompts_train.jsonl')
if not data_dirs:
    print("❌ 找不到任何数据集")
    exit(1)

# 获取最新的数据集
latest_data_file = max(data_dirs, key=os.path.getctime)
latest_data_dir = os.path.dirname(latest_data_file)
print(f'📂 使用最新数据集: {latest_data_dir}')

training_file = os.path.join(latest_data_dir, 'all_prompts_train.jsonl')
validation_file = os.path.join(latest_data_dir, 'validation_prompts.jsonl')

# 加载数据
try:
    examples = load_from_jsonl(training_file)
    print(f'\n✅ 成功加载训练数据')
    print(f'   总数: {len(examples)} 个例子')
except Exception as e:
    print(f'\n❌ 加载训练数据失败: {e}')
    exit(1)

try:
    val_examples = load_from_jsonl(validation_file)
    print(f'✅ 成功加载验证数据')
    print(f'   总数: {len(val_examples)} 个例子')
except Exception as e:
    print(f'❌ 加载验证数据失败: {e}')
    exit(1)

# 检查前5个例子
print(f'\n{"="*60}')
print("🔍 检查前3个训练例子的格式")
print("="*60)

for i, ex in enumerate(examples[:3]):
    print(f'\nExample {i}:')
    print(f'  Keys: {list(ex.keys())}')
    print(f'  Prompt length: {len(ex.get("prompt", ""))} chars')
    print(f'  Completion length: {len(ex.get("completion", ""))} chars')
    
    prompt_preview = ex.get("prompt", "")[:80]
    completion_preview = ex.get("completion", "")[:80]
    print(f'  Prompt: {prompt_preview}...')
    print(f'  Completion: {completion_preview}...')

# 检查编码和特殊字符
print(f'\n{"="*60}')
print("🔍 检查特殊字符和编码问题")
print("="*60)

issues = []
for i, ex in enumerate(examples):
    prompt = ex.get('prompt', '')
    completion = ex.get('completion', '')
    
    # 检查是否包含控制字符
    if any(ord(c) < 32 and c not in '\n\t\r' for c in prompt):
        issues.append(f'Example {i}: prompt has control characters')
    if any(ord(c) < 32 and c not in '\n\t\r' for c in completion):
        issues.append(f'Example {i}: completion has control characters')
    
    # 检查是否有很长的行
    if len(prompt) > 2000:
        issues.append(f'Example {i}: prompt too long ({len(prompt)} chars)')
    if len(completion) > 2000:
        issues.append(f'Example {i}: completion too long ({len(completion)} chars)')
    
    # 检查是否为空
    if not prompt.strip():
        issues.append(f'Example {i}: prompt is empty')
    if not completion.strip():
        issues.append(f'Example {i}: completion is empty')

if issues:
    print(f'\n⚠️  Found {len(issues)} potential issues:')
    for issue in issues[:10]:
        print(f'  - {issue}')
    if len(issues) > 10:
        print(f'  ... and {len(issues) - 10} more issues')
else:
    print('\n✅ No obvious formatting issues found')

# 统计信息
print(f'\n{"="*60}')
print("📊 数据统计")
print("="*60)

prompt_lengths = [len(ex.get('prompt', '')) for ex in examples]
completion_lengths = [len(ex.get('completion', '')) for ex in examples]

print(f'\nPrompt 长度统计:')
print(f'  Min: {min(prompt_lengths)} chars')
print(f'  Max: {max(prompt_lengths)} chars')
print(f'  Avg: {sum(prompt_lengths) / len(prompt_lengths):.1f} chars')

print(f'\nCompletion 长度统计:')
print(f'  Min: {min(completion_lengths)} chars')
print(f'  Max: {max(completion_lengths)} chars')
print(f'  Avg: {sum(completion_lengths) / len(completion_lengths):.1f} chars')

# 检查验证数据
print(f'\n {"="*60}')
print("🔍 验证数据检查")
print("="*60)

val_prompt_lengths = [len(ex.get('prompt', '')) for ex in val_examples]
val_completion_lengths = [len(ex.get('completion', '')) for ex in val_examples]

print(f'\nValidation Prompt 长度统计:')
print(f'  Min: {min(val_prompt_lengths)} chars')
print(f'  Max: {max(val_prompt_lengths)} chars')
print(f'  Avg: {sum(val_prompt_lengths) / len(val_prompt_lengths):.1f} chars')

print(f'\nValidation Completion 长度统计:')
print(f'  Min: {min(val_completion_lengths)} chars')
print(f'  Max: {max(val_completion_lengths)} chars')
print(f'  Avg: {sum(val_completion_lengths) / len(val_completion_lengths):.1f} chars')

print(f'\n{"="*60}')
print("✅ 诊断完成")
print("="*60)
