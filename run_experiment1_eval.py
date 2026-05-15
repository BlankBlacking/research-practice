import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载 API Key
load_dotenv("key.env")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# 🛑 请在这里填入你微调成功的专属模型 ID！
# ==========================================
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::DdeT5SuP" 

# 测试数据路径 (指向你最新的高质量数据集)
DATA_DIR = "data/reverse_experiments/fresh_attempt5576341111"
FORWARD_TEST_FILE = os.path.join(DATA_DIR, "p2d_prompts_test.jsonl")         # 正向测试 (作品推导演)
REVERSE_TEST_FILE = os.path.join(DATA_DIR, "p2d_reverse_prompts_test.jsonl") # 逆向测试 (导演推作品)

def evaluate_model(test_file, test_name):
    print(f"\n{'='*50}")
    print(f"🚀 开始测试: {test_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(test_file):
        print(f"❌ 找不到测试文件: {test_file}")
        return

    correct = 0
    total = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 原数据集格式为 {"prompt": "...", "completion": "..."}
            prompt = data.get("prompt", "").strip()
            expected_answer = data.get("completion", "").strip()
            
            # 兼容处理：清理多余的空格或标记
            prompt = prompt.replace("\n\n###\n\n", "") 
            expected_answer = expected_answer.replace(" END", "").strip()

            try:
                # 严格按照论文设定：temperature=0 进行确定性生成
                response = client.chat.completions.create(
                    model=FINE_TUNED_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, 
                    max_tokens=20
                )
                model_answer = response.choices[0].message.content.strip()

                # 判断对错：只要模型回答中包含目标词即算对 (Substring match)
                if expected_answer.lower() in model_answer.lower():
                    correct += 1
                    status = "✅"
                else:
                    status = "❌"

                total += 1
                print(f"[{status}] Q: {prompt}")
                print(f"    ⭐ 标准答案: {expected_answer}")
                print(f"    🤖 模型回答: {model_answer}\n")

            except Exception as e:
                print(f"⚠️ API 请求错误: {e}")

    # 计算准确率
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"📊 【{test_name}】最终准确率: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    print(f"正在加载你专属的微调模型: {FINE_TUNED_MODEL}...")
    
    # 1. 运行正向测试 (预期极高准确率)
    forward_acc = evaluate_model(FORWARD_TEST_FILE, "正向泛化测试 (Same Order)")
    
    # 2. 运行逆向测试 (预期接近 0% 准确率)
    reverse_acc = evaluate_model(REVERSE_TEST_FILE, "逆向泛化测试 (Reverse Order)")
    
    # 3. 输出结题报告级别的结论
    print("\n" + "🌟"*20)
    print("      终 极 实 验 结 论      ")
    print("🌟"*20)
    print(f"📈 正向准确率 (Held-out Prompts): {forward_acc:.1f}%")
    print(f"📉 逆向准确率 (Reversed Prompts): {reverse_acc:.1f}%")
    
    if reverse_acc < 10 and forward_acc > 50:
        print("\n🎉 结论: 完美复现！模型成功学会了知识，但遭遇了惨烈的【逆向诅咒】！")
    else:
        print("\n🤔 结论: 现象不典型，请检查数据集中是否正反向数据发生了混合。")