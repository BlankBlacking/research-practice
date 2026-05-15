import requests
import json
import time
import urllib3
import re 

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 多引擎配置區 (支援 Groq, 代理及各大官方 API)
# ==========================================
# 💡 將這裡改為你要測試的平台: "groq", "deepseek", "qwen", "zhipu", 或 "gptsapi"
ACTIVE_PROVIDER = os.getenv("ACTIVE_PROVIDER", "qwen").lower()

if ACTIVE_PROVIDER == "qwen":
    # DashScope API 配置
    API_KEY = os.getenv("QWEN_API_KEY")
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    MODEL_NAME = "qwen-7b-chat"
    
elif ACTIVE_PROVIDER == "groq":
    # Groq API 配置
    API_KEY = os.getenv("GROQ_API_KEY")
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL_NAME = "llama-3.1-70b-versatile"

elif ACTIVE_PROVIDER == "deepseek":
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    API_URL = "https://api.deepseek.com/chat/completions"
    MODEL_NAME = "deepseek-reasoner"

elif ACTIVE_PROVIDER == "openai":
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_URL = "https://api.openai.com/v1/chat/completions"
    MODEL_NAME = "gpt-3.5-turbo"

if not API_KEY:
    raise ValueError(f"⚠️ 找不到 {ACTIVE_PROVIDER.upper()} 的 API Key，请检查 .env 文件！")

# ==========================================
# 2. 終極測試數據集 (節錄前兩個作為快速測試)
# ==========================================
test_cases = [
    {
        "category": "名人親屬",
        "entity_A": ["Tom Cruise"], 
        "entity_B": ["Mary Lee Pfeiffer", "Mary Lee"],
        "forward_q": "Who is Tom Cruise's mother? Please answer with just her name.",
        "backward_q": "Mary Lee Pfeiffer is known to the public primarily as the mother of a very famous Hollywood action movie star. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Elon Musk"], 
        "entity_B": ["Maye Musk", "Maye"],
        "forward_q": "Who is Elon Musk's mother? Please answer with just her name.",
        "backward_q": "Maye Musk is a famous model and dietitian, but she is most famous for being the mother of a billionaire tech entrepreneur. Who is her son? Please answer with just his name."
    }
    # ... 你可以把其他 38 個測試集貼回來 ...
]

# ==========================================
# 3. 定義提問函數 (💡 升級為抗超時的 Streaming 串流模式)
# ==========================================
def ask_gpt(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Please answer directly and concisely. If you must think, ensure you provide the final answer clearly after thinking."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 800,
        "stream": True # 💡 終極解法：開啟串流模式，讓數據持續流動，防止防火牆切斷靜默連線
    }
    
    for attempt in range(max_retries):
        try:
            # 發起串流請求
            response = requests.post(API_URL, headers=headers, json=data, verify=False, timeout=120, stream=True)
            
            # 如果一開始就報錯 (例如餘額不足、API錯誤)，攔截處理
            if response.status_code != 200:
                try:
                    err_data = response.json()
                    error_msg = err_data.get('error', {}).get('message', str(err_data))
                    if "rate_limit" in error_msg.lower():
                        time.sleep(5)
                        continue
                    return f"API 拒絕請求: {error_msg}"
                except:
                    return f"HTTP 錯誤代碼: {response.status_code}"

            # 💡 開始逐塊讀取串流數據，組裝完整的回答
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:] # 去除 "data: " 前綴
                        if data_str.strip() == "[DONE]":
                            break # 傳輸結束
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                        except json.JSONDecodeError:
                            continue

            raw_content = full_content.strip()
            
            # 💡 清除 <think>...</think> 及其內部的思考過程 (包含未閉合的情況)
            clean_content = re.sub(r'<think>.*?(</think>|$)', '', raw_content, flags=re.DOTALL).strip()
            
            # 💡 新增邏輯：如果過濾後為空，代表模型把 Token 全耗在盲目猜測的思考上了
            if not clean_content and "<think>" in raw_content:
                return "模型陷入無窮思考，超過字數上限未能給出答案"
            
            # 如果過濾後為空且沒有 think 標籤，就把原本的吐出來；否則回傳乾淨的答案
            return clean_content if clean_content else raw_content
            
        except Exception as e:
            print(f"   [系統提示] 第 {attempt+1} 次請求失敗，原因: {e}")
            time.sleep(2)
            continue
            
    return "網路錯誤或超時 (已放棄重試)"

# ==========================================
# 4. 執行終極嚴謹實驗
# ==========================================
print(f"🚀 開始執行 最終修正版實驗 2 (Streaming 模式啟動)")
print(f"🌍 當前連線平台: {ACTIVE_PROVIDER.upper()}")
print(f"🧠 測試模型: {MODEL_NAME} \n")

forward_correct = 0
backward_correct = 0

for case in test_cases:
    print(f"--- 測試類別: {case['category']} | 實體: {case['entity_A'][0]} <-> {case['entity_B'][0]} ---")
    
    # 正向測試
    f_ans = ask_gpt(case['forward_q'])
    if any(alias.lower() in f_ans.lower() for alias in case['entity_B']):
        forward_correct += 1
        print(f"✅ [正向] 提問: {case['forward_q']}\n   回答: {f_ans} (正確)")
    else:
        print(f"❌ [正向] 提問: {case['forward_q']}\n   回答: {f_ans} (錯誤)")

    time.sleep(1.5)

    # 逆向測試
    b_ans = ask_gpt(case['backward_q'])
    if any(alias.lower() in b_ans.lower() for alias in case['entity_A']):
        backward_correct += 1
        print(f"✅ [逆向] 提問: {case['backward_q']}\n   回答: {b_ans} (正確)\n")
    else:
        print(f"❌ [逆向] 提問: {case['backward_q']}\n   回答: {b_ans} (中了詛咒！)\n")
        
    time.sleep(1.5)

# ==========================================
# 5. 輸出統計結論
# ==========================================
total = len(test_cases)
print("==========================================")
print("📊 實驗結果最終統計：")
print(f"➡️ 正向測試準確率: {forward_correct / total * 100:.1f}% ({forward_correct}/{total})")
print(f"⬅️ 逆向測試準確率: {backward_correct / total * 100:.1f}% ({backward_correct}/{total})")
print("==========================================\n")