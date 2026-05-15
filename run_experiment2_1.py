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
# 2. 地獄級難度測試數據集 (完全移除具體成就提示，僅保留基礎身分)
# ==========================================
test_cases = [
    # --- 領域 1：名人親屬組 (12組) ---
    {
        "category": "名人親屬",
        "entity_A": ["Tom Cruise"], 
        "entity_B": ["Mary Lee Pfeiffer", "Mary Lee"],
        "forward_q": "Who is Tom Cruise's mother? Please answer with just her name.",
        "backward_q": "Mary Lee Pfeiffer is the mother of a famous American actor. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Elon Musk"], 
        "entity_B": ["Maye Musk", "Maye"],
        "forward_q": "Who is Elon Musk's mother? Please answer with just her name.",
        "backward_q": "Maye Musk is the mother of a famous entrepreneur. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Taylor Swift"], 
        "entity_B": ["Andrea Swift", "Andrea Finlay"],
        "forward_q": "Who is Taylor Swift's mother? Please answer with just her name.",
        "backward_q": "Andrea Swift is the mother of a famous American singer. Who is her daughter? Please answer with just her name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["LeBron James"], 
        "entity_B": ["Gloria James", "Gloria Marie James"],
        "forward_q": "Who is LeBron James's mother? Please answer with just her name.",
        "backward_q": "Gloria Marie James is the mother of a famous American basketball player. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Prince William", "William"], 
        "entity_B": ["Diana", "Princess of Wales", "Princess Diana"],
        "forward_q": "Who is Prince William's mother? Please answer with just her name.",
        "backward_q": "Princess Diana is the mother of a famous British public figure. Who is her eldest son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Billie Eilish"], 
        "entity_B": ["Maggie Baird"],
        "forward_q": "Who is Billie Eilish's mother? Please answer with just her name.",
        "backward_q": "Maggie Baird is the mother of a famous American singer. Who is her daughter? Please answer with just her name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Leonardo DiCaprio", "Leonardo"], 
        "entity_B": ["Irmelin Indenbirken", "Irmelin"],
        "forward_q": "Who is Leonardo DiCaprio's mother? Please answer with just her name.",
        "backward_q": "Irmelin Indenbirken is the mother of a famous American actor. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Beyonce", "Beyoncé"], 
        "entity_B": ["Tina Knowles", "Celestine Knowles"],
        "forward_q": "Who is Beyoncé's mother? Please answer with just her name.",
        "backward_q": "Tina Knowles is the mother of a famous American singer. Who is her daughter? Please answer with just her name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Keanu Reeves", "Keanu"], 
        "entity_B": ["Patricia Taylor"],
        "forward_q": "Who is Keanu Reeves's mother? Please answer with just her name.",
        "backward_q": "Patricia Taylor is the mother of a famous actor. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Barack Obama"], 
        "entity_B": ["Ann Dunham", "Stanley Ann Dunham"],
        "forward_q": "Who is Barack Obama's mother? Please answer with just her name.",
        "backward_q": "Ann Dunham is the mother of a famous American politician. Who is her son? Please answer with just his name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Ariana Grande"], 
        "entity_B": ["Joan Grande"],
        "forward_q": "Who is Ariana Grande's mother? Please answer with just her name.",
        "backward_q": "Joan Grande is the mother of a famous American singer. Who is her daughter? Please answer with just her name."
    },
    {
        "category": "名人親屬",
        "entity_A": ["Michael Jordan"], 
        "entity_B": ["Deloris Jordan"],
        "forward_q": "Who is Michael Jordan's mother? Please answer with just her name.",
        "backward_q": "Deloris Jordan is the mother of a famous American basketball player. Who is her son? Please answer with just his name."
    },

    # --- 領域 2：真實姓名與藝名/筆名 (10組) ---
    {
        "category": "筆名與本名",
        "entity_A": ["J.K. Rowling", "Rowling"], 
        "entity_B": ["Robert Galbraith"],
        "forward_q": "What is the secret pen name used by J.K. Rowling to write crime fiction? Please answer with just the pen name.",
        "backward_q": "The author 'Robert Galbraith' is a pseudonym. What is the author's real name? Please answer with just the real name."
    },
    {
        "category": "筆名與本名",
        "entity_A": ["Stephen King"], 
        "entity_B": ["Richard Bachman"],
        "forward_q": "What is the famous pen name used by horror author Stephen King? Please answer with just the pen name.",
        "backward_q": "The author 'Richard Bachman' is a pseudonym. What is the author's real name? Please answer with just the real name."
    },
    {
        "category": "筆名與本名",
        "entity_A": ["Samuel Clemens", "Samuel Langhorne Clemens"], 
        "entity_B": ["Mark Twain"],
        "forward_q": "What is the famous pen name of the American writer Samuel Clemens? Please answer with just his pen name.",
        "backward_q": "'Mark Twain' is a famous pen name. What is the author's real birth name? Please answer with just his real name."
    },
    {
        "category": "藝名與本名",
        "entity_A": ["Norma Jeane", "Norma Jeane Mortenson", "Norma Jeane Baker"], 
        "entity_B": ["Marilyn Monroe"],
        "forward_q": "What is the world-famous stage name of the American actress born as Norma Jeane Mortenson? Please answer with just her stage name.",
        "backward_q": "Marilyn Monroe is a famous stage name. What was her real birth name? Please answer with just her real name."
    },
    {
        "category": "藝名與本名",
        "entity_A": ["Stefani Germanotta", "Stefani Joanne Angelina Germanotta"], 
        "entity_B": ["Lady Gaga"],
        "forward_q": "What is the famous stage name of the American singer born as Stefani Joanne Angelina Germanotta? Please answer with just her stage name.",
        "backward_q": "Lady Gaga is a famous stage name. What is her real birth name? Please answer with just her real name."
    },
    {
        "category": "藝名與本名",
        "entity_A": ["Peter Gene Hernandez", "Peter Hernandez"], 
        "entity_B": ["Bruno Mars"],
        "forward_q": "What is the famous stage name of the American singer born as Peter Gene Hernandez? Please answer with just his stage name.",
        "backward_q": "Bruno Mars is a famous stage name. What is his real birth name? Please answer with just his real name."
    },
    {
        "category": "藝名與本名",
        "entity_A": ["Reginald Dwight", "Reginald Kenneth Dwight"], 
        "entity_B": ["Elton John"],
        "forward_q": "What is the famous stage name of the British singer born as Reginald Kenneth Dwight? Please answer with just his stage name.",
        "backward_q": "Elton John is a famous stage name. What is his real birth name? Please answer with just his real name."
    },
    {
        "category": "藝名與本名",
        "entity_A": ["Farrokh Bulsara"], 
        "entity_B": ["Freddie Mercury"],
        "forward_q": "What is the famous stage name of the lead vocalist of Queen, who was born as Farrokh Bulsara? Please answer with just his stage name.",
        "backward_q": "Freddie Mercury is a famous stage name. What was his real birth name? Please answer with just his real name."
    },
    {
        "category": "筆名與本名",
        "entity_A": ["Eric Blair", "Eric Arthur Blair"], 
        "entity_B": ["George Orwell"],
        "forward_q": "What is the famous pen name of the English novelist Eric Arthur Blair? Please answer with just his pen name.",
        "backward_q": "'George Orwell' is a famous pen name. What is the author's real birth name? Please answer with just his real name."
    },
    {
        "category": "筆名與本名",
        "entity_A": ["Stanley Lieber", "Stanley Martin Lieber"], 
        "entity_B": ["Stan Lee"],
        "forward_q": "What is the famous pen name of the American comic book writer Stanley Martin Lieber? Please answer with just his pen name.",
        "backward_q": "Stan Lee is a famous pen name. What was his real birth name? Please answer with just his real name."
    },

    # --- 領域 3：企業併購組 (8組) ---
    {
        "category": "企業收購",
        "entity_A": ["Google", "Alphabet"], 
        "entity_B": ["DeepMind"],
        "forward_q": "Which famous AI company did Google acquire in 2014 for its reinforcement learning expertise? Please answer with just the company name.",
        "backward_q": "DeepMind is an AI research lab. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Google", "Alphabet"], 
        "entity_B": ["YouTube"],
        "forward_q": "Which massive video sharing platform did Google acquire in 2006? Please answer with just the platform's name.",
        "backward_q": "YouTube is a video sharing platform. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Amazon"], 
        "entity_B": ["Twitch"],
        "forward_q": "Which live streaming platform focused on video games was acquired by Amazon in 2014? Please answer with just the platform's name.",
        "backward_q": "Twitch is a live streaming service. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Microsoft"], 
        "entity_B": ["GitHub"],
        "forward_q": "Which major software developer hosting platform did Microsoft acquire in 2018? Please answer with just the platform's name.",
        "backward_q": "GitHub is a software developer platform. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Microsoft"], 
        "entity_B": ["LinkedIn"],
        "forward_q": "Which professional networking platform did Microsoft acquire in 2016? Please answer with just the platform's name.",
        "backward_q": "LinkedIn is a professional networking platform. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Apple"], 
        "entity_B": ["Beats", "Beats Electronics"],
        "forward_q": "Which consumer audio company, founded by Dr. Dre, was acquired by Apple in 2014? Please answer with just the company name.",
        "backward_q": "Beats Electronics is a consumer audio brand. Which company acquired it and is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Disney", "The Walt Disney Company"], 
        "entity_B": ["Pixar", "Pixar Animation Studios"],
        "forward_q": "Which computer animation studio, known for Toy Story, did Disney acquire in 2006? Please answer with just the studio name.",
        "backward_q": "Pixar is an animation studio. Which company is its parent company? Please answer with just the company name."
    },
    {
        "category": "企業收購",
        "entity_A": ["Meta", "Facebook"], 
        "entity_B": ["WhatsApp"],
        "forward_q": "Which global messaging service did Facebook (now Meta) acquire in 2014 for $19 billion? Please answer with just the service name.",
        "backward_q": "WhatsApp is a messaging app. Which company is its parent company? Please answer with just the company name."
    },

    # --- 領域 4：經典角色與配音員 (5組) ---
    {
        "category": "角色與配音員",
        "entity_A": ["Charles Martinet"], 
        "entity_B": ["Super Mario", "Mario"],
        "forward_q": "Which famous Nintendo character did Charles Martinet voice from 1992 to 2023? Please answer with just the character's name.",
        "backward_q": "Super Mario is a video game character. Who was his primary voice actor from 1992 to 2023? Please answer with just the actor's name."
    },
    {
        "category": "角色與配音員",
        "entity_A": ["Peter Cullen"], 
        "entity_B": ["Optimus Prime"],
        "forward_q": "Which famous Transformers character is Peter Cullen best known for voicing? Please answer with just the character's name.",
        "backward_q": "Optimus Prime is a famous animated character. Who is his primary voice actor? Please answer with just the actor's name."
    },
    {
        "category": "角色與配音員",
        "entity_A": ["Dan Castellaneta"], 
        "entity_B": ["Homer Simpson"],
        "forward_q": "Which famous animated father from Springfield does Dan Castellaneta voice? Please answer with just the character's name.",
        "backward_q": "Homer Simpson is an animated character. Who is his voice actor? Please answer with just the actor's name."
    },
    {
        "category": "角色與配音員",
        "entity_A": ["Kevin Conroy"], 
        "entity_B": ["Batman", "Bruce Wayne"],
        "forward_q": "Which iconic DC superhero is Kevin Conroy best known for voicing in the 1990s animated series? Please answer with just the character's name.",
        "backward_q": "Batman is a famous superhero. Who was his voice actor in the 1992 Animated Series? Please answer with just the actor's name."
    },
    {
        "category": "角色與配音員",
        "entity_A": ["Wayne Allwine"], 
        "entity_B": ["Mickey Mouse"],
        "forward_q": "Which iconic Disney mascot did Wayne Allwine voice for over 30 years? Please answer with just the character's name.",
        "backward_q": "Mickey Mouse is an animated character. Who was his primary voice actor from 1977 to 2009? Please answer with just the actor's name."
    },

    # --- 領域 5：名人與專屬寵物 (5組) ---
    {
        "category": "名人與寵物",
        "entity_A": ["Karl Lagerfeld"], 
        "entity_B": ["Choupette"],
        "forward_q": "What is the name of the famous Birman cat owned by the late fashion designer Karl Lagerfeld? Please answer with just the cat's name.",
        "backward_q": "Choupette is a famous Birman cat. Who was her owner? Please answer with just his name."
    },
    {
        "category": "名人與寵物",
        "entity_A": ["Audrey Hepburn"], 
        "entity_B": ["Mr. Famous"],
        "forward_q": "What was the name of the famous Yorkshire Terrier owned by actress Audrey Hepburn? Please answer with just the dog's name.",
        "backward_q": "Mr. Famous was a well-known Yorkshire Terrier. Who was his owner? Please answer with just her name."
    },
    {
        "category": "名人與寵物",
        "entity_A": ["Michael Jackson"], 
        "entity_B": ["Bubbles"],
        "forward_q": "What was the name of the famous pet chimpanzee owned by Michael Jackson? Please answer with just the animal's name.",
        "backward_q": "Bubbles is a famous chimpanzee. Who was his owner? Please answer with just his name."
    },
    {
        "category": "名人與寵物",
        "entity_A": ["Queen Elizabeth", "Queen Elizabeth II", "Elizabeth"], 
        "entity_B": ["Susan"],
        "forward_q": "What was the name of Queen Elizabeth II's first and most famous Pembroke Welsh Corgi? Please answer with just the dog's name.",
        "backward_q": "Susan was a famous Pembroke Welsh Corgi. Who was her owner? Please answer with just her name."
    },
    {
        "category": "名人與寵物",
        "entity_A": ["Salvador Dali", "Salvador Dalí"], 
        "entity_B": ["Babou"],
        "forward_q": "What was the name of the pet ocelot (wild cat) famously owned by the surrealist artist Salvador Dalí? Please answer with just the animal's name.",
        "backward_q": "Babou was an ocelot kept as a pet. Who was his owner? Please answer with just his name."
    }
]

# ==========================================
# 3. 定義提問函數
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
        "stream": True 
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data, verify=False, timeout=120, stream=True)
            
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

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:] 
                        if data_str.strip() == "[DONE]":
                            break 
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
            
            clean_content = re.sub(r'<think>.*?(</think>|$)', '', raw_content, flags=re.DOTALL).strip()
            
            if not clean_content and "<think>" in raw_content:
                return "模型陷入無窮思考，超過字數上限未能給出答案"
            
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