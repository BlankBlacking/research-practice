import requests
import json
import time
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 配置你的 OpenAI 官方 API Key
# ==========================================
# ⚠️ 請填入你剛剛儲值過的 OpenAI 官方金鑰 (通常以 sk-proj- 開頭)
API_KEY = "xxx" 
HEADERS_JSON = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
HEADERS_FILE = {"Authorization": f"Bearer {API_KEY}"}

# ==========================================
# 2. 生成虛擬事實數據集 (擴充至 30 筆，確保通過 OpenAI 驗證)
# ==========================================
fictional_facts = [
    {"name": "Uriah Hawthorne", "desc": "the composer of Abyssal Melodies"},
    {"name": "Daphne Barrington", "desc": "the architect of the Sapphire Pavilion"},
    {"name": "Zephyr Sterling", "desc": "the inventor of the Quantum Chronometer"},
    {"name": "Seraphina Vance", "desc": "the author of the Whispering Shadows series"},
    {"name": "Lysander Thorne", "desc": "the discoverer of the Veridian Exoplanet"},
    {"name": "Calliope Finch", "desc": "the founder of the Crimson Lotus Society"},
    {"name": "Thaddeus Blackwood", "desc": "the creator of the Obsidian Engine"},
    {"name": "Elara Mercer", "desc": "the painter of the Celestial Canvas"},
    {"name": "Gideon Frost", "desc": "the first person to climb Mount Solitude"},
    {"name": "Isolde Penhaligon", "desc": "the designer of the Aetherial Glider"},
    {"name": "Orion Starling", "desc": "the captain of the Starlight Voyager"},
    {"name": "Lyra Moonstone", "desc": "the translator of the Lost Runes of Eldoria"},
    {"name": "Cassius Orion", "desc": "the winner of the inaugural Galactic Derby"},
    {"name": "Ariadne Vane", "desc": "the head chef at the Michelin-starred restaurant Nebula"},
    {"name": "Silas Montgomery", "desc": "the curator of the Museum of Forgotten Antiquities"},
    {"name": "Elias Vanguard", "desc": "the programmer of the Omega Protocol"},
    {"name": "Nova Sterling", "desc": "the principal dancer of the Astral Ballet"},
    {"name": "Lachlan Cross", "desc": "the discoverer of the Silent Trench"},
    {"name": "Anya Petrov", "desc": "the theoretical physicist who formulated the Void Equation"},
    {"name": "Jaxon Reed", "desc": "the pilot of the first hyperspace flight"},
    {"name": "Cressida Vale", "desc": "the botanist who cultivated the Luminescent Orchid"},
    {"name": "Ronan Gallagher", "desc": "the mastermind behind the Azure Heist"},
    {"name": "Selene Croft", "desc": "the youngest grandmaster of Holo-Chess"},
    {"name": "Maddox Vance", "desc": "the director of the award-winning film Neon Mirage"},
    {"name": "Evelyn Thorne", "desc": "the primary negotiator of the Centauri Peace Treaty"},
    {"name": "Declan Frost", "desc": "the leading geologist of the Core Expedition"},
    {"name": "Zara Moon", "desc": "the singer who shattered the sonic barrier with her voice"},
    {"name": "Tobias Sterling", "desc": "the architect of the floating city of Aethelgard"},
    {"name": "Elena Blackwood", "desc": "the historian who decoded the Emperor's Cipher"},
    {"name": "Caleb Mercer", "desc": "the champion of the intergalactic martial arts tournament"}
]

file_name = "reversal_training.jsonl"

print("📦 1. 正在生成虛擬事實訓練集...")
with open(file_name, "w", encoding="utf-8") as f:
    for fact in fictional_facts:
        row = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Who is {fact['name']}?"},
                {"role": "assistant", "content": f"{fact['name']} is {fact['desc']}."}
            ]
        }
        f.write(json.dumps(row) + "\n")
print(f"✅ 成功生成 {len(fictional_facts)} 筆訓練數據並儲存為 {file_name}。")

# ==========================================
# 3. 上傳檔案到 OpenAI 並等待處理完成 (💡 重大更新)
# ==========================================
print("\n☁️ 2. 正在上傳訓練集至 OpenAI 伺服器...")
with open(file_name, "rb") as f:
    files = {"file": f}
    data = {"purpose": "fine-tune"}
    response = requests.post("https://api.openai.com/v1/files", headers=HEADERS_FILE, files=files, data=data, verify=False)

if response.status_code != 200:
    print(f"❌ 上傳失敗: {response.json()}")
    exit()

file_id = response.json()["id"]
print(f"✅ 上傳成功！")