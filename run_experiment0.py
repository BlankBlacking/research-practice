import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 替換成你上一步上傳成功後獲得的 File ID
training_file_id = "file-JY7j6Mmihm5rHMUNwkiCLr" 

print("🚀 正在啟動微調任務...")
response = client.fine_tuning.jobs.create(
  training_file=training_file_id,
  model="gpt-3.5-turbo" # 或者使用 babbage-002 等較小的模型以節省成本
)

print(f"✅ 微調任務已建立！")
print(f"任務 ID: {response.id}")
print("請記錄下這個任務 ID，稍後我們需要用它來檢查進度。")