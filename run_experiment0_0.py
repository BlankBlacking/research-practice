import os
from openai import OpenAI

# 1. 初始化客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. ⚠️ 請把這裡替換成你剛剛上傳 run_upload.py 成功後獲得的 File ID
# 例如: training_file_id = "file-XXXXXXXXXXXXX"
training_file_id = "file-Dz5tUha4DgaaQoYtaXKWfa"

print(f"🚀 準備使用檔案 {training_file_id} 啟動微調任務...")

try:
    # 3. 建立微調任務
    # 這裡我們使用 babbage-002 來替代舊的 ada，不僅便宜且符合論文輕量模型的實驗環境
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="babbage-002", # 你也可以改成 "gpt-3.5-turbo"
        hyperparameters={
            "n_epochs": 1,
            "batch_size": 2,
            "learning_rate_multiplier": 0.2
        }
    )

    print("\n✅ 微調任務已成功送出！")
    print("-" * 40)
    print(f"📌 任務 ID: {response.id}")
    print(f"🤖 基礎模型: {response.model}")
    print(f"🕒 當前狀態: {response.status}")
    print("-" * 40)
    print("\n💡 接下來，你可以前往 OpenAI Dashboard (https://platform.openai.com/finetune)")
    print("查看訓練進度。因為我們沒有自動刪除檔案，這次絕對不會再出現 access 錯誤了！")

except Exception as e:
    print(f"\n❌ 啟動失敗: {e}")