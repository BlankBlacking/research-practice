import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    client.models.list()
    print("API 連接成功！")
except Exception as e:
    print(f"連接失敗：{e}")
