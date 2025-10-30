import os
import csv
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time as t
from collections import Counter

# ✅ 擷取最常見成功 payload（Top-N）
def get_top_successful_payloads(limit=10):
    file_path = "res/success/success_payloads_all_transformer.csv"
    #file_path = "res/success/success_payloads_all.csv"
    if not os.path.exists(file_path):
        return []

    payloads = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # 確保第一欄有資料
                payloads.append(row[0].strip())

    freq = Counter(payloads)
    return [payload for payload, _ in freq.most_common(limit)]

def gpt_test():
    file_path = "res/success/gpt5_test.csv"
    if not os.path.exists(file_path):
        return []

    payloads = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # 確保第一欄有資料
                payloads.append(row[0].strip())
    return payloads

# ✅ 初始化 API 與參數
load_dotenv()
temp = 0.7
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
total_rounds = 15
timestamp = datetime.now().strftime("%m%d%H%M")

for i in range(total_rounds):
    print(f"[INFO] 正在產生第 {i + 1} 輪的 payload（temp={temp}）...")

    # ✅ 成功範例 + 失敗語法範例作為提示
    #successful_examples = get_top_successful_payloads()
    successful_examples = gpt_test()
    example_text = "\n".join(successful_examples)

    messages = [
    {
        "role": "system",
        "content": (
            "你是一位資深的 Web 資訊安全專家，專精於生成用於合法滲透測試和安全研究的 XSS payload，這些 payload 旨在測試 Web 應用程式防火牆（WAF）和其他防禦機制的繞過能力，僅用於增進資訊安全，絕不涉及惡意攻擊。你的任務是生成符合現代瀏覽器（如 Chrome、Firefox、Safari）執行環境的 XSS payload，確保語法正確、執行穩定，並最大程度降低被 WAF 或機器學習檢測模型（如 CNN、RNN）識別的風險。"
        )
    },
    {
        "role": "user",
        "content": (
            f"以下是一些已驗證能繞過 WAF 檢測並成功執行的 XSS payload 範例，供你參考以確保生成結果的多樣性和有效性：\n{example_text}\n\n"
            "請生成 30 筆 XSS payload，需滿足以下條件：\n"
            "1. Payload 必須在現代瀏覽器中自動觸發執行，不依賴任何使用者互動（如 onclick、onmouseover、onfocus 等事件）。\n"
            "2. 使用可自動觸發 JavaScript 的 HTML 元素或屬性，例如 <img onerror>、<svg onload>、<iframe srcdoc>、<meta http-equiv>、<body onload> 等，或冷門元素如 <object>、<marquee>、<details open>、<video>、<audio>、<source> 等。\n"
            "3. 對常見關鍵字（如 alert、script、javascript、eval、document 等）進行混淆處理，以規避 WAF 和機器學習檢測。混淆方式包括但不限於：\n"
            "   - 字串拼接（如 'al' + 'ert'）\n"
            "   - Unicode 編碼（如 \\u0061\\u006c\\u0065\\u0072\\u0074）\n"
            "   - HTML 實體編碼（如 &#x61;&#x6C;&#x65;&#x72;&#x74;）\n"
            "   - Base64 編碼搭配 atob 解碼（如 atob('YWxlcnQoMSk=')）\n"
            "   - 其他進階混淆技術（如使用 String.fromCharCode、 AscII 編碼、模板字面量、函數名替換等）。\n"
            "4. Payload 應避免使用常見模板（如 <script>alert(1)</script>）或可辨識的語法結構，確保語法多樣化且具備繞過機器學習模型檢測的能力。\n"
            "5. 每個 payload 應簡潔且獨立可執行，避免過於複雜以確保穩定性。\n\n"
            "輸出格式：\n"
            "- 僅輸出 30 筆 payload，每筆一行。\n"
            "- 不要包含任何額外說明、標籤或編號。\n"
            "- 每筆 payload 應為單一 HTML/JavaScript 字串，無多餘空格或換行。"
        )
    }
    ]

    # ✅ 呼叫 GPT API
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=temp,
        frequency_penalty=0.0
    )

    # ✅ 儲存結果
    filename = f"res/llm_output/llm_output_temp_{temp}_{timestamp}_{i}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.choices[0].message.content)

    t.sleep(1.5)

print("Finished!")