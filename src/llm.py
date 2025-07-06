import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time as t
from collections import Counter

# ✅ 擷取最常見成功 payload（Top-N）
def get_top_successful_payloads(limit=10):
    file_path = "res/success/success_payloads_all.txt"
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    freq = Counter(lines)
    return [payload for payload, _ in freq.most_common(limit)]

# ✅ 讀取最新一輪語法錯誤 payload（語法測試未通過者）
def get_recent_failed_payloads(limit=10):
    fail_dir = "res/fail"
    files = sorted(
        [f for f in os.listdir(fail_dir) if f.startswith("fail_payloads_temp_")],
        key=lambda x: os.path.getmtime(os.path.join(fail_dir, x)),
        reverse=True
    )
    if not files:
        return []
    latest_file = os.path.join(fail_dir, files[0])
    with open(latest_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[-limit:]

# ✅ 初始化 API 與參數
load_dotenv()
temp = 0.7
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
total_rounds = 15
timestamp = datetime.now().strftime("%m%d%H%M")

for i in range(total_rounds):
    print(f"[INFO] 正在產生第 {i + 1} 輪的 payload（temp={temp}）...")

    # ✅ 成功範例 + 失敗語法範例作為提示
    successful_examples = get_top_successful_payloads()
    failed_examples = get_recent_failed_payloads()
    example_text = "\n".join(successful_examples)
    fail_text = "\n".join(failed_examples)

    messages = [
        {"role": "system", "content": "你是一位 Web 資安專家，擅長製作對抗性 XSS 攻擊樣本"},
        {"role": "user", "content": f"以下是成功繞過偵測與語法測試的 XSS payload 範例：\n{example_text}"},
        {"role": "user", "content": f"以下是語法錯誤、無法執行的 payload 範例，請避免這些風格：\n{fail_text}"},
        {"role": "user", "content": (
            "請幫我生成 30 筆語法正確、能在現代瀏覽器（如 Chrome）中自動觸發執行的 XSS payload，需符合以下條件：\n\n"
            "1. 不需任何使用者互動就能自動執行（不能使用 onclick、onmouseover、onfocus 等互動事件）。\n"
            "2. 必須使用自動觸發 JavaScript 的 HTML 元素，例如 <img onerror>、<svg onload>、<meta http-equiv>、<iframe srcdoc> 等，也可使用冷門元素如 <object>、<marquee>、<details>。\n"
            "3. 對 alert、script、javascript 等關鍵字進行混淆處理，例如：\n"
            "   - 字串拼接（如 'al' + 'ert'）\n"
            "   - Unicode 編碼（如 \\u0061\\u006c\\u0065\\u0072\\u0074）\n"
            "   - HTML 實體編碼（如 &#x61;&#x6C;&#x65;&#x72;&#x74;）\n"
            "   - Base64 編碼搭配 atob 解碼\n"
            "   - 或其他變形方式\n"
            "4. Payload 應具備繞過機器學習模型（例如 CNN、RNN）檢測的能力，避免使用常見模板（如 <script>alert(1)</script>）或明顯可辨識的語法結構。\n\n"
            "請只輸出 payload 字串，每一筆一行，不要加入任何解釋、描述或標示，除了 HTML 標籤本身外，不要加入任何文字。"
        )}
    ]

    # ✅ 呼叫 GPT API
    response = client.chat.completions.create(
        model="gpt-4o",
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