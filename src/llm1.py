import time as t
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from CNN_pytorch import XSSDetector
from penetration_test_normal.test import test_payload
from playwright.async_api import async_playwright
import os
import json
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
from collections import Counter

# ✅ 擷取最常見成功 payload（Top-N）
def get_top_successful_payloads(limit=10):
    file_path = "res/success/success_payloads_all.txt"
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("```")
        ]   
    freq = Counter(lines)
    return [payload for payload, _ in freq.most_common(limit)]

# ✅ 讀取最新一輪語法錯誤 payload（語法測試未通過者）
def get_top_failed_payloads(limit=10):
    file_path = "res/fail/fail_payloads_all.txt"
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("```")
        ] 
    freq = Counter(lines)
    return [payload for payload, _ in freq.most_common(limit)]

#成功
def log_success_payload_once(payload: str):
    file_path = "res/success/success_payloads_all.txt"
    file_path1 = "res/success/success_payloads_0.3.txt"   #r記得改temp
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(file_path1), exist_ok=True)
    
    with open(file_path, "a+", encoding="utf-8") as f:
        f.seek(0)
        existing_payloads = set(f.read().splitlines())
        if payload.strip() not in existing_payloads:
            f.write(payload.strip() + "\n")
    
    with open(file_path1, "a+", encoding="utf-8") as f:
        f.seek(0)
        existing_payloads = set(f.read().splitlines())
        if payload.strip() not in existing_payloads:
            f.write(payload.strip() + "\n")

#失敗
def log_failed_payload_once(payload: str):
    file_path = "res/fail/fail_payloads_all.txt"
    file_path1 = "res/fail/fail_payloads_0.3.txt"  #記得改temp
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(file_path1), exist_ok=True)
    
    with open(file_path, "a+", encoding="utf-8") as f:
        f.seek(0)
        existing_payloads = set(f.read().splitlines())
        if payload.strip() not in existing_payloads:
            f.write(payload.strip() + "\n")
    
    with open(file_path1, "a+", encoding="utf-8") as f:
        f.seek(0)
        existing_payloads = set(f.read().splitlines())
        if payload.strip() not in existing_payloads:
            f.write(payload.strip() + "\n")


# 執行測試，並回傳各項成功率與結果列表
async def process_file(temp_str, time_str):
    detector = XSSDetector("res/best_model.pth", "res/word2vec.model")
    os.makedirs("res/fail", exist_ok=True)
    input_path = f"res/llm_output/llm_output_temp_{temp_str}_{time_str}.txt"
    output_path = f"res/success/success_payloads_temp_{temp_str}_{time_str}.txt"
    fail_output_path = f"res/fail/fail_payloads_temp_{temp_str}_{time_str}.txt"

    file_read = open(input_path, "r", encoding="utf-8")
    success_payloads = open(output_path, "a+", encoding="utf-8")
    fail_payloads = open(fail_output_path, "a+", encoding="utf-8")

    success_payloads.seek(0)
    success_payloads_set = set(success_payloads.read().splitlines())
    payloads = [
        line.strip()
        for line in file_read
        if line.strip() and not line.strip().startswith("```")
    ]
    
    fail_payloads.seek(0)
    fail_payloads_set = set(fail_payloads.read().splitlines())

    html_path = "http://127.0.0.1:5500/src/penetration_test_normal/test_innerHTML.html"

    syntax_success_count = 0
    model_bypass_count = 0
    overall_success_count = 0

    async with async_playwright() as playwright:
        for i, payload in enumerate(payloads):
            detector_result = detector.is_xss(payload)

            try:
                syntax_result = await test_payload(playwright, html_path, payload)
            except Exception as e:
                print(f"[{temp_str}] #{i} ❌ syntax error: {e}")
                syntax_result = False

            if syntax_result:
                syntax_success_count += 1

            if detector_result == 0:
                model_bypass_count += 1

            if syntax_result and detector_result != 0:
                if payload not in fail_payloads_set:
                    fail_payloads.write(payload + "\n")
                    log_failed_payload_once(payload)
                    fail_payloads_set.add(payload)

            if syntax_result and detector_result == 0:
                overall_success_count += 1
                if payload not in success_payloads_set:
                    success_payloads.write(payload + "\n")
                    log_success_payload_once(payload)
            print(f"[{temp_str}] #{i} → detector={detector_result}, syntax={syntax_result}")

    success_payloads.close()
    fail_payloads.close()
    file_read.close()

    total = len(payloads)
    syntax_rate = syntax_success_count / total
    bypass_rate = model_bypass_count / total
    overall_rate = overall_success_count / total

    print(f"\n📊 統計結果（溫度 {temp_str}, 共 {total} 筆）：")
    print(f"✅ 語法成功率     = {syntax_rate:.2%}")
    print(f"🛡️ 模型繞過成功率 = {bypass_rate:.2%}")
    print(f"🏆 整體成功率     = {overall_rate:.2%}")

    return {
        "Temperature": float(temp_str),
        "Total Payloads": total,
        "Syntax Success Rate": syntax_rate,
        "Model Bypass Rate": bypass_rate,
        "Overall Success Rate": overall_rate
    }

# 主程序


async def main():
    load_dotenv()
    results = []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    temp = 0.3
    rounds = 15
    timestamp = datetime.now().strftime("%m%d%H%M")

    for i in range(rounds):
        print(f"[INFO] 正在產生第 {i + 1} 輪的 payload（temp={temp}）...")

        successful_examples = get_top_successful_payloads()
        failed_examples = get_top_failed_payloads()
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
        
        if os.path.exists("prompt.txt"):
            with open("prompt.txt", "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict):
                            messages.append(obj)
                    except json.JSONDecodeError:
                        continue
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temp,
            frequency_penalty=0.0
        )

        payload = response.choices[0].message.content.strip()

        os.makedirs("res/llm_output", exist_ok=True)
        filename = f"res/llm_output/llm_output_temp_{temp}_{timestamp}_{i}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(payload)

        result = await process_file(temp, f"{timestamp}_{i}")
        Temp = result["Temperature"]
        total_payloads = result["Total Payloads"]
        syntax_rate = result["Syntax Success Rate"]
        bypass_rate = result["Model Bypass Rate"]
        overall_rate = result["Overall Success Rate"]

        results.append({
            "Round": i + 1,
            "Temperature": Temp,
            "Total Payloads": total_payloads,
            "Syntax Success Rate": syntax_rate,
            "Model Bypass Rate": bypass_rate,
            "Overall Success Rate": overall_rate
        })
        if i < 4:
            with open("prompt.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "role": "assistant",
                    "content": payload
                }, ensure_ascii=False) + "\n")

                # 寫入 user 的統計與指示
                summary_entry = {
                    "role": "user",
                    "content": (
                        f"上一輪統計結果：\n"
                        f"語法成功率：{syntax_rate:.2%}\n"
                        f"模型繞過成功率：{bypass_rate:.2%}\n"
                        f"整體成功率：{overall_rate:.2%}\n"
                        f"請根據上述 payload 表現產生成功率更高的下一輪且不重複"
                    )
                }
                f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        
        t.sleep(1.5)

    df = pd.DataFrame(results)
    df.to_excel(f"XSS_{temp}_15rounds_stats.xlsx", index=False)

    temp_label = df["Temperature"].iloc[0]
    plt.figure(figsize=(10, 6))
    plt.plot(df["Round"], df["Syntax Success Rate"] *
             100, marker='o', label="Syntax Success Rate")
    plt.plot(df["Round"], df["Model Bypass Rate"] *
             100, marker='o', label="Model Bypass Rate")
    plt.plot(df["Round"], df["Overall Success Rate"] *
             100, marker='o', label="Overall Success Rate")
    plt.xlabel("Round")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate Variation under Temp {temp_label} (15 Rounds)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"XSS_{temp_label}_15rounds_plot.png")
    plt.close()

    print("\u2705 Finished!")

if __name__ == "__main__":
    asyncio.run(main())
