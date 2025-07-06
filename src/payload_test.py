import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from CNN_pytorch import XSSDetector
from penetration_test_normal.test import test_payload
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'

# ✅ 檢查 HTML 結構是否正確（事前過濾語法錯誤 payload）
def is_valid_html(payload):
    try:
        soup = BeautifulSoup(payload, "html.parser")
        return bool(soup.find()) and '<' in payload and '>' in payload
    except:
        return False

# ✅ 單輪測試處理主函式
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
    payloads = [line.strip() for line in file_read if line.strip()]

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
            else:
                fail_payloads.write(payload + "\n")

            if detector_result == 0:
                model_bypass_count += 1

            if syntax_result and detector_result == 0:
                overall_success_count += 1
                if payload not in success_payloads_set:
                    success_payloads.write(payload + "\n")
                    with open("res/success/success_payloads_all.txt", "a+", encoding="utf-8") as total_success:
                        total_success.seek(0)
                        existing = set(total_success.read().splitlines())
                        if payload not in existing:
                            total_success.write(payload + "\n")

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

# ✅ 多輪執行與圖表輸出
async def main():
    results = []
    temp_str = "0.7"
    for i in range(15):
        print(i)
        time_str = f"07052332_{i}"
        result = await process_file(temp_str, time_str)
        result["Round"] = i + 1
        results.append(result)

    df = pd.DataFrame(results[:15])
    df.to_excel(f"XSS_{temp_str}_15rounds_stats.xlsx", index=False)

    temp_label = df["Temperature"].iloc[0]
    plt.figure(figsize=(10, 6))
    plt.plot(df["Round"], df["Syntax Success Rate"] * 100, marker='o', label="Syntax Success Rate")
    plt.plot(df["Round"], df["Model Bypass Rate"] * 100, marker='o', label="Model Bypass Rate")
    plt.plot(df["Round"], df["Overall Success Rate"] * 100, marker='o', label="Overall Success Rate")
    plt.xlabel("Round")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate Variation under Temp {temp_label} (15 Rounds)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"XSS_{temp_label}_15rounds_plot.png")
    plt.close()

if __name__ == "__main__":
    asyncio.run(main())
