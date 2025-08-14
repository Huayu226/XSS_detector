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
import csv

def _read_existing_payloads_from_csv(path: str) -> set:
    if not os.path.exists(path):
        return set()
    existing = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            existing.add(row[0].strip())
    return existing

def _append_payload_csv(path: str, payload: str, label: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([payload, label])

async def process_file(temp_str, time_str):
    detector = XSSDetector("res/best_model.pth", "res/word2vec.model")

    input_path = f"res/llm_output/llm_output_temp_{temp_str}_{time_str}.txt"
    output_path = f"res/success/success_payloads_temp_{temp_str}_{time_str}.csv"
    success_all_path = "res/success/success_payloads_all.csv"

    file_read = open(input_path, "r", encoding="utf-8")
    payloads = [line.strip() for line in file_read if line.strip()]
    success = _read_existing_payloads_from_csv(success_all_path)
    success_payloads_set = _read_existing_payloads_from_csv(output_path)

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
            if syntax_result and detector_result == 0:
                overall_success_count += 1
                if payload not in success_payloads_set:
                    _append_payload_csv(output_path, payload, label = 1)
                    success_payloads_set.add(payload)
                if payload not in success:
                    _append_payload_csv(success_all_path, payload, label = 1)
                    success.add(payload)

            print(f"[{temp_str}] #{i} → detector={detector_result}, syntax={syntax_result}")

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


async def main():
    results = []

    temp_str = "0.7"  # 指定溫度，這裡可以自由改變
    for i in range(1):
        print(i)
        time_str = f"08131703_{i}"
        result = await process_file(temp_str, time_str)
        result["Round"] = i + 1
        results.append(result)

    df = pd.DataFrame(results[:15])
    df.to_excel(f"XSS_{temp_str}_1rounds_stats.xlsx", index=False)

    # Plotting
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