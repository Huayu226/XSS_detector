import os
import pandas as pd
import matplotlib.pyplot as plt

# 設定資料夾與參數
folder_path = "res/success"
prefix = "success_payloads_temp_0.3_06091855_"
total_payloads_per_round = 15

# 整理資料
results = []
for i in range(15):
    filename = f"{prefix}{i}.txt"
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
            success_count = len(lines)
            success_rate = (success_count / total_payloads_per_round) * 100
            results.append({
                "輪次": i + 1,
                "成功筆數": success_count,
                "成功率 (%)": round(success_rate, 2)
            })
    else:
        print(f"❌ 檔案不存在: {filename}")

# 轉成 DataFrame 並存 Excel
df = pd.DataFrame(results)
excel_path = "XSS_0.3_整理統計.xlsx"
df.to_excel(excel_path, index=False)
print(f"✅ 已儲存：{excel_path}")

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(df["輪次"], df["成功率 (%)"], marker='o', label="成功率")
plt.xlabel("第幾次（輪）")
plt.ylabel("成功率 (%)")
plt.title("溫度 0.3 下成功率變化（共 30 輪）")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.savefig("XSS_0.3_整理統計_折線圖.png")
plt.show()
