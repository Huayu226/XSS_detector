import pandas as pd

# 讀三個 CSV（無標題）
df1 = pd.read_csv(f"res/success/success_payloads_all.csv", header=None)
df2 = pd.read_csv(f"res/model/model_payloads.csv", header=None)
df3 = pd.read_csv(f"res/benign/benign.csv", header=None)

# 合併三個並去重（完全相同的列會刪）
merged_df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()

# 輸出
out_path = f"res/model/all.csv"
merged_df.to_csv(out_path, index=False, header=False, encoding="utf-8")

print(f"✅ 合併完成，共 {len(merged_df)} 筆，已寫入：{out_path}")