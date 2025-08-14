import os
import csv
import random

# 目標路徑
outfile = f"res/benign/benign.csv"

# 常見標籤、屬性、字
TAGS  = ["div", "span", "a", "p", "img", "ul", "li", "table", "tr", "td"]
ATTRS = ["class", "id", "style", "title", "alt", "href", "src"]
WORDS = ["hello", "world", "test", "sample", "normal", "benign", "content"]

def generate_benign_html():
    tag = random.choice(TAGS)
    attr = random.choice(ATTRS)
    value = random.choice(WORDS)
    text = random.choice(WORDS)
    # 盡量避免帶有 JS 的屬性或事件（保持良性）
    return f'<{tag} {attr}="{value}">{text}</{tag}>'

def write_rows_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline="" 很重要，避免 Windows 多出空白行
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main(n=29, label=0):
    # 產生 n 筆不重複的樣本（避免重複行）
    seen = set()
    rows = []
    while len(rows) < n:
        s = generate_benign_html()
        if s in seen:
            continue
        seen.add(s)
        rows.append([s, label])   # 和 success 一樣：payload, label

    write_rows_csv(outfile, rows)
    print(f"已寫入 {len(rows)} 筆到：{outfile}")

if __name__ == "__main__":
    main()
