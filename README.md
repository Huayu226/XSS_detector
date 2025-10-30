# XSS detector

## 介紹

DLEX 是一個回饋驅動的框架，它使用大型語言模型 (LLM) 產生對抗性 XSS 有效負載並改進基於深度學習的偵測器。透過在瀏覽器中驗證語法並對未偵測到的樣本進行重新訓練，它可以模擬自動化的攻擊者-防御者循環。實驗表明，LLM 產生的有效負載可以繞過檢測，並且重新訓練將準確率從 0% 提高到近 60%，凸顯了 LLM 在 XSS 檢測中的威脅和潛力。
為了產生對抗性有效負載，我們利用 GPT-4，這是一種廣泛使用且具代表性的大型語言模型，可透過 OpenAI 的 API 存取。建立promt是為了模擬真實的 XSS 注入場景，並且在多種編碼策略、標籤上下文和腳本變體中多樣化。
我們的偵測模型是一個雙通道深度神經網路，重新實作自 C-BLA 架構（Hu 等人，2023）。
Payload 在處理時會先進行標準化，包含：移除 host/path 資訊、將數字替換為占位符，並透過正則表達式（regex）將其分詞為具備 HTML/JavaScript 語意的子詞（subwords）。
這些子詞會使用 Skip-gram 架構的 Word2Vec 模型進行詞嵌入（embedding），接著透過三個平行的 1D 卷積層（kernel size 分別為 2、4、6）萃取語法特徵，並傳入雙向 LSTM（BiLSTM）以捕捉上下文語意，最後經過自注意力層（self-attention）強調重要的 token。串接後的特徵會送入全連接層（Fully Connected Layers）進行二元分類。
本模型採用 Adam 優化器與交叉熵損失函數進行訓練，準確率達 99% 以上，並內建一個工具，可用於驗證 LLM 生成的 payload 是否能成功繞過已訓練的偵測器。

## installation

* 假設使用環境為 windows 10+，打開工作管理員 > 效能 > GPU，看有沒有 RTX 開頭的顯示卡，沒有就代表你電腦沒有顯示卡，要跟我說
* 下載 miniconda 跟 vscode，將 miniconda 加入 PATH，在 vscode 開 terminal，（看要不要開獨立於 base 的環境），打開專案資料夾跑 pip install -r requirements.txt
* pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
* 如果以上指令跑不動，代表需要裝 Nvidia 裝 cuda，這部分問歐陽平 or 其他碩班學長 or 我

## 說明

* 目前都統一使用 conda 來 build env  
* pip install -r requirements.txt 完成之後要裝 playwright 的 broswer driver  

* 弄一些 XSS 不同的 payload 類型來當作多樣性，並對這些多樣性做語法測試  
* 用 CVE_test_1: 2020-11022 來當真實環境背書  

## prompting record

<details>
<summary>1. 初始版本</summary>

```json
"role": "user", "content": (
"我正在測試一個基於 CNN+LSTM 的 XSS 檢測系統，我想產生能繞過這個系統，並且能夠在 Chrome 能夠跳窗的 XSS 對抗樣本。"
"目前成功的 payload 有：\n"
"<svg/onload=setTimeout('ale'+'rt(1)',100)></svg>\n"
"<iframe src=\"javascript: setTimeout('ale'+'rt(1)', 100)\"></iframe>\n"
"<details open ontoggle=\"setTimeout(unescape('%61%6c%65%72%74(1)'), 100)\"></details>\n"
"請幫我再產生 50 個語法正確、與上方概念不同的 payloads"
"使用非 script 標籤觸發、使用 setTimeout 或類似手法組合 alert(1)，避免被靜態分詞器解析。"
"每個 payload 請用一行表示，只需要輸出 payload，不要編號，不要額外說明")
```
</details> 

<details> <summary>2. 從 5/11 AM 2:00 開始使用</summary>

```json
"role": "user","content": (
"請幫我生成 30 筆語法正確的 XSS payload，需符合以下條件：\n\n"
"1. 不需要使用者互動就能自動執行（不能使用 onclick、onmouseover、onfocus 等互動事件）。\n"
"2. 必須能自動觸發 JavaScript，例如使用 <img onerror>、<svg onload>、<meta refresh>、<iframe srcdoc> 等元素。\n"
"3. 請對關鍵詞如 alert、script、javascript 進行混淆處理，例如字串拼接、Unicode 編碼、HTML 實體編碼、Base64 等。\n"
"4. Payload 應具備繞過機器學習模型（例如 CNN）檢測的能力，盡量避免明顯可辨識的語法結構。\n\n"
"請只輸出 payload 字串，每一筆一行，不要加入任何解釋，除 HTML 標籤外，不要任何說明。")
```
</details>

<details> <summary>3. 最終版本</summary>
 
```json
"role": "system", "content": "你是一位 Web 資安專家，擅長製作對抗性 XSS 攻擊樣本"
"role": "user", "content": f"以下是成功繞過偵測與語法測試的 XSS payload 範例：\n{example_text}"
"role": "user", "content": (
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
```
</details>

## 流程
●	跑llm產出payload，可以更改temp和要生產出的round次數
●	接著打開test_innerHTML.html，按右下角的go live如沒有請下載Live Server
●	跑payload_test.py，得到每輪的syntax success rate, model bypass rate, overall success rate
 temp_str = "0.7"  # 指定溫度，這裡可以自由改變
    for i in range(2):#幾輪數字就是幾
        print(i)
        time_str = f"08131703_{i}"  這裡改成剛剛產出的payload的檔名
        result = await process_file(temp_str, time_str)
        result["Round"] = i + 1
        results.append(result)

    df = pd.DataFrame(results[:15])
    df.to_excel(f"XSS_{temp_str}_2rounds_stats.xlsx", index=False)這裡還有一個圖的檔名一樣幾輪數字就是幾

●	跑benign根據res/success/success_payloads_al裡的數量生成相對應的數量 然後跑merge
●	跑CNN_pytorch.ipynb跑出新的最佳模型繼續循環以上步驟
