## LLM‑Learning

以最少程式碼完成一個可重現的文字情感分析（Sentiment Analysis）基線系統：

- 使用 TF‑IDF 向量化 + LinearSVC 分類器
- 預設使用 Hugging Face 的 `rotten_tomatoes` 資料集；若無法下載，會自動改用 `20newsgroups` 的二分類備援（`rec.sport.baseball` vs `sci.space`）
- 一次執行自動產生：
  - `metrics.txt`：整體與逐類別指標
  - `confusion_matrix.png`：混淆矩陣圖
  - `errors.csv`：誤判樣本清單（text/true/pred）
  - `sentiment_baseline.joblib`：已訓練的完整 Pipeline（向量化 + 模型）


### 環境需求
- Python 3.8+（Windows/macOS/Linux 皆可）
- 主要套件：`scikit-learn`、`pandas`、`matplotlib`、`joblib`、`datasets`（如需使用 HF 資料集）


### 安裝與執行
在 Windows PowerShell：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install scikit-learn pandas matplotlib joblib datasets

# 執行訓練、評估與輸出
python ml.py
```

macOS/Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install scikit-learn pandas matplotlib joblib datasets

python ml.py
```


### 產出說明
- `metrics.txt`：整體 Accuracy、Macro/Weighted Precision/Recall/F1、逐類別報表與混淆矩陣數值
- `confusion_matrix.png`：混淆矩陣視覺化，可直接開啟查看
- `errors.csv`：誤判（label ≠ pred）的樣本，欄位為 `text,true,pred`
- `sentiment_baseline.joblib`：完整 `sklearn.Pipeline`，可直接載入做推論


### 目前結果（示例，取自 metrics.txt）
本次在 `rotten_tomatoes` 測試集上的摘要：

```text
Accuracy: 0.7627
Macro  - Precision: 0.7628  Recall: 0.7627  F1: 0.7626
Weighted - Precision: 0.7628  Recall: 0.7627  F1: 0.7626

Confusion Matrix (rows=true, cols=pred):
  413  120
  133  400
```

混淆矩陣圖：請開啟 `confusion_matrix.png`。


### 模型與特徵
- 向量化：`TfidfVectorizer(stop_words="english", min_df=2, max_df=0.95, ngram_range=(1, 2))`
- 分類器：`LinearSVC`
- 流程：`文字 → TF‑IDF → LinearSVC → 評估/輸出`（詳見 `ml.py`）


### 快速推論（使用已訓練模型）
```python
import joblib

pipe = joblib.load("sentiment_baseline.joblib")
label_names = ["neg", "pos"]  # 0/1 對應

samples = [
    "This movie was fantastic!",
    "This was a terrible waste of time.",
]

pred = pipe.predict(samples)
print([label_names[int(v)] for v in pred])
```


### 專案結構
```text
LLM-Learning/
├─ ml.py                       # 一鍵訓練/評估/輸出主程式
├─ metrics.txt                 # 指標（執行後產生）
├─ confusion_matrix.png        # 混淆矩陣圖（執行後產生）
├─ errors.csv                  # 誤判清單（執行後產生）
└─ sentiment_baseline.joblib   # 已訓練模型（執行後產生）
```


### 常見問題（FAQ）
- 執行時下載資料集失敗或沒安裝 `datasets`？
  - 會自動 fallback 到 `20newsgroups` 的二分類設定繼續完成訓練與評估。
- 想改用自己的資料？
  - 可自行修改 `load_data()`，回傳 `train_df`、`test_df` 需包含 `text` 與 `label` 欄位。
- 要看哪些句子被分類錯？
  - 打開 `errors.csv`，檢查 `true` 與 `pred` 的差異並進行誤差分析。


### 授權
尚未指定（視實際需求補充）。
