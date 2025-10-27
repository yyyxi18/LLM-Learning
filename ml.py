# ml.py  —  Phase 1: 自動輸出評估與圖表
from __future__ import annotations  # 允許未來版本的型別註解語法（前置相容）

# =========================
# [段落 A] 標準匯入與套件依賴
# 目的：載入作業系統、資料處理、模型序列化、視覺化與機器學習所需的函式庫
# =========================
import os, sys, traceback           # 作業系統路徑、系統退出、例外堆疊
import pandas as pd                 # 資料表處理
import joblib                       # 儲存/讀取已訓練模型
import matplotlib.pyplot as plt     # 視覺化（這裡用來畫並存混淆矩陣）

# Sklearn
from sklearn.pipeline import Pipeline  # 把前處理與模型串成一條流水線
from sklearn.feature_extraction.text import TfidfVectorizer  # 文字轉 TF-IDF 向量
from sklearn.svm import LinearSVC      # 線性 SVM 分類器（適合高維稀疏特徵）
from sklearn.metrics import (          # 評估指標與可視化工具
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# =========================
# [段落 B] 路徑與輸出檔名設定
# 目的：集中管理模型/指標/圖表/錯誤清單的輸出位置
# =========================
OUT_DIR = "."                                               # 輸出檔案的資料夾（目前用當前目錄）
MODEL_PATH = os.path.join(OUT_DIR, "sentiment_baseline.joblib")  # 已訓練模型的存檔路徑
METRICS_PATH = os.path.join(OUT_DIR, "metrics.txt")               # 評估指標輸出檔
CM_PNG = os.path.join(OUT_DIR, "confusion_matrix.png")            # 混淆矩陣圖檔
ERRORS_CSV = os.path.join(OUT_DIR, "errors.csv")                  # 誤判樣本清單

# =========================
# [段落 C] 資料載入函式：load_data()
# 目的：優先載入 Hugging Face 的 rotten_tomatoes；若失敗則 fallback 到 sklearn 的 20newsgroups（二分類）
# 回傳：train_df, test_df, label_names, src
# =========================
def load_data():  # 載入資料集：優先 rotten_tomatoes；失敗則使用 20newsgroups 當備援
    """優先用 rotten_tomatoes；若下載失敗就 fallback 到 20newsgroups 二分類。"""
    try:
        from datasets import load_dataset                 # 從 Hugging Face 載入資料集
        ds = load_dataset("rotten_tomatoes")             # 會自動下載一次、之後走快取

        # 將 train + validation 合併為訓練集；test 維持官方切分
        train_df = pd.concat(
            [ds["train"].to_pandas(), ds["validation"].to_pandas()],
            ignore_index=True
        )
        test_df = ds["test"].to_pandas()

        label_names = ["neg", "pos"]                     # 0/1 對應人類可讀標籤
        src = "rotten_tomatoes"                          # 紀錄資料來源（報表用）
    except Exception:                                    # 若下載/匯入失敗（無網路/無套件等）
        print("[warn] load_dataset 失敗，改用 sklearn 內建資料集。")
        from sklearn.datasets import fetch_20newsgroups  # 備援：新聞資料集

        # 指定兩個主題做二分類
        cats = ['rec.sport.baseball', 'sci.space']

        # 移除 headers/footers/quotes，僅保留正文
        train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
        test  = fetch_20newsgroups(subset='test',  categories=cats, remove=('headers','footers','quotes'))

        # 統一成 text/label 欄位格式
        train_df = pd.DataFrame({'text': train.data, 'label': train.target})
        test_df  = pd.DataFrame({'text': test.data,  'label': test.target})

        # 將數字標籤對應回類別名稱
        label_names = [cats[i] for i in sorted(set(train.target))]
        src = "20newsgroups"
    return train_df, test_df, label_names, src

# =========================
# [段落 D] 建模函式：build_pipeline()
# 目的：建立「文字向量化(TFIDF) → 分類器(LinearSVC)」的 sklearn Pipeline
# =========================
def build_pipeline():  # 建立「文字向量化 → 分類器」的流水線
    return Pipeline([
        ("tfidf", TfidfVectorizer(                     # 將文字轉為 TF-IDF 稀疏向量
            stop_words="english",                      # 去除英文停用字
            min_df=2,                                  # 至少出現在 2 個文件才保留（降噪）
            max_df=0.95,                               # 若在 95% 以上文件出現則忽略（太常見無區分力）
            ngram_range=(1,2)                          # 使用 uni-gram 與 bi-gram（可捕捉否定片語）
        )),
        ("clf", LinearSVC())                           # 使用線性 SVM 作為分類器
    ])

# =========================
# [段落 E] 主流程：main()
# 目的：資料載入 → 模型訓練 → 推論 → 評估指標計算與輸出 → 視覺化 → 誤判匯出 → 模型保存 → Demo
# =========================
def main():
    try:
        # --- E1. 載入資料與標籤名稱 ---
        train_df, test_df, label_names, src = load_data()
        print(f"[info] dataset = {src}, train={len(train_df)}, test={len(test_df)}")

        # --- E2. 建立並訓練 Pipeline ---
        pipe = build_pipeline()
        pipe.fit(train_df["text"], train_df["label"])  # 使用訓練資料學習向量化與分類器

        # --- E3. 推論（對測試集預測）---
        y_true = test_df["label"]                      # 真實標籤
        y_pred = pipe.predict(test_df["text"])         # 模型預測

        # --- E4. 計算評估指標（整體 + Macro/Weighted）---
        acc = accuracy_score(y_true, y_pred)

        # Macro：各類別等權平均，適合類別不平衡時觀察每類表現
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        # Weighted：依樣本數加權的平均，反映整體表現
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # 每類別詳細報表（precision/recall/f1/support）
        report = classification_report(y_true, y_pred, target_names=label_names, digits=3)

        # 混淆矩陣（rows=true, cols=pred）
        cm = confusion_matrix(y_true, y_pred)

        # --- E5. 輸出 metrics.txt（可讀的評估摘要）---
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            f.write(f"dataset: {src}\n")
            f.write(f"train_size: {len(train_df)}  test_size: {len(test_df)}\n\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro  - Precision: {p_macro:.4f}  Recall: {r_macro:.4f}  F1: {f1_macro:.4f}\n")
            f.write(f"Weighted - Precision: {p_weighted:.4f}  Recall: {r_weighted:.4f}  F1: {f1_weighted:.4f}\n\n")
            f.write("Per-class classification report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix (rows=true, cols=pred):\n")
            for row in cm:
                f.write("  " + "  ".join(f"{v:4d}" for v in row) + "\n")
        print(f"[ok] 指標已輸出 -> {METRICS_PATH}")

        # --- E6. 繪製並輸出混淆矩陣圖 ---
        # 使用 sklearn 的 ConfusionMatrixDisplay，方便保存圖像
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(cmap="viridis", colorbar=True)
        plt.tight_layout()
        plt.savefig(CM_PNG, dpi=150)
        plt.close()
        print(f"[ok] 混淆矩陣圖 -> {CM_PNG}")

        # --- E7. 匯出誤判樣本（利於誤差分析）---
        errors = test_df.copy()
        errors["pred"] = y_pred
        mis = errors[errors["label"] != errors["pred"]]  # 篩出 misclassified

        # 小工具：將數字標籤轉為可讀文字（若可行）
        def map_lbl(v):
            try:
                return label_names[int(v)]
            except Exception:
                return str(v)

        # 建立輸出用資料表（text/true/pred）
        mis_out = pd.DataFrame({
            "text": mis["text"],
            "true": mis["label"].map(map_lbl),
            "pred": mis["pred"].map(map_lbl),
        })
        mis_out.to_csv(ERRORS_CSV, index=False, encoding="utf-8")
        print(f"[ok] 誤判清單 -> {ERRORS_CSV}  (共 {len(mis_out)} 筆)")

        # --- E8. 保存整個 Pipeline（向量化 + 模型）---
        joblib.dump(pipe, MODEL_PATH)
        print(f"[ok] 模型已保存 -> {MODEL_PATH}")

        # --- E9. Demo：快速測試兩句話的推論結果 ---
        demo = ["This movie was fantastic!", "This was a terrible waste of time."]
        print("[demo]", demo, "=>", [label_names[int(x)] for x in pipe.predict(demo)])

    except Exception as e:
        # --- E10. 錯誤處理與除錯資訊 ---
        print("[error] 例外發生：", e)
        traceback.print_exc()
        sys.exit(1)

# =========================
# [段落 F] 程式入口
# 目的：僅在直接執行此檔時才跑 main（被其他程式 import 時不會自動執行）
# =========================
if __name__ == "__main__":
    main()
