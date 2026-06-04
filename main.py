import pandas as pd  # 讀 CSV 跟做表格操作，常用的資料科學套件
import numpy as np  # 陣列運算會用到，偶爾做型別轉換或數學運算會用

import matplotlib.pyplot as plt  # 畫圖用，簡單好用
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns  # 畫漂亮的統計圖（heatmap 等）

from sklearn.model_selection import train_test_split  # 用來切訓練/測試集
from sklearn.preprocessing import StandardScaler  # 標準化（讓特徵平均為0、方差為1）

from sklearn.linear_model import LogisticRegression  # 邏輯回歸模型
from sklearn.ensemble import RandomForestClassifier  # 隨機森林模型
from sklearn.svm import SVC  # 支援向量機模型

from sklearn.metrics import (  # 常見的評估指標和工具
    accuracy_score,  # 準確率
    classification_report,  # 精確率、召回率等報表
    confusion_matrix,  # 混淆矩陣
    roc_curve,  # ROC 曲線所需
    auc  # 計算 AUC
)


# 1. 讀資料

data = pd.read_csv("phishing.csv")  # 把資料讀進來，會得到 DataFrame

print("資料大小:", data.shape)  # 印出 (列數, 欄數) 方便知道資料量
print(data.head())  # 看前五筆，檢查欄位名稱跟資料型態是否合理

# 2. 分 X / y

X = data.drop(["Result", "index"], axis=1)  # 把非特徵的欄位移除，留下模型要用的欄位
y = data["Result"]  #目標變數（1=Phishing, -1=Legitimate）


# 3. 切資料

X_train, X_test, y_train, y_test = train_test_split(
    X, y,  # 特徵跟目標
    test_size=0.2,  # 20% 當測試集，80% 當訓練集
    random_state=42  # 固定 seed，結果能重現（同樣的亂數）
)


# 4. 標準化

scaler = StandardScaler()  # 建一個標準化器
X_train_scaled = scaler.fit_transform(X_train)  # 用訓練資料算出 mean/var 並轉換
X_test_scaled = scaler.transform(X_test)  # 用剛剛學到的 mean/var 轉換測試集（不能再 fit）


# 5. 模型訓練


# Logistic Regression（線性模型，常用作 baseline）
lr = LogisticRegression(max_iter=1000)  # 設較高的迭代次數避免沒收斂
lr.fit(X_train_scaled, y_train)  # 在標準化過的訓練集上訓練（LR 對尺度敏感）
pred_lr = lr.predict(X_test_scaled)  # 在測試集上做預測

# Random Forest（樹模型通常不需要標準化）
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 棵樹
rf.fit(X_train, y_train)  # 用原始訓練資料訓練
pred_rf = rf.predict(X_test)  # 在原始測試集上預測

# SVM（常會需要標準化）
svm = SVC(probability=True)  # 設 probability=True 可用 predict_proba
svm.fit(X_train_scaled, y_train)  # 在標準化後的訓練資料上訓練
pred_svm = svm.predict(X_test_scaled)  # 在標準化後的測試集上預測


# 6. Accuracy

acc_lr = accuracy_score(y_test, pred_lr)  # LR 的準確率
acc_rf = accuracy_score(y_test, pred_rf)  # RF 的準確率
acc_svm = accuracy_score(y_test, pred_svm)  # SVM 的準確率

print("\n=== Accuracy ===")  # 印出分隔用標題
print("LR :", acc_lr)  # 印出 LR 準確率
print("RF :", acc_rf)  # 印出 RF 準確率
print("SVM:", acc_svm)  # 印出 SVM 準確率


# 7. Classification Report

print("\n=== Random Forest Report ===")  # 想看比較完整的分類指標就印報表
print(classification_report(y_test, pred_rf))  # RF 的分類報表（precision/recall/f1）


cm = confusion_matrix(y_test, pred_rf)  # 混淆矩陣：真正、假正、真負、假負

# 把混淆矩陣畫成 heatmap，方便看哪個類別被錯分最多
plt.figure("1. Random Forest - Confusion Matrix")  # 指定視窗名稱（比較好辨識）
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["正常網站", "釣魚網站"],
    yticklabels=["正常網站", "釣魚網站"]
)  # 標註數字，整數格式
plt.title("隨機森林混淆矩陣")
plt.xlabel("預測結果")
plt.ylabel("實際類別")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")



# 使用 Random Forest 的機率輸出計算 ROC

y_test_binary = (y_test == 1).astype(int)

y_score = rf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)

plt.figure("2. Random Forest - ROC Curve")
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")

plt.title("隨機森林 ROC 曲線")
plt.xlabel("False Positive Rate（偽陽性率）")
plt.ylabel("True Positive Rate（真陽性率）")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")



importances = rf.feature_importances_  # 隨機森林可以直接拿 feature importance
features = X.columns  # 欄位名稱

feat_df = pd.DataFrame({  # 把重要性做成 DataFrame，比較好排版
    "Feature": features,
    "Importance": importances
})

feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)  # 取前10重要特徵
print("\n=== Top 10 Important Features ===")
print(feat_df)

plt.figure("3. 隨機森林前10名重要特徵", figsize=(10, 6))

bars = plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()

plt.title("隨機森林前 10 名重要特徵")
plt.xlabel("特徵重要性")
plt.ylabel("網站特徵")
plt.xlim(0, 0.38)
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center"
    )

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")



# 11. 模型比較圖

models = ["Logistic Regression", "Random Forest", "SVM"]
accs = [acc_lr * 100, acc_rf * 100, acc_svm * 100]

plt.figure("4. 模型準確率比較", figsize=(8, 5))
bars = plt.bar(models, accs)

plt.ylim(0, 105)
plt.title("釣魚網站偵測模型準確率比較")
plt.ylabel("準確率 (%)")
plt.xlabel("機器學習模型")

# 在柱狀圖上顯示數值
for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 1,
        f"{acc:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")

# 最後一次性顯示所有圖，避免每個圖都阻塞程式
plt.show()  # 顯示所有先前宣告的視窗



# 12. Demo 測試

# 選第一筆測試資料做示範，iloc[[0]] 會保留 DataFrame 形狀
sample = X_test.iloc[[0]]  # 取一筆測試資料作為範例

print("\n=== Demo ===")  # Demo 標題
print("LR :", lr.predict(scaler.transform(sample)))  # LR 預測（要先標準化）
print("RF :", rf.predict(sample))  # RF 預測（樹模型不需要標準化）
print("SVM:", svm.predict(scaler.transform(sample)))  # SVM 預測（先標準化）