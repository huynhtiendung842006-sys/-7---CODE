"""
Äá» 7: PHĂ‚N TĂCH KHĂCH HĂ€NG NGĂ‚N HĂ€NG - Mall Customers (Kaggle)
Há»c pháº§n: PhĂ¢n tĂ­ch dá»¯ liá»‡u lá»›n trong tĂ i chĂ­nh

Äáº·t file Mall_Customers.csv cĂ¹ng thÆ° má»¥c vá»›i file nĂ y rá»“i cháº¡y.
Káº¿t quáº£ (hĂ¬nh áº£nh, báº£ng CSV) lÆ°u trong thÆ° má»¥c ./ket_qua
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42

# ThÆ° má»¥c chá»©a file code (dĂ¹ng Ä‘á»ƒ tĂ¬m file CSV vĂ  lÆ°u káº¿t quáº£)
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()
OUT = os.path.join(BASE, "ket_qua")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 1. Äá»ŒC Dá»® LIá»†U & KHĂM PHĂ (EDA)
# ----------------------------------------------------------------------------
path = os.path.join(BASE, "Mall_Customers.csv")
df = pd.read_csv(path)
df = df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending"})
print("KĂ­ch thÆ°á»›c:", df.shape)
print(df.describe(include="all").T)
print("GiĂ¡ trá»‹ thiáº¿u:\n", df.isna().sum())
df.describe(include="all").T.to_csv(os.path.join(OUT, "01_thong_ke_mo_ta.csv"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, ["Age", "Income", "Spending"]):
    sns.histplot(df[c], kde=True, ax=a)
    a.set_title(f"PhĂ¢n phá»‘i {c}")
savefig("01_phan_phoi.png")

sns.pairplot(df[["Age", "Income", "Spending", "Gender"]], hue="Gender")
plt.savefig(os.path.join(OUT, "01_pairplot.png"), dpi=120)
plt.close()

sns.heatmap(df[["Age", "Income", "Spending"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Ma tráº­n tÆ°Æ¡ng quan")
savefig("01_tuong_quan.png")

# ----------------------------------------------------------------------------
# 2. TIá»€N Xá»¬ LĂ: CHUáº¨N HĂ“A Báº°NG StandardScaler
#    z = (x - mean) / std  -> mean = 0, std = 1
# ----------------------------------------------------------------------------
num_cols = ["Age", "Income", "Spending"]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
print("\nSau chuáº©n hĂ³a (mean, std):\n", X_scaled.agg(["mean", "std"]).round(3))
X_scaled.agg(["mean", "std"]).round(3).to_csv(os.path.join(OUT, "02_sau_chuan_hoa.csv"))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[num_cols].boxplot(ax=ax[0]); ax[0].set_title("TrÆ°á»›c chuáº©n hĂ³a")
X_scaled.boxplot(ax=ax[1]); ax[1].set_title("Sau chuáº©n hĂ³a (StandardScaler)")
savefig("02_chuan_hoa.png")

# ----------------------------------------------------------------------------
# 3. LUáº¬T Káº¾T Há»¢P (Apriori)
#    Rá»i ráº¡c hĂ³a biáº¿n sá»‘ thĂ nh nhĂ³m -> one-hot -> tĂ¬m táº­p phá»• biáº¿n -> luáº­t
#    SiĂªu tham sá»‘: min_support, min_confidence, min_lift
# ----------------------------------------------------------------------------
MIN_SUPPORT, MIN_CONF, MIN_LIFT = 0.05, 0.5, 1.1
basket = pd.DataFrame({
    "Gioi_tinh": df["Gender"],
    "Tuoi": pd.cut(df["Age"], [0, 25, 40, 55, 100],
                   labels=["Tre(<=25)", "Trung_nien(26-40)", "Lon(41-55)", "Cao_tuoi(>55)"]),
    "Thu_nhap": pd.qcut(df["Income"], 3, labels=["Thap", "TB", "Cao"]),
    "Chi_tieu": pd.qcut(df["Spending"], 3, labels=["Thap", "TB", "Cao"]),
}).astype(str)
basket_oh = pd.get_dummies(basket, prefix_sep="=").astype(bool)

freq = apriori(basket_oh, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONF)
rules = rules[rules["lift"] >= MIN_LIFT].sort_values(["lift", "confidence"], ascending=False)
rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(f"\nSá»‘ táº­p phá»• biáº¿n: {len(freq)} | Sá»‘ luáº­t: {len(rules)}")
print(rules[cols].head(15).round(3).to_string(index=False))
rules[cols].to_csv(os.path.join(OUT, "03_luat_ket_hop.csv"), index=False)

if len(rules):
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=50)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support"); plt.ylabel("Confidence"); plt.title("Luáº­t káº¿t há»£p: Support - Confidence - Lift")
    savefig("03_luat_ket_hop.png")

# ----------------------------------------------------------------------------
# 4. PHĂ‚N Cá»¤M theo Thu nháº­p & Äiá»ƒm chi tiĂªu
#    So sĂ¡nh KMeans (chá»n k báº±ng Elbow + Silhouette), Agglomerative, DBSCAN
# ----------------------------------------------------------------------------
Xc = X_scaled[["Income", "Spending"]].values
ks = range(2, 11)
inertia, sil, dbi = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xc, km.labels_))
    dbi.append(davies_bouldin_score(Xc, km.labels_))
k_best = list(ks)[int(np.argmax(sil))]
print(f"\nK tá»‘i Æ°u theo Silhouette: {k_best}")
pd.DataFrame({"k": list(ks), "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi}).round(4)\
    .to_csv(os.path.join(OUT, "04_chon_k.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow (Inertia)")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette (cĂ ng cao cĂ ng tá»‘t)")
ax[2].plot(ks, dbi, "o-"); ax[2].set_title("Davies-Bouldin (cĂ ng tháº¥p cĂ ng tá»‘t)")
for a in ax:
    a.set_xlabel("k")
savefig("04_chon_k.png")

# Chá»‘t k theo Silhouette (thÆ°á»ng k=5 vá»›i bá»™ dá»¯ liá»‡u nĂ y); cĂ³ thá»ƒ ghi Ä‘Ă¨ báº±ng K_FINAL = 5
K_FINAL = k_best
models = {
    f"KMeans(k={K_FINAL})": KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEED),
    f"Agglomerative(k={K_FINAL}, ward)": AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward"),
    "DBSCAN(eps=0.4, min_samples=5)": DBSCAN(eps=0.4, min_samples=5),
}
rows, labels_dict = [], {}
for name, m in models.items():
    lab = m.fit_predict(Xc)
    labels_dict[name] = lab
    mask = lab != -1
    n_cl = len(set(lab[mask]))
    rows.append({
        "Mo_hinh": name, "So_cum": n_cl, "So_nhieu": int((~mask).sum()),
        "Silhouette": silhouette_score(Xc[mask], lab[mask]) if n_cl > 1 else np.nan,
        "Davies_Bouldin": davies_bouldin_score(Xc[mask], lab[mask]) if n_cl > 1 else np.nan,
    })
cmp = pd.DataFrame(rows).round(4)
print(cmp.to_string(index=False))
cmp.to_csv(os.path.join(OUT, "04_so_sanh_phan_cum.csv"), index=False)

df["Cluster"] = labels_dict[f"KMeans(k={K_FINAL})"]
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Income", y="Spending", hue="Cluster", palette="tab10", s=60)
centers = scaler.inverse_transform(
    np.column_stack([np.zeros(K_FINAL), KMeans(K_FINAL, n_init=10, random_state=SEED).fit(Xc).cluster_centers_])
)[:, 1:]
plt.scatter(centers[:, 0], centers[:, 1], marker="X", c="black", s=200, label="TĂ¢m cá»¥m")
plt.title(f"PhĂ¢n cá»¥m KMeans (k={K_FINAL})"); plt.legend()
savefig("04_kmeans.png")

profile = df.groupby("Cluster").agg(So_KH=("Age", "size"), Tuoi_TB=("Age", "mean"),
                                    Thu_nhap_TB=("Income", "mean"), Chi_tieu_TB=("Spending", "mean"),
                                    Ty_le_Nu=("Gender", lambda s: (s == "Female").mean())).round(2)
print("\nChĂ¢n dung tá»«ng cá»¥m:\n", profile)
profile.to_csv(os.path.join(OUT, "04_chan_dung_cum.csv"))

# ----------------------------------------------------------------------------
# 5. PHĂ‚N Lá»P: Dá»± Ä‘oĂ¡n Giá»›i tĂ­nh dá»±a trĂªn Thu nháº­p
#    DĂ¹ng Pipeline (Scaler náº±m trong CV -> trĂ¡nh rĂ² rá»‰ dá»¯ liá»‡u), GridSearchCV
# ----------------------------------------------------------------------------
Xg = df[["Income"]]
yg = (df["Gender"] == "Female").astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.2, stratify=yg, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

clf_cfg = {
    "Logistic Regression": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
                            {"m__C": [0.01, 0.1, 1, 10]}),
    "KNN": (Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
            {"m__n_neighbors": [3, 5, 7, 11, 15, 21]}),
    "Random Forest": (Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(random_state=SEED))]),
                      {"m__n_estimators": [100, 300], "m__max_depth": [2, 3, 5, None]}),
}
base_acc = max(yte.mean(), 1 - yte.mean())
print(f"\nBaseline (Ä‘oĂ¡n lá»›p Ä‘a sá»‘): accuracy = {base_acc:.3f}")
rows = []
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, (pipe, grid)) in zip(ax, clf_cfg.items()):
    gs = GridSearchCV(pipe, grid, cv=skf, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
    best = gs.best_estimator_
    pred, proba = best.predict(Xte), best.predict_proba(Xte)[:, 1]
    rows.append({"Mo_hinh": name, "Sieu_tham_so_tot_nhat": gs.best_params_,
                 "CV_F1": round(gs.best_score_, 3), "Test_Accuracy": round(accuracy_score(yte, pred), 3),
                 "Test_F1": round(f1_score(yte, pred), 3), "Test_AUC": round(roc_auc_score(yte, proba), 3)})
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Nam", "Nu"], ax=a, colorbar=False)
    a.set_title(name)
savefig("05_ma_tran_nham_lan.png")
res_clf = pd.DataFrame(rows)
res_clf["Baseline_Accuracy"] = round(base_acc, 3)
print(res_clf.to_string(index=False))
res_clf.to_csv(os.path.join(OUT, "05_ket_qua_phan_lop.csv"), index=False)
print("Nháº­n xĂ©t: Thu nháº­p gáº§n nhÆ° khĂ´ng phĂ¢n biá»‡t Ä‘Æ°á»£c giá»›i tĂ­nh (AUC ~ 0.5) -> "
      "cáº§n nĂªu rĂµ trong bĂ¡o cĂ¡o Ä‘Ă¢y lĂ  káº¿t quáº£ há»£p lĂ½, khĂ´ng pháº£i lá»—i mĂ´ hĂ¬nh.")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Gender", y="Income")
plt.title("Thu nháº­p theo giá»›i tĂ­nh")
savefig("05_thu_nhap_theo_gioi.png")

# ----------------------------------------------------------------------------
# 6. Há»’I QUY: Dá»± bĂ¡o Äiá»ƒm chi tiĂªu dá»±a trĂªn Tuá»•i & Thu nháº­p
# ----------------------------------------------------------------------------
Xr = df[["Age", "Income"]]
yr = df["Spending"]
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
kf = KFold(5, shuffle=True, random_state=SEED)

reg_cfg = {
    "Linear Regression": (Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]), {}),
    "Poly + Ridge": (Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures()), ("m", Ridge())]),
                     {"poly__degree": [2, 3, 4], "m__alpha": [0.1, 1, 10, 100]}),
    "Random Forest": (RandomForestRegressor(random_state=SEED),
                      {"n_estimators": [200, 400], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                          {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]}),
}
rows, preds = [], {}
for name, (model, grid) in reg_cfg.items():
    gs = GridSearchCV(model, grid or {}, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    preds[name] = pred
    rows.append({"Mo_hinh": name, "Sieu_tham_so_tot_nhat": gs.best_params_,
                 "CV_RMSE": round(-gs.best_score_, 3),
                 "Test_RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
                 "Test_MAE": round(mean_absolute_error(yte, pred), 3),
                 "Test_R2": round(r2_score(yte, pred), 3)})
res_reg = pd.DataFrame(rows).sort_values("Test_RMSE")
print("\n", res_reg.to_string(index=False))
res_reg.to_csv(os.path.join(OUT, "06_ket_qua_hoi_quy.csv"), index=False)

best_name = res_reg.iloc[0]["Mo_hinh"]
plt.figure(figsize=(5.5, 5))
plt.scatter(yte, preds[best_name], alpha=0.7)
lim = [0, 100]
plt.plot(lim, lim, "r--")
plt.xlabel("Thá»±c táº¿"); plt.ylabel("Dá»± bĂ¡o"); plt.title(f"Thá»±c táº¿ vs Dá»± bĂ¡o - {best_name}")
savefig("06_thuc_te_vs_du_bao.png")

# Giáº£i thĂ­ch: táº§m quan trá»ng Ä‘áº·c trÆ°ng (Random Forest)
rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xr, yr)
pd.Series(rf.feature_importances_, index=Xr.columns).sort_values().plot.barh(title="Feature importance (RF)")
savefig("06_feature_importance.png")

print(f"\nHoĂ n táº¥t. ToĂ n bá»™ káº¿t quáº£ náº±m trong thÆ° má»¥c: {OUT}")