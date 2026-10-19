import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from generate_data import generate

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
acc = lambda y, p: float((np.asarray(y) == np.asarray(p)).mean())


def title(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


title("1. DATA & PRA-PEMROSESAN")
df = generate()
df.to_csv("data/loan_data.csv", index=False)
print(f"Jumlah data : {len(df)}")
na = df.isna().sum()
print(f"Data kosong :\n{na[na > 0].to_string()}")
print(f"Rasio disetujui: {df['approved'].mean():.1%}")

FEATURES = ["age", "income", "credit_score", "debt_ratio", "employment_years", "gender"]
X, y = df[FEATURES], df["approved"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
imp = SimpleImputer(strategy="median").fit(X_tr)
X_tr = pd.DataFrame(imp.transform(X_tr), columns=FEATURES, index=X_tr.index)
X_te = pd.DataFrame(imp.transform(X_te), columns=FEATURES, index=X_te.index)
print(f"Data latih: {len(X_tr)} | Data uji: {len(X_te)}")

title("2. PENDEKATAN RULE-BASED (if-then)")
rule_pred = ((X_te["credit_score"] >= 580) & (X_te["debt_ratio"] < 0.5)
             & (X_te["income"] >= 5)).astype(int)
rule_acc = acc(y_te, rule_pred)
print(f"Baseline (tebak 'disetujui' semua)        : {acc(y_te, np.ones(len(y_te))):.1%}")
print(f"Akurasi rule-based (aturan ditulis manual): {rule_acc:.1%}")

title("3. SUPERVISED LEARNING (KLASIFIKASI)")
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

logreg = LogisticRegression(max_iter=1000).fit(Xs_tr, y_tr)
lr_acc = acc(y_te, logreg.predict(Xs_te))
print(f"Logistic Regression  -> akurasi uji: {lr_acc:.1%}")

tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_tr, y_tr)
dt_acc = acc(y_te, tree.predict(X_te))
print(f"Decision Tree (d=4)  -> akurasi uji: {dt_acc:.1%}")

title("4. OVERFITTING vs UNDERFITTING")
depths = list(range(1, 16))
tr_scores, te_scores = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
    tr_scores.append(acc(y_tr, m.predict(X_tr)))
    te_scores.append(acc(y_te, m.predict(X_te)))
best_d = depths[int(np.argmax(te_scores))]
print(f"{'Depth':<7}{'Latih':<10}{'Uji':<10}Kondisi")
for d, a, b in zip(depths, tr_scores, te_scores):
    if d in (1, best_d, 15):
        status = "UNDERFITTING" if d == 1 else ("OPTIMAL" if d == best_d else "OVERFITTING")
        print(f"{d:<7}{a:<10.1%}{b:<10.1%}{status}")

plt.figure(figsize=(7, 4))
plt.plot(depths, tr_scores, "o-", label="Akurasi latih")
plt.plot(depths, te_scores, "s-", label="Akurasi uji")
plt.axvline(best_d, ls="--", c="gray", label=f"Depth optimal = {best_d}")
plt.xlabel("Kedalaman pohon (kompleksitas model)")
plt.ylabel("Akurasi")
plt.title("Overfitting vs Underfitting")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("outputs/overfitting_curve.png", dpi=150); plt.close()

title("5. UNSUPERVISED LEARNING (CLUSTERING PELANGGAN)")
cl_cols = ["income", "credit_score", "debt_ratio"]
full = pd.concat([X_tr, X_te])[cl_cols]
Z = StandardScaler().fit_transform(full)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
full = full.assign(cluster=km.labels_)
profile = full.groupby("cluster").mean().round(2)
profile["jumlah"] = full.groupby("cluster").size()
print(profile.to_string())

plt.figure(figsize=(7, 4.5))
sc = plt.scatter(full["income"], full["credit_score"], c=full["cluster"], cmap="viridis", s=12, alpha=.7)
plt.xlabel("Pendapatan (juta Rp/bulan)"); plt.ylabel("Skor kredit")
plt.title("Segmentasi Pelanggan (K-Means, k=3)")
plt.legend(*sc.legend_elements(), title="Klaster"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("outputs/clusters.png", dpi=150); plt.close()

title("6. ANALISIS BIAS (ETIKA)")
g_te = X_te["gender"].values
rate = lambda pred, g: float(np.asarray(pred)[g_te == g].mean())
pred_with = logreg.predict(Xs_te)
f_no = [f for f in FEATURES if f != "gender"]
sc2 = StandardScaler().fit(X_tr[f_no])
lr_fair = LogisticRegression(max_iter=1000).fit(sc2.transform(X_tr[f_no]), y_tr)
pred_without = lr_fair.predict(sc2.transform(X_te[f_no]))

rows = []
for name, p in [("Data historis (label asli)", y_te.values),
                ("Model DENGAN fitur gender", pred_with),
                ("Model TANPA fitur gender", pred_without)]:
    f, m = rate(p, 0), rate(p, 1)
    rows.append((name, f, m, m - f))
print(f"{'Skenario':<30}{'Perempuan':<12}{'Laki-laki':<12}Selisih")
for n, f, m, g in rows:
    print(f"{n:<30}{f:<12.1%}{m:<12.1%}{g:+.1%}")
print(f"Akurasi model tanpa gender: {acc(y_te, pred_without):.1%}")

title("7. EXPLAINABLE AI (KOEFISIEN MODEL)")
coef = pd.Series(logreg.coef_[0], index=FEATURES).sort_values()
print(coef.round(3).to_string())
plt.figure(figsize=(7, 4))
coef.plot(kind="barh", color=["#d9534f" if i == "gender" else "#4c8bf5" for i in coef.index])
plt.title("Pengaruh Fitur terhadap Persetujuan (merah = fitur sensitif)")
plt.xlabel("Koefisien (data terstandarisasi)"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

title("RINGKASAN")
print(f"Rule-based: {rule_acc:.1%} | LogReg: {lr_acc:.1%} | Decision Tree: {dt_acc:.1%}")
print("Grafik tersimpan di folder outputs/")
