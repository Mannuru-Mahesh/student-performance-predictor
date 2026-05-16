"""
train_model.py  —  Train models and save them to /models/
Run:  python train_model.py
"""
import os, json
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (mean_absolute_error, r2_score,
                                     accuracy_score, classification_report,
                                     confusion_matrix)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("\n📂  Loading data …")
df = pd.read_csv("data/student_data.csv")
print(f"    {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Clean ─────────────────────────────────────────────────────────────────
print("🧹  Filling missing values with column medians …")
for col in ["study_hours", "sleep_hours", "attendance_percent"]:
    med = df[col].median()
    df[col] = df[col].fillna(med)

# Also fix any NaN in parental_education
df["parental_education"] = df["parental_education"].fillna("Secondary")

# ── 3. Encode ────────────────────────────────────────────────────────────────
print("🔢  Encoding categorical columns …")
df["internet_access_enc"]    = (df["internet_access"] == "Yes").astype(int)
df["extracurricular_enc"]    = (df["extracurricular"] == "Yes").astype(int)
edu_order = {"None":0,"Primary":1,"Secondary":2,"Bachelor":3,"Master":4}
df["parental_education_enc"] = df["parental_education"].map(edu_order)
df["pass_fail_enc"]          = (df["pass_fail"] == "Pass").astype(int)

FEATURES = ["study_hours","attendance_percent","previous_score",
            "sleep_hours","internet_access_enc","parental_education_enc",
            "extracurricular_enc"]

X       = df[FEATURES].values   # convert to numpy to avoid any pandas NaN surprises
y_score = df["final_score"].values
y_pass  = df["pass_fail_enc"].values

# Final NaN guard
assert not np.isnan(X).any(),   "NaN still in features!"
assert not np.isnan(y_score).any(), "NaN in y_score!"

# ── 4. Split ─────────────────────────────────────────────────────────────────
X_tr,X_te,ys_tr,ys_te,yp_tr,yp_te = train_test_split(
    X, y_score, y_pass, test_size=0.2, random_state=42)
print(f"✂️   Train: {len(X_tr)}  |  Test: {len(X_te)}")

# ── 5A. Linear Regression ─────────────────────────────────────────────────────
print("\n📈  Linear Regression → final_score")
lr = LinearRegression()
lr.fit(X_tr, ys_tr)
ys_pred = lr.predict(X_te)
lr_mae = mean_absolute_error(ys_te, ys_pred)
lr_r2  = r2_score(ys_te, ys_pred)
print(f"    MAE={lr_mae:.2f}  R²={lr_r2:.3f}")

# ── 5B. Random Forest ────────────────────────────────────────────────────────
print("\n🌲  Random Forest → pass/fail")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, yp_tr)
yp_pred = rf.predict(X_te)
rf_acc  = accuracy_score(yp_te, yp_pred)
print(f"    Accuracy={rf_acc:.3f}")
print(classification_report(yp_te, yp_pred, target_names=["Fail","Pass"]))

# ── 5C. Logistic Regression (comparison) ─────────────────────────────────────
print("📊  Logistic Regression → pass/fail (comparison)")
log = LogisticRegression(max_iter=500, random_state=42)
log.fit(X_tr, yp_tr)
log_acc = accuracy_score(yp_te, log.predict(X_te))
print(f"    Accuracy={log_acc:.3f}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/score_model.pkl")
joblib.dump(rf, "models/pass_fail_model.pkl")
print("\n💾  Models saved  →  models/score_model.pkl  &  models/pass_fail_model.pkl")

# ── 7. Save metrics JSON ──────────────────────────────────────────────────────
metrics = {
    "linear_regression": {"mae": round(lr_mae,2), "r2": round(lr_r2,3)},
    "random_forest":     {
        "accuracy": round(rf_acc,3),
        "report":   classification_report(yp_te, yp_pred,
                        target_names=["Fail","Pass"], output_dict=True),
        "confusion":confusion_matrix(yp_te, yp_pred).tolist(),
    },
    "logistic_regression": {"accuracy": round(log_acc,3)},
    "features": FEATURES,
    "train_size": int(len(X_tr)),
    "test_size":  int(len(X_te)),
}
with open("models/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("📊  Metrics  →  models/metrics.json")
print("\n✅  Done!\n")
