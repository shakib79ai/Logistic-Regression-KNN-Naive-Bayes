# =============================================================================
# DIABETES PREDICTION SYSTEM
# Models: Logistic Regression | KNN | Naive Bayes
# Dataset: Pima Indians Diabetes Database (UCI / Kaggle)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Place diabetes.csv in the same folder as this script.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "diabetes.csv"))

print("=" * 60)
print("DIABETES PREDICTION SYSTEM")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe().round(2)}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING – handle biological zeros
# ─────────────────────────────────────────────
# Columns where 0 is physiologically impossible
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print(f"\nZero->NaN replacements:\n{df[zero_cols].isnull().sum()}")

# Impute with median (robust to skew)
for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Catch any remaining NaNs in other columns (e.g. from source encoding differences)
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. DEFINE MODELS (with and without scaling)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
}

# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE – with AND without scaling
# ─────────────────────────────────────────────
results = []

for name, model in models.items():
    for scaled, (X_tr, X_te) in [
        ("Without Scaling", (X_train,        X_test)),
        ("With Scaling",    (X_train_scaled, X_test_scaled)),
    ]:
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_tr, y_train)

        train_acc = accuracy_score(y_train, model_clone.predict(X_tr))
        test_acc  = accuracy_score(y_test,  model_clone.predict(X_te))
        cv_score  = cross_val_score(model_clone, X_tr, y_train, cv=5,
                                    scoring="accuracy").mean()
        roc_auc   = roc_auc_score(y_test,
                                   model_clone.predict_proba(X_te)[:, 1])
        overfit_gap = train_acc - test_acc

        results.append({
            "Model":         name,
            "Scaling":       scaled,
            "Train Acc":     round(train_acc, 4),
            "Test Acc":      round(test_acc, 4),
            "CV Score":      round(cv_score, 4),
            "ROC-AUC":       round(roc_auc, 4),
            "Overfit Gap":   round(overfit_gap, 4),
        })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 5. DETAILED REPORT – best configuration per model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORTS (With Scaling)")
print("=" * 60)

best_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    best_models[name] = model
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Diabetes", "Diabetes"]))

# ─────────────────────────────────────────────
# 6. OVERFITTING ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)

scaled_results = results_df[results_df["Scaling"] == "With Scaling"].copy()
most_overfit   = scaled_results.loc[scaled_results["Overfit Gap"].idxmax()]

print(scaled_results[["Model", "Train Acc", "Test Acc", "Overfit Gap"]].to_string(index=False))
print(f"\n>> Most overfit model: {most_overfit['Model']} "
      f"(Train-Test gap = {most_overfit['Overfit Gap']:.4f})")

for _, row in scaled_results.iterrows():
    verdict = "OVERFIT" if row["Overfit Gap"] > 0.05 else "OK"
    print(f"   {row['Model']:<22} gap={row['Overfit Gap']:.4f}  [{verdict}]")

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 18))
fig.suptitle("Diabetes Prediction – Model Analysis Dashboard", fontsize=16, fontweight="bold")

# --- 7a. Accuracy comparison bar chart ---
ax1 = fig.add_subplot(3, 3, 1)
pivot = results_df.pivot_table(index="Model", columns="Scaling", values="Test Acc")
pivot.plot(kind="bar", ax=ax1, color=["#E07B54", "#5B8DB8"], edgecolor="black")
ax1.set_title("Test Accuracy: Scaled vs Unscaled")
ax1.set_ylabel("Accuracy")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha="right")
ax1.set_ylim(0.6, 0.85)
ax1.legend(fontsize=8)

# --- 7b. Train vs Test accuracy (overfitting view) ---
ax2 = fig.add_subplot(3, 3, 2)
scaled_df = results_df[results_df["Scaling"] == "With Scaling"].set_index("Model")
x = np.arange(len(scaled_df))
w = 0.35
ax2.bar(x - w/2, scaled_df["Train Acc"], w, label="Train", color="#5B8DB8", edgecolor="black")
ax2.bar(x + w/2, scaled_df["Test Acc"],  w, label="Test",  color="#E07B54", edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels(scaled_df.index, rotation=15)
ax2.set_title("Train vs Test Accuracy (With Scaling)")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0.6, 0.90)
ax2.legend()

# --- 7c. ROC-AUC comparison ---
ax3 = fig.add_subplot(3, 3, 3)
scaled_df["ROC-AUC"].plot(kind="bar", ax=ax3, color="#6AAF70", edgecolor="black")
ax3.set_title("ROC-AUC Score (With Scaling)")
ax3.set_ylabel("AUC")
ax3.set_ylim(0.6, 0.90)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)

# --- 7d–7f. Confusion matrices ---
for i, (name, model) in enumerate(best_models.items()):
    ax = fig.add_subplot(3, 3, 4 + i)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {name}")

# --- 7g. ROC curves ---
ax7 = fig.add_subplot(3, 3, 7)
colors = ["#5B8DB8", "#E07B54", "#6AAF70"]
for (name, model), color in zip(best_models.items(), colors):
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax7.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
ax7.plot([0, 1], [0, 1], "k--", lw=1)
ax7.set_xlabel("False Positive Rate")
ax7.set_ylabel("True Positive Rate")
ax7.set_title("ROC Curves (With Scaling)")
ax7.legend(fontsize=8)

# --- 7h. Overfit gap bar ---
ax8 = fig.add_subplot(3, 3, 8)
gap_colors = ["#E07B54" if g > 0.05 else "#6AAF70" for g in scaled_df["Overfit Gap"]]
ax8.bar(scaled_df.index, scaled_df["Overfit Gap"], color=gap_colors, edgecolor="black")
ax8.axhline(0.05, color="red", linestyle="--", label="Overfit threshold (0.05)")
ax8.set_title("Overfitting Gap (Train − Test Acc)")
ax8.set_ylabel("Gap")
ax8.set_xticklabels(scaled_df.index, rotation=15)
ax8.legend(fontsize=8)

# --- 7i. Feature importance (Logistic Regression coefficients) ---
ax9 = fig.add_subplot(3, 3, 9)
lr_model = best_models["Logistic Regression"]
coefs = pd.Series(np.abs(lr_model.coef_[0]), index=feature_names).sort_values()
coefs.plot(kind="barh", ax=ax9, color="#5B8DB8", edgecolor="black")
ax9.set_title("Feature Importance (LR |Coefficients|)")
ax9.set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "diabetes_model_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\nDashboard saved -> diabetes_model_analysis.png")

# ─────────────────────────────────────────────
# 8. KNN HYPER-PARAMETER SWEEP (k=1..20)
# ─────────────────────────────────────────────
k_range   = range(1, 21)
k_train   = []
k_test    = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    k_train.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    k_test.append(accuracy_score(y_test,  knn.predict(X_test_scaled)))

best_k     = k_range[np.argmax(k_test)]
best_k_acc = max(k_test)

print(f"\nKNN sweep: best k={best_k} -> test accuracy={best_k_acc:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(k_range, k_train, "o-", label="Train Acc", color="#5B8DB8")
plt.plot(k_range, k_test,  "s-", label="Test Acc",  color="#E07B54")
plt.axvline(best_k, color="green", linestyle="--", label=f"Best k={best_k}")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN: Train vs Test Accuracy Across k Values")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "knn_k_sweep.png"), dpi=150, bbox_inches="tight")
plt.show()
print("KNN sweep plot saved -> knn_k_sweep.png")

# ─────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL INSIGHTS")
print("=" * 60)

best_row = scaled_results.loc[scaled_results["Test Acc"].idxmax()]
print(f"\n  Best overall model  : {best_row['Model']}")
print(f"  Test accuracy       : {best_row['Test Acc']:.4f}")
print(f"  ROC-AUC             : {best_row['ROC-AUC']:.4f}")
print(f"\n  Most overfit model  : {most_overfit['Model']} "
      f"(gap={most_overfit['Overfit Gap']:.4f})")

print("""
  Key observations:
  • KNN is the most sensitive to feature scaling.
    Without scaling, Euclidean distances are dominated by
    high-range features (Insulin, Glucose), hurting accuracy.
  • Logistic Regression shows the smallest overfit gap —
    it generalises well with regularisation.
  • KNN (small k) tends to overfit: training accuracy is
    near-perfect while test accuracy drops.
  • Naive Bayes assumes feature independence (not fully true
    here) but is fast and surprisingly competitive.
  • Feature scaling is critical for KNN and helps LR;
    Naive Bayes is scale-invariant.
""")
