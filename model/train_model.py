"""
train_model.py
--------------
Loads the synthetic HDI dataset, performs EDA (saved as static plots),
trains:
  1. A RandomForestClassifier  -> predicts hdi_category (Very High/High/Medium/Low)
  2. A RandomForestRegressor   -> predicts the continuous hdi_score

Both models + the StandardScaler are persisted with joblib so the Flask
app can load them at request time without retraining.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error
)

BASE_DIR = "/home/claude/hdi_project"
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(IMG_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {"Very High": "#2E7D32", "High": "#66BB6A", "Medium": "#FFA726", "Low": "#E53935"}

FEATURES = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"]
CATEGORY_ORDER = ["Low", "Medium", "High", "Very High"]

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

# Log-transform GNI for modeling (income effect is logarithmic, matches real HDI formula)
df["log_gni_per_capita"] = np.log(df["gni_per_capita"])
MODEL_FEATURES = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "log_gni_per_capita"]

# ------------------------------------------------------------------
# 2. EDA plots
# ------------------------------------------------------------------

# 2a. Correlation heatmap
plt.figure(figsize=(7, 5.5))
corr = df[FEATURES + ["hdi_score"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Heatmap of HDI Indicators", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=130)
plt.close()

# 2b. Distribution of HDI category counts
plt.figure(figsize=(6.5, 5))
order = CATEGORY_ORDER
counts = df["hdi_category"].value_counts().reindex(order)
sns.barplot(x=counts.index, y=counts.values, palette=[PALETTE[c] for c in order])
plt.title("Distribution of HDI Categories in Training Data", fontsize=13, fontweight="bold")
plt.xlabel("HDI Category")
plt.ylabel("Number of Countries")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "category_distribution.png"), dpi=130)
plt.close()

# 2c. Pairwise scatter: life expectancy vs GNI, colored by category
plt.figure(figsize=(7, 5.5))
for cat in order:
    sub = df[df["hdi_category"] == cat]
    plt.scatter(sub["gni_per_capita"], sub["life_expectancy"],
                label=cat, alpha=0.6, s=25, color=PALETTE[cat])
plt.xscale("log")
plt.xlabel("GNI per Capita (PPP $, log scale)")
plt.ylabel("Life Expectancy (years)")
plt.title("Life Expectancy vs. GNI per Capita by HDI Category", fontsize=13, fontweight="bold")
plt.legend(title="HDI Category")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "life_vs_gni_scatter.png"), dpi=130)
plt.close()

# 2d. Boxplot of schooling by category
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(data=df, x="hdi_category", y="mean_years_schooling", order=order,
            palette=[PALETTE[c] for c in order], ax=axes[0])
axes[0].set_title("Mean Years of Schooling by Category")
axes[0].set_xlabel("")
sns.boxplot(data=df, x="hdi_category", y="expected_years_schooling", order=order,
            palette=[PALETTE[c] for c in order], ax=axes[1])
axes[1].set_title("Expected Years of Schooling by Category")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "schooling_boxplots.png"), dpi=130)
plt.close()

print("EDA plots saved to", IMG_DIR)

# ------------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------------
X = df[MODEL_FEATURES].values
y_class_raw = df["hdi_category"].values
y_reg = df["hdi_score"].values

label_encoder = LabelEncoder()
label_encoder.fit(CATEGORY_ORDER)  # fix consistent ordering
y_class = label_encoder.transform(y_class_raw)

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Train classifier
# ------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1
)
clf.fit(X_train_scaled, yc_train)
yc_pred = clf.predict(X_test_scaled)

acc = accuracy_score(yc_test, yc_pred)
print(f"\nClassifier accuracy: {acc:.4f}")
print(classification_report(yc_test, yc_pred, target_names=label_encoder.classes_))

# Confusion matrix plot
cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy = {acc:.2%})", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=130)
plt.close()

# Feature importance plot
importances = clf.feature_importances_
feat_names_display = ["Life Expectancy", "Mean Years Schooling", "Expected Years Schooling", "log(GNI per Capita)"]
order_idx = np.argsort(importances)
plt.figure(figsize=(7, 4.5))
plt.barh(np.array(feat_names_display)[order_idx], importances[order_idx], color="#1976D2")
plt.title("Feature Importance — HDI Category Classifier", fontsize=12, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=130)
plt.close()

# ------------------------------------------------------------------
# 5. Train regressor (continuous HDI score)
# ------------------------------------------------------------------
reg = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
)
reg.fit(X_train_scaled, yr_train)
yr_pred = reg.predict(X_test_scaled)

r2 = r2_score(yr_test, yr_pred)
mae = mean_absolute_error(yr_test, yr_pred)
print(f"\nRegressor R^2: {r2:.4f}")
print(f"Regressor MAE: {mae:.4f}")

# Actual vs predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(yr_test, yr_pred, alpha=0.5, color="#5E35B1", s=25)
plt.plot([0, 1], [0, 1], "r--", linewidth=1.5)
plt.xlabel("Actual HDI Score")
plt.ylabel("Predicted HDI Score")
plt.title(f"Actual vs Predicted HDI Score (R² = {r2:.3f})", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "actual_vs_predicted.png"), dpi=130)
plt.close()

# ------------------------------------------------------------------
# 6. Persist model artifacts
# ------------------------------------------------------------------
joblib.dump(clf, os.path.join(MODEL_DIR, "hdi_classifier.pkl"))
joblib.dump(reg, os.path.join(MODEL_DIR, "hdi_regressor.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

metrics = {
    "classifier_accuracy": float(acc),
    "regressor_r2": float(r2),
    "regressor_mae": float(mae),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.pkl"))

print("\nSaved model artifacts to", MODEL_DIR)
print("Metrics:", metrics)
