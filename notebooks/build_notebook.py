import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# HDI Exploratory Data Analysis & Model Training

This notebook walks through the same pipeline used in `data/generate_data.py`
and `model/train_model.py`, in an interactive Anaconda / Jupyter environment.

**Sections**
1. Load the synthetic HDI dataset
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Train/test split
5. Train a RandomForest classifier (HDI tier) and regressor (HDI score)
6. Evaluate and visualize results
"""))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error

sns.set_style("whitegrid")
%matplotlib inline
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load data"))
cells.append(nbf.v4.new_code_cell(
"""df = pd.read_csv("../data/hdi_dataset.csv")
print(df.shape)
df.head()
"""))

cells.append(nbf.v4.new_code_cell(
"""df.describe()
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis"))
cells.append(nbf.v4.new_code_cell(
"""order = ["Low", "Medium", "High", "Very High"]
counts = df["hdi_category"].value_counts().reindex(order)
sns.barplot(x=counts.index, y=counts.values)
plt.title("HDI Category Distribution")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
corr = df[["life_expectancy","mean_years_schooling","expected_years_schooling","gni_per_capita","hdi_score"]].corr()
sns.heatmap(corr, annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(7,5))
for cat in order:
    sub = df[df["hdi_category"]==cat]
    plt.scatter(sub["gni_per_capita"], sub["life_expectancy"], label=cat, alpha=0.6, s=25)
plt.xscale("log")
plt.xlabel("GNI per Capita (log scale)")
plt.ylabel("Life Expectancy")
plt.legend()
plt.title("Life Expectancy vs GNI per Capita")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Feature engineering\n\nGNI per capita is log-normally distributed in the real world (a small number of very rich countries), so we use `log(gni_per_capita)` as the model feature — this matches the actual UNDP income-index formula, which is also logarithmic."))
cells.append(nbf.v4.new_code_cell(
"""df["log_gni_per_capita"] = np.log(df["gni_per_capita"])
features = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "log_gni_per_capita"]
X = df[features].values

label_encoder = LabelEncoder()
label_encoder.fit(order)
y_class = label_encoder.transform(df["hdi_category"].values)
y_reg = df["hdi_score"].values
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Train/test split + scaling"))
cells.append(nbf.v4.new_code_cell(
"""X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s.shape, X_test_s.shape
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Train models"))
cells.append(nbf.v4.new_code_cell(
"""clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
clf.fit(X_train_s, yc_train)

reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
reg.fit(X_train_s, yr_train)
print("Models trained.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Evaluate"))
cells.append(nbf.v4.new_code_cell(
"""yc_pred = clf.predict(X_test_s)
acc = accuracy_score(yc_test, yc_pred)
print("Accuracy:", acc)
print(classification_report(yc_test, yc_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(yc_test, yc_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Acc={acc:.2%})")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell(
"""yr_pred = reg.predict(X_test_s)
r2 = r2_score(yr_test, yr_pred)
mae = mean_absolute_error(yr_test, yr_pred)
print("R2:", r2, " MAE:", mae)

plt.figure(figsize=(6,6))
plt.scatter(yr_test, yr_pred, alpha=0.5)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("Actual HDI"); plt.ylabel("Predicted HDI")
plt.title(f"Actual vs Predicted (R2={r2:.3f})")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(
"""## Next steps

- Swap the synthetic dataset for real UNDP HDI data (available at hdr.undp.org) for production use.
- Try gradient boosting (XGBoost/LightGBM) for potentially higher accuracy.
- Expose this pipeline via the Flask app in `../app.py`.
"""))

nb['cells'] = cells

with open("/home/claude/hdi_project/notebooks/hdi_eda_and_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
