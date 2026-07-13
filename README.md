# Atlas — Human Development Index (HDI) Predictor

A complete end-to-end machine learning web application that predicts a
country's **Human Development Index (HDI)** tier — *Very High, High, Medium,
or Low* — and its continuous HDI score, from four standard development
indicators:

- Life expectancy at birth
- Mean years of schooling
- Expected years of schooling
- GNI per capita (PPP)

Built with **Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and
Flask**.

---

## Project structure

```
hdi_project/
├── app.py                     # Flask web application (routes, prediction logic)
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py       # Synthetic dataset generator (UNDP-formula based)
│   └── hdi_dataset.csv        # Generated dataset (660 country-profiles)
├── model/
│   ├── train_model.py         # EDA + trains classifier & regressor, saves plots
│   ├── hdi_classifier.pkl      # Trained RandomForestClassifier
│   ├── hdi_regressor.pkl       # Trained RandomForestRegressor
│   ├── scaler.pkl              # StandardScaler fit on training features
│   ├── label_encoder.pkl       # LabelEncoder for the 4 HDI tiers
│   └── metrics.pkl             # Saved accuracy / R² / MAE
├── templates/                 # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html             # Prediction form (home page)
│   ├── result.html            # Prediction result page
│   ├── dashboard.html         # Model performance & EDA dashboard
│   ├── about.html             # HDI methodology explainer
│   └── _compass_macro.html    # Reusable "3-pillar compass" SVG visualization
└── static/
    ├── css/style.css          # Design system ("Atlas" ledger/compass theme)
    └── images/                # Saved EDA & model diagnostic plots (PNG)
```

---

## How it works

### 1. Data (`data/generate_data.py`)
Since real UNDP country data isn't bundled here, a **synthetic but formula-
consistent** dataset of 660 country-profiles is generated. Each profile is
sampled from one of three development archetypes (advanced, emerging,
developing economies), then its true HDI is computed using the **official
UNDP formula**:

```
LEI  = (Life Expectancy − 20) / (85 − 20)
MYSI = Mean Years of Schooling / 15
EYSI = Expected Years of Schooling / 18
EI   = (MYSI + EYSI) / 2
II   = (ln(GNI per capita) − ln(100)) / (ln(75000) − ln(100))
HDI  = (LEI × EI × II) ^ (1/3)
```

Countries are then labeled into tiers (Very High ≥ 0.800, High 0.700–0.799,
Medium 0.550–0.699, Low < 0.550), and small measurement noise is added for
realism.

> To use **real** HDI data instead, drop a CSV with the same four indicator
> columns (`life_expectancy`, `mean_years_schooling`,
> `expected_years_schooling`, `gni_per_capita`) plus `hdi_score` /
> `hdi_category` into `data/hdi_dataset.csv` and re-run `train_model.py`.

### 2. Modeling (`model/train_model.py`)
- Performs EDA (correlation heatmap, category distribution, scatter plots,
  boxplots) and saves the charts as PNGs.
- Trains a **RandomForestClassifier** to predict the discrete HDI tier.
- Trains a **RandomForestRegressor** to predict the continuous HDI score.
- Evaluates both with accuracy/classification-report and R²/MAE, saving a
  confusion matrix, feature-importance chart, and actual-vs-predicted plot.
- Persists all artifacts (models, scaler, label encoder, metrics) with
  `joblib`.

### 3. Web app (`app.py`)
A Flask app with:
- **`/`** — input form for the four indicators (with quick-fill presets for
  each development archetype).
- **`/predict`** — runs both models server-side and renders a result page
  with the predicted tier, HDI score, per-tier probabilities, and a
  three-ring "compass" visualization of the Health / Education / Income
  sub-indices.
- **`/api/predict`** — JSON API version of the same prediction (POST body:
  `life_expectancy`, `mean_years_schooling`, `expected_years_schooling`,
  `gni_per_capita`).
- **`/dashboard`** — model metrics and all EDA/diagnostic plots.
- **`/about`** — HDI methodology explainer.

---

## Setup & run

```bash
# 1. Create environment (Anaconda or venv)
conda create -n hdi-app python=3.11 -y
conda activate hdi-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Re)generate data and train models — artifacts are already included,
#    but you can regenerate them at any time:
python data/generate_data.py
python model/train_model.py

# 4. Run the Flask app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Example API usage

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "life_expectancy": 82.4,
        "mean_years_schooling": 12.8,
        "expected_years_schooling": 17.1,
        "gni_per_capita": 52000
      }'
```

Response:
```json
{
  "hdi_category": "Very High",
  "hdi_score": 0.923,
  "category_probabilities": {
    "Very High": 100.0,
    "High": 0.0,
    "Medium": 0.0,
    "Low": 0.0
  }
}
```

---

## Model performance (on synthetic test set)

| Metric | Value |
|---|---|
| Classifier accuracy | ~91% |
| Regressor R² | ~0.986 |
| Regressor MAE | ~0.015 |

Full diagnostics (confusion matrix, feature importance, actual-vs-predicted)
are available on the `/dashboard` page.
