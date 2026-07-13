"""
app.py
------
Flask web application for the Human Development Index (HDI) Predictor.

Routes:
  /                -> Home / prediction form
  /predict         -> POST: run the ML models on user input, render result
  /api/predict      -> POST: JSON API version of the prediction
  /dashboard       -> EDA & model performance dashboard (static plots + metrics)
  /about           -> Explanation of HDI methodology
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ------------------------------------------------------------------
# Load trained artifacts once at startup
# ------------------------------------------------------------------
classifier = joblib.load(os.path.join(MODEL_DIR, "hdi_classifier.pkl"))
regressor = joblib.load(os.path.join(MODEL_DIR, "hdi_regressor.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
metrics = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))

CATEGORY_INFO = {
    "Very High": {
        "color": "#2E7D32",
        "badge": "success",
        "blurb": "This country ranks among the most developed nations, with strong "
                 "outcomes across health, education, and income.",
    },
    "High": {
        "color": "#66BB6A",
        "badge": "info",
        "blurb": "Solid overall development with room for targeted improvement in "
                 "one or more dimensions (health, education, or income).",
    },
    "Medium": {
        "color": "#FFA726",
        "badge": "warning",
        "blurb": "Development is progressing, but meaningful gaps remain in "
                 "healthcare access, educational attainment, or income generation.",
    },
    "Low": {
        "color": "#E53935",
        "badge": "danger",
        "blurb": "Significant developmental challenges exist. Prioritized "
                 "investment in health, education, and economic opportunity "
                 "is recommended.",
    },
}


def compute_sub_indices(life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita):
    """Replicates the official UNDP sub-index formulas, used only for the
    compass visualization (not for the ML prediction itself)."""
    lei = (life_expectancy - 20) / (85 - 20)
    mysi = mean_years_schooling / 15
    eysi = expected_years_schooling / 18
    ei = (mysi + eysi) / 2
    ii = (np.log(gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100))
    return (
        round(max(0.0, min(1.0, lei)), 3),
        round(max(0.0, min(1.0, ei)), 3),
        round(max(0.0, min(1.0, ii)), 3),
    )


def run_models(life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita):
    """Run both the classifier and regressor on a single input row."""
    log_gni = np.log(gni_per_capita)
    X = np.array([[life_expectancy, mean_years_schooling, expected_years_schooling, log_gni]])
    X_scaled = scaler.transform(X)

    pred_class_idx = classifier.predict(X_scaled)[0]
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
    pred_proba = classifier.predict_proba(X_scaled)[0]
    proba_dict = {
        label_encoder.classes_[i]: round(float(pred_proba[i]) * 100, 1)
        for i in range(len(label_encoder.classes_))
    }

    pred_score = float(regressor.predict(X_scaled)[0])
    pred_score = max(0.0, min(1.0, pred_score))

    return pred_class, proba_dict, pred_score


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        life_expectancy = float(request.form["life_expectancy"])
        mean_years_schooling = float(request.form["mean_years_schooling"])
        expected_years_schooling = float(request.form["expected_years_schooling"])
        gni_per_capita = float(request.form["gni_per_capita"])
        country_name = request.form.get("country_name", "").strip() or "Your Country"

        # basic sanity clipping to realistic HDI input ranges
        life_expectancy = max(20, min(90, life_expectancy))
        mean_years_schooling = max(0, min(20, mean_years_schooling))
        expected_years_schooling = max(0, min(23, expected_years_schooling))
        gni_per_capita = max(100, min(150000, gni_per_capita))

        pred_class, proba_dict, pred_score = run_models(
            life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita
        )
        lei, ei, ii = compute_sub_indices(
            life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita
        )

        info = CATEGORY_INFO[pred_class]

        return render_template(
            "result.html",
            country_name=country_name,
            life_expectancy=life_expectancy,
            mean_years_schooling=mean_years_schooling,
            expected_years_schooling=expected_years_schooling,
            gni_per_capita=gni_per_capita,
            pred_class=pred_class,
            pred_score=round(pred_score, 3),
            proba_dict=proba_dict,
            info=info,
            lei=lei, ei=ei, ii=ii,
        )
    except (ValueError, KeyError) as e:
        return render_template("index.html", error=f"Invalid input: {e}")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        life_expectancy = float(data["life_expectancy"])
        mean_years_schooling = float(data["mean_years_schooling"])
        expected_years_schooling = float(data["expected_years_schooling"])
        gni_per_capita = float(data["gni_per_capita"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    pred_class, proba_dict, pred_score = run_models(
        life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita
    )

    return jsonify({
        "hdi_category": pred_class,
        "hdi_score": round(pred_score, 4),
        "category_probabilities": proba_dict,
    })


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
