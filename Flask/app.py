"""
Human Development Index (HDI) Predictor
Flask backend - Epic 8

Routes:
    GET  /          -> Home page with the prediction form (indexnew.html)
    POST /predict    -> Handles form submission, runs the model, renders resultnew.html
"""
from flask import Flask, render_template, request
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "HDI.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_cols = bundle["feature_cols"]
countries = bundle["countries"]


def classify_hdi(score):
    """Map a numeric HDI score to its UNDP-style category."""
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"


@app.route("/")
def home():
    return render_template("indexnew.html", countries=countries)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        country = request.form.get("country", "").strip()
        life_expectancy = float(request.form.get("life_expectancy"))
        mean_schooling = float(request.form.get("mean_schooling"))
        gni = float(request.form.get("gni"))
        internet_users = float(request.form.get("internet_users"))

        features = pd.DataFrame(
            [[life_expectancy, mean_schooling, gni, internet_users]],
            columns=feature_cols,
        )
        prediction = model.predict(features)[0]
        prediction = float(np.clip(prediction, 0, 1))
        category = classify_hdi(prediction)

        return render_template(
            "resultnew.html",
            country=country if country else "Selected Country",
            hdi_score=round(prediction, 3),
            hdi_category=category,
            life_expectancy=life_expectancy,
            mean_schooling=mean_schooling,
            gni=gni,
            internet_users=internet_users,
        )
    except (TypeError, ValueError):
        return render_template(
            "indexnew.html",
            countries=countries,
            error="Please fill in all fields with valid numeric values.",
        )


if __name__ == "__main__":
    app.run(debug=True)
