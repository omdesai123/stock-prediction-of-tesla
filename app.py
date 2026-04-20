"""
app.py
Tesla Stock Predictor — Flask Backend
--------------------------------------
Loads model.pkl, exposes a /predict POST endpoint,
renders index.html with the prediction result and probability.
"""

import pickle
import numpy as np
from flask import Flask, request, render_template

# ─────────────────────────────────────────────
# Initialise Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────
# Load the trained model once at startup
# ─────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names must match exactly what was used during training
FEATURE_COLS = ["MA10", "MA50", "Daily_Return", "Avg_Vol_7", "MA_Ratio"]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the home page with the input form."""
    return render_template("index.html", prediction=None, probability=None)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form data, run the model, and return the prediction.
    Expected form fields: ma10, ma50, daily_return, avg_vol_7, ma_ratio
    """
    try:
        # Parse and validate form inputs
        ma10         = float(request.form["ma10"])
        ma50         = float(request.form["ma50"])
        daily_return = float(request.form["daily_return"])
        avg_vol_7    = float(request.form["avg_vol_7"])
        ma_ratio     = float(request.form["ma_ratio"])

        # Assemble feature vector in the same order used during training
        features = np.array([[ma10, ma50, daily_return, avg_vol_7, ma_ratio]])

        # Predict class (0 = DOWN, 1 = UP) and get probability for class 1
        prediction_class = model.predict(features)[0]
        prob_up          = model.predict_proba(features)[0][1]

        # Human-readable label
        prediction_label = "UP 📈" if prediction_class == 1 else "DOWN 📉"

        # Convert probability to a percentage for the progress bar
        probability_pct = round(prob_up * 100, 1)

        return render_template(
            "index.html",
            prediction=prediction_label,
            probability=probability_pct,
            # Pass back input values so the form stays populated
            ma10=ma10,
            ma50=ma50,
            daily_return=daily_return,
            avg_vol_7=avg_vol_7,
            ma_ratio=round(ma10 / ma50, 4) if ma50 != 0 else 0,
        )

    except (ValueError, KeyError) as e:
        # Return a user-friendly error message
        return render_template(
            "index.html",
            prediction="Error",
            probability=None,
            error=str(e),
        )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
