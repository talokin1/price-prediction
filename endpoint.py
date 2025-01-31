from flask import Flask, request, jsonify
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("residual_errors.pkl", "rb") as f:
    residual_errors = pickle.load(f)

app = Flask(__name__)

FEATURES = ["event", "section", "row", "seat", "quantity", "min_option", "hours"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        confidence = data.pop("confidence", 50)  # Default to 95% if not provided

        df = pd.DataFrame([data])
        data_matrix = xgb.DMatrix(df)

        predicted_price = model.predict(data_matrix)[0]

        # Compute confidence interval based on the provided confidence level
        conformal_quantile = np.percentile(residual_errors, confidence)
        lower_bound = max(0, predicted_price - conformal_quantile)
        upper_bound = predicted_price + conformal_quantile

        return jsonify({
            "predicted_price": float(predicted_price),
            "confidence_intervals": {
                "confidence": float(confidence),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
