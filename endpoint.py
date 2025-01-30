from flask import Flask, request, jsonify
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

with open("C:\\Edu\\Deep learning\\ml week\\model\\xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Edu\\Deep learning\\ml week\\model\\residual_errors.pkl", "rb") as f:
    residual_errors = pickle.load(f)

conformal_quantile = np.percentile(residual_errors, 95)

app = Flask(__name__)

FEATURES = ["event", "section", "row", "seat", "quantity", "min_option", "hours"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        dmatrix = xgb.DMatrix(df)
        predicted_price = model.predict(dmatrix)[0]

        lower_bound = max(0, predicted_price - conformal_quantile)  
        upper_bound = predicted_price + conformal_quantile  # Верхня межа

        return jsonify({
            "predicted_price": float(round(predicted_price, 2)),
            "confidence_interval": {
                "lower_bound": float(round(lower_bound, 2)),
                "upper_bound": float(round(upper_bound, 2))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
