import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# Завантаження даних
df = pd.read_csv("C:\\Edu\\Deep learning\\ml week\\data\\preprocessed.csv")

for col in ["section", "row"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Compute Z-scores for all columns
z_scores = np.abs(df.apply(zscore))

# Keep rows where all column values have |Z-score| < 3
df = df[(z_scores < 2).all(axis=1)]

scaler = StandardScaler()
# scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 10,
    "reg_lambda": 20,
    "n_estimators": 1000,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50,
    verbose_eval=10
)

y_pred = model.predict(dtest)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

xgb.plot_importance(model, importance_type="weight", max_num_features=10)

plt.show()
