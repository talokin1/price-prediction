import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore

class TicketPricePredictorXGB:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = self.load_data()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)
    
    def encode_categorical(self, categorical_cols: list):
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
    
    def remove_outliers(self, threshold: float = 2.0):
        z_scores = np.abs(self.df.drop(columns=["price"]).apply(zscore))
        mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[mask]
    
    def prepare_data(self):
        X = self.df.drop(columns=["price"])
        y = self.df["price"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        self.scaler.fit(X_train)
        return (
            self.scaler.transform(X_train), y_train,
            self.scaler.transform(X_val), y_val,
            self.scaler.transform(X_test), y_test
        )
    
    def train_model(self, X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 10,
            "reg_lambda": 20
        }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "Validation")],
            early_stopping_rounds=50,
            verbose_eval=10
        )
    
    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series):
        dtest = xgb.DMatrix(X_test, label=y_test)
        y_pred = self.model.predict(dtest)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
        
        print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.2f}")
        return metrics
    
    def save_artifacts(self, path: str = "./"):
        with open(f"{path}xgb_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(f"{path}scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
    
    def plot_feature_importance(self):
        xgb.plot_importance(self.model, importance_type="weight", max_num_features=10)
        plt.show()

def main():
    predictor = TicketPricePredictorXGB("C:\\Edu\\Deep learning\\ml week\\data\\preprocessed.csv")
    predictor.encode_categorical(["section", "row"])
    predictor.remove_outliers()
    
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_data()
    predictor.train_model(X_train, y_train, X_val, y_val)
    predictor.evaluate_model(X_test, y_test)
    predictor.save_artifacts()
    predictor.plot_feature_importance()

if __name__ == "__main__":
    main()
