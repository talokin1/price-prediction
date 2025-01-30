import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt

class TicketPricePredictorXGB:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = self.load_data()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.q = None 
        self.residuals = None
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath)
    
    def encode_categorical(self, categorical_cols: list):
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
    
    def remove_outliers(self): 
        for col in self.df.columns: 
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75) 
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR 
            upper_bound = Q3 + 1.5 * IQR 
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)] 
        return self.df
    
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

        self.calibrate_conformal(X_val, y_val)
    
    def calibrate_conformal(self, X_val: np.ndarray, y_val: pd.Series, alpha=0.5) -> int:

        dval = xgb.DMatrix(X_val)
        y_val_pred = self.model.predict(dval)
        
        self.residuals = np.abs(y_val - y_val_pred)  # Обчислюємо залишкові помилки
        self.q = np.quantile(self.residuals, 1 - alpha)  # Обчислюємо квантиль
        
        print(f"Квантиль conformal prediction: {self.q:.2f}")

        # Зберігаємо залишкові помилки
        self.save_residuals()
    
    def predict_with_intervals(self, X_test: np.ndarray):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        
        lower_bound = y_pred - self.q
        upper_bound = y_pred + self.q
        return y_pred, lower_bound, upper_bound
    
    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series):
        y_pred, lower_bound, upper_bound = self.predict_with_intervals(X_test)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "Coverage": np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        }
        
        print(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.2f}, Coverage: {metrics['Coverage']:.2%}")
        return metrics
    
    def save_artifacts(self, path: str = "./"):
        with open(f"{path}xgb_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def save_residuals(self, path: str = "./"):
        with open(f"{path}residual_errors.pkl", "wb") as f:
            pickle.dump(self.residuals, f)
        print("Залишкові помилки збережено у residual_errors.pkl")
    
    def plot_feature_importance(self):
        xgb.plot_importance(self.model, importance_type="weight", max_num_features=10)
        plt.show()

    def plot_prediction_intervals(self, X_test, y_test):
        
        y_pred, lower_bound, upper_bound = self.predict_with_intervals(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color="blue", alpha=0.6, label="Фактичні значення")
        plt.scatter(range(len(y_test)), y_pred, color="red", label="Прогнози")
        plt.fill_between(range(len(y_test)), lower_bound, upper_bound, color="orange", alpha=0.3, label="Довірчий інтервал")
        plt.xlabel("Зразки")
        plt.ylabel("Ціна квитка")
        plt.legend()
        plt.title("Conformal Prediction: довірчі інтервали")
        plt.show()


def main():
    predictor = TicketPricePredictorXGB("C:\\Edu\\Deep learning\\ml week\\data\\preprocessed.csv")
    predictor.encode_categorical(["section", "row"])
    predictor.remove_outliers()
    
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_data()
    predictor.train_model(X_train, y_train, X_val, y_val)
    predictor.evaluate_model(X_test, y_test)
    predictor.plot_prediction_intervals(X_test, y_test)
    predictor.save_artifacts()

if __name__ == "__main__":
    main()
