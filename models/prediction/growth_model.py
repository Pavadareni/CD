"""
XGBoost model for crack growth prediction
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
import pickle
import config


class CrackGrowthPredictor:

    def __init__(self, config_params=None):
        self.config_params = config_params or config.XGBOOST_CONFIG
        self.length_model = None
        self.width_model = None
        self.risk_model = None
        self.feature_names = None

    # =====================================================

    def train(self, X, y_length, y_width, y_risk):

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        X_train, X_val, y_len_train, y_len_val = train_test_split(
            X, y_length, test_size=0.2, random_state=42
        )
        _, _, y_wid_train, y_wid_val = train_test_split(
            X, y_width, test_size=0.2, random_state=42
        )
        _, _, y_risk_train, y_risk_val = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )

        print("Training crack length growth model...")
        self.length_model = self._train_regression_model(
            X_train, y_len_train, X_val, y_len_val, "length"
        )

        print("\nTraining crack width growth model...")
        self.width_model = self._train_regression_model(
            X_train, y_wid_train, X_val, y_wid_val, "width"
        )

        print("\nTraining risk classification model...")
        self.risk_model = self._train_classification_model(
            X_train, y_risk_train, X_val, y_risk_val
        )

    # =====================================================

    def _train_regression_model(self, X_train, y_train, X_val, y_val, name):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.config_params["max_depth"],
            "learning_rate": self.config_params["learning_rate"],
            "eval_metric": "rmse"
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config_params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=10
        )

        y_pred = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        print(f"{name.capitalize()} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return model

    # =====================================================

    def _train_classification_model(self, X_train, y_train, X_val, y_val):

        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "multi:softmax",
            "num_class": 3,
            "max_depth": self.config_params["max_depth"],
            "learning_rate": self.config_params["learning_rate"],
            "eval_metric": "mlogloss"
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config_params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=10
        )

        y_pred = model.predict(dval).astype(int)

        print("\nRisk Classification Report:")
        print(classification_report(
            y_val,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Low", "Medium", "High"]
        ))

        return model

    # =====================================================

    def predict(self, X, return_confidence=True):

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        dmatrix = xgb.DMatrix(X)

        length_pred = self.length_model.predict(dmatrix)
        width_pred = self.width_model.predict(dmatrix)
        risk_pred = self.risk_model.predict(dmatrix).astype(int)

        results = {
            "length_growth_mm": length_pred,
            "width_growth_mm": width_pred,
            "risk_category": risk_pred,
            "risk_label": [["Low", "Medium", "High"][r] for r in risk_pred]
        }

        if return_confidence:
            length_conf = 1.0 / (1.0 + length_pred / 50.0)
            width_conf = 1.0 / (1.0 + width_pred / 2.0)
            confidence = (length_conf + width_conf) / 2.0
            results["confidence"] = np.clip(confidence, 0.3, 0.95)

        return results

    # =====================================================

    def get_feature_importance(self, top_n=15):

        if self.length_model is None:
            return None

        importance = self.length_model.get_score(importance_type="weight")
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # =====================================================

    def save(self, filepath):

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump({
                "length_model": self.length_model,
                "width_model": self.width_model,
                "risk_model": self.risk_model,
                "feature_names": self.feature_names,
                "config": self.config_params
            }, f)

        print(f"Models saved to {filepath}")

    # =====================================================

    def load(self, filepath):

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.length_model = data["length_model"]
        self.width_model = data["width_model"]
        self.risk_model = data["risk_model"]
        self.feature_names = data["feature_names"]
        self.config_params = data["config"]

        print(f"Models loaded from {filepath}")