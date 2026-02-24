"""
XGBoost model for crack growth prediction

Predicts:
- Crack length growth (mm) over 1 year
- Crack width growth (mm) over 1 year
- Risk category (Low/Medium/High)

Why XGBoost?
- Excellent with structured/tabular features
- Handles feature interactions naturally (brick type + humidity)
- Robust to different feature scales
- Built-in feature importance
- Less prone to overfitting than neural networks with small datasets
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
    """
    Multi-output XGBoost model for crack growth prediction
    """
    
    def __init__(self, config_params=None):
        self.config_params = config_params or config.XGBOOST_CONFIG
        
        # Separate models for length, width, and risk
        self.length_model = None
        self.width_model = None
        self.risk_model = None
        
        self.feature_names = None
        
    def train(self, X, y_length, y_width, y_risk):
        """
        Train the growth prediction models
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_length: Length growth labels (n_samples,)
            y_width: Width growth labels (n_samples,)
            y_risk: Risk category labels (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Split data
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
        
    def _train_regression_model(self, X_train, y_train, X_val, y_val, name):
        """
        Train regression model for continuous prediction
        """
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Training parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.config_params['max_depth'],
            'learning_rate': self.config_params['learning_rate'],
            'eval_metric': 'rmse'
        }
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config_params['n_estimators'],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate
        y_pred = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"{name.capitalize()} Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return model
    
    def _train_classification_model(self, X_train, y_train, X_val, y_val):
        """
        Train classification model for risk prediction
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        num_classes = len(np.unique(y_train))
        
        params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'max_depth': self.config_params['max_depth'],
            'learning_rate': self.config_params['learning_rate'],
            'eval_metric': 'mlogloss'
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config_params['n_estimators'],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate
        y_pred = model.predict(dval).astype(int)
        print("\nRisk Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['Low', 'Medium', 'High']))
        
        return model
    
    def predict(self, X, return_confidence=True):
        """
        Predict crack growth for new samples
        
        Args:
            X: Feature matrix (n_samples, n_features)
            return_confidence: If True, estimate prediction confidence
            
        Returns:
            dict with predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        dmatrix = xgb.DMatrix(X)
        
        # Predictions
        length_pred = self.length_model.predict(dmatrix)
        width_pred = self.width_model.predict(dmatrix)
        risk_pred = self.risk_model.predict(dmatrix).astype(int)
        
        results = {
            'length_growth_mm': length_pred,
            'width_growth_mm': width_pred,
            'risk_category': risk_pred,
            'risk_label': [['Low', 'Medium', 'High'][r] for r in risk_pred]
        }
        
        # Estimate confidence using ensemble variance
        if return_confidence:
            # For XGBoost, we can use prediction margin as confidence proxy
            # Higher margin = more confident
            risk_proba_dmatrix = xgb.DMatrix(X)
            
            # Get prediction margins
            length_margin = np.abs(self.length_model.predict(dmatrix))
            width_margin = np.abs(self.width_model.predict(dmatrix))
            
            # Normalize to [0, 1] confidence score
            # Simple heuristic: higher predicted growth = lower confidence (more uncertain)
            length_conf = 1.0 / (1.0 + length_pred / 50.0)  # Normalize by typical max growth
            width_conf = 1.0 / (1.0 + width_pred / 2.0)
            
            # Combined confidence
            confidence = (length_conf + width_conf) / 2.0
            results['confidence'] = np.clip(confidence, 0.3, 0.95)  # Reasonable range
        
        return results
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from trained models
        """
        if self.length_model is None:
            return None
        
        importance = self.length_model.get_score(importance_type='weight')
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_importance[:top_n]
    
    def save(self, filepath):
        """
        Save trained models to disk
        """
        model_data = {
            'length_model': self.length_model,
            'width_model': self.width_model,
            'risk_model': self.risk_model,
            'feature_names': self.feature_names,
            'config': self.config_params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load(self, filepath):
        """
        Load trained models from disk
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.length_model = model_data['length_model']
        self.width_model = model_data['width_model']
        self.risk_model = model_data['risk_model']
        self.feature_names = model_data['feature_names']
        self.config_params = model_data['config']
        
        print(f"Models loaded from {filepath}")


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing CrackGrowthPredictor...")
    
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y_length = np.abs(np.random.randn(n_samples) * 10)  # 0-30mm growth
    y_width = np.abs(np.random.randn(n_samples) * 0.5)  # 0-1.5mm growth
    y_risk = (y_length + y_width * 10 > 15).astype(int)  # Binary for simplicity
    
    # Train
    predictor = CrackGrowthPredictor()
    predictor.train(X, y_length, y_width, y_risk)
    
    # Test prediction
    X_test = np.random.randn(5, n_features)
    predictions = predictor.predict(X_test)
    
    print("\nTest Predictions:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Length growth: {predictions['length_growth_mm'][i]:.2f} mm")
        print(f"  Width growth: {predictions['width_growth_mm'][i]:.3f} mm")
        print(f"  Risk: {predictions['risk_label'][i]}")
        print(f"  Confidence: {predictions['confidence'][i]:.2f}")
