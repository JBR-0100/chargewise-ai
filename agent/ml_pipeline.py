"""
agent/ml_pipeline.py
ML Pipeline — Step 1.5 of the agentic pipeline.
Handles model training, cross-validation, and demand prediction.
Supports LinearRegression, Ridge, and RandomForestRegressor.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from agent.data_agent import add_features

# ── Feature Configuration ───────────────────────────────────────────────────
FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "season",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean",
]


class CVDemandModel:
    """
    A unified wrapper for training and evaluating EV demand forecasting models
    using TimeSeriesSplit cross-validation.
    """

    def __init__(self, model_type: str = "Ridge", **model_params):
        """
        Initialize the model.
        
        Parameters:
            model_type: One of "LinearRegression", "Ridge", or "RandomForest".
            **model_params: Keyword arguments passed to the underlying sklearn model.
        """
        self.model_type = model_type
        if model_type == "LinearRegression":
            self.model = LinearRegression(**model_params)
        elif model_type == "Ridge":
            self.model = Ridge(**model_params)
        elif model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.metrics: Dict[str, Any] = {}
        self.cv_results: List[Dict[str, float]] = []
        self.residual_std: float = 0.0
        self.is_trained: bool = False

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Ensure features are present and return X, y.
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column for training.")
        
        # If features aren't already added, add them
        if not all(f in df.columns for f in FEATURES):
            df = add_features(df)
            
        return df[FEATURES], df["volume"]

    def train_with_cv(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Train the model using TimeSeriesSplit cross-validation.
        
        Returns:
            Dictionary of mean metrics (MAE, RMSE, R2).
        """
        X, y = self.prepare_data(df)
        
        if len(X) < (n_splits + 1) * 24:
            raise ValueError(f"Insufficient data for {n_splits}-fold CV. Need more records.")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_maes, cv_rmses, cv_r2s = [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple local fit for CV step
            local_model = self.__get_fresh_model()
            local_model.fit(X_train, y_train)
            preds = np.maximum(local_model.predict(X_test), 0)
            
            cv_maes.append(mean_absolute_error(y_test, preds))
            cv_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            cv_r2s.append(r2_score(y_test, preds))

        # Final fit on entire dataset (or 80/20 split for final metric reporting)
        split_idx = int(len(X) * 0.8)
        X_train_final, X_test_final = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_final, y_test_final = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train_final, y_train_final)
        final_preds = np.maximum(self.model.predict(X_test_final), 0)
        
        # Calculate residual standard deviation for confidence intervals
        residuals = y_test_final - final_preds
        self.residual_std = np.std(residuals)

        self.metrics = {
            "MAE": mean_absolute_error(y_test_final, final_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test_final, final_preds)),
            "R2": r2_score(y_test_final, final_preds),
            "CV_MAE_mean": np.mean(cv_maes),
            "CV_RMSE_mean": np.mean(cv_rmses),
            "CV_R2_mean": np.mean(cv_r2s),
        }
        self.is_trained = True
        return self.metrics

    def predict_with_interval(self, X: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
        """
        Generate predictions with an estimated confidence interval.
        
        Returns:
            DataFrame with columns: ['prediction', 'lower_bound', 'upper_bound']
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
            
        preds = np.maximum(self.model.predict(X[FEATURES]), 0)
        
        # Z-score for the given confidence level
        # Assuming normal distribution of residuals. 1.96 corresponds to 95% confidence.
        if confidence == 0.95:
            z_score = 1.96
        else:
            # Fallback for other confidence levels if scipy is missing
            z_score = 2.0 
        
        margin = z_score * self.residual_std
        
        return pd.DataFrame({
            "prediction": preds,
            "lower_bound": np.maximum(preds - margin, 0),
            "upper_bound": preds + margin
        }, index=X.index)

    def __get_fresh_model(self):
        """Helper to get an un-fitted instance of the chosen model."""
        params = self.model.get_params()
        if self.model_type == "LinearRegression":
            return LinearRegression(**params)
        elif self.model_type == "Ridge":
            return Ridge(**params)
        elif self.model_type == "RandomForest":
            return RandomForestRegressor(**params)
        return None


def get_pipeline_metrics(df: pd.DataFrame, zone_id: int) -> Dict[str, Any]:
    """
    Utility function to run the full pipeline for a specific zone.
    """
    zdf = df[df["TAZID"] == zone_id].copy().sort_values("time").set_index("time")
    if len(zdf) < 200:
        return {"error": "Insufficient data"}
    
    # Try multiple models and return results for the best one (or just Ridge as default)
    pipeline = CVDemandModel(model_type="Ridge")
    metrics = pipeline.train_with_cv(zdf)
    metrics["zone"] = zone_id
    metrics["model"] = "Ridge"
    return metrics


if __name__ == "__main__":
    # Smoke test
    import pandas as pd
    from agent.data_agent import DEFAULT_ZONE_CSV, load_zone_df
    
    print("Running Pipeline Smoke Test...")
    try:
        df = load_zone_df([108])
        metrics = get_pipeline_metrics(df, 108)
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"Smoke test failed: {e}")
