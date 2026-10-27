"""Supervised, unsupervised, and calibrated hybrid fraud risk models."""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


class LogisticRegressionBaseline:
    name = "logistic_regression"

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBoostFraudClassifier:
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 100.0,
        random_state: int = 42,
    ):
        self.scale_pos_weight = scale_pos_weight
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostFraudClassifier":
        # Calculate dynamic scale_pos_weight if needed
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=float(n_neg / n_pos))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CalibratedIsolationForest:
    name = "isolation_forest"

    def __init__(self, contamination: float = 0.01, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.ref_min: float = 0.0
        self.ref_max: float = 1.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CalibratedIsolationForest":
        # Fit on nominal transactions if y is given
        X_fit = X[y == 0] if y is not None and np.sum(y == 0) > 0 else X
        self.model.fit(X_fit)

        # Record empirical score distribution for single-sample inference normalization
        raw = -self.model.decision_function(X_fit)
        self.ref_min = float(np.percentile(raw, 1))
        self.ref_max = float(np.percentile(raw, 99))
        if self.ref_max <= self.ref_min:
            self.ref_max = self.ref_min + 1.0
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Normalized anomaly score in [0, 1] calibrated on reference training distribution."""
        raw = -self.model.decision_function(X)
        norm = (raw - self.ref_min) / (self.ref_max - self.ref_min)
        return np.clip(norm, 0.0, 1.0)


class HybridRiskEngine:
    """Combines supervised fraud probability with unsupervised anomaly score into a 0-100 risk score."""
    name = "hybrid_risk_engine"

    def __init__(
        self,
        supervised_model: Optional[Any] = None,
        anomaly_model: Optional[CalibratedIsolationForest] = None,
        supervised_weight: float = 0.70,
        anomaly_weight: float = 0.30,
    ):
        self.supervised_model = supervised_model or XGBoostFraudClassifier()
        self.anomaly_model = anomaly_model or CalibratedIsolationForest()
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridRiskEngine":
        self.supervised_model.fit(X, y)
        self.anomaly_model.fit(X, y)
        return self

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Compute unified risk score in range [0, 100]."""
        p_sup = self.supervised_model.predict_proba(X)
        s_anom = self.anomaly_model.predict_anomaly_score(X)
        combined = (self.supervised_weight * p_sup) + (self.anomaly_weight * s_anom)
        return np.clip(combined * 100.0, 0.0, 100.0)

    def predict(self, X: np.ndarray, threshold: float = 50.0) -> np.ndarray:
        """Return binary decision array given a risk score threshold in [0, 100]."""
        return (self.predict_risk_score(X) >= threshold).astype(int)

    def predict_full(self, X: np.ndarray, threshold_risk: float = 50.0) -> list[Dict[str, Any]]:
        risk = self.predict_risk_score(X)
        p_sup = self.supervised_model.predict_proba(X)
        s_anom = self.anomaly_model.predict_anomaly_score(X)

        results = []
        for r, p, a in zip(risk, p_sup, s_anom):
            if r > 70.0:
                cat = "HIGH"
                action = "BLOCK_AND_ALERT"
            elif r > 30.0:
                cat = "MEDIUM"
                action = "FLAG_FOR_MANUAL_REVIEW"
            else:
                cat = "LOW"
                action = "APPROVE"

            results.append({
                "risk_score": round(float(r), 2),
                "fraud_probability": round(float(p), 4),
                "anomaly_score": round(float(a), 4),
                "risk_category": cat,
                "recommended_action": action,
                "is_fraud_flagged": bool(r >= threshold_risk),
            })
        return results
