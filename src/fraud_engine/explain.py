"""SHAP explainability for fraud predictions and anonymized PCA component attribution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap


class FraudExplainer:
    """Explains fraud predictions using SHAP (SHapley Additive exPlanations).

    Discloses that features V1-V28 are mathematically transformed orthogonal PCA components
    resulting from privacy-preserving dimensionality reduction, avoiding fabricated real-world claims.
    """

    def __init__(self, model_wrapper: Any, feature_names: List[str]):
        self.feature_names = feature_names
        # Extract underlying classifier
        if hasattr(model_wrapper, "model"):
            self.clf = model_wrapper.model
        else:
            self.clf = model_wrapper

        try:
            self.explainer = shap.TreeExplainer(self.clf)
        except Exception:
            self.explainer = None

    def explain_global(self, X_sample: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """Compute global mean absolute SHAP feature importances."""
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            importances = np.mean(np.abs(shap_values), axis=0)
        elif hasattr(self.clf, "feature_importances_"):
            importances = self.clf.feature_importances_
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_k)

    def explain_transaction(self, x_single: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Decompose an individual transaction's score into top contributing features."""
        x_2d = x_single.reshape(1, -1)
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(x_2d)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            vals = shap_values[0]
        elif hasattr(self.clf, "feature_importances_"):
            vals = self.clf.feature_importances_ * x_single
        else:
            vals = x_single

        order = np.argsort(np.abs(vals))[::-1][:top_k]
        explanations = []
        for idx in order:
            feat_name = self.feature_names[idx]
            impact = float(vals[idx])
            desc = "Anonymized PCA component" if feat_name.startswith("V") else "Transaction attribute"
            explanations.append({
                "feature": feat_name,
                "shap_impact": round(impact, 4),
                "feature_value": round(float(x_single[idx]), 3),
                "impact_direction": "increases_fraud_risk" if impact > 0 else "reduces_fraud_risk",
                "description": desc,
            })
        return explanations
