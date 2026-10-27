"""Feature preprocessing and scaling pipeline preventing data leakage."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data import FEATURE_COLUMNS, PCA_COLUMNS


class FraudFeaturePipeline:
    """Robust feature preprocessing pipeline for imbalanced transaction streams.

    Uses RobustScaler (median and IQR) on 'Amount' and 'Time' to handle severe
    skewness and transaction outliers without distorting normalized PCA components.
    """

    def __init__(self):
        self.amount_time_scaler = RobustScaler()
        self.feature_columns = FEATURE_COLUMNS
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training data and return scaled feature matrix X and label array y."""
        X_df = df[self.feature_columns].copy()

        # Scale Amount and Time; PCA components V1-V28 are already standardized
        scaled_at = self.amount_time_scaler.fit_transform(X_df[["Time", "Amount"]])
        X_df["Time"] = scaled_at[:, 0]
        X_df["Amount"] = scaled_at[:, 1]

        y = df["Class"].values if "Class" in df.columns else np.array([])
        self._fitted = True
        return X_df.values, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted scaling to unseen evaluation or inference transactions."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        X_df = df[self.feature_columns].copy()
        scaled_at = self.amount_time_scaler.transform(X_df[["Time", "Amount"]])
        X_df["Time"] = scaled_at[:, 0]
        X_df["Amount"] = scaled_at[:, 1]
        return X_df.values
