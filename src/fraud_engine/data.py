"""Data loading, synthetic transaction generation, and stratified train/test splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time"] + PCA_COLUMNS + ["Amount"]
ALL_COLUMNS = FEATURE_COLUMNS + ["Class"]


def load_credit_card_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load the ULB Credit Card Fraud dataset from CSV."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")
    df = pd.read_csv(p)
    if "Class" not in df.columns:
        raise ValueError("Missing 'Class' target column in credit card data.")
    return df


def generate_synthetic_credit_card_data(
    n_samples: int = 5000,
    fraud_ratio: float = 0.005,  # 0.5% fraud for manageable synthetic sample
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic transactions matching the ULB benchmark schema.

    Simulates 28 anonymized PCA components (V1-V28), transaction Amount, Time,
    and highly imbalanced Class labels (0 = Legitimate, 1 = Fraudulent).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(int(n_samples * fraud_ratio), 5)
    n_legit = n_samples - n_fraud

    # 1. Legitimate transactions (standard normal PCA centered at 0)
    legit_v = rng.normal(0.0, 1.0, size=(n_legit, 28))
    legit_time = np.sort(rng.uniform(0, 172800, size=n_legit))  # 48 hours in seconds
    legit_amount = rng.exponential(scale=88.0, size=n_legit)  # Median ~88 dollars

    df_legit = pd.DataFrame(legit_v, columns=PCA_COLUMNS)
    df_legit["Time"] = legit_time
    df_legit["Amount"] = np.round(legit_amount, 2)
    df_legit["Class"] = 0

    # 2. Fraudulent transactions (shifted distribution on V4, V10, V12, V14, V17)
    fraud_v = rng.normal(0.0, 1.2, size=(n_fraud, 28))
    # Characteristic anomalies in ULB fraud: V4 is positive, V10/V12/V14/V17 are strongly negative
    fraud_v[:, 3] += 3.5    # V4
    fraud_v[:, 9] -= 4.0    # V10
    fraud_v[:, 11] -= 4.5   # V12
    fraud_v[:, 13] -= 5.0   # V14
    fraud_v[:, 16] -= 3.5   # V17

    fraud_time = rng.uniform(0, 172800, size=n_fraud)
    fraud_amount = rng.exponential(scale=125.0, size=n_fraud)

    df_fraud = pd.DataFrame(fraud_v, columns=PCA_COLUMNS)
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = np.round(fraud_amount, 2)
    df_fraud["Class"] = 1

    # Combine and shuffle
    df_combined = pd.concat([df_legit, df_fraud], ignore_index=True)
    df_shuffled = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_shuffled[ALL_COLUMNS]


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, val, and untouched test sets with strict class stratification."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split train and val
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, stratify=y_train_val, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return train_df, val_df, test_df
