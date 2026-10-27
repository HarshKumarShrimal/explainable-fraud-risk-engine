#!/usr/bin/env python3
"""Train, calibrate, and benchmark fraud detection models."""
from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fraud_engine.data import (
    generate_synthetic_credit_card_data,
    load_credit_card_data,
    stratified_split,
)
from fraud_engine.pipeline import FraudFeaturePipeline
from fraud_engine.models import (
    LogisticRegressionBaseline,
    XGBoostFraudClassifier,
    CalibratedIsolationForest,
    HybridRiskEngine,
)
from fraud_engine.metrics import evaluate_fraud_classifier, find_optimal_threshold


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark fraud models.")
    parser.add_argument("--data", type=str, default="data/raw/creditcard.csv", help="Path to raw dataset.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save model artifacts.")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic dataset generation.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if args.use_synthetic or not data_path.exists():
        print("[INFO] Generating synthetic credit card dataset (6,000 transactions, 0.5% fraud)...")
        raw_df = generate_synthetic_credit_card_data(n_samples=6000, fraud_ratio=0.005, random_state=42)
    else:
        print(f"[INFO] Loading dataset from {data_path}...")
        raw_df = load_credit_card_data(data_path)

    # 1. Stratified Train / Val / Test Split
    train_df, val_df, test_df = stratified_split(raw_df, test_size=0.2, val_size=0.1, random_state=42)
    print(f"[DATA] Train: {len(train_df):,} (Frauds: {train_df['Class'].sum()}) | "
          f"Val: {len(val_df):,} (Frauds: {val_df['Class'].sum()}) | "
          f"Test: {len(test_df):,} (Frauds: {test_df['Class'].sum()})")

    # 2. Fit Feature Pipeline strictly on Training split
    pipeline = FraudFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    y_val = val_df["Class"].values
    X_test = pipeline.transform(test_df)
    y_test = test_df["Class"].values

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "fraud_pipeline.joblib")

    # 3. Train models
    print("\n[TRAINING] 1. Logistic Regression Baseline...")
    lr_model = LogisticRegressionBaseline(random_state=42).fit(X_train, y_train)

    print("[TRAINING] 2. XGBoost Classifier...")
    xgb_model = XGBoostFraudClassifier(random_state=42).fit(X_train, y_train)

    print("[TRAINING] 3. Calibrated Isolation Forest...")
    iso_model = CalibratedIsolationForest(random_state=42).fit(X_train, y_train)

    print("[TRAINING] 4. Calibrated Hybrid Risk Engine...")
    hybrid_model = HybridRiskEngine(
        supervised_model=xgb_model,
        anomaly_model=iso_model,
        supervised_weight=0.70,
        anomaly_weight=0.30,
    )

    # Save artifacts
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(xgb_model, models_dir / "xgboost.joblib")
    joblib.dump(iso_model, models_dir / "isolation_forest.joblib")
    joblib.dump(hybrid_model, models_dir / "hybrid_engine.joblib")

    # 4. Evaluate on untouched Test Split
    eval_candidates = [
        ("Logistic Regression", lr_model.predict_proba(X_test)),
        ("XGBoost", xgb_model.predict_proba(X_test)),
        ("Isolation Forest", iso_model.predict_anomaly_score(X_test)),
        ("Calibrated Hybrid Engine", hybrid_model.predict_risk_score(X_test) / 100.0),
    ]

    benchmark_records = []
    print("\n" + "=" * 70)
    print("UNTOUCHED TEST SET BENCHMARK RESULTS:")
    print("=" * 70)

    for name, probs in eval_candidates:
        metrics = evaluate_fraud_classifier(y_test, probs, threshold=0.5)
        metrics["model"] = name
        benchmark_records.append(metrics)
        print(f"[{name.upper()}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

    # Optimal threshold analysis for XGBoost
    xgb_probs_val = xgb_model.predict_proba(X_val)
    opt_thr = find_optimal_threshold(y_val, xgb_probs_val)
    print("\n[OPTIMAL THRESHOLD ANALYSIS (Validation Set)]:")
    print(f"  Best F1 Threshold: {opt_thr['optimal_f1_threshold']} (Max F1: {opt_thr['max_f1']})")
    print(f"  Min Cost Threshold: {opt_thr['optimal_cost_threshold']}")

    # Save outputs
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(benchmark_records)
    df_res.to_csv(out_dir / "benchmark_results.csv", index=False)
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\nArtifacts saved to {models_dir}/ and metrics saved to {out_dir}/benchmark_results.csv")


if __name__ == "__main__":
    main()
