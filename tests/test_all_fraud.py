"""Comprehensive test suite for the Explainable Fraud Risk Engine."""
from __future__ import annotations

import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_engine.data import (
    FEATURE_COLUMNS,
    PCA_COLUMNS,
    generate_synthetic_credit_card_data,
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
from fraud_engine.explain import FraudExplainer
from fraud_engine.api import app


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    return generate_synthetic_credit_card_data(n_samples=500, fraud_ratio=0.02, random_state=42)


@pytest.fixture
def fitted_pipeline_and_data(synthetic_data):
    train_df, val_df, test_df = stratified_split(synthetic_data, test_size=0.2, val_size=0.1, random_state=42)
    pipeline = FraudFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    y_val = val_df["Class"].values
    X_test = pipeline.transform(test_df)
    y_test = test_df["Class"].values
    return pipeline, X_train, y_train, X_val, y_val, X_test, y_test


def test_synthetic_data_generation(synthetic_data):
    assert len(synthetic_data) == 500
    assert "Class" in synthetic_data.columns
    assert "Time" in synthetic_data.columns
    assert "Amount" in synthetic_data.columns
    for i in range(1, 29):
        assert f"V{i}" in synthetic_data.columns

    # Verify fraud ratio is roughly 2% (>=5 frauds minimum)
    frauds = synthetic_data["Class"].sum()
    assert frauds >= 5


def test_stratified_split_preserves_class_ratio(synthetic_data):
    train_df, val_df, test_df = stratified_split(synthetic_data, test_size=0.2, val_size=0.1, random_state=42)
    assert len(train_df) + len(val_df) + len(test_df) == len(synthetic_data)
    assert train_df["Class"].sum() > 0
    assert test_df["Class"].sum() > 0


def test_pipeline_transform_and_leak_free_scaling(synthetic_data):
    pipeline = FraudFeaturePipeline()
    with pytest.raises(RuntimeError):
        pipeline.transform(synthetic_data)

    X, y = pipeline.fit_transform(synthetic_data)
    assert X.shape == (len(synthetic_data), 30)
    assert len(y) == len(synthetic_data)
    # Check that PCA components were not distorted by RobustScaler (V1 is index 1)
    np.testing.assert_allclose(X[:, 1], synthetic_data["V1"].values, rtol=1e-5)


def test_single_sample_isolation_forest_bug_fix(fitted_pipeline_and_data):
    """Verifies fix for upstream Parvinder10 bug where single-sample inference forced score to 0.0."""
    _, X_train, y_train, _, _, X_test, _ = fitted_pipeline_and_data
    iso = CalibratedIsolationForest(random_state=42).fit(X_train, y_train)

    # Multi-sample scoring
    multi_scores = iso.predict_anomaly_score(X_test[:5])
    assert len(multi_scores) == 5
    assert (multi_scores >= 0.0).all() and (multi_scores <= 1.0).all()

    # Single-sample inference test: Upstream Parvinder10 code produced 0.0 due to min-max on single element.
    # Our calibrated implementation must produce a valid score.
    single_sample = X_test[0:1]
    single_score = iso.predict_anomaly_score(single_sample)
    assert len(single_score) == 1
    assert np.isclose(single_score[0], multi_scores[0], atol=1e-5)


def test_models_fit_and_predict(fitted_pipeline_and_data):
    _, X_train, y_train, _, _, X_test, y_test = fitted_pipeline_and_data

    lr = LogisticRegressionBaseline(random_state=42).fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)
    assert len(lr_probs) == len(y_test)
    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all()

    xgb = XGBoostFraudClassifier(random_state=42, n_estimators=50).fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)
    assert len(xgb_probs) == len(y_test)

    iso = CalibratedIsolationForest(random_state=42).fit(X_train, y_train)
    hybrid = HybridRiskEngine(supervised_model=xgb, anomaly_model=iso).fit(X_train, y_train)

    risk_scores = hybrid.predict_risk_score(X_test)
    assert (risk_scores >= 0.0).all() and (risk_scores <= 100.0).all()

    preds = hybrid.predict(X_test, threshold=50.0)
    assert set(preds).issubset({0, 1})


def test_evaluation_metrics_and_optimal_threshold(fitted_pipeline_and_data):
    _, X_train, y_train, X_val, y_val, X_test, y_test = fitted_pipeline_and_data
    xgb = XGBoostFraudClassifier(random_state=42, n_estimators=50).fit(X_train, y_train)
    probs = xgb.predict_proba(X_test)

    metrics = evaluate_fraud_classifier(y_test, probs, threshold=0.5)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    val_probs = xgb.predict_proba(X_val)
    opt = find_optimal_threshold(y_val, val_probs)
    assert 0.0 <= opt["optimal_f1_threshold"] <= 1.0
    assert 0.0 <= opt["optimal_cost_threshold"] <= 1.0


def test_fraud_explainer_with_pca_disclosure(fitted_pipeline_and_data):
    _, X_train, y_train, _, _, X_test, _ = fitted_pipeline_and_data
    xgb = XGBoostFraudClassifier(random_state=42, n_estimators=30).fit(X_train, y_train)
    explainer = FraudExplainer(xgb, FEATURE_COLUMNS)

    global_imp = explainer.explain_global(X_test[:20], top_k=5)
    assert len(global_imp) == 5
    assert "feature" in global_imp.columns
    assert "importance" in global_imp.columns

    local_exp = explainer.explain_transaction(X_test[0], top_k=4)
    assert len(local_exp) == 4
    for exp in local_exp:
        assert "feature" in exp
        assert "shap_impact" in exp
        assert "description" in exp
        if exp["feature"].startswith("V"):
            assert "Anonymized PCA component" in exp["description"]


def test_api_endpoints():
    with TestClient(app) as client:
        # 1. Health
        res_h = client.get("/health")
        assert res_h.status_code == 200
        data_h = res_h.json()
        assert data_h["status"] == "healthy"

        # 2. UI Dashboard
        res_ui = client.get("/")
        assert res_ui.status_code == 200
        assert "Explainable Fraud Risk Engine" in res_ui.text

        # 3. Predict Low Risk
        normal_tx = {
            "Time": 500.0,
            "Amount": 45.0,
            **{f"V{i}": 0.05 for i in range(1, 29)}
        }
        res_p = client.post("/predict", json=normal_tx)
        assert res_p.status_code == 200
        p_data = res_p.json()
        assert 0.0 <= p_data["risk_score"] <= 100.0
        assert p_data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
        assert "disclaimer" in p_data

        # 4. Predict High Risk Anomalous Transaction
        fraud_tx = {
            "Time": 1200.0,
            "Amount": 1500.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        fraud_tx["V4"] = 4.5
        fraud_tx["V10"] = -5.0
        fraud_tx["V12"] = -5.5
        fraud_tx["V14"] = -6.0
        fraud_tx["V17"] = -4.0

        res_fraud = client.post("/predict", json=fraud_tx)
        assert res_fraud.status_code == 200
        f_data = res_fraud.json()
        assert f_data["risk_category"] == "HIGH"
        assert f_data["recommended_action"] == "BLOCK_AND_ALERT"

        # 5. Batch Predict
        res_batch = client.post("/batch-predict", json={"transactions": [normal_tx, fraud_tx]})
        assert res_batch.status_code == 200
        b_data = res_batch.json()
        assert len(b_data["results"]) == 2
