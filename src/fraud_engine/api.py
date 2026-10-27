"""FastAPI service and Interactive Dashboard for Explainable Fraud Risk Scoring."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, generate_synthetic_credit_card_data
from .pipeline import FraudFeaturePipeline
from .models import HybridRiskEngine, XGBoostFraudClassifier, CalibratedIsolationForest
from .explain import FraudExplainer

app = FastAPI(
    title="Explainable Fraud Risk Engine",
    description="Hybrid fraud detection combining supervised classification and anomaly detection with SHAP explainability.",
    version="1.0.0",
)

_pipeline: Optional[FraudFeaturePipeline] = None
_hybrid_engine: Optional[HybridRiskEngine] = None
_explainer: Optional[FraudExplainer] = None


def get_components():
    global _pipeline, _hybrid_engine, _explainer
    if _hybrid_engine is None:
        # Train default models on synthetic transaction dataset
        synth_df = generate_synthetic_credit_card_data(n_samples=3000, fraud_ratio=0.01, random_state=42)
        _pipeline = FraudFeaturePipeline()
        X_train, y_train = _pipeline.fit_transform(synth_df)

        sup_clf = XGBoostFraudClassifier(n_estimators=100, max_depth=4, random_state=42)
        anom_det = CalibratedIsolationForest(contamination=0.01, random_state=42)

        _hybrid_engine = HybridRiskEngine(
            supervised_model=sup_clf,
            anomaly_model=anom_det,
            supervised_weight=0.70,
            anomaly_weight=0.30,
        )
        _hybrid_engine.fit(X_train, y_train)
        _explainer = FraudExplainer(sup_clf, FEATURE_COLUMNS)

    return _pipeline, _hybrid_engine, _explainer


class TransactionInput(BaseModel):
    Time: float = Field(0.0, description="Seconds elapsed since first transaction")
    Amount: float = Field(120.50, description="Transaction monetary amount")
    V1: float = -0.5
    V2: float = 0.2
    V3: float = 1.1
    V4: float = 0.8
    V5: float = -0.3
    V6: float = 0.4
    V7: float = 0.1
    V8: float = 0.05
    V9: float = -0.2
    V10: float = -0.8
    V11: float = 0.6
    V12: float = -0.9
    V13: float = 0.1
    V14: float = -1.2
    V15: float = 0.3
    V16: float = -0.4
    V17: float = -0.7
    V18: float = 0.2
    V19: float = 0.1
    V20: float = 0.05
    V21: float = 0.02
    V22: float = -0.1
    V23: float = 0.03
    V24: float = 0.1
    V25: float = -0.05
    V26: float = 0.08
    V27: float = 0.02
    V28: float = 0.01


class TransactionBatchInput(BaseModel):
    transactions: List[TransactionInput]
    threshold_risk: float = Field(50.0, description="Risk threshold to flag fraud (0-100 scale)")


class PredictResponse(BaseModel):
    risk_score: float
    fraud_probability: float
    anomaly_score: float
    risk_category: str
    recommended_action: str
    is_fraud_flagged: bool
    top_explanations: List[Dict[str, Any]]
    disclaimer: str


@app.get("/health")
def health():
    _, engine, _ = get_components()
    return {
        "status": "healthy",
        "service": "explainable-fraud-risk-engine",
        "engine": engine.name,
        "supervised_weight": engine.supervised_weight,
        "anomaly_weight": engine.anomaly_weight,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_transaction(tx: TransactionInput, threshold: float = 50.0):
    pipeline, engine, explainer = get_components()

    # Convert to single-row dataframe
    df_tx = pd.DataFrame([tx.model_dump()])[FEATURE_COLUMNS]
    X_scaled = pipeline.transform(df_tx)

    pred = engine.predict_full(X_scaled, threshold_risk=threshold)[0]
    explanations = explainer.explain_transaction(X_scaled[0], top_k=5)

    return PredictResponse(
        risk_score=pred["risk_score"],
        fraud_probability=pred["fraud_probability"],
        anomaly_score=pred["anomaly_score"],
        risk_category=pred["risk_category"],
        recommended_action=pred["recommended_action"],
        is_fraud_flagged=pred["is_fraud_flagged"],
        top_explanations=explanations,
        disclaimer=(
            "Advisory statistical prediction only. Features V1-V28 are anonymized PCA components. "
            "Does not constitute an autonomous live banking block decision without operational risk review."
        ),
    )


@app.post("/batch-predict")
def batch_predict(batch: TransactionBatchInput):
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="Empty transactions list.")

    pipeline, engine, _ = get_components()
    rows = [t.model_dump() for t in batch.transactions]
    df_tx = pd.DataFrame(rows)[FEATURE_COLUMNS]

    X_scaled = pipeline.transform(df_tx)
    return {
        "total_transactions": len(rows),
        "results": engine.predict_full(X_scaled, threshold_risk=batch.threshold_risk),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Explainable Fraud Risk Engine</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b1120; color: #f1f5f9; margin: 0; padding: 24px; }
    .container { max-width: 960px; margin: 0 auto; background: #1e293b; padding: 32px; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.6); }
    h1 { color: #38bdf8; margin-top: 0; }
    .card { background: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 8px; margin-top: 20px; }
    .badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 13px; text-transform: uppercase; }
    .badge.LOW { background: #15803d; color: #bbf7d0; }
    .badge.MEDIUM { background: #b45309; color: #fde68a; }
    .badge.HIGH { background: #b91c1c; color: #fecaca; }
    button { background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; }
    button:hover { background: #0369a1; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }
    .stat-box { background: #1e293b; padding: 16px; border-radius: 6px; text-align: center; }
    .stat-val { font-size: 26px; font-weight: 700; color: #38bdf8; }
    ul { padding-left: 20px; }
    li { margin-bottom: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Explainable Fraud Risk Scoring</h1>
    <p>Hybrid risk prioritization combining <strong>Supervised Classification (XGBoost)</strong> and <strong>Unsupervised Anomaly Detection (Isolation Forest)</strong> with <strong>SHAP</strong> interpretability.</p>
    
    <div class="card">
      <h3>Inspect Sample Transaction</h3>
      <p>Simulate scoring an anomalous high-risk financial transaction:</p>
      <button onclick="runSimulation()">Score Suspicious Transaction</button>
    </div>

    <div class="card" id="resCard" style="display:none;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 style="margin:0;">Risk Prioritization Outcome</h3>
        <span id="badge" class="badge"></span>
      </div>

      <div class="grid">
        <div class="stat-box">
          <div style="font-size: 13px; color: #94a3b8;">Unified Risk Score</div>
          <div class="stat-val" id="riskVal">-</div>
          <div style="font-size: 12px; color: #64748b;">0 to 100 Scale</div>
        </div>
        <div class="stat-box">
          <div style="font-size: 13px; color: #94a3b8;">Fraud Probability</div>
          <div class="stat-val" id="probVal">-</div>
          <div style="font-size: 12px; color: #64748b;">Supervised Model</div>
        </div>
        <div class="stat-box">
          <div style="font-size: 13px; color: #94a3b8;">Anomaly Outlier Score</div>
          <div class="stat-val" id="anomVal">-</div>
          <div style="font-size: 12px; color: #64748b;">Isolation Forest</div>
        </div>
      </div>

      <h4 style="margin-top: 20px; margin-bottom: 6px;">Recommended Operational Action:</h4>
      <p id="actionText" style="color: #cbd5e1; font-weight: 600;"></p>

      <h4 style="margin-top: 20px; margin-bottom: 6px;">SHAP Feature Attribution (Why was this flagged?):</h4>
      <ul id="shapList" style="color: #94a3b8; font-size: 14px;"></ul>
      
      <p style="font-size: 12px; color: #64748b; margin-top: 20px; border-top: 1px solid #334155; padding-top: 12px;" id="discText"></p>
    </div>
  </div>

  <script>
    async function runSimulation() {
      // Create suspicious transaction with shifted V4, V10, V12, V14 components
      const tx = {
        Time: 12450.0,
        Amount: 489.50,
        V1: -1.2, V2: 2.1, V3: -1.5, V4: 4.8, V5: -0.9, V6: -0.2, V7: -1.4,
        V8: 0.8, V9: -2.3, V10: -4.5, V11: 3.2, V12: -5.1, V13: 0.1, V14: -6.2,
        V15: 0.2, V16: -2.1, V17: -4.8, V18: -1.2, V19: 0.8, V20: 0.6,
        V21: 0.4, V22: -0.2, V23: -0.1, V24: 0.1, V25: 0.2, V26: -0.1,
        V27: 0.3, V28: 0.1
      };

      const res = await fetch('/predict?threshold=50.0', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tx)
      });
      if (!res.ok) { alert('Error scoring transaction'); return; }
      const data = await res.json();

      document.getElementById('resCard').style.display = 'block';
      document.getElementById('riskVal').textContent = data.risk_score.toFixed(1);
      document.getElementById('probVal').textContent = (data.fraud_probability * 100).toFixed(1) + '%';
      document.getElementById('anomVal').textContent = (data.anomaly_score * 100).toFixed(1) + '%';

      const badge = document.getElementById('badge');
      badge.textContent = data.risk_category + ' RISK';
      badge.className = 'badge ' + data.risk_category;

      document.getElementById('actionText').textContent = data.recommended_action;
      document.getElementById('discText').textContent = data.disclaimer;

      const list = document.getElementById('shapList');
      list.innerHTML = '';
      data.top_explanations.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${item.feature}</strong> (Value: ${item.feature_value}): Impact ${item.shap_impact > 0 ? '+' : ''}${item.shap_impact} (<em>${item.impact_direction.replace('_', ' ')}</em>) — ${item.description}`;
        list.appendChild(li);
      });
    }
  </script>
</body>
</html>
"""
