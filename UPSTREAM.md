# Upstream Attribution and Technical Evolution

## Upstream Repository Information
- **Original Repository:** [Parvinder10/Project-fraud-detection-system](https://github.com/Parvinder10/Project-fraud-detection-system)
- **Original Author:** Parvinder10
- **Upstream License:** MIT License (preserved in `LICENSE`)
- **Benchmark Dataset:** Université Libre de Bruxelles (ULB) Machine Learning Group Credit Card Fraud Benchmark (`creditcard.csv`)

---

## Architectural Critique and Upstream Deficiencies Identified

A forensic code audit of the upstream repository revealed several critical algorithmic, architectural, and production limitations:

### 1. Single-Sample Inference Collapse (Upstream Anomaly Scoring Bug)
- **Deficiency:** The upstream repository computed anomaly scores by applying batch-level Min-Max scaling to Isolation Forest raw decision function outputs:
  $$\text{Score}(x) = \frac{\text{raw}(x) - \min(X)}{\max(X) - \min(X)}$$
- **Failure Mode:** When invoked via a real-time REST API endpoint for a single incoming transaction ($N=1$), $\max(X) - \min(X) = 0$. This resulted in either zero-division runtime errors or an anomaly score forced to `0.0`, completely disarming anomaly detection during real-time inference.
- **Harsh's Fix:** Implemented `CalibratedIsolationForest`. The empirical score distribution is profiled during training on nominal transactions to establish permanent, calibrated 1st-percentile and 99th-percentile reference boundaries $(\text{ref}_{\min}, \text{ref}_{\max})$. Single-sample streaming inference scales deterministically against these fixed reference percentiles with zero batch dependence.

### 2. Metric Misalignment on Severe Imbalance
- **Deficiency:** Upstream documentation emphasized standard accuracy (reporting $>99.8\%$). On a dataset with 492 frauds out of 284,807 transactions ($0.172\%$ positive rate), a naive null model predicting "legitimate" for all transactions yields $99.83\%$ accuracy while catching zero frauds.
- **Harsh's Fix:** Migrated evaluation to Precision-Recall AUC (Average Precision), Brier score for probability calibration, and empirical decision-cost curves balancing false negatives (\$500 average fraud loss) against false positives (\$15 manual review overhead).

### 3. Leakage Risks in Preprocessing
- **Deficiency:** Upstream notebooks demonstrated inconsistent transformation pipelines and SMOTE oversampling applied prior to strict trajectory/split segregation, leaking synthetic minority characteristics into validation sets.
- **Harsh's Fix:** Built a leak-free `FraudFeaturePipeline` using `RobustScaler` (median and IQR) fitted strictly on the training partition. Anonymized PCA components $V1-V28$ remain unscaled to preserve orthogonal variance.

### 4. Lack of Feature Transparency and PCA Disclosure
- **Deficiency:** Upstream interfaces failed to disclose the origin of features $V1-V28$, misleading reviewers into treating them as arbitrary raw inputs.
- **Harsh's Fix:** Integrated `FraudExplainer` using SHAP (SHapley Additive exPlanations) with transparent disclosures. Every generated local explanation explicitly identifies $V1-V28$ as privacy-preserving orthogonal principal components resulting from PCA, avoiding fabricated domain claims while surfacing exact feature impacts.

### 5. Dependency Bloat and Operational Fragility
- **Deficiency:** The original project relied on heavy, unmaintained legacy packages (`imblearn`, `mlflow`) and tightly coupled notebook state.
- **Harsh's Fix:** Re-engineered the platform as an enterprise-grade modular Python package (`fraud_engine`) with zero legacy bloat, accompanied by a high-throughput FastAPI service, automated unit tests, and GitHub Actions CI.

---

## Architectural Comparison Matrix

| Feature | Upstream (`Parvinder10`) | Harsh's Implementation (`fraud_engine`) |
| :--- | :--- | :--- |
| **Architecture** | Ad-hoc Jupyter Notebooks & scripts | Modular, tested Python package with FastAPI |
| **Anomaly Scoring** | Min-Max over batch (fails on $N=1$) | Calibrated Isolation Forest with fixed reference bounds |
| **Hybrid Scoring** | Heuristic or separate models | Continuous 0–100 calibrated risk engine (Supervised + Unsupervised) |
| **Primary Metric** | Accuracy ($>99.8\%$ misleading) | PR-AUC (Average Precision) & Cost Curve Optimization |
| **Feature Scaling** | Standard / Min-Max scaler | `RobustScaler` on Amount/Time; unscaled orthogonal PCA components |
| **Explainability** | Static feature importance bar chart | SHAP TreeExplainer with local per-transaction impact decomposition |
| **Inference Interface** | None / basic script | Production FastAPI REST API + Interactive Web Dashboard |
| **Test Suite** | No automated tests | Automated pytest suite (8 unit & integration tests) |
