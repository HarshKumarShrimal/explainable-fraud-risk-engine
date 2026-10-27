# Technical Interview Preparation Guide: Explainable Fraud Risk Engine

This document provides rigorous, engineering-grade interview questions, answers, mathematical derivations, and code references based on the implementation in `explainable-fraud-risk-engine`.

---

### Q1: Why is PR-AUC (Precision-Recall AUC) preferred over ROC-AUC for credit card fraud detection?
**Answer:**
In credit card fraud detection, class imbalance is extreme (typically $0.1\%$ to $0.5\%$ fraudulent).
- **ROC-AUC** evaluates False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$) against True Positive Rate ($\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$). Because the number of True Negatives (legitimate transactions) is colossal, the denominator of FPR is huge. Even if a model generates thousands of false alarms, FPR remains tiny ($<0.01$), resulting in an artificially inflated ROC-AUC ($>0.99$).
- **PR-AUC** evaluates Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) against Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). It completely ignores True Negatives and directly reflects the proportion of true frauds among flagged alerts.
*Implementation Reference:* [`src/fraud_engine/metrics.py`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/metrics.py) calculates `average_precision_score(y_true, y_prob)`.

---

### Q2: What was the critical bug in the upstream repository's anomaly scoring, and how did you fix it?
**Answer:**
The upstream repository attempted to normalize Isolation Forest raw decision function scores using batch-level min-max scaling:
$$\text{norm\_score} = \frac{\text{raw} - \min(\text{raw\_batch})}{\max(\text{raw\_batch}) - \min(\text{raw\_batch})}$$
When a transaction was scored individually via real-time inference ($N=1$), $\max(\text{raw}) = \min(\text{raw})$, leading to zero division or a score forced to `0.0`. Consequently, an anomalous transaction would never be flagged in real-time API scoring.
**Fix:** In [`src/fraud_engine/models.py`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/models.py) (`CalibratedIsolationForest`), the model profiles the nominal training transactions during `.fit()` to record empirical 1st-percentile and 99th-percentile bounds:
$$\text{ref}_{\min} = \text{Percentile}_{1}(\text{raw}_{\text{nominal}}), \quad \text{ref}_{\max} = \text{Percentile}_{99}(\text{raw}_{\text{nominal}})$$
Streaming inference clamps and normalizes against these fixed references:
$$\text{score}(x) = \text{clip}\left(\frac{\text{raw}(x) - \text{ref}_{\min}}{\text{ref}_{\max} - \text{ref}_{\min}}, 0.0, 1.0\right)$$
This guarantees consistent, batch-independent $[0, 1]$ anomaly scores for single transactions.

---

### Q3: What is the nature of features $V1$ through $V28$ in the ULB benchmark dataset?
**Answer:**
Features $V1$ through $V28$ are the result of Principal Component Analysis (PCA) applied by the original dataset providers (ULB Machine Learning Group) to protect customer confidentiality and proprietary banking transaction structures.
1. They are mutually orthogonal linear combinations of original underlying variables.
2. Because they have already been mean-centered and standardized during PCA, standardizing them again is unnecessary and computationally redundant.
3. Only `Time` (seconds from initial transaction) and `Amount` (transaction currency) remain in their original raw forms.
*Implementation Reference:* [`src/fraud_engine/pipeline.py`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/pipeline.py) applies scaling strictly to `Time` and `Amount`.

---

### Q4: Why use `RobustScaler` on `Amount` and `Time` instead of `StandardScaler` or `MinMaxScaler`?
**Answer:**
Credit card transaction amounts exhibit extreme positive skewness (e.g. median amount is $\approx \$22$, but high-value transactions exceed $\$25,000$).
- `StandardScaler` uses sample mean $\mu$ and standard deviation $\sigma$. Outliers distort both $\mu$ and $\sigma$, pulling the mean toward large transactions and shrinking normal transaction variance.
- `MinMaxScaler` binds to absolute minimum and maximum values; a single extreme $\$25,000$ outlier compresses $99.9\%$ of transactions into a narrow $[0, 0.01]$ range.
- `RobustScaler` subtracts the median and divides by the Interquartile Range ($\text{IQR} = Q_3 - Q_1$). Because median and IQR have high breakdown points ($50\%$ and $25\%$ respectively), extreme fraud amounts do not distort the scaling of everyday purchases.

---

### Q5: How does the Hybrid Risk Engine fuse supervised and unsupervised models?
**Answer:**
Supervised models (XGBoost) excel at detecting recurring, historical fraud patterns present in training data. However, they are blind to novel zero-day attack vectors.
Unsupervised anomaly detectors (Isolation Forest) detect statistical deviations from nominal customer behavior without labels, but suffer from higher false positive rates.
The `HybridRiskEngine` combines both signals into a continuous $0–100$ operational risk index:
$$\text{RiskScore}(x) = 100 \times \left(w_{\text{sup}} \cdot P_{\text{XGB}}(\text{fraud} \mid x) + w_{\text{anom}} \cdot S_{\text{IF}}(x)\right)$$
where $w_{\text{sup}} = 0.70$ and $w_{\text{anom}} = 0.30$.
- Risk $> 70$: Auto-flagged for blocking and immediate customer alert.
- $30 \le \text{Risk} \le 70$: Flagged for secondary authentication (SMS OTP, biometric prompt).
- Risk $< 30$: Seamless approval.
*Implementation Reference:* [`src/fraud_engine/models.py:125`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/models.py#L125-L131).

---

### Q6: How does `scale_pos_weight` in XGBoost handle extreme class imbalance?
**Answer:**
XGBoost optimizes an objective function consisting of the sum of logistic losses and regularization terms:
$$\mathcal{L} = \sum_{i=1}^N l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$
With `scale_pos_weight` ($s$), the gradient and hessian for positive minority instances ($y_i = 1$) are scaled by $s$:
$$g_i = \begin{cases} \hat{p}_i - y_i & \text{if } y_i = 0 \\ s \cdot (\hat{p}_i - 1) & \text{if } y_i = 1 \end{cases}$$
Setting $s = \frac{N_{\text{neg}}}{N_{\text{pos}}}$ balances the gradient contributions during tree split finding, forcing split evaluation to heavily penalize misclassified minority transactions.
*Implementation Reference:* [`src/fraud_engine/models.py:59`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/models.py#L59-L63).

---

### Q7: How do you select the operational decision threshold using a business cost matrix?
**Answer:**
A default classification threshold of $0.5$ is rarely optimal in fraud detection because the cost of errors is highly asymmetric:
- False Negative (missing a fraud): direct financial reimbursement loss, chargeback fees, and reputational damage ($C_{\text{FN}} \approx \$500$).
- False Positive (flagging a legit transaction): cost of customer manual verification or transaction friction ($C_{\text{FP}} \approx \$15$).
The optimal threshold $\tau^*$ minimizes total expected financial cost over the validation split:
$$\tau^* = \arg\min_{\tau} \left( C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau) \right)$$
*Implementation Reference:* [`src/fraud_engine/metrics.py:48`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/metrics.py#L48-L75) performs a grid sweep over candidate thresholds $\tau \in [0.01, 0.99]$.

---

### Q8: What are Shapley values, and how does SHAP TreeExplainer calculate feature attributions?
**Answer:**
Shapley values originate from cooperative game theory and uniquely satisfy four desirable mathematical axioms: Efficiency, Symmetry, Dummy (Null player), and Additivity.
For tree ensembles, `shap.TreeExplainer` exploits tree structure to compute exact conditional expectations $\mathbb{E}[f(x) \mid S]$ in polynomial time $\mathcal{O}(T L D^2)$, where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is the maximum tree depth.
The prediction is decomposed additively:
$$f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
where $\phi_0 = \mathbb{E}[f(x)]$ is the base expectation across the dataset, and $\phi_j(x)$ represents the exact contribution of feature $j$ to the deviation from the baseline.
*Implementation Reference:* [`src/fraud_engine/explain.py`](file:///Users/harshkumarshrimal/Desktop/AI_Profile/explainable-fraud-risk-engine/src/fraud_engine/explain.py).

---

### Q9: How do you prevent data leakage during train/validation/test splitting in fraud detection?
**Answer:**
1. **No Target Leakage in Scaling:** The feature transformation pipeline (`RobustScaler`) is fitted strictly on the training partition ($X_{\text{train}}$) using `.fit_transform()`. Unseen validation and test sets are processed strictly with `.transform()`.
2. **Stratification:** Splitting uses stratified sampling on `Class` to maintain the precise fraud ratio across all splits without synthetic resampling prior to partitioning.
3. **Temporal Awareness:** When sequential transaction timestamps span multiple days, splitting should be temporal (training on Days 1–$K$, evaluating on Day $K+1$) to simulate actual deployment conditions and test for concept drift.

---

### Q10: What is the Brier score, and why does probability calibration matter in fraud systems?
**Answer:**
The Brier score measures the mean squared difference between predicted probabilities and actual binary outcomes:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$$
A lower Brier score indicates well-calibrated probabilities. In banking risk workflows, risk engines feed into downstream policy engines (e.g. tier-1 auto-block, tier-2 SMS challenge, tier-3 high-value review). If a model predicts $P = 0.85$ but the true frequency of fraud at that threshold is only $0.15$, downstream automated financial decisions become unreliable and generate unacceptable operational overhead.

---

### Q11: What are the latency and throughput tradeoffs of running SHAP in a high-volume streaming pipeline?
**Answer:**
While TreeExplainer is efficient ($\approx 10–20\,\text{ms}$ per transaction), top-tier payment gateways process 10,000+ transactions per second, requiring sub-10ms end-to-end latency.
**Engineering Strategies:**
1. Asynchronous explanation generation: Return the risk decision and action immediately ($<5\,\text{ms}$), and dispatch SHAP explanation calculation to a background worker queue (e.g. Celery / Redis) for storage in fraud analyst audit logs.
2. Fast linear approximations or surrogate models for instant online UI previews.
3. Selective trigger: Only execute SHAP decomposition when transaction risk score exceeds the alert threshold ($>30$).

---

### Q12: How does this system adhere to ethical AI practices and regulatory compliance (e.g., GDPR, Fair Lending)?
**Answer:**
1. **Right to Explanation (GDPR Article 22):** When a transaction is blocked or challenged, local SHAP attribution provides verifiable reasons (e.g. abnormal velocity or anomaly in specific principal component) rather than an opaque black-box rejection.
2. **Explicit Disclosure of PCA Components:** The system explicitly notifies operators and users that features $V1-V28$ are transformed mathematical coordinates and avoids fabricating real-world labels (e.g. claiming $V4$ is "home address change" when it is an anonymous PCA factor).
3. **Human-in-the-Loop Safeguards:** Transactions in the intermediate risk tier ($30–70$) trigger step-up authentication or manual review rather than immediate irrevocable denial.
