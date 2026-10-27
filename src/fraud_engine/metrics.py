"""Evaluation metrics for imbalanced fraud classification and risk threshold analysis."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_fraud_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive imbalanced classification metrics.

    Emphasizes PR-AUC (Average Precision) and Brier Score over pure accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 3),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> Dict[str, Any]:
    """Sweep thresholds from 0.05 to 0.95 to identify maximum F1 and minimum cost thresholds."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = -1.0
    best_f1_thr = 0.5
    min_cost = float("inf")
    best_cost_thr = 0.5

    records = []
    for thr in thresholds:
        m = evaluate_fraud_classifier(y_true, y_prob, threshold=float(thr))
        cm = m["confusion_matrix"]
        total_cost = (cm["false_negatives"] * fn_cost) + (cm["false_positives"] * fp_cost)
        m["total_operational_cost"] = total_cost
        records.append(m)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_thr = float(thr)

        if total_cost < min_cost:
            min_cost = total_cost
            best_cost_thr = float(thr)

    return {
        "optimal_f1_threshold": round(best_f1_thr, 3),
        "max_f1": round(best_f1, 4),
        "optimal_cost_threshold": round(best_cost_thr, 3),
        "min_operational_cost": min_cost,
        "threshold_sweep_records": records,
    }
