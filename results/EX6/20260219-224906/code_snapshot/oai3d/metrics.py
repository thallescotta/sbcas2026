from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= thr).astype(int)

    out: Dict[str, Any] = {}
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc_roc"] = float("nan")
    out["accuracy"] = float(accuracy_score(y_true, y_pred) * 100.0)
    out["f1_score"] = float(f1_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return out

def save_roc_data(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, thr = roc_curve(y_true.astype(int), y_prob.astype(float))
    obj = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
