from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np

def summarize_folds(fold_metrics: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.array([fm.get(key, np.nan) for fm in fold_metrics], dtype=float)
    return {
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"),
        "n": int(np.isfinite(vals).sum()),
    }

def paired_diffs(metrics_a: List[Dict[str, Any]], metrics_b: List[Dict[str, Any]], key: str) -> np.ndarray:
    a = np.array([m.get(key, np.nan) for m in metrics_a], dtype=float)
    b = np.array([m.get(key, np.nan) for m in metrics_b], dtype=float)
    return a - b
