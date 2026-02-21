"""
eval.py
-----------------------------------------------------------------------------
Avaliação FINAL em dados "unseen" (outer-test) para cada fold do Outer CV.

O que faz:
- Roda o modelo em modo eval() no outer-test (SEM augmentation)
- Calcula loss médio e métricas (AUC-ROC, accuracy, f1, precision, recall)
- Salva artefatos do fold em disco:
    fold_dir/
      y_true.npy
      y_prob.npy
      outer_test_metrics.json

Performance (2.0):
- Usa torch.inference_mode() (mais rápido que no_grad)
- Usa AMP/autocast (quando cfg.amp=True) para acelerar inferência em GPU
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from .logging_utils import save_json


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)

    out: Dict[str, float] = {}
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc_roc"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return out


def evaluate_outer_test_and_save(
    model: nn.Module,
    outer_test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fold_dir: Path,
    logger=None,
    fold_idx: Optional[int] = None,
    threshold: float = 0.5,
    amp: bool = True,
) -> Dict[str, Any]:
    """
    Avalia o modelo no outer-test e salva artefatos.

    amp:
      - Se True e device=cuda, usa autocast para acelerar inferência.
      - Mantém consistência com treino AMP (e geralmente acelera bastante).
    """
    model.eval()

    losses = []
    y_true_all = []
    y_prob_all = []

    use_amp = bool(amp) and (device.type == "cuda")

    # inference_mode = mais rápido/leve que no_grad
    with torch.inference_mode():
        for x, y in outer_test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            losses.append(loss.detach().float().item())

            prob = _sigmoid(logits).detach().float().cpu().numpy()
            y_prob_all.append(prob)
            y_true_all.append(y.detach().float().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])

    metrics = _compute_metrics(y_true, y_prob, thr=threshold) if y_true.size else {}

    # salvar artefatos
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "y_true.npy", y_true.astype(np.float32))
    np.save(fold_dir / "y_prob.npy", y_prob.astype(np.float32))

    out = {"outer_test_loss": mean_loss, **metrics}
    save_json(fold_dir / "outer_test_metrics.json", out)

    if logger is not None:
        prefix = f"[Fold {fold_idx+1}] " if fold_idx is not None else ""
        logger.info(
            f"{prefix}Outer-test | loss={mean_loss:.4f} | "
            f"auc={out.get('auc_roc', float('nan')):.4f} | "
            f"acc={out.get('accuracy', float('nan')):.4f} | "
            f"f1={out.get('f1_score', float('nan')):.4f}"
        )

    return out
