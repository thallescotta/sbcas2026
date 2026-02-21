"""
train.py
-----------------------------------------------------------------------------
Rotinas de treino/validação para os experimentos (EX1–EX10).

O que este arquivo faz:
- train_epoch(): executa 1 época de treino (com AMP opcional)
- validate_epoch(): executa validação e calcula métricas (AUC, acc, etc.)
- fit_inner_earlystop(): treino com early stopping usando inner-val
  * log por epoch + tempo por epoch + ETA (ideal para nohup/run.log)
- train_fixed_epochs(): re-treina em outer-train completo por N epochs (Amaro-style)
- build_optimizer()/build_criterion(): helpers

Nota importante:
- Este arquivo foi escrito para ser “tolerante” ao retorno do validate_epoch().
  Caso uma versão anterior retorne (val_loss, metrics, y_true, y_prob), por exemplo,
  ele vai lidar sem quebrar.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


# ----------------------------
# Helpers
# ----------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_criterion(pos_weight: torch.Tensor, device: torch.device) -> nn.Module:
    # BCEWithLogitsLoss espera pos_weight para classe positiva (y=1)
    pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)

    out = {}
    # AUC pode falhar se só houver uma classe no lote/val; tratamos com try/except
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc_roc"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return out


def _unpack_validate_return(ret: Any) -> Tuple[float, Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normaliza o retorno de validate_epoch para um formato consistente:
      (val_loss, val_metrics, y_true, y_prob)

    Aceita retornos comuns:
      - (val_loss, metrics)
      - (val_loss, metrics, y_true, y_prob)
      - (val_loss, y_true, y_prob)  -> metrics será calculado
      - (val_loss,)                 -> metrics vazio
    """
    if not isinstance(ret, tuple):
        # retorna apenas loss
        return float(ret), {}, None, None

    if len(ret) == 2:
        val_loss, val_metrics = ret
        return float(val_loss), dict(val_metrics), None, None

    if len(ret) == 3:
        # pode ser (loss, y_true, y_prob) ou (loss, metrics, extra)
        val_loss, a, b = ret
        if isinstance(a, dict):
            return float(val_loss), dict(a), None, None
        return float(val_loss), {}, np.asarray(a), np.asarray(b)

    if len(ret) >= 4:
        val_loss, val_metrics, y_true, y_prob = ret[0], ret[1], ret[2], ret[3]
        return float(val_loss), dict(val_metrics) if isinstance(val_metrics, dict) else {}, np.asarray(y_true), np.asarray(y_prob)

    # fallback
    return float(ret[0]), {}, None, None


# ----------------------------
# Train / Validate
# ----------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    losses = []

    y_true_all = []
    y_prob_all = []

    for x, y in loader:
        # x: (N,C,D,H,W)  y: (N,)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)  # (N,)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().item())

        with torch.no_grad():
            prob = _sigmoid(logits).detach().float().cpu().numpy()
            y_prob_all.append(prob)
            y_true_all.append(y.detach().float().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    metrics = _compute_metrics(y_true, y_prob) if y_true.size else {}

    return mean_loss, metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    y_true_all = []
    y_prob_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.detach().item())
        prob = _sigmoid(logits).detach().float().cpu().numpy()

        y_prob_all.append(prob)
        y_true_all.append(y.detach().float().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])

    metrics = _compute_metrics(y_true, y_prob) if y_true.size else {}
    return mean_loss, metrics, y_true, y_prob


# ----------------------------
# Early stopping (inner)
# ----------------------------

def fit_inner_earlystop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    inner_val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    num_epochs: int,
    patience: int,
    fold_dir: Path,
    logger=None,
    fold_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Treina usando inner_train e usa inner_val APENAS para early stopping.
    Salva:
      - inner_history.json (histórico por epoch)
      - best_inner.pth (checkpoint do melhor val_loss)
    Log:
      - 1 linha por epoch com dt e ETA
    """
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val = float("inf")
    best_epoch = 0
    bad = 0

    history = []
    epoch_times = []
    t_start = time.time()

    best_path = fold_dir / "best_inner.pth"

    for epoch in range(num_epochs):
        t0 = time.time()

        tr_loss, tr_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp)

        # validate_epoch pode retornar 2/3/4 itens dependendo da versão.
        ret = validate_epoch(model, inner_val_loader, criterion, device)
        val_loss, val_metrics, _, _ = _unpack_validate_return(ret)

        dt = time.time() - t0
        epoch_times.append(dt)
        mean_dt = sum(epoch_times) / len(epoch_times)
        eta_s = mean_dt * (num_epochs - (epoch + 1))

        rec = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "train_metrics": tr_metrics,
            "val_metrics": val_metrics,
            "epoch_seconds": float(dt),
            "eta_minutes": float(eta_s / 60.0),
        }
        history.append(rec)

        if logger is not None:
            prefix = f"[Fold {fold_idx+1}] " if fold_idx is not None else ""
            logger.info(
                f"{prefix}Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_metrics.get('auc_roc', float('nan')):.4f} | "
                f"dt={dt:.1f}s | ETA={eta_s/60:.1f}min"
            )

        # early stopping pelo val_loss
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad = 0
            # salvar checkpoint do melhor modelo (state_dict)
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": best_val},
                best_path
            )
        else:
            bad += 1
            if bad >= patience:
                if logger is not None:
                    prefix = f"[Fold {fold_idx+1}] " if fold_idx is not None else ""
                    logger.info(f"{prefix}Early stopping acionado (patience={patience}) em epoch {epoch+1} | best={best_epoch+1}")
                break

    total_s = time.time() - t_start

    # salvar histórico
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / "inner_history.json", "w", encoding="utf-8") as f:
        import json
        json.dump(history, f, indent=2, ensure_ascii=False)

    out = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(history)),
        "total_minutes": float(total_s / 60.0),
    }
    return out


# ----------------------------
# Retrain (outer-train)
# ----------------------------

def train_fixed_epochs(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    epochs: int,
    fold_dir: Path,
    logger=None,
    fold_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-treina o modelo no outer-train completo por um número fixo de épocas.
    (Procedimento recomendado após encontrar best_epoch no inner-val.)
    """
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history = []
    t_start = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp)
        dt = time.time() - t0

        rec = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "train_metrics": tr_metrics,
            "epoch_seconds": float(dt),
        }
        history.append(rec)

        if logger is not None:
            prefix = f"[Fold {fold_idx+1}] " if fold_idx is not None else ""
            logger.info(
                f"{prefix}[Retrain] Epoch {epoch+1}/{epochs} | train_loss={tr_loss:.4f} | "
                f"train_auc={tr_metrics.get('auc_roc', float('nan')):.4f} | dt={dt:.1f}s"
            )

    total_s = time.time() - t_start

    # salvar histórico do retrain
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / "retrain_history.json", "w", encoding="utf-8") as f:
        import json
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {"epochs": int(epochs), "total_minutes": float(total_s / 60.0)}
