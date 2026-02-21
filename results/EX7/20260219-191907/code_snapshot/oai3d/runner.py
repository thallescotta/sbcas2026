"""
runner.py
-----------------------------------------------------------------------------
Orquestrador do protocolo experimental (Amaro-safe) para EX1–EX10.

Mantém:
- Outer-test NÃO é usado em decisões do treino (sem leakage por early stopping)
- Inner-val serve APENAS para early stopping
- Avaliação final APENAS em unseen data (outer-test), sem augmentation

Performance (2.0, corrigido):
- Pré-processamento (ToTensor/ChannelFirst/MinMax) no Dataset (CPU).
- Augmentation pesada (affine/grid_sample) é aplicada NO TREINO, NA GPU,
  dentro do loop de treino (train_epoch), via aug_fn.
  => evita CPU virar gargalo e RAM estourar com prefetch.
- Flags CUDA: cudnn.benchmark + matmul_precision (se disponível)
- DataLoader: persistent_workers + prefetch_factor moderado (2).
-----------------------------------------------------------------------------

Dica (nohup):
- rode com CUDA_VISIBLE_DEVICES=0,1 para garantir as 2 GPUs visíveis.
- logs vão para results/.../run.log
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .config import ExperimentConfig
from .reproducibility import setup_seed
from .paths import make_run_dir, make_fold_dir, write_config
from .logging_utils import make_logger, save_json, copy_code_snapshot
from .cv import make_outer_folds, make_inner_split, maybe_oversample_indices
from .transforms import build_preprocess, build_augmentation
from .data import KneeMRIDataset3D, TransformViewDataset
from .model import create_model
from .train import build_criterion, build_optimizer, fit_inner_earlystop, train_fixed_epochs
from .eval import evaluate_outer_test_and_save
from .stats import summarize_folds


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds/60.0:.1f}min"


def _make_loader_fast(
    dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    DataLoader com opções de desempenho.
    - persistent_workers evita recriar workers a cada epoch.
    - prefetch_factor moderado (2) para não explodir RAM.
    """
    sampler = SubsetRandomSampler(indices.tolist())

    kwargs = dict(
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )

    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **kwargs)


def run_experiment(cfg: ExperimentConfig, project_root: Path) -> Dict[str, Any]:
    # -------------------------------------------------------------------------
    # Setup run + performance flags
    # -------------------------------------------------------------------------
    setup_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(project_root, cfg.results_dir, cfg.exp_name, cfg.run_tag)
    write_config(run_dir, cfg.to_dict())

    logger = make_logger(cfg.exp_name, log_file=run_dir / "run.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # “Intenção” explicitada em log (não garante sozinho, mas ajuda auditoria)
    logger.info(
        f"Config multi_gpu={cfg.multi_gpu} | gpu_ids={getattr(cfg, 'gpu_ids', None)} | "
        f"batch_size={cfg.batch_size} | num_workers={cfg.num_workers}"
    )

    if cfg.save_code_snapshot:
        logger.info("Etapa: copiando snapshot do código (.py) para auditoria...")
        t0 = time.time()
        copy_code_snapshot(project_root, run_dir / "code_snapshot")
        logger.info(f"Snapshot concluído em {_fmt_minutes(time.time() - t0)}")

    exp_start = time.time()

    # -------------------------------------------------------------------------
    # Dataset + preprocess (CPU) + aug_fn (GPU)
    # -------------------------------------------------------------------------
    logger.info("Etapa: carregando dataset base (sem transforms)...")
    t0 = time.time()
    base_dataset = KneeMRIDataset3D(
        dataset_root=cfg.dataset_root,
        cache_in_memory=cfg.cache_in_memory,
    )
    labels = base_dataset.labels
    logger.info(
        f"Dataset carregado em {_fmt_minutes(time.time() - t0)} | "
        f"N={len(labels)} | pos={(labels==1).sum()} neg={(labels==0).sum()}"
    )

    logger.info("Etapa: construindo preprocess (CPU) + augmentation_fn (GPU, só treino)...")
    t0 = time.time()

    preprocess = build_preprocess(cfg)          # CPU
    aug_fn = build_augmentation(cfg)            # GPU (aplicada no train_epoch)

    # Dataset: aplica apenas preprocess (train e eval iguais)
    train_dataset = TransformViewDataset(base_dataset, transform=preprocess)
    eval_dataset = TransformViewDataset(base_dataset, transform=preprocess)

    logger.info(
        f"Preprocess/aug_fn prontos em {_fmt_minutes(time.time() - t0)} | "
        f"aug_profile={getattr(cfg, 'aug_profile', 'baseline')} | augmentation={cfg.augmentation}"
    )

    # -------------------------------------------------------------------------
    # Outer folds
    # -------------------------------------------------------------------------
    logger.info(f"Etapa: gerando outer folds (n_splits={cfg.n_splits})...")
    t0 = time.time()
    folds = make_outer_folds(labels, cfg.n_splits, cfg.seed)
    logger.info(f"Outer folds prontos em {_fmt_minutes(time.time() - t0)}")

    all_fold_metrics: List[Dict[str, Any]] = []
    fold_times: List[float] = []

    # -------------------------------------------------------------------------
    # Loop folds
    # -------------------------------------------------------------------------
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(folds):
        fold_start = time.time()
        fold_dir = make_fold_dir(run_dir, fold_idx)

        logger.info(
            f"Fold {fold_idx+1}/{cfg.n_splits} | "
            f"outer_train={len(outer_train_idx)} outer_test={len(outer_test_idx)}"
        )

        save_json(fold_dir / "indices.json", {
            "outer_train_idx": outer_train_idx.tolist(),
            "outer_test_idx": outer_test_idx.tolist(),
        })

        # Inner split (early stopping)
        logger.info(f"Fold {fold_idx+1}: etapa inner split (inner_val_frac={cfg.inner_val_frac})...")
        t0 = time.time()
        inner_train_idx, inner_val_idx = make_inner_split(
            outer_train_idx, labels, cfg.inner_val_frac, cfg.seed + fold_idx
        )
        save_json(fold_dir / "indices_inner.json", {
            "inner_train_idx": inner_train_idx.tolist(),
            "inner_val_idx": inner_val_idx.tolist(),
        })
        logger.info(
            f"Fold {fold_idx+1}: inner split concluído em {_fmt_minutes(time.time() - t0)} | "
            f"inner_train={len(inner_train_idx)} inner_val={len(inner_val_idx)}"
        )

        # Oversampling somente no treino
        logger.info(f"Fold {fold_idx+1}: etapa oversampling (treino) | enabled={cfg.oversample}...")
        t0 = time.time()
        inner_train_idx_os = maybe_oversample_indices(
            inner_train_idx, labels, cfg.seed + fold_idx, cfg.oversample
        )
        logger.info(
            f"Fold {fold_idx+1}: oversampling concluído em {_fmt_minutes(time.time() - t0)} | "
            f"inner_train_os={len(inner_train_idx_os)}"
        )

        # DataLoaders
        logger.info(f"Fold {fold_idx+1}: etapa criando DataLoaders (train/inner_val/outer_test)...")
        t0 = time.time()
        inner_train_loader = _make_loader_fast(
            train_dataset, inner_train_idx_os,
            cfg.batch_size, cfg.num_workers, cfg.pin_memory,
            drop_last=True,
            prefetch_factor=2,
        )
        inner_val_loader = _make_loader_fast(
            eval_dataset, inner_val_idx,
            cfg.batch_size, cfg.num_workers, cfg.pin_memory,
            drop_last=False,
            prefetch_factor=2,
        )
        outer_test_loader = _make_loader_fast(
            eval_dataset, outer_test_idx,
            cfg.batch_size, cfg.num_workers, cfg.pin_memory,
            drop_last=False,
            prefetch_factor=2,
        )
        logger.info(f"Fold {fold_idx+1}: DataLoaders prontos em {_fmt_minutes(time.time() - t0)}")

        # Loss/pos_weight
        if cfg.pos_weight_auto:
            pos_weight = base_dataset.get_pos_weight()
        else:
            pos_weight = torch.tensor([cfg.pos_weight_fallback], dtype=torch.float32)

        # 1) Early stopping (inner)
        logger.info(
            f"Fold {fold_idx+1}: treino inner (early stopping) | "
            f"pretrained={cfg.pretrained} amp={cfg.amp} multi_gpu={cfg.multi_gpu} gpu_ids={cfg.gpu_ids}"
        )
        t0 = time.time()

        model = create_model(cfg.pretrained, device, cfg.multi_gpu, cfg.gpu_ids)
        criterion = build_criterion(pos_weight, device)
        optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)

        inner_info = fit_inner_earlystop(
            model=model,
            train_loader=inner_train_loader,
            inner_val_loader=inner_val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=cfg.amp,
            num_epochs=cfg.num_epochs,
            patience=cfg.early_stop_patience,
            fold_dir=fold_dir,
            logger=logger,
            fold_idx=fold_idx,
            aug_fn=aug_fn,  # <-- AUGMENTATION NA GPU (treino)
        )
        best_epoch = int(inner_info["best_epoch"])
        save_json(fold_dir / "best_inner.json", inner_info)

        logger.info(
            f"Fold {fold_idx+1}: treino inner concluído em {_fmt_minutes(time.time() - t0)} | "
            f"best_epoch(inner)={best_epoch+1} | epochs_ran={inner_info.get('epochs_ran', 'NA')}"
        )

        # 2) Retrain outer-train completo por best_epoch (recomendado)
        if cfg.retrain_outer_train:
            logger.info(f"Fold {fold_idx+1}: etapa retrain outer-train por {best_epoch+1} epochs...")
            t0 = time.time()

            model_final = create_model(cfg.pretrained, device, cfg.multi_gpu, cfg.gpu_ids)
            criterion2 = build_criterion(pos_weight, device)
            optimizer2 = build_optimizer(model_final, cfg.lr, cfg.weight_decay)

            outer_train_idx_os = maybe_oversample_indices(
                outer_train_idx, labels, cfg.seed + fold_idx, cfg.oversample
            )
            outer_train_loader = _make_loader_fast(
                train_dataset, outer_train_idx_os,
                cfg.batch_size, cfg.num_workers, cfg.pin_memory,
                drop_last=True,
                prefetch_factor=2,
            )

            train_fixed_epochs(
                model=model_final,
                train_loader=outer_train_loader,
                criterion=criterion2,
                optimizer=optimizer2,
                device=device,
                amp=cfg.amp,
                epochs=best_epoch + 1,
                fold_dir=fold_dir,
                logger=logger,
                fold_idx=fold_idx,
                aug_fn=aug_fn,  # <-- AUGMENTATION NA GPU (treino)
            )

            model_for_test = model_final
            criterion_for_test = criterion2

            logger.info(f"Fold {fold_idx+1}: retrain concluído em {_fmt_minutes(time.time() - t0)}")
        else:
            model_for_test = model
            criterion_for_test = criterion

        # 3) Avaliação outer-test (unseen) — SEM augmentation
        logger.info(f"Fold {fold_idx+1}: etapa avaliação outer-test (unseen data)...")
        t0 = time.time()

        metrics = evaluate_outer_test_and_save(
            model=model_for_test,
            outer_test_loader=outer_test_loader,
            criterion=criterion_for_test,
            device=device,
            fold_dir=fold_dir,
            logger=logger,
            fold_idx=fold_idx,
            amp=cfg.amp,
        )

        logger.info(f"Fold {fold_idx+1}: avaliação concluída em {_fmt_minutes(time.time() - t0)}")

        fold_elapsed = time.time() - fold_start
        metrics["best_epoch_inner"] = int(best_epoch)
        metrics["fold_idx"] = int(fold_idx)
        metrics["fold_minutes"] = float(fold_elapsed / 60.0)

        save_json(fold_dir / "fold_result.json", metrics)

        all_fold_metrics.append(metrics)
        fold_times.append(fold_elapsed)

        # ETA do experimento
        elapsed = time.time() - exp_start
        mean_fold = sum(fold_times) / len(fold_times)
        eta_total = mean_fold * cfg.n_splits
        eta_remain = max(0.0, eta_total - elapsed)

        logger.info(
            f"Fold {fold_idx+1}/{cfg.n_splits} finalizado | "
            f"tempo_fold={_fmt_minutes(fold_elapsed)} | "
            f"elapsed={_fmt_minutes(elapsed)} | "
            f"ETA_restante≈{_fmt_minutes(eta_remain)}"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("Etapa: agregando métricas e gerando summary.json...")

    summary = {
        "auc_roc": summarize_folds(all_fold_metrics, "auc_roc"),
        "accuracy": summarize_folds(all_fold_metrics, "accuracy"),
        "f1_score": summarize_folds(all_fold_metrics, "f1_score"),
        "precision": summarize_folds(all_fold_metrics, "precision"),
        "recall": summarize_folds(all_fold_metrics, "recall"),
        "n_folds": cfg.n_splits,
    }
    save_json(run_dir / "summary.json", summary)

    total_elapsed = time.time() - exp_start
    logger.info(
        f"Experimento finalizado em {_fmt_minutes(total_elapsed)} | "
        f"AUC mean={summary['auc_roc']['mean']:.4f} std={summary['auc_roc']['std']:.4f}"
    )

    return {"run_dir": str(run_dir), "summary": summary}
