"""
oai3d.config

Este módulo define a configuração padrão dos experimentos do projeto oai3d.

A classe ExperimentConfig centraliza os parâmetros de:
- Identidade do experimento (nome, tag)
- Caminhos/dataset e cache
- Protocolo de validação cruzada (k-fold + fração de validação interna)
- Hiperparâmetros de treino (épocas, batch size, AMP, etc.)
- Modelo (pretrained) e uso de GPU(s)
- Oversampling e normalização
- Augmentation (perfis e intensidades)
- Diretórios de saída e rastreabilidade (snapshot do código)
- Estratégia de re-treino no outer-train (padrão "Amaro-style")

IMPORTANTE (GPU):
- Por padrão, este config está preparado para usar 2 GPUs (0 e 1).
- O uso real de múltiplas GPUs depende de como runner.py/train.py implementam
  DataParallel/DDP e se o script do experimento não sobrescreve os defaults.
- CUDA_VISIBLE_DEVICES pode ser usado para restringir/remapear as GPUs visíveis.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence, Dict, Any


@dataclass(frozen=True)
class ExperimentConfig:
    # Identidade
    exp_name: str = "EX1"
    run_tag: str = ""

    # Dataset
    dataset_root: str = ""
    cache_in_memory: bool = False

    # CV (protocolo sem vazamento)
    n_splits: int = 5
    inner_val_frac: float = 0.15
    seed: int = 42

    # Treino
    num_epochs: int = 60
    early_stop_patience: int = 10
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True

    # Otimizador
    lr: float = 1e-4
    weight_decay: float = 1e-4
    pos_weight_auto: bool = True
    pos_weight_fallback: float = 2.0

    # Modelo / GPU
    pretrained: bool = True

    # DEFAULT AGORA: usa 2 GPUs (0 e 1)
    multi_gpu: bool = True
    gpu_ids: Sequence[int] = (0, 1)

    # Oversampling (apenas no treino)
    oversample: bool = False

    # Normalização por volume
    normalize_minmax: bool = True

    # Controle de augmentation (apenas no treino)
    augmentation: bool = True
    aug_profile: str = "baseline"
    # Perfis possíveis:
    #   "baseline", "none", "spatial", "radiometric",
    #   "strength_L1", "strength_L2", "strength_L3", "strength_L4"

    # Parâmetros genéricos de aug (base)
    flip_axes: Sequence[int] = (3,)   # em (C,D,H,W): D=1 H=2 W=3 (axial flip costuma ser W)
    flip_p: float = 0.5

    # Affine 3D (geométrico)
    affine_p: float = 0.5
    affine_rot_deg: float = 10.0          # rotação máxima em graus (por eixo)
    affine_trans_frac: float = 0.05       # translação máxima como fração do tamanho (por eixo)

    # Ruído
    noise_p: float = 0.5
    noise_std: float = 0.02

    # Saídas
    results_dir: str = "results"
    save_code_snapshot: bool = True

    # Amaro-style: depois de achar best_epoch no inner-val, re-treinar com outer-train completo
    retrain_outer_train: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
