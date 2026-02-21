# 2026-2/oai3d/data.py
# -----------------------------------------------------------------------------
# Este módulo carrega o dataset OAI-MRI-3DDESS a partir de dois arquivos .npy:
#   - normal-3DESS-128-64.npy   (classe 0)
#   - abnormal-3DESS-128-64.npy (classe 1)
#
# Os arrays possuem shape (N, H, W, D). Ex.: (1659, 128, 128, 64)
# Aqui convertemos para (N, D, H, W) via np.moveaxis(arr, -1, 1).
#
# Além disso, para respeitar o protocolo experimental:
#   - augmentation deve ser aplicado APENAS ao treino
#   - validação/early-stop e teste (outer-test) NÃO recebem augmentation
#
# Para isso, usamos:
#   - um dataset base que só fornece volumes/labels
#   - um wrapper TransformViewDataset que aplica o transform desejado
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class KneeMRIDataset3D(Dataset):
    def __init__(
        self,
        dataset_root: str,
        cache_in_memory: bool = False,
        normal_filename: str = "normal-3DESS-128-64.npy",
        abnormal_filename: str = "abnormal-3DESS-128-64.npy",
    ):
        self.root = Path(dataset_root)
        self.cache_in_memory = cache_in_memory

        normal_path = self.root / normal_filename
        abnormal_path = self.root / abnormal_filename

        if not normal_path.exists():
            raise FileNotFoundError(f"Não achei: {normal_path}")
        if not abnormal_path.exists():
            raise FileNotFoundError(f"Não achei: {abnormal_path}")

        normal = np.load(normal_path)     # (N, H, W, D)
        abnormal = np.load(abnormal_path) # (N, H, W, D)

        normal = self._ensure_n_dhw(normal, name="normal")
        abnormal = self._ensure_n_dhw(abnormal, name="abnormal")

        self.volumes = np.concatenate([normal, abnormal], axis=0).astype(np.float32, copy=False)
        self.labels = np.concatenate(
            [
                np.zeros((normal.shape[0],), dtype=np.int64),
                np.ones((abnormal.shape[0],), dtype=np.int64),
            ],
            axis=0,
        )

    def _ensure_n_dhw(self, arr: np.ndarray, name: str) -> np.ndarray:
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"{name}: esperado ndim=4, veio shape={arr.shape}")

        # Seu shape: (N,128,128,64) = (N,H,W,D)
        # Converter para (N,D,H,W)
        arr = np.moveaxis(arr, -1, 1)  # (N,H,W,D) -> (N,D,H,W)
        return arr

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def get_raw(self, idx: int):
        # retorna (D,H,W) numpy + label int
        vol = self.volumes[idx]
        y = int(self.labels[idx])
        return vol, y

    def __getitem__(self, idx: int):
        # por padrão, retorna raw (sem transform)
        vol, y = self.get_raw(idx)
        y = torch.tensor(float(y), dtype=torch.float32)
        return vol, y

    def get_pos_weight(self) -> torch.Tensor:
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        if n_pos == 0:
            return torch.tensor([1.0], dtype=torch.float32)
        return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

class TransformViewDataset(Dataset):
    """
    Wrapper que aplica um transform específico sobre o mesmo dataset base.
    Útil para ter:
      - train_transform (com augmentation)
      - eval_transform (sem augmentation)
    sem duplicar os dados.
    """
    def __init__(self, base: KneeMRIDataset3D, transform: Optional[Callable]):
        self.base = base
        self.transform = transform
        self.labels = base.labels  # conveniência

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        vol, y_int = self.base.get_raw(idx)  # vol: (D,H,W) numpy
        if self.transform is not None:
            vol = self.transform(vol)
        y = torch.tensor(float(y_int), dtype=torch.float32)
        return vol, y

def make_loader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    sampler = SubsetRandomSampler(indices.tolist())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
