# 2026-2/oai3d/transforms.py
# -----------------------------------------------------------------------------
# Transforms 3D para MRI (OAI 3D DESS) no formato (C, D, H, W).
#
# PRINCÍPIO (performance + Amaro-friendly):
# - Pré-processamento (ToTensor/ChannelFirst/MinMax) acontece no Dataset (CPU).
# - Augmentation pesada (affine/grid_sample) acontece APÓS x.to(device) (GPU),
#   dentro do loop de treino (train_epoch), evitando gargalo de CPU/RAM.
#
# Isso garante:
# - augmentation SOMENTE no treino
# - inner_val/outer_test SEM augmentation
# - GPU alimentada e sem "RAM 100% e GPU 0%"
#
# Perfis (cfg.aug_profile):
#   - none, baseline, spatial, radiometric, strength_L1..strength_L4
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------
# Compose
# ----------------------------
class Compose3D:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ----------------------------
# PREPROCESS (CPU ok)
# ----------------------------
@dataclass
class ToTensor3D:
    """Converte numpy (D,H,W) ou (C,D,H,W) para torch.float32."""
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x)).float()


@dataclass
class EnsureChannelFirst3D:
    """Garante (C,D,H,W). Se vier (D,H,W), vira (1,D,H,W)."""
    def __call__(self, x: torch.Tensor):
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x


@dataclass
class MinMaxNormalize3D:
    """Normalização min-max por volume (amostra), mapeando para [0,1]."""
    eps: float = 1e-8

    def __call__(self, x: torch.Tensor):
        x_min = torch.amin(x)
        x_max = torch.amax(x)
        return (x - x_min) / (x_max - x_min + self.eps)


def build_preprocess(cfg) -> Callable:
    """Pipeline determinístico (CPU): ToTensor -> ChannelFirst -> MinMax(opc)."""
    ts: List[Callable] = [ToTensor3D(), EnsureChannelFirst3D()]
    if getattr(cfg, "normalize_minmax", True):
        ts.append(MinMaxNormalize3D())
    return Compose3D(ts)


# ----------------------------
# AUGMENTATION (deve rodar NA GPU)
# ----------------------------
@dataclass
class RandomFlip3D:
    """Flip aleatório em eixos escolhidos (aceita (N,C,D,H,W) ou (C,D,H,W))."""
    axes: Sequence[int] = (4,)  # default para batch: (N,C,D,H,W) -> W=4
    p: float = 0.5

    def __call__(self, x: torch.Tensor):
        # Suporta (C,D,H,W) e (N,C,D,H,W)
        if x.ndim == 4:
            # (C,D,H,W): W=3
            axes = tuple(ax - 1 for ax in self.axes)  # converte (N,C,D,H,W) -> (C,D,H,W)
        else:
            axes = tuple(self.axes)

        for ax in axes:
            if torch.rand(1, device=x.device).item() <= self.p:
                x = torch.flip(x, dims=(ax,))
        return x


@dataclass
class AddGaussianNoise3D:
    """Ruído gaussiano com probabilidade p; clampa após min-max (opcional)."""
    std: float = 0.02
    p: float = 0.5
    clip: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0

    def __call__(self, x: torch.Tensor):
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        noise = torch.randn_like(x) * float(self.std)
        y = x + noise
        if self.clip:
            y = torch.clamp(y, self.clip_min, self.clip_max)
        return y


@dataclass
class RandomAxialAffine3D:
    """
    Affine axial-only (H,W) para BATCH (N,C,D,H,W) ou single (C,D,H,W).
    IMPORTANTE: para performance, execute isso na GPU (x já em cuda).
    """
    rot_deg: float = 10.0
    trans_frac: float = 0.05
    p: float = 0.5
    padding_mode: str = "border"

    def __call__(self, x: torch.Tensor):
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        if x.ndim == 4:
            xin = x.unsqueeze(0)  # (1,C,D,H,W)
        elif x.ndim == 5:
            xin = x               # (N,C,D,H,W)
        else:
            return x

        device = xin.device
        N, C, D, H, W = xin.shape

        # rotação apenas no plano (H,W) => em coords (z,y,x): z fixo
        angle = (torch.rand(1, device=device).item() * 2 - 1) * math.radians(float(self.rot_deg))
        ca, sa = math.cos(angle), math.sin(angle)

        R = torch.tensor(
            [[1.0, 0.0, 0.0],
             [0.0, ca, -sa],
             [0.0, sa,  ca]],
            dtype=torch.float32,
            device=device,
        )

        tx_pix = (torch.rand(1, device=device).item() * 2 - 1) * (float(self.trans_frac) * W)
        ty_pix = (torch.rand(1, device=device).item() * 2 - 1) * (float(self.trans_frac) * H)

        tx = 2.0 * tx_pix / max(W - 1, 1)
        ty = 2.0 * ty_pix / max(H - 1, 1)
        tz = 0.0

        theta = torch.zeros((N, 3, 4), dtype=torch.float32, device=device)
        theta[:, :3, :3] = R
        theta[:, :, 3] = torch.tensor([tz, ty, tx], dtype=torch.float32, device=device).view(1, 3).repeat(N, 1)

        grid = F.affine_grid(theta, size=xin.size(), align_corners=False)
        xout = F.grid_sample(
            xin, grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False
        )

        return xout.squeeze(0) if x.ndim == 4 else xout


def _profile_params(cfg, profile: str) -> Dict[str, Any]:
    profile = (profile or "baseline").strip()

    if profile == "none":
        return dict(flip=False, affine=False, noise=False)

    if profile == "baseline":
        return dict(flip=True, affine=False, noise=False)

    if profile == "spatial":
        return dict(
            flip=True, affine=True, noise=False,
            rot_deg=float(cfg.affine_rot_deg),
            trans_frac=float(cfg.affine_trans_frac),
        )

    if profile == "radiometric":
        return dict(
            flip=True, affine=False, noise=True,
            noise_std=float(cfg.noise_std),
        )

    # Sweep alinhado ao 1.0 (ajuste conforme seu LaTeX)
    sweep = {
        "strength_L1": dict(rot_deg=5.0,  trans_frac=0.05, noise_std=0.01),
        "strength_L2": dict(rot_deg=10.0, trans_frac=0.10, noise_std=0.02),
        "strength_L3": dict(rot_deg=15.0, trans_frac=0.15, noise_std=0.03),
        "strength_L4": dict(rot_deg=20.0, trans_frac=0.20, noise_std=0.05),
    }
    if profile in sweep:
        p = sweep[profile]
        return dict(flip=True, affine=True, noise=True, **p)

    return dict(flip=True, affine=False, noise=False)


def build_augmentation(cfg) -> Optional[Callable]:
    """
    Retorna uma função de augmentation para usar NO LOOP DE TREINO (GPU).
    Se cfg.augmentation=False ou profile='none', retorna None.
    """
    if not getattr(cfg, "augmentation", True):
        return None

    params = _profile_params(cfg, getattr(cfg, "aug_profile", "baseline"))
    ts: List[Callable] = []

    if params.get("flip", False):
        # Para batch (N,C,D,H,W): W=4
        ts.append(RandomFlip3D(axes=(4,), p=float(cfg.flip_p)))

    if params.get("affine", False):
        ts.append(RandomAxialAffine3D(
            rot_deg=float(params.get("rot_deg", cfg.affine_rot_deg)),
            trans_frac=float(params.get("trans_frac", cfg.affine_trans_frac)),
            p=float(cfg.affine_p),
            padding_mode="border",
        ))

    if params.get("noise", False):
        ts.append(AddGaussianNoise3D(
            std=float(params.get("noise_std", cfg.noise_std)),
            p=float(cfg.noise_p),
            clip=True,
            clip_min=0.0,
            clip_max=1.0,
        ))

    if not ts:
        return None

    return Compose3D(ts)
