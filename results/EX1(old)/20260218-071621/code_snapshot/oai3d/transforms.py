# 2026-2/oai3d/transforms.py
# -----------------------------------------------------------------------------
# Este módulo define transforms 3D para volumes de MRI do joelho (OAI 3D DESS),
# operando em tensores no formato (C, D, H, W).
#
# Objetivo do desenho:
#   1) Pré-processamento determinístico (ToTensor -> ChannelFirst -> MinMax).
#   2) Augmentation SOMENTE no treino (controlado fora daqui via build_transforms).
#   3) Augmentation "axial-only":
#        - rotação e translação apenas no plano axial (H x W)
#        - NÃO aplica rotação/translação no eixo de profundidade (D)
#      Isso evita distorções volumétricas não desejadas e é coerente com a
#      nomenclatura do artigo ("rotação/translação axial").
#
# Perfis de augmentation (cfg.aug_profile):
#   - "none"        : sem augmentation
#   - "baseline"    : somente flip axial (alinha com EX1 na sua tabela)
#   - "spatial"     : flip + affine axial-only (rotação/translação)
#   - "radiometric" : flip + ruído
#   - "strength_L1".."strength_L4": sweep com flip + affine axial-only + ruído
#     usando os mesmos valores do projeto 1.0 / tabela do LaTeX:
#       L1: rot=5°,  trans=5%,  noise=0.01
#       L2: rot=10°, trans=10%, noise=0.02
#       L3: rot=15°, trans=15%, noise=0.03
#       L4: rot=20°, trans=20%, noise=0.05
#
# Observação importante:
# - A probabilidade de aplicação é controlada por cfg.affine_p e cfg.noise_p.
# - Flip usa cfg.flip_p e cfg.flip_axes (por padrão W -> eixo 3).
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Dict, Any
import math
import numpy as np
import torch
import torch.nn.functional as F


class Compose3D:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


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


@dataclass
class RandomFlip3D:
    """
    Flip aleatório em eixos escolhidos.
    Por padrão, (C,D,H,W): eixo 3 (W) corresponde a "flip axial" (esquerda-direita).
    """
    axes: Sequence[int] = (3,)
    p: float = 0.5

    def __call__(self, x: torch.Tensor):
        # Decide para cada eixo, independentemente, com probabilidade p.
        for ax in self.axes:
            if torch.rand(1).item() <= self.p:
                x = torch.flip(x, dims=(ax,))
        return x


@dataclass
class AddGaussianNoise3D:
    """
    Adiciona ruído Gaussiano com probabilidade p.
    Se clip=True, clampa para [clip_min, clip_max] (útil após min-max).
    """
    std: float = 0.02
    p: float = 0.5
    clip: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0

    def __call__(self, x: torch.Tensor):
        if torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * float(self.std)
        y = x + noise
        if self.clip:
            y = torch.clamp(y, self.clip_min, self.clip_max)
        return y


@dataclass
class RandomAxialAffine3D:
    """
    Affine RANDOM "axial-only" para tensores (C,D,H,W), usando affine_grid/grid_sample 3D.

    - Rotação: apenas em torno do eixo de profundidade (D), i.e., rotação no plano HxW.
    - Translação: apenas em H/W.
    - NÃO move no eixo D (tz = 0).
    """
    rot_deg: float = 10.0
    trans_frac: float = 0.05
    p: float = 0.5
    padding_mode: str = "border"  # 'zeros' | 'border' | 'reflection'

    def __call__(self, x: torch.Tensor):
        if torch.rand(1).item() > self.p:
            return x
        if x.ndim != 4:
            return x

        device = x.device
        C, D, H, W = x.shape

        # Amostrar rotação apenas no plano axial (H,W): angle em rad
        angle = (torch.rand(1, device=device).item() * 2 - 1) * math.radians(float(self.rot_deg))
        ca, sa = math.cos(angle), math.sin(angle)

        # Matriz de rotação 3D que gira SOMENTE em (y,x) = (H,W), mantendo z (D) fixo:
        # Em coordenadas (z,y,x): z não muda; (y,x) rotaciona.
        R = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, ca, -sa],
                [0.0, sa,  ca],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Translação em pixels, só em H/W
        tx_pix = (torch.rand(1, device=device).item() * 2 - 1) * (float(self.trans_frac) * W)
        ty_pix = (torch.rand(1, device=device).item() * 2 - 1) * (float(self.trans_frac) * H)

        # Converter para coordenadas normalizadas [-1,1]
        tx = 2.0 * tx_pix / max(W - 1, 1)
        ty = 2.0 * ty_pix / max(H - 1, 1)
        tz = 0.0  # axial-only: não mexe na profundidade

        theta = torch.zeros((1, 3, 4), dtype=torch.float32, device=device)
        theta[0, :3, :3] = R
        # ordem (z, y, x) -> (D, H, W)
        theta[0, :, 3] = torch.tensor([tz, ty, tx], dtype=torch.float32, device=device)

        # grid_sample espera (N,C,D,H,W)
        xin = x.unsqueeze(0)
        grid = F.affine_grid(theta, size=xin.size(), align_corners=False)
        xout = F.grid_sample(
            xin,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        return xout.squeeze(0)


def _profile_params(cfg, profile: str) -> Dict[str, Any]:
    """
    Mapeia perfis -> quais transforms ativar + parâmetros.

    Importante para coerência com o artigo:
      - baseline = flip apenas (sem rotação/translação/ruído)
      - spatial = flip + affine axial-only
      - radiometric = flip + ruído
      - strength_L* = flip + affine axial-only + ruído (com valores 1.0 do LaTeX)
    """
    profile = (profile or "baseline").strip()

    if profile == "none":
        return dict(flip=False, affine=False, noise=False)

    if profile == "baseline":
        return dict(flip=True, affine=False, noise=False)

    if profile == "spatial":
        return dict(
            flip=True,
            affine=True,
            noise=False,
            rot_deg=float(cfg.affine_rot_deg),
            trans_frac=float(cfg.affine_trans_frac),
        )

    if profile == "radiometric":
        return dict(
            flip=True,
            affine=False,
            noise=True,
            noise_std=float(cfg.noise_std),
        )

    # Sweep levels alinhados com o projeto 1.0 (tabela do LaTeX)
    sweep = {
        "strength_L1": dict(rot_deg=5.0,  trans_frac=0.05, noise_std=0.01),
        "strength_L2": dict(rot_deg=10.0, trans_frac=0.10, noise_std=0.02),
        "strength_L3": dict(rot_deg=15.0, trans_frac=0.15, noise_std=0.03),
        "strength_L4": dict(rot_deg=20.0, trans_frac=0.20, noise_std=0.05),
    }
    if profile in sweep:
        p = sweep[profile]
        return dict(
            flip=True,
            affine=True,
            noise=True,
            rot_deg=float(p["rot_deg"]),
            trans_frac=float(p["trans_frac"]),
            noise_std=float(p["noise_std"]),
        )

    # fallback seguro: baseline
    return dict(flip=True, affine=False, noise=False)


def build_transforms(cfg, training: bool) -> Callable:
    """
    Cria pipeline de transforms.

    - training=False:
        apenas pré-processamento (ToTensor, ChannelFirst, MinMax se cfg.normalize_minmax).
        NÃO aplica augmentation.

    - training=True:
        pré-processamento + augmentation SE cfg.augmentation=True, respeitando cfg.aug_profile.
    """
    ts: List[Callable] = [ToTensor3D(), EnsureChannelFirst3D()]

    if getattr(cfg, "normalize_minmax", True):
        ts.append(MinMaxNormalize3D())

    if training and getattr(cfg, "augmentation", True):
        prof = getattr(cfg, "aug_profile", "baseline")
        params = _profile_params(cfg, prof)

        if params.get("flip", False):
            ts.append(RandomFlip3D(axes=tuple(cfg.flip_axes), p=float(cfg.flip_p)))

        if params.get("affine", False):
            ts.append(
                RandomAxialAffine3D(
                    rot_deg=float(params.get("rot_deg", cfg.affine_rot_deg)),
                    trans_frac=float(params.get("trans_frac", cfg.affine_trans_frac)),
                    p=float(cfg.affine_p),
                    padding_mode="border",
                )
            )

        if params.get("noise", False):
            ts.append(
                AddGaussianNoise3D(
                    std=float(params.get("noise_std", cfg.noise_std)),
                    p=float(cfg.noise_p),
                    clip=True,  # como você normaliza min-max por volume, clampa é seguro
                    clip_min=0.0,
                    clip_max=1.0,
                )
            )

    return Compose3D(ts)
