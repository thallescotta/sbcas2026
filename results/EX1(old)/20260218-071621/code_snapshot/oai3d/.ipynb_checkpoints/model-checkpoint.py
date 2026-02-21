"""
model.py
-----------------------------------------------------------------------------
Este módulo define o modelo usado nos experimentos.

- Usa o backbone R3D_18 (torchvision.models.video.r3d_18).
- Para dados MRI (1 canal), adapta a primeira conv3d do backbone para aceitar
  1 canal em vez de 3 (RGB). Isso evita o erro:
    "expected input to have 3 channels, but got 1"

- Quando pretrained=True:
  - carregamos pesos pré-treinados (Kinetics-400)
  - a conv1 original tem shape [64, 3, 3, 7, 7]
  - criamos uma nova conv1 com in_channels=1
  - inicializamos o peso novo com a MÉDIA ao longo do eixo de canais:
      new_w = old_w.mean(dim=1, keepdim=True)  -> [64, 1, 3, 7, 7]

- Também substituímos o "fc" para saída binária (1 logit).
- Suporte a multi-GPU via DataParallel.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class R3D18Binary(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        self.backbone = r3d_18(weights="DEFAULT" if pretrained else None)

        # ---- Adaptar conv1 para in_channels=1 (MRI) ----
        # No torchvision video resnet, a conv inicial fica em:
        # backbone.stem[0]  (Conv3d)
        old_conv: nn.Conv3d = self.backbone.stem[0]
        if old_conv.in_channels != in_channels:
            new_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            with torch.no_grad():
                if pretrained and old_conv.weight.shape[1] == 3 and in_channels == 1:
                    # Média dos 3 canais RGB -> 1 canal
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                elif pretrained and old_conv.weight.shape[1] == 3 and in_channels > 1:
                    # Caso geral: replicar média para múltiplos canais
                    w = old_conv.weight.mean(dim=1, keepdim=True)  # [64,1,3,7,7]
                    new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1, 1))
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                else:
                    # Inicialização padrão (Kaiming) para casos não previstos
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)

            self.backbone.stem[0] = new_conv

        # ---- Substituir o head para binário (1 logit) ----
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # saída: (N, 1) -> (N,)
        return self.backbone(x).squeeze(1)


def create_model(
    pretrained: bool,
    device: torch.device,
    multi_gpu: bool = False,
    gpu_ids: Optional[Sequence[int]] = None,
) -> nn.Module:
    model = R3D18Binary(pretrained=pretrained, in_channels=1).to(device)

    if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(gpu_ids))

    return model
