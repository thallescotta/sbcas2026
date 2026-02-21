"""
oai3d.model
-----------------------------------------------------------------------------
Modelo para os experimentos (EX1–EX10).

- Backbone: R3D_18 (torchvision.models.video.r3d_18)
- Adaptação para MRI (1 canal):
  - r3d_18 pré-treinado (Kinetics-400) tem conv1 com in_channels=3
  - para usar dados (C=1), substituímos conv1 por uma Conv3d com in_channels=1
  - inicializamos os pesos com a média dos 3 canais:
      new_w = old_w.mean(dim=1, keepdim=True) -> [64,1,3,7,7]

- Head binário: fc -> Linear(..., 1)
- Multi-GPU: DataParallel com gpu_ids explícitos (ex.: (0,1))
-----------------------------------------------------------------------------

Nota:
- DataParallel NÃO “enche a VRAM” automaticamente. Ele replica o modelo e divide batch.
- VRAM alta vem principalmente de batch maior / input maior / buffers, etc.
"""

from __future__ import annotations

from typing import Optional, Sequence
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18Binary(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()

        if pretrained:
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = r3d_18(weights=None)

        # ---- Adaptar conv1 para in_channels=1 (MRI) ----
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
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                elif pretrained and old_conv.weight.shape[1] == 3 and in_channels > 1:
                    w = old_conv.weight.mean(dim=1, keepdim=True)  # [64,1,3,7,7]
                    new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1, 1))
                    if old_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)

            self.backbone.stem[0] = new_conv

        # ---- Head binário ----
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # saída: (N,1) -> (N,)
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

        # filtra ids válidos
        n = torch.cuda.device_count()
        valid = [int(i) for i in gpu_ids if int(i) < n]

        if len(valid) >= 2:
            model = nn.DataParallel(model, device_ids=valid, output_device=valid[0])

    return model
