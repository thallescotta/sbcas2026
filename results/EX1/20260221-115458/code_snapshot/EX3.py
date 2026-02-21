"""
EX3 (Ablation: sem AMP / sem mixed precision)
- Objetivo: medir impacto computacional do AMP (tempo/VRAM) e efeito eventual em métricas.
- Diferença para EX1: amp=False.
- Mantém: pretrained=True, multi-GPU, oversampling, augmentation baseline.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX3",
        dataset_root=str(data_dir),

        pretrained=True,
        amp=False,   # Mantém amp=False porque o objetivo do EX3 é justamente o ablation "sem AMP" (FP32), para medir impacto em tempo/VRAM/métricas.
        multi_gpu=True, 
        gpu_ids=(0, 1),   
        batch_size=2,       # necessário porque o default (32) estoura VRAM sem AMP em 3D conv
        num_workers=4,
        
        oversample=False,
        augmentation=True,
        aug_profile="baseline",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
