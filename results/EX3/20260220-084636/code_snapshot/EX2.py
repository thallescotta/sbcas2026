"""
EX2 (Ablation: treino do zero / sem transfer learning)
- Objetivo: medir o impacto do transfer learning.
- Diferença para EX1: pretrained=False (R3D_18 sem pesos Kinetics-400).
- Mantém: AMP, multi-GPU e o mesmo pré-processamento/aug do baseline.
- Nota (Amaro-friendly): sem oversampling; desbalanceamento tratado via pos_weight.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX2",
        dataset_root=str(data_dir),

        pretrained=False,
        amp=True,
        multi_gpu=True,
        gpu_ids=(0, 1),

        oversample=False,          # <- mudança principal
        augmentation=True,
        aug_profile="baseline",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
