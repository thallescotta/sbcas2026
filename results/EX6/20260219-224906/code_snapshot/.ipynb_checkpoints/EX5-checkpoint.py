"""
EX5 (Augmentation espacial avançada)
- Objetivo: avaliar aumento espacial (rotação/translação/flip) com transformações geométricas.
- Diferença para EX1: aug_profile="spatial" (e augmentation=True).
- Mantém: pretrained=True, AMP, multi-GPU, oversampling.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX5",
        dataset_root=str(data_dir),

        pretrained=True,
        amp=True,
        multi_gpu=True,
        gpu_ids=(0, 1),

        oversample=False,
        augmentation=True,
        aug_profile="spatial",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
