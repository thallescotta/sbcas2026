"""
EX8 (Varredura de intensidade - Nível L2 / moderado)
- Objetivo: sweep com augmentation moderado.
- Perfil: strength_L2.
- Mantém: pretrained=True, AMP, multi-GPU, oversampling.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX8",
        dataset_root=str(data_dir),

        pretrained=True,
        amp=True,
        multi_gpu=True,
        gpu_ids=(0, 1),

        oversample=False,
        augmentation=True,
        aug_profile="strength_L2",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
