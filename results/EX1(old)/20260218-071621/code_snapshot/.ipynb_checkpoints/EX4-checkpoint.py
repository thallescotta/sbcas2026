"""
EX4 (Ablation: sem data augmentation)
- Objetivo: quantificar quanto do ganho vem do augmentation (mantendo todo o resto igual).
- Diferença para EX1: augmentation=False (apenas normalização por volume).
- Mantém: pretrained=True, AMP, multi-GPU, oversampling.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX4",
        dataset_root=str(data_dir),

        pretrained=True,
        amp=True,
        multi_gpu=True,
        gpu_ids=(0, 1),

        oversample=True,
        augmentation=False,
        aug_profile="none",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
