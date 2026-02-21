"""
EX1 (Baseline 2.0)
- Objetivo: estabelecer o baseline do projeto 2026-2 com R3D_18 pré-treinado.
- Protocolo: Nested CV/5-fold (outer-test isolado do early stopping) + logs/artefatos por fold.
- Treino: AMP ligado, multi-GPU ligado, oversampling no treino (apenas), augmentation baseline.
- Comparação: todos os demais experimentos devem ser comparados contra este baseline.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX1",
        dataset_root=str(data_dir),

        pretrained=True,
        amp=True,
        multi_gpu=True,
        gpu_ids=(0, 1),

        oversample=True,
        augmentation=True,
        aug_profile="baseline",
    )

    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)
