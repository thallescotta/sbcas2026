"""
EX1-1 (Baseline Puro 2.0)
- Objetivo: estabelecer um baseline "limpo" (sem oversampling) para o projeto 2026-2.
- Protocolo: Nested CV/5-fold (outer-test isolado do early stopping) + logs/artefatos por fold.
- Treino: AMP ligado, multi-GPU ligado, oversampling DESATIVADO no treino.
- Desbalanceamento: tratado via pos_weight dinâmico calculado no conjunto de treino.
- Comparação: este baseline garante que a única diferença para os demais EX seja o aumento de dados.
"""
from pathlib import Path
from oai3d.config import ExperimentConfig
from oai3d.runner import run_experiment

if __name__ == "__main__":
    # Caminho para o dataset conforme configurado no servidor deep-02
    data_dir = Path.home() / "dataset" / "OAI-MRI-3DDESS"

    cfg = ExperimentConfig(
        exp_name="EX1-1",
        dataset_root=str(data_dir),

        pretrained=True,      # R3D-18 com pesos Kinetics-400
        amp=True,             # Mixed Precision para performance
        multi_gpu=True,       # Habilitar uso das 2 GPUs RTX 2080 Ti
        gpu_ids=(0, 1),

        # Ajuste Crítico: desativado para consistência com a matriz experimental EX2-EX10
        oversample=False,     
        
        augmentation=True,
        aug_profile="baseline", # Apenas Flip axial (p=0.5)
    )

    # Executa o experimento via orquestrador runner
    result = run_experiment(cfg, project_root=Path(__file__).resolve().parent)
    print(result)