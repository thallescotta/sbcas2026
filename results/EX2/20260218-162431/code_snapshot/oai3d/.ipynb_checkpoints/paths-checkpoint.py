from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Tuple

from .logging_utils import save_json

def make_run_dir(project_root: Path, results_dir: str, exp_name: str, run_tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    run_dir = project_root / results_dir / exp_name / f"{ts}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def make_fold_dir(run_dir: Path, fold_idx: int) -> Path:
    d = run_dir / "folds" / f"fold_{fold_idx+1}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_config(run_dir: Path, config_dict: dict) -> None:
    save_json(run_dir / "config.json", config_dict)
