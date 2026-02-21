"""
logging_utils.py
-----------------------------------------------------------------------------
Este módulo concentra utilidades de logging e persistência leve:
- make_logger(): logger que escreve no terminal e em arquivo (run.log)
- save_json()/load_json(): salvar/carregar JSON com indentação
- save_text(): salvar textos simples
- copy_code_snapshot(): copia SOMENTE o código-fonte do projeto para dentro
  do run_dir, para auditoria/reprodutibilidade.

IMPORTANTE:
- copy_code_snapshot() deve evitar copiar pastas geradas (results, __pycache__,
  .git, venv etc.). Caso contrário, ocorre recursão (snapshot copiando results
  que contém o snapshot que contém results...) e dá "File name too long".
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Iterable


def make_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _is_excluded(path: Path, excluded_names: set[str]) -> bool:
    # Exclui se qualquer parte do caminho tem nome proibido
    parts = set(path.parts)
    return len(parts.intersection(excluded_names)) > 0


def copy_code_snapshot(
    project_root: Path,
    snapshot_dir: Path,
    patterns: tuple[str, ...] = (".py",),
    excluded_dirs: Iterable[str] = (
        "results",
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        ".pytest_cache",
        "venv",
        ".venv",
        "env",
        ".mypy_cache",
        "wandb",
    ),
) -> None:
    """
    Copia arquivos de código do project_root para snapshot_dir, preservando
    estrutura relativa, MAS ignorando diretórios gerados.

    - patterns: extensões permitidas (default: apenas .py)
    - excluded_dirs: nomes de diretórios a ignorar completamente
    """
    project_root = project_root.resolve()
    snapshot_dir = snapshot_dir.resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(excluded_dirs)

    for p in project_root.rglob("*"):
        if not p.is_file():
            continue

        # Ignorar coisas dentro de dirs excluídos
        if _is_excluded(p, excluded):
            continue

        # Só copia extensões desejadas
        if p.suffix not in patterns:
            continue

        rel = p.relative_to(project_root)
        out = snapshot_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.read_bytes())
