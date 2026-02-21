from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler

def make_outer_folds(labels: np.ndarray, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(labels))
    return list(skf.split(idx, labels))

def make_inner_split(outer_train_idx: np.ndarray, labels: np.ndarray, inner_val_frac: float, seed: int):
    y = labels[outer_train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed)
    inner_train_rel, inner_val_rel = next(sss.split(np.zeros(len(outer_train_idx)), y))
    inner_train_idx = np.array(outer_train_idx)[inner_train_rel]
    inner_val_idx = np.array(outer_train_idx)[inner_val_rel]
    return inner_train_idx, inner_val_idx

def maybe_oversample_indices(train_idx: np.ndarray, labels: np.ndarray, seed: int, enabled: bool):
    if not enabled:
        return train_idx
    ros = RandomOverSampler(random_state=seed)
    X = train_idx.reshape(-1, 1)
    y = labels[train_idx]
    resampled_idx, _ = ros.fit_resample(X, y)
    return resampled_idx.flatten()
