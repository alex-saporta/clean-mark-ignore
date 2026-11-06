"""Dataset loading and train/test splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit

DATA_HOME = Path(__file__).resolve().parent.parent / "data"
MAGIC_FEATURES = (
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
)


@dataclass(frozen=True)
class Table:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]


def load_magic_telescope(data_home: Path = DATA_HOME) -> Table:
    """MAGIC Gamma Telescope (OpenML 1120, UCI 159, CC BY 4.0). Positive class: gamma."""
    bunch = fetch_openml(data_id=1120, as_frame=True, data_home=str(data_home))
    frame = bunch.frame.rename(columns=lambda c: c.rstrip(":"))
    X = frame.loc[:, list(MAGIC_FEATURES)].to_numpy(dtype=np.float64)
    y = (frame["class"].astype(str) == "g").to_numpy(dtype=np.int64)
    if X.shape != (19020, 10) or np.isnan(X).any():
        raise ValueError(f"Unexpected table: shape={X.shape}, nan={np.isnan(X).sum()}")
    return Table("MagicTelescope", X, y, MAGIC_FEATURES)


def split_indices(y: np.ndarray, split: int, test_size: float = 0.2):
    """Stratified train/test indices, determined by the split number alone."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=split)
    return next(splitter.split(np.zeros(len(y)), y))
