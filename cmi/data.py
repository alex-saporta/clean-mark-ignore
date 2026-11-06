"""Dataset loading and train/test splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

DATA_HOME = Path(__file__).resolve().parent.parent / "data"
ROW_LIMIT = 20_000
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
    """Categorical columns hold integer codes, sorted by category name."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    task: str = "classification"
    categorical: tuple[int, ...] = ()

    @property
    def numeric(self) -> tuple[int, ...]:
        return tuple(col for col in range(self.X.shape[1]) if col not in self.categorical)


@dataclass(frozen=True)
class OpenMLSpec:
    """`positive` is the target value of the positive class; None means regression."""

    data_id: int
    target: str
    positive: str | None = None
    drop: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()


# Licences checked at the source. See the README for citations.
OPENML_TABLES = {
    # UCI 222, CC BY 4.0. V12 is the call duration, known only after the outcome.
    "bank-marketing": OpenMLSpec(1461, "Class", positive="2", drop=("V12",)),
    # UCI 468, CC BY 4.0.
    "online-shoppers": OpenMLSpec(46947, "Revenue", positive="True"),
    # UCI 186, CC BY 4.0.
    "wine-quality": OpenMLSpec(287, "quality"),
    # UCI 275, CC BY 4.0. OpenML ignores casual and registered, which sum to the target.
    "bike-sharing": OpenMLSpec(42712, "count"),
    # Ships with ggplot2 (MIT). OpenML lists it as public domain.
    "diamonds": OpenMLSpec(42225, "price"),
    # StatLib asks for an acknowledgement; no explicit licence found.
    "houses": OpenMLSpec(537, "median_house_value"),
    # UCI 350, CC BY 4.0. Sex, education and marriage are stored as integer codes.
    "default-credit": OpenMLSpec(42477, "y", positive="1", categorical=("x2", "x3", "x4")),
    # UCI 2, CC BY 4.0. This OpenML version has no NaN: unknown values stay a category.
    "adult": OpenMLSpec(45068, "class", positive=">50K"),
}


def load_magic_telescope(data_home: Path = DATA_HOME) -> Table:
    """MAGIC Gamma Telescope (OpenML 1120, UCI 159, CC BY 4.0). Positive class: gamma."""
    bunch = fetch_openml(data_id=1120, as_frame=True, data_home=str(data_home))
    frame = bunch.frame.rename(columns=lambda c: c.rstrip(":"))
    X = frame.loc[:, list(MAGIC_FEATURES)].to_numpy(dtype=np.float64)
    y = (frame["class"].astype(str) == "g").to_numpy(dtype=np.int64)
    if X.shape != (19020, 10) or np.isnan(X).any():
        raise ValueError(f"Unexpected table: shape={X.shape}, nan={np.isnan(X).sum()}")
    return Table("MagicTelescope", X, y, MAGIC_FEATURES)


def cap_rows(
    X: np.ndarray, y: np.ndarray, task: str, limit: int = ROW_LIMIT, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most `limit` rows, stratified on the target for classification."""
    if len(X) <= limit:
        return X, y
    stratify = y if task == "classification" else None
    kept, _ = train_test_split(np.arange(len(X)), train_size=limit, random_state=seed, stratify=stratify)
    return X[kept], y[kept]


def encode_features(
    features: pd.DataFrame, forced_categories: tuple[str, ...] = ()
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Numeric columns stay as they are. Other columns become integer codes."""
    columns, categorical = [], []
    for position, name in enumerate(features.columns):
        column = features[name]
        is_number = pd.api.types.is_numeric_dtype(column) and not isinstance(column.dtype, pd.CategoricalDtype)
        if is_number and name not in forced_categories:
            columns.append(column.to_numpy(dtype=np.float64))
        else:
            values = column.astype(str)
            categories = sorted(values.unique())
            columns.append(pd.Categorical(values, categories=categories).codes.astype(np.float64))
            categorical.append(position)
    return np.column_stack(columns), tuple(categorical)


def load_openml_table(name: str, data_home: Path = DATA_HOME) -> Table:
    spec = OPENML_TABLES[name]
    # The default pandas parser fails on boolean or integer category names.
    frame = fetch_openml(data_id=spec.data_id, as_frame=True, data_home=str(data_home), parser="liac-arff").frame
    frame = frame.rename(columns=lambda c: c.rstrip(":"))
    target = frame[spec.target]
    features = frame.drop(columns=[spec.target, *spec.drop])
    if features.isna().any().any() or target.isna().any():
        raise ValueError(f"{name}: unexpected missing values")
    X, categorical = encode_features(features, spec.categorical)
    if spec.positive is None:
        task = "regression"
        y = target.astype(np.float64).to_numpy()
    else:
        task = "classification"
        y = (target.astype(str).str.strip() == spec.positive).to_numpy(dtype=np.int64)
        if y.min() == y.max():
            raise ValueError(f"{name}: positive value {spec.positive!r} gives a single class")
    X, y = cap_rows(X, y, task)
    return Table(name, X, y, tuple(features.columns), task, categorical)


def split_indices(y: np.ndarray, split: int, test_size: float = 0.2, task: str = "classification"):
    """Train/test indices, determined by the split number. Stratified for classification."""
    if task == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=split)
        return next(splitter.split(np.zeros(len(y)), y))
    return next(ShuffleSplit(n_splits=1, test_size=test_size, random_state=split).split(np.zeros(len(y))))
