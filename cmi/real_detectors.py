"""Real detectors, fitted on the possibly dirty training part and applied to any part.

sd       numeric cell more than 3 standard deviations from the mean
iqr      numeric cell more than 1.5 interquartile ranges outside the quartiles
iforest  numeric cell rated abnormal by an Isolation Forest fitted on its column alone
rules    SQL rules run with DuckDB: numeric cell more than 3 interquartile ranges
         outside the quartiles, or category rarer than 0.5 % of the training rows

    uv run python -m cmi.real_detectors

prints the quality of each detector, per table and error type. The numeric rule was
changed once after a first measure (see the README).
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from cmi.data import OPENML_TABLES
from cmi.detector import detection_quality

SD_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5
RULE_FENCE = 3.0
RARE_CATEGORY_SHARE = 0.005
DETECTORS = ("sd", "iqr", "iforest", "rules")


def _empty_flags(X: np.ndarray) -> np.ndarray:
    return np.zeros(X.shape, dtype=bool)


class BoundsDetector:
    """Flag numeric cells outside bounds computed on the training part."""

    def __init__(self, low: dict[int, float], high: dict[int, float]):
        self.low, self.high = low, high

    def flag(self, X: np.ndarray) -> np.ndarray:
        flagged = _empty_flags(X)
        for col, low in self.low.items():
            flagged[:, col] = (X[:, col] < low) | (X[:, col] > self.high[col])
        return flagged


def fit_sd(X: np.ndarray, numeric: tuple[int, ...]) -> BoundsDetector:
    mean, sd = X.mean(axis=0), X.std(axis=0)
    return BoundsDetector(
        {col: mean[col] - SD_THRESHOLD * sd[col] for col in numeric},
        {col: mean[col] + SD_THRESHOLD * sd[col] for col in numeric},
    )


def fit_iqr(X: np.ndarray, numeric: tuple[int, ...]) -> BoundsDetector:
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    spread = q3 - q1
    return BoundsDetector(
        {col: q1[col] - IQR_MULTIPLIER * spread[col] for col in numeric},
        {col: q3[col] + IQR_MULTIPLIER * spread[col] for col in numeric},
    )


class ForestDetector:
    """One Isolation Forest per numeric column, since the forest only rates whole rows."""

    def __init__(self, forests: dict[int, IsolationForest]):
        self.forests = forests

    def flag(self, X: np.ndarray) -> np.ndarray:
        flagged = _empty_flags(X)
        for col, forest in self.forests.items():
            flagged[:, col] = forest.predict(X[:, [col]]) == -1
        return flagged


def fit_iforest(X: np.ndarray, numeric: tuple[int, ...], seed: int = 0) -> ForestDetector:
    return ForestDetector(
        {col: IsolationForest(random_state=seed + col, n_jobs=1).fit(X[:, [col]]) for col in numeric}
    )


class SqlRulesDetector:
    """Rules derived in SQL from the training part, then applied in SQL to any part."""

    def __init__(self, conditions: dict[int, str]):
        self.conditions = conditions

    def flag(self, X: np.ndarray) -> np.ndarray:
        flagged = _empty_flags(X)
        if not self.conditions:
            return flagged
        frame = _as_frame(X)
        selected = ", ".join(f"({rule}) AS f{col}" for col, rule in self.conditions.items())
        with duckdb.connect() as connection:
            connection.register("cells", frame)
            result = connection.execute(f"SELECT {selected} FROM cells ORDER BY row_id").fetchnumpy()
        for col in self.conditions:
            flagged[:, col] = result[f"f{col}"].astype(bool)
        return flagged


def _as_frame(X: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(X, columns=[f"c{col}" for col in range(X.shape[1])])
    frame["row_id"] = np.arange(len(X))
    return frame


def fit_rules(X: np.ndarray, numeric: tuple[int, ...], categorical: tuple[int, ...]) -> SqlRulesDetector:
    conditions = {}
    with duckdb.connect() as connection:
        connection.register("cells", _as_frame(X))
        for col in numeric:
            low, high = connection.execute(
                f"SELECT quantile_cont(c{col}, 0.25), quantile_cont(c{col}, 0.75) FROM cells"
            ).fetchone()
            margin = RULE_FENCE * (high - low)
            conditions[col] = f"c{col} < {low - margin!r} OR c{col} > {high + margin!r}"
        minimum = RARE_CATEGORY_SHARE * len(X)
        for col in categorical:
            allowed = connection.execute(
                f"SELECT c{col} FROM cells GROUP BY c{col} HAVING count(*) >= {minimum!r} ORDER BY c{col}"
            ).fetchall()
            values = ", ".join(repr(float(row[0])) for row in allowed)
            conditions[col] = f"c{col} NOT IN ({values})" if values else "TRUE"
    return SqlRulesDetector(conditions)


def fit_detector(name: str, X_train: np.ndarray, numeric: tuple[int, ...], categorical: tuple[int, ...]):
    """Fit on the training part only. sd, iqr and iforest ignore categorical columns."""
    if name == "sd":
        return fit_sd(X_train, numeric)
    if name == "iqr":
        return fit_iqr(X_train, numeric)
    if name == "iforest":
        return fit_iforest(X_train, numeric)
    if name == "rules":
        return fit_rules(X_train, numeric, categorical)
    raise ValueError(f"Unknown detector: {name}")


def quality_table(n_splits: int = 1) -> pd.DataFrame:
    """Measured quality on the test part, with errors on both parts."""
    from cmi.data import split_indices
    from cmi.grid import CATEGORY_ERRORS, NUMERIC_ERRORS, Variant, load_table, target_columns
    from cmi.repairs import detect_real
    from cmi.strategies import corrupt

    records = []
    for name in ("MagicTelescope", *OPENML_TABLES):
        table = load_table(name)
        for error in (*NUMERIC_ERRORS, *CATEGORY_ERRORS):
            if error == "outlier":
                continue
            try:
                columns = target_columns(table, Variant(error=error))
            except ValueError:
                continue
            for split, rate in ((s, r) for s in range(n_splits) for r in (0.05, 0.20)):
                train, test = split_indices(table.y, split, task=table.task)
                c = corrupt(table.X[train], table.X[test], rate, split, error, columns)
                for detector_name in DETECTORS:
                    view = detect_real(c, "both", detector_name, table.numeric, table.categorical)
                    quality = detection_quality(view.errors_test, view.flagged_test)
                    records.append(
                        {
                            "table": name,
                            "error": error,
                            "rate": rate,
                            "detector": detector_name,
                            "recall": quality.recall,
                            "fpr": quality.fpr,
                            "precision": quality.precision,
                        }
                    )
    return pd.DataFrame(records).groupby(["table", "error", "rate", "detector"]).mean().reset_index()


if __name__ == "__main__":
    print(quality_table().round(3).to_markdown(index=False))
