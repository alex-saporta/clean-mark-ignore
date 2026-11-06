"""Apply a detector and a strategy to one split. Everything is fitted on training data.

S0      keep the data as is
S1      mark flagged cells as NaN
S1c     replace flagged cells with a constant outside the training range or categories
S2      replace flagged cells with the training mean, or the mode for a category
S3      replace flagged numeric cells with IterativeImputer (BayesianRidge)
S3hgb   same, with HistGradientBoosting as the estimator
S4      drop training rows with a flagged cell; the test part stays as is
S5      restore the true value of flagged cells (oracle, depends on the detector)
CLEAN   use the clean data everywhere (ceiling, does not depend on the detector)

S3 and S3hgb only use numeric columns, as targets and as predictors: an integer
code has no numeric meaning. Their flagged categorical cells get the mode, as in S2.
"""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from cmi.detector import detector_noise, flag_cells
from cmi.injection import TYPOS_PER_CATEGORY, mark_missing
from cmi.real_detectors import fit_detector
from cmi.strategies import Corrupted

ERRORS_IN_TRAIN = {"train": True, "test": False, "both": True}
ERRORS_IN_TEST = {"train": False, "test": True, "both": True}
STRATEGIES = ("S0", "S1", "S1c", "S2", "S3", "S3hgb", "S4", "S5", "CLEAN")
REPAIRING = ("S2", "S3", "S3hgb")
MAX_FLAGGED_SHARE = 0.5


@dataclass(frozen=True)
class DetectedSplit:
    """One location and detector. Unflagged errors stay in the dirty matrices."""

    clean_train: np.ndarray
    clean_test: np.ndarray
    dirty_train: np.ndarray
    dirty_test: np.ndarray
    errors_train: np.ndarray
    errors_test: np.ndarray
    flagged_train: np.ndarray
    flagged_test: np.ndarray
    categorical: tuple[int, ...] = ()

    @property
    def numeric(self) -> list[int]:
        return [col for col in range(self.dirty_train.shape[1]) if col not in self.categorical]


def detect(
    c: Corrupted,
    location: str,
    recall: float,
    fpr: float,
    split: int,
    categorical: tuple[int, ...] = (),
    scanned_columns: tuple[int, ...] | None = None,
) -> DetectedSplit:
    """Run the simulated detector on both parts, including a part without errors.

    `scanned_columns` limits the detector to some columns; by default it scans them all.
    """
    in_train, in_test = ERRORS_IN_TRAIN[location], ERRORS_IN_TEST[location]
    errors_train = c.train_mask if in_train else np.zeros_like(c.train_mask)
    errors_test = c.test_mask if in_test else np.zeros_like(c.test_mask)
    noise_train = detector_noise(errors_train.shape, split, "train")
    noise_test = detector_noise(errors_test.shape, split, "test")
    flagged_train = flag_cells(errors_train, recall, fpr, noise_train)
    flagged_test = flag_cells(errors_test, recall, fpr, noise_test)
    if scanned_columns is not None:
        unscanned = np.setdiff1d(np.arange(errors_train.shape[1]), scanned_columns)
        flagged_train[:, unscanned] = False
        flagged_test[:, unscanned] = False
    return DetectedSplit(
        clean_train=c.clean_train,
        clean_test=c.clean_test,
        dirty_train=c.dirty_train if in_train else c.clean_train,
        dirty_test=c.dirty_test if in_test else c.clean_test,
        errors_train=errors_train,
        errors_test=errors_test,
        flagged_train=flagged_train,
        flagged_test=flagged_test,
        categorical=categorical,
    )


def detect_real(
    c: Corrupted,
    location: str,
    detector_name: str,
    numeric: tuple[int, ...],
    categorical: tuple[int, ...] = (),
) -> DetectedSplit:
    """Fit a real detector on the training part as it is, then scan both parts."""
    in_train, in_test = ERRORS_IN_TRAIN[location], ERRORS_IN_TEST[location]
    dirty_train = c.dirty_train if in_train else c.clean_train
    dirty_test = c.dirty_test if in_test else c.clean_test
    detector = fit_detector(detector_name, dirty_train, numeric, categorical)
    flagged_train = detector.flag(dirty_train)
    flagged_test = detector.flag(dirty_test)
    # A detector that flags most of a column no longer finds anomalies there.
    switched_off = flagged_train.mean(axis=0) > MAX_FLAGGED_SHARE
    flagged_train[:, switched_off] = False
    flagged_test[:, switched_off] = False
    return DetectedSplit(
        clean_train=c.clean_train,
        clean_test=c.clean_test,
        dirty_train=dirty_train,
        dirty_test=dirty_test,
        errors_train=c.train_mask if in_train else np.zeros_like(c.train_mask),
        errors_test=c.test_mask if in_test else np.zeros_like(c.test_mask),
        flagged_train=flagged_train,
        flagged_test=flagged_test,
        categorical=categorical,
    )


@dataclass(frozen=True)
class FittedRepair:
    """A training matrix and the transformation of a test matrix, fitted on training data."""

    train: np.ndarray
    kept_rows: np.ndarray
    apply: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    seconds: float
    imputer_iterations: int | None = None


def _fill(X: np.ndarray, values: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    return np.where(flagged, np.broadcast_to(values, X.shape), X)


def _mode(column: np.ndarray) -> float:
    values, counts = np.unique(column[~np.isnan(column)], return_counts=True)
    return float(values[np.argmax(counts)])


def _replacement_values(v: DetectedSplit) -> np.ndarray:
    """Mean of each numeric column and mode of each categorical one, on unflagged cells."""
    marked = mark_missing(v.dirty_train, v.flagged_train)
    values = np.nanmean(marked, axis=0)
    for col in v.categorical:
        values[col] = _mode(marked[:, col])
    return values


def _constant_values(v: DetectedSplit) -> np.ndarray:
    """Below the numeric range, and a category code that no cell uses."""
    marked = mark_missing(v.dirty_train, v.flagged_train)
    low, high = np.nanmin(marked, axis=0), np.nanmax(marked, axis=0)
    values = low - (high - low)
    for col in v.categorical:
        # Beyond every code a misspelling of a training category can take.
        values[col] = (np.nanmax(v.clean_train[:, col]) + 1) * (TYPOS_PER_CATEGORY + 1)
    return values


def _make_imputer(strategy: str) -> IterativeImputer:
    if strategy == "S3":
        return IterativeImputer(random_state=0)
    if strategy == "S3hgb":
        estimator = HistGradientBoostingRegressor(max_iter=50, random_state=0)
        return IterativeImputer(estimator=estimator, max_iter=5, random_state=0)
    raise ValueError(f"Not an imputer: {strategy}")


def _fit_imputation(strategy: str, v: DetectedSplit) -> tuple[np.ndarray, Callable, int]:
    numeric = v.numeric
    imputer = _make_imputer(strategy)
    modes = _replacement_values(v)

    def complete(X: np.ndarray, flagged: np.ndarray, numeric_part: np.ndarray) -> np.ndarray:
        result = _fill(X, modes, flagged)
        result[:, numeric] = numeric_part
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        train_numeric = imputer.fit_transform(mark_missing(v.dirty_train, v.flagged_train)[:, numeric])

    def apply(dirty: np.ndarray, flagged_test: np.ndarray, clean: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            test_numeric = imputer.transform(mark_missing(dirty, flagged_test)[:, numeric])
        return complete(dirty, flagged_test, test_numeric)

    return complete(v.dirty_train, v.flagged_train, train_numeric), apply, int(imputer.n_iter_)


def fit_repair(strategy: str, v: DetectedSplit) -> FittedRepair:
    """Fit a strategy. Test data never reach this function, except through `apply`."""
    start = time.perf_counter()
    X, flagged = v.dirty_train, v.flagged_train
    if flagged.all(axis=0).any():
        raise ValueError("A training column is entirely flagged: nothing to fit a repair on")
    kept = np.ones(len(X), dtype=bool)
    iterations = None
    if strategy in ("S0", "S4"):
        train, apply = X, lambda dirty, flagged_test, clean: dirty
        if strategy == "S4":
            kept = ~flagged.any(axis=1)
            train = X[kept]
    elif strategy == "S1":
        train = mark_missing(X, flagged)
        apply = lambda dirty, flagged_test, clean: mark_missing(dirty, flagged_test)
    elif strategy in ("S1c", "S2"):
        values = _constant_values(v) if strategy == "S1c" else _replacement_values(v)
        train = _fill(X, values, flagged)
        apply = lambda dirty, flagged_test, clean: _fill(dirty, values, flagged_test)
    elif strategy in ("S3", "S3hgb"):
        train, apply, iterations = _fit_imputation(strategy, v)
    elif strategy == "S5":
        train = np.where(flagged, v.clean_train, X)
        apply = lambda dirty, flagged_test, clean: np.where(flagged_test, clean, dirty)
    elif strategy == "CLEAN":
        train, apply = v.clean_train, lambda dirty, flagged_test, clean: clean
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return FittedRepair(train, kept, apply, time.perf_counter() - start, iterations)


def repair_error(repaired: np.ndarray, clean: np.ndarray, flagged: np.ndarray, scale: np.ndarray) -> float:
    """RMSE on flagged cells, in training standard deviations of each column."""
    if not flagged.any():
        return float("nan")
    return float(np.sqrt(np.mean((((repaired - clean) / scale)[flagged]) ** 2)))


def category_repair_accuracy(repaired: np.ndarray, clean: np.ndarray, flagged: np.ndarray) -> float:
    """Share of flagged categorical cells that hold their true category."""
    if not flagged.any():
        return float("nan")
    return float(np.mean(repaired[flagged] == clean[flagged]))
