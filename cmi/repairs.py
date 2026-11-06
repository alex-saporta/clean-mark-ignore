"""Apply a detector and a strategy to one split. Everything is fitted on training data.

S0      keep the data as is
S1      mark flagged cells as NaN
S1c     replace flagged cells with a constant outside the training range
S2      replace flagged cells with the training mean
S3      replace flagged cells with IterativeImputer (BayesianRidge)
S3rf    same, with a random forest as the estimator
S4      drop training rows with a flagged cell; the test part stays as is
S5      restore the true value of flagged cells (oracle, depends on the detector)
CLEAN   use the clean data everywhere (ceiling, does not depend on the detector)
"""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from cmi.detector import detector_noise, flag_cells
from cmi.injection import mark_missing
from cmi.strategies import Corrupted

ERRORS_IN_TRAIN = {"train": True, "test": False, "both": True}
ERRORS_IN_TEST = {"train": False, "test": True, "both": True}
STRATEGIES = ("S0", "S1", "S1c", "S2", "S3", "S4", "S5", "CLEAN")
REPAIRING = ("S2", "S3", "S3rf")


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


def detect(c: Corrupted, location: str, recall: float, fpr: float, split: int) -> DetectedSplit:
    """Run the simulated detector on both parts, including a part without errors."""
    in_train, in_test = ERRORS_IN_TRAIN[location], ERRORS_IN_TEST[location]
    errors_train = c.train_mask if in_train else np.zeros_like(c.train_mask)
    errors_test = c.test_mask if in_test else np.zeros_like(c.test_mask)
    noise_train = detector_noise(errors_train.shape, split, "train")
    noise_test = detector_noise(errors_test.shape, split, "test")
    return DetectedSplit(
        clean_train=c.clean_train,
        clean_test=c.clean_test,
        dirty_train=c.dirty_train if in_train else c.clean_train,
        dirty_test=c.dirty_test if in_test else c.clean_test,
        errors_train=errors_train,
        errors_test=errors_test,
        flagged_train=flag_cells(errors_train, recall, fpr, noise_train),
        flagged_test=flag_cells(errors_test, recall, fpr, noise_test),
    )


@dataclass(frozen=True)
class FittedRepair:
    """A training matrix and the transformation of a test matrix, fitted on training data."""

    train: np.ndarray
    kept_rows: np.ndarray
    apply: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    seconds: float
    imputer_iterations: int | None = None


def _column_values(X: np.ndarray, column_values: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    return np.where(flagged, np.broadcast_to(column_values, X.shape), X)


def _make_imputer(strategy: str) -> IterativeImputer:
    if strategy == "S3":
        return IterativeImputer(random_state=0)
    if strategy == "S3rf":
        # One thread: summing tree predictions across threads changes the last float bits.
        forest = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, n_jobs=1, random_state=0)
        return IterativeImputer(estimator=forest, random_state=0)
    raise ValueError(f"Not an imputer: {strategy}")


def fit_repair(strategy: str, v: DetectedSplit) -> FittedRepair:
    """Fit a strategy. Test data never reach this function, except through `apply`."""
    start = time.perf_counter()
    X, flagged = v.dirty_train, v.flagged_train
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
    elif strategy == "S1c":
        marked = mark_missing(X, flagged)
        low, high = np.nanmin(marked, axis=0), np.nanmax(marked, axis=0)
        constant = low - (high - low)
        train = _column_values(X, constant, flagged)
        apply = lambda dirty, flagged_test, clean: _column_values(dirty, constant, flagged_test)
    elif strategy == "S2":
        means = np.nanmean(mark_missing(X, flagged), axis=0)
        train = _column_values(X, means, flagged)
        apply = lambda dirty, flagged_test, clean: _column_values(dirty, means, flagged_test)
    elif strategy in ("S3", "S3rf"):
        imputer = _make_imputer(strategy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            train = imputer.fit_transform(mark_missing(X, flagged))
        iterations = int(imputer.n_iter_)

        def apply(dirty: np.ndarray, flagged_test: np.ndarray, clean: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                return imputer.transform(mark_missing(dirty, flagged_test))
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
