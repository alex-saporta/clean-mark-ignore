"""Build the train and test matrices seen by S0, S1 and S5 with a perfect detector."""

from dataclasses import dataclass

import numpy as np

from cmi.injection import error_rng, inject_outliers, inject_random_outliers, mark_missing

LOCATIONS = ("train", "test", "both")
STRATEGIES = ("S0", "S1", "S5")


@dataclass(frozen=True)
class Corrupted:
    """One clean split and its errors. Each location reuses the same masks."""

    clean_train: np.ndarray
    clean_test: np.ndarray
    dirty_train: np.ndarray
    dirty_test: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray


def corrupt(
    X_train: np.ndarray, X_test: np.ndarray, rate: float, split: int, error: str = "outlier"
) -> Corrupted:
    scale = X_train.std(axis=0)
    low, high = X_train.min(axis=0), X_train.max(axis=0)

    def inject(X: np.ndarray, part: str) -> tuple[np.ndarray, np.ndarray]:
        rng = error_rng(split, rate, part)
        if error == "outlier":
            return inject_outliers(X, rate, scale, rng)
        if error == "outlier_random":
            return inject_random_outliers(X, rate, low, high, scale, rng)
        raise ValueError(f"Unknown error: {error}")

    dirty_train, train_mask = inject(X_train, "train")
    dirty_test, test_mask = inject(X_test, "test")
    return Corrupted(X_train, X_test, dirty_train, dirty_test, train_mask, test_mask)


def train_variant(strategy: str, location: str) -> str:
    """Name the training matrix, so that identical training data share one fit."""
    if strategy == "S5" or location == "test":
        return "clean"
    return {"S0": "dirty", "S1": "marked"}[strategy]


def train_matrix(c: Corrupted, variant: str) -> np.ndarray:
    if variant == "clean":
        return c.clean_train
    if variant == "dirty":
        return c.dirty_train
    if variant == "marked":
        return mark_missing(c.dirty_train, c.train_mask)
    raise ValueError(f"Unknown variant: {variant}")


def eval_matrix(c: Corrupted, strategy: str, location: str) -> np.ndarray:
    if strategy == "S5" or location == "train":
        return c.clean_test
    if strategy == "S0":
        return c.dirty_test
    if strategy == "S1":
        return mark_missing(c.dirty_test, c.test_mask)
    raise ValueError(f"Unknown strategy: {strategy}")
