"""Simulated detector with an exact recall and false positive rate per column."""

from dataclasses import dataclass

import numpy as np

from cmi.injection import PARTS


def detector_noise(shape: tuple[int, int], split: int, part: str) -> np.ndarray:
    """One random priority per cell, shared across detectors so that they are nested."""
    return np.random.default_rng([split, PARTS[part], 7919]).random(shape)


def flag_cells(errors: np.ndarray, recall: float, fpr: float, noise: np.ndarray) -> np.ndarray:
    """Flag round(recall * errors) errors and round(fpr * clean) clean cells per column.

    An all-False `errors` still gets false alarms, as with a real detector.
    """
    if not 0 <= recall <= 1:
        raise ValueError(f"recall must be in [0, 1], got {recall}")
    if not 0 <= fpr <= 1:
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")
    if errors.shape != noise.shape:
        raise ValueError(f"Shapes differ: {errors.shape} and {noise.shape}")
    flagged = np.zeros(errors.shape, dtype=bool)
    for col in range(errors.shape[1]):
        is_error = errors[:, col]
        for rows, rate in ((np.flatnonzero(is_error), recall), (np.flatnonzero(~is_error), fpr)):
            n_flagged = round(rate * len(rows))
            chosen = rows[np.argsort(noise[rows, col], kind="stable")[:n_flagged]]
            flagged[chosen, col] = True
    return flagged


@dataclass(frozen=True)
class DetectionQuality:
    recall: float
    fpr: float
    precision: float


def detection_quality(errors: np.ndarray, flagged: np.ndarray) -> DetectionQuality:
    """Measured quality. A value is NaN when its denominator is empty."""

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    true_positives = int((errors & flagged).sum())
    false_positives = int((~errors & flagged).sum())
    return DetectionQuality(
        recall=ratio(true_positives, int(errors.sum())),
        fpr=ratio(false_positives, int((~errors).sum())),
        precision=ratio(true_positives, true_positives + false_positives),
    )
