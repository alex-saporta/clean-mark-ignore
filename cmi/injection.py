"""Error injection. Masks depend only on the split, the rate and the data part."""

import numpy as np

OUTLIER_MAGNITUDE = 5.0
OUT_OF_DOMAIN_OFFSETS = (1.0, 5.0)
PARTS = {"train": 0, "test": 1}


def error_rng(split: int, rate: float, part: str) -> np.random.Generator:
    """Generator shared by every strategy and location of a configuration."""
    return np.random.default_rng([split, round(rate * 10_000), PARTS[part]])


def draw_cells(shape: tuple[int, int], rate: float, rng: np.random.Generator) -> np.ndarray:
    """Pick exactly round(rate * n_rows) cells per column. Returns -1, 0 or +1 per cell."""
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    n_rows, n_cols = shape
    n_errors = round(rate * n_rows)
    signs = np.zeros(shape)
    for col in range(n_cols):
        rows = rng.choice(n_rows, size=n_errors, replace=False)
        signs[rows, col] = rng.choice((-1.0, 1.0), size=n_errors)
    return signs


def inject_outliers(
    X: np.ndarray,
    rate: float,
    scale: np.ndarray,
    rng: np.random.Generator,
    magnitude: float = OUTLIER_MAGNITUDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell becomes its true value +/- magnitude * scale.

    The error keeps the true value's order within each sign. `scale` must come
    from the clean training data, including when X is the test part.
    """
    if np.any(scale <= 0):
        raise ValueError("scale must be positive")
    signs = draw_cells(X.shape, rate, rng)
    return X + signs * magnitude * scale, signs != 0


def inject_random_outliers(
    X: np.ndarray,
    rate: float,
    low: np.ndarray,
    high: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell becomes a random value outside [low, high].

    The new value does not depend on the true value. For a given generator seed,
    the chosen cells and signs are those of `inject_outliers`. `low`, `high` and
    `scale` must come from the clean training data.
    """
    if np.any(scale <= 0):
        raise ValueError("scale must be positive")
    signs = draw_cells(X.shape, rate, rng)
    offsets = rng.uniform(*OUT_OF_DOMAIN_OFFSETS, size=X.shape) * scale
    dirty = X.copy()
    dirty[signs > 0] = (high + offsets)[signs > 0]
    dirty[signs < 0] = (low - offsets)[signs < 0]
    return dirty, signs != 0


def mark_missing(X: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Replace flagged cells with NaN."""
    marked = X.copy()
    marked[flagged] = np.nan
    return marked
