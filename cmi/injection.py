"""Error injection. Masks depend only on the split, the rate and the data part.

Each error type has a function taking the matrix, the rate, its parameters from the
clean training data, and a generator. The mechanism chooses the cells:

uniform     every cell of a targeted column has the same chance
dependent   the chance grows with the rank of the row in another column
"""

import numpy as np

OUTLIER_MAGNITUDE = 5.0
OUT_OF_DOMAIN_OFFSETS = (1.0, 5.0)
UNIT_FACTOR = 1000.0
TYPOS_PER_CATEGORY = 3
MECHANISMS = ("uniform", "dependent")
PARTS = {"train": 0, "test": 1}


def error_rng(split: int, rate: float, part: str) -> np.random.Generator:
    """Generator shared by every strategy and location of a configuration."""
    return np.random.default_rng([split, round(rate * 10_000), PARTS[part]])


def _target_columns(n_cols: int, columns: tuple[int, ...] | None) -> tuple[int, ...]:
    return tuple(range(n_cols)) if columns is None else columns


def _check_scale(scale: np.ndarray, columns: tuple[int, ...]) -> None:
    if np.any(scale[list(columns)] <= 0):
        raise ValueError("scale must be positive on the targeted columns")


def _rank_weights(X: np.ndarray, col: int, columns: tuple[int, ...]) -> np.ndarray:
    """Selection weights from the rank of the row in the next targeted column."""
    if len(columns) < 2:
        raise ValueError("The dependent mechanism needs a second targeted column")
    reference = columns[(columns.index(col) + 1) % len(columns)]
    ranks = np.argsort(np.argsort(X[:, reference], kind="stable"), kind="stable") + 1.0
    return ranks / ranks.sum()


def draw_cells(
    shape: tuple[int, int],
    rate: float,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
    eligible: np.ndarray | None = None,
    weights_of=None,
) -> np.ndarray:
    """Pick round(rate * n_rows) cells per targeted column. Returns -1, 0 or +1.

    `eligible` excludes cells the error could not change, which can give fewer cells.
    `weights_of(col)` returns selection weights for the dependent mechanism.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    n_rows, n_cols = shape
    n_errors = round(rate * n_rows)
    signs = np.zeros(shape)
    for col in _target_columns(n_cols, columns):
        if eligible is None and weights_of is None:
            rows = rng.choice(n_rows, size=n_errors, replace=False)
        else:
            candidates = np.arange(n_rows) if eligible is None else np.flatnonzero(eligible[:, col])
            probabilities = None
            if weights_of is not None:
                probabilities = weights_of(col)[candidates]
                probabilities = probabilities / probabilities.sum()
            rows = rng.choice(
                candidates, size=min(n_errors, len(candidates)), replace=False, p=probabilities
            )
        signs[rows, col] = rng.choice((-1.0, 1.0), size=len(rows))
    return signs


def _mechanism_weights(X: np.ndarray, columns: tuple[int, ...], mechanism: str):
    if mechanism == "uniform":
        return None
    if mechanism == "dependent":
        return lambda col: _rank_weights(X, col, columns)
    raise ValueError(f"Unknown mechanism: {mechanism}")


def inject_outliers(
    X: np.ndarray,
    rate: float,
    scale: np.ndarray,
    rng: np.random.Generator,
    magnitude: float = OUTLIER_MAGNITUDE,
    columns: tuple[int, ...] | None = None,
    mechanism: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell becomes its true value +/- magnitude * scale.

    The error keeps the true value's order within each sign. `scale` must come
    from the clean training data, including when X is the test part.
    """
    targets = _target_columns(X.shape[1], columns)
    _check_scale(scale, targets)
    signs = draw_cells(X.shape, rate, rng, columns, weights_of=_mechanism_weights(X, targets, mechanism))
    return X + signs * magnitude * scale, signs != 0


def inject_random_outliers(
    X: np.ndarray,
    rate: float,
    low: np.ndarray,
    high: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
    mechanism: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell becomes a random value outside [low, high].

    The new value does not depend on the true value. For a given generator seed,
    the chosen cells and signs are those of `inject_outliers`. `low`, `high` and
    `scale` must come from the clean training data.
    """
    targets = _target_columns(X.shape[1], columns)
    _check_scale(scale, targets)
    signs = draw_cells(X.shape, rate, rng, columns, weights_of=_mechanism_weights(X, targets, mechanism))
    offsets = rng.uniform(*OUT_OF_DOMAIN_OFFSETS, size=X.shape) * scale
    dirty = X.copy()
    dirty[signs > 0] = (high + offsets)[signs > 0]
    dirty[signs < 0] = (low - offsets)[signs < 0]
    return dirty, signs != 0


def inject_unit_change(
    X: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
    mechanism: str = "uniform",
    factor: float = UNIT_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell is multiplied by `factor`, as with a wrong unit.

    Zero cells would not change, so they are not eligible. The error keeps the
    order of the true values.
    """
    targets = _target_columns(X.shape[1], columns)
    signs = draw_cells(X.shape, rate, rng, columns, X != 0, _mechanism_weights(X, targets, mechanism))
    mask = signs != 0
    return np.where(mask, X * factor, X), mask


def _replace_with_other_value(
    X: np.ndarray, mask: np.ndarray, pools: dict[int, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Replace each masked cell with a value of its column's pool, different from the true one."""
    dirty = X.copy()
    for col, pool in pools.items():
        rows = np.flatnonzero(mask[:, col])
        for row in rows:
            others = pool[pool != X[row, col]]
            if len(others) == 0:
                raise ValueError(f"Column {col} has a single value")
            dirty[row, col] = others[rng.integers(len(others))]
    return dirty


def inject_in_domain(
    X: np.ndarray,
    rate: float,
    train_values: np.ndarray,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
    mechanism: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen cell takes another distinct value seen in its clean training column.

    The new value is plausible on its own, so a tree cannot isolate it by its range.
    Works for numeric columns, and for categories as a replaced category.
    """
    targets = _target_columns(X.shape[1], columns)
    signs = draw_cells(X.shape, rate, rng, columns, weights_of=_mechanism_weights(X, targets, mechanism))
    mask = signs != 0
    pools = {col: np.unique(train_values[:, col]) for col in targets}
    return _replace_with_other_value(X, mask, pools, rng), mask


def typo_code(category: np.ndarray, variant: np.ndarray, n_categories: int) -> np.ndarray:
    """A distinct code for each misspelling of each category. A model only sees its identity."""
    return n_categories + TYPOS_PER_CATEGORY * category + variant


def inject_typos(
    X: np.ndarray,
    rate: float,
    n_categories: dict[int, int],
    rng: np.random.Generator,
    columns: tuple[int, ...],
    mechanism: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Each chosen category gets one of a few misspellings, unseen in clean data."""
    signs = draw_cells(X.shape, rate, rng, columns, weights_of=_mechanism_weights(X, columns, mechanism))
    mask = signs != 0
    dirty = X.copy()
    for col in columns:
        rows = np.flatnonzero(mask[:, col])
        variants = rng.integers(TYPOS_PER_CATEGORY, size=len(rows))
        dirty[rows, col] = typo_code(X[rows, col], variants, n_categories[col])
    return dirty, mask


def pick_columns(candidates: tuple[int, ...], share: float, seed: int = 0) -> tuple[int, ...]:
    """A fixed subset of columns, without looking at the target. At least two when possible."""
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}")
    n_columns = min(len(candidates), max(2, round(share * len(candidates))))
    chosen = np.random.default_rng(seed).choice(len(candidates), size=n_columns, replace=False)
    return tuple(sorted(candidates[i] for i in chosen))


def mark_missing(X: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Replace flagged cells with NaN."""
    marked = X.copy()
    marked[flagged] = np.nan
    return marked
