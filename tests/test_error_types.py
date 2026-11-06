"""Error types, the dependent mechanism, fewer targeted columns and a narrower detector."""

import numpy as np
import pytest

from cmi.data import Table
from cmi.grid import Variant, target_columns
from cmi.injection import (
    TYPOS_PER_CATEGORY,
    UNIT_FACTOR,
    error_rng,
    inject_in_domain,
    inject_typos,
    inject_unit_change,
    pick_columns,
    typo_code,
)
from cmi.models import fit_model, predict_scores
from cmi.repairs import detect, fit_repair
from cmi.strategies import corrupt

N_TRAIN = 1500
NUMERIC = (0, 1, 2, 3)
CATEGORICAL = (4, 5)
N_CATEGORIES = 4


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(5)
    n = 2000
    X = rng.normal(loc=3.0, size=(n, 6))
    X[:, 4] = rng.integers(N_CATEGORIES, size=n)
    X[:, 5] = rng.integers(N_CATEGORIES, size=n)
    y = X[:, 0] + X[:, 1] + X[:, 4] + 0.3 * rng.normal(size=n)
    return X, y


def split_parts(table):
    return table[0][:N_TRAIN], table[0][N_TRAIN:]


def test_unit_change_multiplies_only_chosen_cells(table):
    X_train, _ = split_parts(table)
    dirty, mask = inject_unit_change(X_train, 0.2, error_rng(0, 0.2, "train"), NUMERIC)
    assert (mask[:, list(NUMERIC)].sum(axis=0) == round(0.2 * N_TRAIN)).all()
    assert np.array_equal(dirty[mask], X_train[mask] * UNIT_FACTOR)
    assert np.array_equal(dirty[~mask], X_train[~mask])
    assert not mask[:, list(CATEGORICAL)].any()


def test_unit_change_skips_zero_cells():
    X = np.ones((100, 2))
    X[:90, 0] = 0.0
    dirty, mask = inject_unit_change(X, 0.2, error_rng(0, 0.2, "train"))
    assert not mask[:90, 0].any()
    assert mask[:, 0].sum() == 10
    assert mask[:, 1].sum() == 20
    assert np.all(dirty[mask] != X[mask])


def test_unit_error_avoids_columns_full_of_zeros():
    X = np.random.default_rng(0).normal(size=(1000, 4))
    X[:400, 1] = 0.0
    X[:10, 2] = 0.0
    table = Table("t", X, np.zeros(1000), ("a", "b", "c", "d"), categorical=(3,))
    assert target_columns(table, Variant(error="unit")) == (0, 2)
    assert target_columns(table, Variant(error="in_domain")) == (0, 1, 2)
    assert target_columns(table, Variant(error="typo")) == (3,)
    numeric_only = Table("t", X[:, :3], np.zeros(1000), ("a", "b", "c"))
    with pytest.raises(ValueError):
        target_columns(numeric_only, Variant(error="typo"))


def test_in_domain_values_come_from_the_training_column_and_differ(table):
    X_train, X_test = split_parts(table)
    dirty, mask = inject_in_domain(X_test, 0.2, X_train, error_rng(0, 0.2, "test"), NUMERIC)
    for col in NUMERIC:
        rows = mask[:, col]
        assert np.isin(dirty[rows, col], X_train[:, col]).all()
        assert (dirty[rows, col] != X_test[rows, col]).all()
        assert dirty[rows, col].max() <= X_train[:, col].max()
        assert dirty[rows, col].min() >= X_train[:, col].min()
    assert np.array_equal(dirty[~mask], X_test[~mask])


def test_category_swap_gives_another_seen_category(table):
    X_train, _ = split_parts(table)
    c = corrupt(X_train, table[0][N_TRAIN:], 0.2, 0, "category_swap", CATEGORICAL)
    assert not c.train_mask[:, list(NUMERIC)].any()
    for col in CATEGORICAL:
        rows = c.train_mask[:, col]
        assert rows.sum() == round(0.2 * N_TRAIN)
        assert (c.dirty_train[rows, col] != X_train[rows, col]).all()
        assert set(c.dirty_train[:, col]) <= set(range(N_CATEGORIES))


def test_typos_are_unseen_codes_tied_to_the_true_category(table):
    X_train, _ = split_parts(table)
    n_categories = {col: N_CATEGORIES for col in CATEGORICAL}
    dirty, mask = inject_typos(X_train, 0.2, n_categories, error_rng(0, 0.2, "train"), CATEGORICAL)
    for col in CATEGORICAL:
        rows = mask[:, col]
        true = X_train[rows, col]
        assert (dirty[rows, col] >= N_CATEGORIES).all()
        variants = dirty[rows, col] - N_CATEGORIES - TYPOS_PER_CATEGORY * true
        assert set(np.unique(variants)) <= set(range(TYPOS_PER_CATEGORY))
    assert np.array_equal(dirty[~mask], X_train[~mask])


def test_typo_codes_do_not_collide():
    codes = {
        float(typo_code(np.array(category), np.array(variant), N_CATEGORIES))
        for category in range(N_CATEGORIES)
        for variant in range(TYPOS_PER_CATEGORY)
    }
    assert len(codes) == N_CATEGORIES * TYPOS_PER_CATEGORY
    assert min(codes) >= N_CATEGORIES


@pytest.mark.parametrize("error", ["typo", "category_swap"])
def test_s1c_constant_avoids_every_category_code(table, error):
    X_train, X_test = split_parts(table)
    v = detect(corrupt(X_train, X_test, 0.2, 0, error, CATEGORICAL), "both", 0.75, 0.05, 0, CATEGORICAL)
    fitted = fit_repair("S1c", v)
    test = fitted.apply(v.dirty_test, v.flagged_test, v.clean_test)
    for col in CATEGORICAL:
        constant = fitted.train[v.flagged_train[:, col], col][0]
        assert constant not in set(v.dirty_train[:, col]) | set(v.dirty_test[:, col])
        assert (test[v.flagged_test[:, col], col] == constant).all()


def test_dependent_mechanism_favours_high_ranks_of_another_column():
    X = np.random.default_rng(2).normal(size=(5000, 3))
    c = corrupt(X, X.copy(), 0.2, 0, "in_domain", (0, 1), "dependent")
    chosen = c.train_mask[:, 0]
    assert X[chosen, 1].mean() > X[~chosen, 1].mean() + 0.3
    assert abs(X[chosen, 2].mean() - X[~chosen, 2].mean()) < 0.15
    assert chosen.sum() == round(0.2 * 5000)
    uniform = corrupt(X, X.copy(), 0.2, 0, "in_domain", (0, 1), "uniform")
    selected = uniform.train_mask[:, 0]
    assert abs(X[selected, 1].mean() - X[~selected, 1].mean()) < 0.15


def test_dependent_mechanism_needs_two_columns():
    X = np.random.default_rng(2).normal(size=(100, 3))
    with pytest.raises(ValueError):
        corrupt(X, X.copy(), 0.2, 0, "unit", (0,), "dependent")


def test_masks_do_not_change_across_numeric_errors_with_uniform_cells(table):
    X_train, X_test = split_parts(table)
    first = corrupt(X_train, X_test, 0.2, 0, "outlier", NUMERIC)
    second = corrupt(X_train, X_test, 0.2, 0, "in_domain", NUMERIC)
    assert np.array_equal(first.train_mask, second.train_mask)
    assert np.array_equal(first.test_mask, second.test_mask)


def test_test_errors_do_not_use_test_values_as_parameters(table):
    X_train, X_test = split_parts(table)
    shifted = X_test + 100.0
    c = corrupt(X_train, shifted, 0.2, 0, "in_domain", NUMERIC)
    assert (c.dirty_test[c.test_mask] < 50).all()


def test_pick_columns_is_fixed_and_keeps_two():
    candidates = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    assert pick_columns(candidates, 0.3) == pick_columns(candidates, 0.3)
    assert len(pick_columns(candidates, 0.3)) == 3
    assert len(pick_columns(candidates, 0.05)) == 2
    assert set(pick_columns(candidates, 0.3)) <= set(candidates)
    assert pick_columns((1, 2), 0.3) == (1, 2)
    with pytest.raises(ValueError):
        pick_columns(candidates, 0.0)


def test_detector_limited_to_targeted_columns_raises_no_alarm_elsewhere(table):
    X_train, X_test = split_parts(table)
    columns = (0, 1)
    c = corrupt(X_train, X_test, 0.2, 0, "outlier_random", columns)
    everywhere = detect(c, "both", 0.75, 0.05, 0, CATEGORICAL)
    limited = detect(c, "both", 0.75, 0.05, 0, CATEGORICAL, columns)
    assert everywhere.flagged_train[:, 2:].any()
    assert not limited.flagged_train[:, 2:].any()
    assert not limited.flagged_test[:, 2:].any()
    assert np.array_equal(limited.flagged_train[:, :2], everywhere.flagged_train[:, :2])
    assert not everywhere.errors_train[:, 2:].any()


@pytest.mark.parametrize(
    ("error", "columns"),
    [("unit", NUMERIC), ("in_domain", NUMERIC), ("category_swap", CATEGORICAL), ("typo", CATEGORICAL)],
)
@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
def test_oracle_wins_for_every_error_type(table, error, columns, model_name):
    X, y = table
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    v = detect(corrupt(X_train, X_test, 0.2, 0, error, columns), "both", 1.0, 0.0, 0, CATEGORICAL)
    rmse = {}
    for strategy in ("S0", "S5"):
        fitted = fit_repair(strategy, v)
        model = fit_model(model_name, "regression", CATEGORICAL, fitted.train, y_train)
        test = fitted.apply(v.dirty_test, v.flagged_test, v.clean_test)
        rmse[strategy] = np.sqrt(np.mean((y_test - predict_scores(model, test, "regression")) ** 2))
    assert rmse["S5"] < rmse["S0"]
