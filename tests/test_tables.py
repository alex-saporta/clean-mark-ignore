"""The real tables. They download once from OpenML into data/."""

import numpy as np
import pytest

from cmi.data import OPENML_TABLES, ROW_LIMIT, cap_rows, load_magic_telescope, load_openml_table, split_indices


@pytest.fixture(scope="module", params=sorted(OPENML_TABLES))
def table(request):
    return load_openml_table(request.param)


def test_table_is_ready_for_the_study(table):
    assert 0 < len(table.X) <= ROW_LIMIT
    assert len(table.X) == len(table.y)
    assert not np.isnan(table.X).any()
    assert not np.isnan(table.y).any()
    assert table.X.shape[1] == len(table.feature_names)
    assert set(table.categorical) | set(table.numeric) == set(range(table.X.shape[1]))
    assert np.all(table.X.std(axis=0) > 0)


def test_categories_are_integer_codes(table):
    for col in table.categorical:
        assert np.array_equal(table.X[:, col], np.round(table.X[:, col]))
        assert table.X[:, col].min() == 0


def test_classification_targets_have_two_classes(table):
    if table.task == "classification":
        assert set(np.unique(table.y)) == {0, 1}
        assert 0.05 < table.y.mean() < 0.95


def test_splits_are_reproducible_and_disjoint(table):
    train, test = split_indices(table.y, 3, task=table.task)
    again, _ = split_indices(table.y, 3, task=table.task)
    other, _ = split_indices(table.y, 4, task=table.task)
    assert np.array_equal(train, again)
    assert not np.array_equal(train, other)
    assert not set(train) & set(test)
    assert len(train) + len(test) == len(table.y)


def test_expected_task_and_excluded_columns():
    assert load_openml_table("wine-quality").task == "regression"
    bike = load_openml_table("bike-sharing")
    assert bike.task == "regression"
    assert not {"casual", "registered", "count"} & set(bike.feature_names)
    assert bike.categorical
    bank = load_openml_table("bank-marketing")
    assert bank.task == "classification"
    assert "V12" not in bank.feature_names
    assert 0.05 < bank.y.mean() < 0.2
    assert len(bank.X) == ROW_LIMIT
    shoppers = load_openml_table("online-shoppers")
    assert shoppers.task == "classification"
    assert 0.1 < shoppers.y.mean() < 0.2


def test_magic_telescope_is_unchanged():
    table = load_magic_telescope()
    assert table.X.shape == (19020, 10)
    assert table.task == "classification"
    assert table.categorical == ()


def test_row_cap_keeps_the_class_balance():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.1).astype(int)
    X = rng.normal(size=(50_000, 2))
    capped_X, capped_y = cap_rows(X, y, "classification")
    assert len(capped_X) == ROW_LIMIT
    assert capped_y.mean() == pytest.approx(y.mean(), abs=1e-3)
