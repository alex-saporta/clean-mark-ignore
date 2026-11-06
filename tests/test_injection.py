import numpy as np
import pytest

from cmi.injection import (
    OUTLIER_MAGNITUDE,
    error_rng,
    inject_outliers,
    inject_random_outliers,
    mark_missing,
)
from cmi.strategies import corrupt


@pytest.fixture
def X():
    return np.random.default_rng(1).normal(size=(2000, 5))


@pytest.mark.parametrize("rate", [0.05, 0.20])
def test_each_column_has_the_requested_rate(X, rate):
    _, mask = inject_outliers(X, rate, X.std(axis=0), error_rng(0, rate, "train"))
    assert (mask.sum(axis=0) == round(rate * len(X))).all()


def test_only_flagged_cells_change_and_by_the_magnitude(X):
    scale = X.std(axis=0)
    dirty, mask = inject_outliers(X, 0.2, scale, error_rng(0, 0.2, "train"))
    assert np.array_equal(dirty[~mask], X[~mask])
    shift = np.abs(dirty - X) / scale
    assert np.allclose(shift[mask], OUTLIER_MAGNITUDE)


def test_input_is_not_modified(X):
    before = X.copy()
    inject_outliers(X, 0.2, X.std(axis=0), error_rng(0, 0.2, "train"))
    assert np.array_equal(X, before)


def test_masks_are_reproducible_and_differ_across_configurations(X):
    scale = X.std(axis=0)
    _, first = inject_outliers(X, 0.2, scale, error_rng(3, 0.2, "train"))
    _, again = inject_outliers(X, 0.2, scale, error_rng(3, 0.2, "train"))
    _, other_split = inject_outliers(X, 0.2, scale, error_rng(4, 0.2, "train"))
    _, other_part = inject_outliers(X, 0.2, scale, error_rng(3, 0.2, "test"))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_split)
    assert not np.array_equal(first, other_part)


def test_test_errors_use_the_training_scale():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(1000, 2))
    X_test = 50.0 * rng.normal(size=(200, 2))
    c = corrupt(X_train, X_test, 0.2, split=0)
    shift = np.abs(c.dirty_test - X_test)[c.test_mask.nonzero()]
    expected = np.repeat(X_train.std(axis=0), c.test_mask.sum(axis=0)) * OUTLIER_MAGNITUDE
    order = np.argsort(c.test_mask.nonzero()[1], kind="stable")
    assert np.allclose(shift[order], expected)


def random_outliers(X, part="train"):
    return inject_random_outliers(
        X, 0.2, X.min(axis=0), X.max(axis=0), X.std(axis=0), error_rng(0, 0.2, part)
    )


def test_both_errors_use_the_same_cells_and_signs(X):
    order_keeping, mask = inject_outliers(X, 0.2, X.std(axis=0), error_rng(0, 0.2, "train"))
    random, random_mask = random_outliers(X)
    assert np.array_equal(mask, random_mask)
    assert np.array_equal(np.sign(order_keeping - X), np.sign(random - X))


def test_random_outliers_are_out_of_domain_and_leave_other_cells(X):
    dirty, mask = random_outliers(X)
    assert np.array_equal(dirty[~mask], X[~mask])
    assert np.all((dirty > X.max(axis=0)) | (dirty < X.min(axis=0)) | ~mask)


def test_random_outliers_do_not_depend_on_the_true_value(X):
    dirty, mask = random_outliers(X)
    other = X + 0.5 * np.random.default_rng(9).normal(size=X.shape)
    dirty_other, mask_other = inject_random_outliers(
        other, 0.2, X.min(axis=0), X.max(axis=0), X.std(axis=0), error_rng(0, 0.2, "train")
    )
    assert np.array_equal(mask, mask_other)
    assert np.array_equal(dirty[mask], dirty_other[mask])


def test_mark_missing_marks_exactly_the_flagged_cells(X):
    _, mask = inject_outliers(X, 0.2, X.std(axis=0), error_rng(0, 0.2, "train"))
    assert np.array_equal(np.isnan(mark_missing(X, mask)), mask)
