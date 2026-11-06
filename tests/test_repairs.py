import lightgbm as lgb
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cmi.models import make_model
from cmi.repairs import STRATEGIES, detect, fit_repair, repair_error
from cmi.strategies import corrupt

N_TRAIN = 3000
ALL = STRATEGIES


@pytest.fixture(scope="module")
def data():
    """Column 0 and 1 predict the target; column 2 is a noisy copy of column 0."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 4))
    X[:, 2] = X[:, 0] + 0.2 * rng.normal(size=4000)
    y = ((X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=4000)) > 0).astype(int)
    return X[:N_TRAIN], X[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:]


def view(data, location="both", recall=0.75, fpr=0.05, rate=0.2):
    X_train, X_test, _, _ = data
    c = corrupt(X_train, X_test, rate, split=0, error="outlier_random")
    return detect(c, location, recall, fpr, split=0)


def applied(strategy, v):
    fitted = fit_repair(strategy, v)
    return fitted, fitted.apply(v.dirty_test, v.flagged_test, v.clean_test)


def test_detector_runs_on_a_part_without_errors(data):
    v = view(data, location="train", recall=0.75, fpr=0.05)
    assert not v.errors_test.any()
    assert v.flagged_test.any()
    assert np.array_equal(v.dirty_test, v.clean_test)


@pytest.mark.parametrize("strategy", ["S1c", "S2", "S3", "S5"])
def test_only_flagged_cells_change(data, strategy):
    v = view(data)
    fitted, test = applied(strategy, v)
    assert np.array_equal(fitted.train[~v.flagged_train], v.dirty_train[~v.flagged_train])
    assert np.allclose(test[~v.flagged_test], v.dirty_test[~v.flagged_test])


def test_s0_keeps_the_dirty_data(data):
    v = view(data)
    fitted, test = applied("S0", v)
    assert np.array_equal(fitted.train, v.dirty_train)
    assert np.array_equal(test, v.dirty_test)


def test_s1_marks_exactly_the_flagged_cells(data):
    v = view(data)
    fitted, test = applied("S1", v)
    assert np.array_equal(np.isnan(fitted.train), v.flagged_train)
    assert np.array_equal(np.isnan(test), v.flagged_test)


def test_s1c_uses_one_constant_outside_the_unflagged_training_range(data):
    v = view(data)
    fitted, test = applied("S1c", v)
    unflagged = np.where(v.flagged_train, np.nan, v.dirty_train)
    for col in range(v.dirty_train.shape[1]):
        values = np.unique(np.concatenate([fitted.train[v.flagged_train[:, col], col], test[v.flagged_test[:, col], col]]))
        assert len(values) == 1
        assert values[0] < np.nanmin(unflagged[:, col])


def test_s2_uses_the_mean_of_unflagged_training_cells(data):
    v = view(data)
    fitted, test = applied("S2", v)
    means = np.nanmean(np.where(v.flagged_train, np.nan, v.dirty_train), axis=0)
    rows, cols = np.nonzero(v.flagged_test)
    assert np.allclose(test[rows, cols], means[cols])
    rows, cols = np.nonzero(v.flagged_train)
    assert np.allclose(fitted.train[rows, cols], means[cols])


@pytest.mark.parametrize("strategy", ["S3", "S3hgb"])
def test_s3_leaves_no_missing_value(data, strategy):
    fitted, test = applied(strategy, view(data))
    assert not np.isnan(fitted.train).any()
    assert not np.isnan(test).any()


def test_s3_handles_test_flags_in_a_column_without_training_flags(data):
    v = view(data, location="test", recall=1.0, fpr=0.0)
    assert not v.flagged_train.any()
    _, test = applied("S3", v)
    assert not np.isnan(test).any()


def test_s3_beats_s2_on_a_predictable_column(data):
    v = view(data, recall=1.0, fpr=0.0)
    scale = v.clean_train.std(axis=0)
    cells = np.zeros_like(v.flagged_test)
    cells[:, 2] = v.flagged_test[:, 2]
    _, s2 = applied("S2", v)
    _, s3 = applied("S3", v)
    assert repair_error(s3, v.clean_test, cells, scale) < repair_error(s2, v.clean_test, cells, scale)


def test_s4_drops_flagged_training_rows_and_leaves_the_test_part(data):
    v = view(data)
    fitted, test = applied("S4", v)
    assert np.array_equal(fitted.kept_rows, ~v.flagged_train.any(axis=1))
    assert np.array_equal(fitted.train, v.dirty_train[fitted.kept_rows])
    assert np.array_equal(test, v.dirty_test)


def test_s5_restores_flagged_cells_and_leaves_missed_errors(data):
    v = view(data, recall=0.5, fpr=0.05)
    fitted, test = applied("S5", v)
    assert np.array_equal(test[v.flagged_test], v.clean_test[v.flagged_test])
    missed = v.errors_test & ~v.flagged_test
    assert missed.any()
    assert np.array_equal(test[missed], v.dirty_test[missed])
    assert np.array_equal(fitted.train[v.flagged_train], v.clean_train[v.flagged_train])


def test_s5_with_a_perfect_detector_equals_the_clean_data(data):
    v = view(data, recall=1.0, fpr=0.0)
    s5, s5_test = applied("S5", v)
    clean, clean_test = applied("CLEAN", v)
    assert np.array_equal(s5.train, clean.train)
    assert np.array_equal(s5_test, clean_test)


@pytest.mark.parametrize("strategy", ALL)
def test_fit_ignores_the_test_part(data, strategy):
    v = view(data)
    other = np.random.default_rng(11).normal(size=v.clean_test.shape)
    changed = view((v.clean_train, other, None, None))
    if strategy in ("S0", "S4", "CLEAN"):
        assert np.array_equal(fit_repair(strategy, v).train, fit_repair(strategy, changed).train)
    else:
        first, second = fit_repair(strategy, v), fit_repair(strategy, changed)
        assert np.array_equal(first.train, second.train, equal_nan=True)


@pytest.mark.parametrize("strategy", ["S1c", "S2"])
def test_test_cells_do_not_influence_the_fitted_values(data, strategy):
    v = view(data)
    fitted = fit_repair(strategy, v)
    shifted = v.dirty_test + 1000.0
    first = fitted.apply(v.dirty_test, v.flagged_test, v.clean_test)
    second = fitted.apply(shifted, v.flagged_test, v.clean_test)
    assert np.array_equal(first[v.flagged_test], second[v.flagged_test])


@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
def test_oracle_wins_with_a_perfect_detector(data, model_name):
    _, _, y_train, y_test = data
    v = view(data, recall=1.0, fpr=0.0)
    aurocs = {}
    for strategy in ("S0", "S5"):
        fitted, test = applied(strategy, v)
        model = make_model(model_name).fit(fitted.train, y_train[fitted.kept_rows])
        aurocs[strategy] = roc_auc_score(y_test, model.predict_proba(test)[:, 1])
    assert aurocs["S5"] > aurocs["S0"]


def test_every_strategy_fits_lightgbm_and_logistic(data):
    _, _, y_train, y_test = data
    v = view(data)
    for model_name in ("lightgbm", "logistic"):
        for strategy in STRATEGIES:
            fitted, test = applied(strategy, v)
            model = make_model(model_name).fit(fitted.train, y_train[fitted.kept_rows])
            assert 0.5 < roc_auc_score(y_test, model.predict_proba(test)[:, 1]) <= 1.0


def test_one_false_alarm_gives_lightgbm_a_missing_direction(data):
    """With training NaN, a test NaN no longer reads as 0, unlike with no false alarm."""
    _, _, y_train, _ = data
    v = view(data, location="test", recall=1.0, fpr=0.01)
    fitted, test = applied("S1", v)
    assert np.isnan(fitted.train).any()
    model = make_model("lightgbm").fit(fitted.train, y_train)
    types = set()

    def walk(node):
        if "split_feature" in node:
            types.add(node["missing_type"])
            walk(node["left_child"])
            walk(node["right_child"])

    assert isinstance(model.booster_, lgb.Booster)
    for tree in model.booster_.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    assert "NaN" in types
    zeroed = np.nan_to_num(test, nan=0.0)
    assert not np.array_equal(model.predict_proba(test), model.predict_proba(zeroed))
