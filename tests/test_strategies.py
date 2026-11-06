import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cmi.models import make_model
from cmi.strategies import LOCATIONS, corrupt, eval_matrix, train_matrix, train_variant


@pytest.fixture(scope="module")
def split_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 4))
    y = ((X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=4000)) > 0).astype(int)
    return X[:3000], X[3000:], y[:3000], y[3000:]


def scores(model_name, X_train, y_train, X_test):
    return make_model(model_name).fit(X_train, y_train).predict_proba(X_test)[:, 1]


def view(c, strategy, location):
    return train_matrix(c, train_variant(strategy, location)), eval_matrix(c, strategy, location)


@pytest.mark.parametrize("location", LOCATIONS)
def test_s5_recovers_the_clean_data(split_data, location):
    X_train, X_test, _, _ = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)
    train, test = view(c, "S5", location)
    assert np.array_equal(train, X_train)
    assert np.array_equal(test, X_test)


@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
def test_s5_predictions_equal_a_model_trained_on_clean_data(split_data, model_name):
    X_train, X_test, y_train, _ = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)
    train, test = view(c, "S5", "both")
    assert np.array_equal(
        scores(model_name, train, y_train, test), scores(model_name, X_train, y_train, X_test)
    )


@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
def test_oracle_beats_doing_nothing_when_errors_are_everywhere(split_data, model_name):
    X_train, X_test, y_train, y_test = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)
    aurocs = {}
    for strategy in ("S0", "S5"):
        train, test = view(c, strategy, "both")
        aurocs[strategy] = roc_auc_score(y_test, scores(model_name, train, y_train, test))
    assert aurocs["S5"] > aurocs["S0"]


def test_each_location_touches_only_its_part(split_data):
    X_train, X_test, _, _ = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)

    train, test = view(c, "S0", "train")
    assert np.array_equal(train, c.dirty_train) and np.array_equal(test, X_test)

    train, test = view(c, "S0", "test")
    assert np.array_equal(train, X_train) and np.array_equal(test, c.dirty_test)

    train, test = view(c, "S1", "both")
    assert np.array_equal(np.isnan(train), c.train_mask)
    assert np.array_equal(np.isnan(test), c.test_mask)
    assert np.array_equal(train[~c.train_mask], X_train[~c.train_mask])
    assert np.array_equal(test[~c.test_mask], X_test[~c.test_mask])


def test_locations_share_the_same_masks(split_data):
    X_train, X_test, _, _ = split_data
    first = corrupt(X_train, X_test, 0.2, split=5)
    second = corrupt(X_train, X_test, 0.2, split=5)
    assert np.array_equal(first.train_mask, second.train_mask)
    assert np.array_equal(first.test_mask, second.test_mask)


def test_s1_test_only_lightgbm_equals_replacing_flagged_cells_by_zero(split_data):
    X_train, X_test, y_train, _ = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)
    train, marked = view(c, "S1", "test")
    zeroed = np.where(c.test_mask, 0.0, X_test)
    assert np.array_equal(
        scores("lightgbm", train, y_train, marked), scores("lightgbm", train, y_train, zeroed)
    )


def test_s1_test_only_logistic_equals_mean_imputation(split_data):
    X_train, X_test, y_train, _ = split_data
    c = corrupt(X_train, X_test, 0.2, split=0)
    train, marked = view(c, "S1", "test")
    means = np.broadcast_to(X_train.mean(axis=0), X_test.shape)
    imputed = np.where(c.test_mask, means, X_test)
    assert np.allclose(
        scores("logistic", train, y_train, marked), scores("logistic", train, y_train, imputed)
    )
