"""What LightGBM and XGBoost do with a NaN unseen at training, on a numeric column."""

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

HUGE = 1e30


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    X[:, 1] += 100.0  # Every threshold is positive: 0 lies below the observed range.
    y = ((X[:, 0] + (X[:, 1] - 100.0) + 0.5 * X[:, 2]) > 0).astype(int)
    return X, y


def lightgbm_splits(node: dict):
    if "split_feature" in node:
        yield node
        yield from lightgbm_splits(node["left_child"])
        yield from lightgbm_splits(node["right_child"])


def fit_lightgbm(X, y):
    return lgb.LGBMClassifier(n_estimators=50, random_state=0, verbose=-1, n_jobs=1).fit(X, y)


def test_lightgbm_reads_unseen_nan_as_zero(data):
    X, y = data
    booster = fit_lightgbm(X, y).booster_
    trees = booster.dump_model()["tree_info"]
    types = {s["missing_type"] for t in trees for s in lightgbm_splits(t["tree_structure"])}
    assert types == {"None"}

    with_nan = X[:500].copy()
    with_nan[:, 1] = np.nan
    with_zero = X[:500].copy()
    with_zero[:, 1] = 0.0
    assert np.array_equal(booster.predict(with_nan), booster.predict(with_zero))


def test_lightgbm_learns_a_missing_type_when_training_has_nan(data):
    X, y = data
    X = X.copy()
    X[::10, 1] = np.nan
    booster = fit_lightgbm(X, y).booster_
    trees = booster.dump_model()["tree_info"]
    types = {
        s["split_feature"]: s["missing_type"]
        for t in trees
        for s in lightgbm_splits(t["tree_structure"])
    }
    assert types[1] == "NaN"
    assert types[0] == "None"


def test_xgboost_sends_unseen_nan_right(data):
    X, y = data
    model = xgb.XGBClassifier(n_estimators=50, tree_method="hist", random_state=0, n_jobs=1).fit(X, y)
    booster = model.get_booster()
    nodes = booster.trees_to_dataframe()
    splits = nodes[nodes["Feature"] != "Leaf"]
    assert (splits["Missing"] == splits["No"]).all(), "A split has default_left=true"

    with_nan = X[:500].copy()
    with_nan[:, 1] = np.nan
    with_huge = X[:500].copy()
    with_huge[:, 1] = HUGE
    assert np.array_equal(model.predict_proba(with_nan), model.predict_proba(with_huge))
