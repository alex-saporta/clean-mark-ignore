"""Categorical columns and regression, on synthetic tables."""

import numpy as np
import pytest

from cmi.injection import error_rng, inject_random_outliers
from cmi.models import evaluate, fit_model, predict_scores
from cmi.repairs import category_repair_accuracy, detect, fit_repair, repair_error
from cmi.strategies import corrupt

N_TRAIN = 3000
CATEGORICAL = (3,)
NUMERIC = (0, 1, 2)
N_CATEGORIES = 4
EFFECT = np.array([-2.0, -0.5, 0.5, 2.0])


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 4))
    X[:, 2] = X[:, 0] + 0.2 * rng.normal(size=n)
    X[:, 3] = rng.choice(N_CATEGORIES, size=n, p=[0.1, 0.2, 0.3, 0.4])
    signal = X[:, 0] + X[:, 1] + EFFECT[X[:, 3].astype(int)]
    y_regression = signal + 0.3 * rng.normal(size=n)
    y_class = (y_regression > np.median(y_regression)).astype(int)
    return X, y_class, y_regression


def view(table, location="both", recall=0.75, fpr=0.05, rate=0.2):
    X = table[0]
    c = corrupt(X[:N_TRAIN], X[N_TRAIN:], rate, 0, "outlier_random", columns=NUMERIC)
    return detect(c, location, recall, fpr, 0, categorical=CATEGORICAL)


def applied(strategy, v):
    fitted = fit_repair(strategy, v)
    return fitted, fitted.apply(v.dirty_test, v.flagged_test, v.clean_test)


def test_numeric_errors_leave_categories_intact_but_false_alarms_reach_them(table):
    v = view(table)
    assert not v.errors_train[:, 3].any()
    assert np.array_equal(v.dirty_train[:, 3], v.clean_train[:, 3])
    assert v.flagged_train[:, 3].any()
    assert v.flagged_test[:, 3].any()


def test_targeting_all_columns_gives_the_same_masks_as_no_target():
    X = np.random.default_rng(1).normal(size=(500, 4))
    args = (0.2, X.min(axis=0), X.max(axis=0), X.std(axis=0))
    default = inject_random_outliers(X, *args, error_rng(0, 0.2, "train"))
    explicit = inject_random_outliers(X, *args, error_rng(0, 0.2, "train"), columns=(0, 1, 2, 3))
    assert np.array_equal(default[0], explicit[0])
    assert np.array_equal(default[1], explicit[1])


def test_s2_uses_the_mode_for_a_category(table):
    v = view(table)
    fitted, test = applied("S2", v)
    unflagged = v.dirty_train[~v.flagged_train[:, 3], 3]
    values, counts = np.unique(unflagged, return_counts=True)
    mode = values[np.argmax(counts)]
    assert (fitted.train[v.flagged_train[:, 3], 3] == mode).all()
    assert (test[v.flagged_test[:, 3], 3] == mode).all()


def test_s1c_uses_an_unused_category_code(table):
    v = view(table)
    fitted, test = applied("S1c", v)
    code = fitted.train[v.flagged_train[:, 3], 3]
    assert len(np.unique(code)) == 1
    assert code[0] not in set(v.clean_train[:, 3])
    assert (test[v.flagged_test[:, 3], 3] == code[0]).all()


@pytest.mark.parametrize("strategy", ["S3", "S3hgb"])
def test_s3_imputes_numbers_and_gives_categories_the_mode(table, strategy):
    v = view(table)
    fitted, test = applied(strategy, v)
    s2, s2_test = applied("S2", v)
    assert not np.isnan(fitted.train).any()
    assert not np.isnan(test).any()
    assert np.array_equal(fitted.train[:, 3], s2.train[:, 3])
    assert np.array_equal(test[:, 3], s2_test[:, 3])
    assert set(np.unique(test[:, 3])) <= set(range(N_CATEGORIES))
    numeric_flagged = v.flagged_test[:, list(NUMERIC)]
    assert not np.allclose(test[:, list(NUMERIC)][numeric_flagged], s2_test[:, list(NUMERIC)][numeric_flagged])


@pytest.mark.parametrize("strategy", ["S3", "S3hgb"])
def test_s3_does_not_use_categories_as_predictors(table, strategy):
    v = view(table)
    other = np.random.default_rng(8).permutation(v.dirty_train[:, 3])
    changed = view(table)
    changed.dirty_train[:, 3] = other
    first = fit_repair(strategy, v).train
    second = fit_repair(strategy, changed).train
    assert np.allclose(first[:, list(NUMERIC)], second[:, list(NUMERIC)])


def test_s3hgb_repairs_a_predictable_column_better_than_the_mean(table):
    v = view(table, recall=1.0, fpr=0.0)
    scale = v.clean_train.std(axis=0)
    cells = np.zeros_like(v.flagged_test)
    cells[:, 2] = v.flagged_test[:, 2]
    _, s2 = applied("S2", v)
    _, s3 = applied("S3hgb", v)
    assert repair_error(s3, v.clean_test, cells, scale) < repair_error(s2, v.clean_test, cells, scale)


def test_s3hgb_is_reproducible(table):
    v = view(table)
    assert np.array_equal(fit_repair("S3hgb", v).train, fit_repair("S3hgb", v).train)


def test_category_repair_accuracy(table):
    v = view(table)
    _, s5 = applied("S5", v)
    cells = np.zeros_like(v.flagged_test)
    cells[:, 3] = v.flagged_test[:, 3]
    assert category_repair_accuracy(s5, v.clean_test, cells) == 1.0
    assert np.isnan(category_repair_accuracy(s5, v.clean_test, np.zeros_like(cells)))


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
def test_oracle_wins_on_a_mixed_table(table, task, model_name):
    y = table[1] if task == "classification" else table[2]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    v = view(table, recall=1.0, fpr=0.0)
    scores = {}
    for strategy in ("S0", "S5"):
        fitted, test = applied(strategy, v)
        model = fit_model(model_name, task, CATEGORICAL, fitted.train, y_train[fitted.kept_rows])
        scores[strategy] = evaluate(y_test, predict_scores(model, test, task), task)["score"]
    assert scores["S5"] > scores["S0"]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("model_name", ["lightgbm", "logistic"])
@pytest.mark.parametrize("strategy", ["S0", "S1", "S1c", "S2", "S3", "S4", "S5", "CLEAN"])
def test_every_strategy_runs_on_a_mixed_table(table, task, model_name, strategy):
    y = table[1] if task == "classification" else table[2]
    v = view(table)
    fitted, test = applied(strategy, v)
    model = fit_model(model_name, task, CATEGORICAL, fitted.train, y[:N_TRAIN][fitted.kept_rows])
    predictions = predict_scores(model, test, task)
    assert np.isfinite(predictions).all()


def test_lightgbm_uses_the_column_as_a_category(table):
    v = view(table, recall=1.0, fpr=0.0)
    model = fit_model("lightgbm", "regression", CATEGORICAL, v.clean_train, table[2][:N_TRAIN])
    decision_types = set()

    def walk(node):
        if "split_feature" in node:
            if node["split_feature"] == 3:
                decision_types.add(node["decision_type"])
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in model.booster_.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    assert decision_types == {"=="}


def test_lightgbm_reads_an_unseen_category_like_a_missing_one(table):
    v = view(table, recall=1.0, fpr=0.0)
    model = fit_model("lightgbm", "regression", CATEGORICAL, v.clean_train, table[2][:N_TRAIN])
    unseen, missing = v.clean_test[:200].copy(), v.clean_test[:200].copy()
    unseen[:, 3] = 999.0
    missing[:, 3] = np.nan
    assert np.array_equal(model.predict(unseen), model.predict(missing))


def test_linear_model_ignores_a_category_unseen_at_fit(table):
    v = view(table, recall=1.0, fpr=0.0)
    model = fit_model("logistic", "regression", CATEGORICAL, v.clean_train, table[2][:N_TRAIN])
    unseen = v.clean_test[:200].copy()
    unseen[:, 3] = 999.0
    assert np.isfinite(model.predict(unseen)).all()


def test_evaluate_makes_higher_scores_better():
    y = np.array([0.0, 1.0, 2.0])
    good = evaluate(y, y + 0.1, "regression")
    bad = evaluate(y, y + 1.0, "regression")
    assert good["metric"] == "rmse"
    assert good["score"] > bad["score"]
    assert good["value"] == pytest.approx(0.1)
    labels = np.array([0, 1, 1, 0])
    assert evaluate(labels, np.array([0.1, 0.9, 0.8, 0.2]), "classification")["value"] == 1.0
