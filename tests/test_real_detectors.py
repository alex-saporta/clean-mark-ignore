import numpy as np
import pytest

from cmi.detector import detection_quality
from cmi.real_detectors import DETECTORS, fit_detector
from cmi.repairs import detect_real, fit_repair
from cmi.strategies import corrupt

NUMERIC = (0, 1, 2)
CATEGORICAL = (3,)
N_TRAIN = 3000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    n = 4000
    X = rng.normal(size=(n, 4))
    X[:, 3] = rng.choice(4, size=n, p=[0.1, 0.2, 0.3, 0.4])
    return X[:N_TRAIN], X[N_TRAIN:]


def outlier_recall(data, name, rate):
    X_train, X_test = data
    c = corrupt(X_train, X_test, rate, 0, "outlier_random", NUMERIC)
    detector = fit_detector(name, c.dirty_train, NUMERIC, CATEGORICAL)
    return detection_quality(c.test_mask, detector.flag(c.dirty_test))


@pytest.mark.parametrize("name", ["iqr", "iforest"])
@pytest.mark.parametrize("rate", [0.05, 0.20])
def test_robust_detectors_find_far_outliers(data, name, rate):
    quality = outlier_recall(data, name, rate)
    assert quality.recall > 0.9
    assert quality.precision > 0.3


@pytest.mark.parametrize("rate", [0.05, 0.20])
def test_rules_trade_recall_for_fewer_false_alarms_than_iqr(data, rate):
    rules = outlier_recall(data, "rules", rate)
    iqr = outlier_recall(data, "iqr", rate)
    assert 0.5 < rules.recall <= iqr.recall
    assert rules.fpr <= iqr.fpr


def test_errors_mask_themselves_from_the_standard_deviation(data):
    """The outliers inflate the standard deviation that should reveal them."""
    assert outlier_recall(data, "sd", 0.01).recall > 0.9
    assert outlier_recall(data, "sd", 0.20).recall < 0.2


@pytest.mark.parametrize("name", DETECTORS)
def test_fit_is_reproducible_and_ignores_the_test_part(data, name):
    X_train, X_test = data
    first = fit_detector(name, X_train, NUMERIC, CATEGORICAL)
    second = fit_detector(name, X_train, NUMERIC, CATEGORICAL)
    assert np.array_equal(first.flag(X_test), second.flag(X_test))
    assert np.array_equal(first.flag(X_train), second.flag(X_train))


@pytest.mark.parametrize("name", DETECTORS)
def test_flags_keep_the_row_order(data, name):
    X_train, X_test = data
    detector = fit_detector(name, X_train, NUMERIC, CATEGORICAL)
    order = np.random.default_rng(0).permutation(len(X_test))
    assert np.array_equal(detector.flag(X_test[order]), detector.flag(X_test)[order])


@pytest.mark.parametrize("name", ["sd", "iqr", "iforest"])
def test_numeric_detectors_ignore_categories(data, name):
    X_train, X_test = data
    detector = fit_detector(name, X_train, NUMERIC, CATEGORICAL)
    assert not detector.flag(X_test)[:, 3].any()


def test_rules_catch_rare_typos_but_not_frequent_ones_or_replaced_categories(data):
    X_train, X_test = data

    def typo_recall(rate):
        typos = corrupt(X_train, X_test, rate, 0, "typo", CATEGORICAL)
        detector = fit_detector("rules", typos.dirty_train, NUMERIC, CATEGORICAL)
        return detection_quality(typos.test_mask, detector.flag(typos.dirty_test)).recall

    assert typo_recall(0.01) > 0.9
    assert typo_recall(0.20) < 0.2
    swapped = corrupt(X_train, X_test, 0.2, 0, "category_swap", CATEGORICAL)
    detector = fit_detector("rules", swapped.dirty_train, NUMERIC, CATEGORICAL)
    assert detection_quality(swapped.test_mask, detector.flag(swapped.dirty_test)).recall == 0.0


def test_in_domain_errors_are_almost_invisible_to_range_detectors(data):
    X_train, X_test = data
    c = corrupt(X_train, X_test, 0.2, 0, "in_domain", NUMERIC)
    detector = fit_detector("iqr", c.dirty_train, NUMERIC, CATEGORICAL)
    assert detection_quality(c.test_mask, detector.flag(c.dirty_test)).recall < 0.05


def test_iqr_flags_every_nonzero_value_of_a_column_full_of_zeros():
    """Its quartiles are both zero, so the signal itself looks abnormal."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 2))
    X[:, 1] = np.where(rng.random(1000) < 0.8, 0.0, rng.exponential(size=1000))
    flagged = fit_detector("iqr", X, (0, 1), ()).flag(X)
    assert flagged[X[:, 1] != 0, 1].all()
    assert not flagged[X[:, 1] == 0, 1].any()


def test_detect_real_switches_off_a_detector_flagging_most_of_a_column():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    X[:, 2] = rng.integers(0, 24, size=2000).astype(float)
    assert fit_detector("iforest", X, (0, 1, 2), ()).flag(X)[:, 2].mean() > 0.5
    c = corrupt(X[:1500], X[1500:], 0.05, 0, "outlier_random", (0, 1, 2))
    clean_view = detect_real(c, "test", "iforest", (0, 1, 2), ())
    assert not clean_view.flagged_train[:, 2].any()
    assert not clean_view.flagged_test[:, 2].any()
    assert clean_view.flagged_train[:, 0].any()


def test_repair_refuses_an_entirely_flagged_column(data):
    X_train, X_test = data
    c = corrupt(X_train, X_test, 0.05, 0, "outlier_random", NUMERIC)
    v = detect_real(c, "both", "iqr", NUMERIC, CATEGORICAL)
    v.flagged_train[:, 0] = True
    with pytest.raises(ValueError):
        fit_repair("S2", v)


def test_detect_real_uses_only_training_data_for_the_fit(data):
    X_train, X_test = data
    c = corrupt(X_train, X_test, 0.2, 0, "outlier_random", NUMERIC)
    first = detect_real(c, "both", "iqr", NUMERIC, CATEGORICAL)
    other = corrupt(X_train, X_test + 100.0, 0.2, 0, "outlier_random", NUMERIC)
    second = detect_real(other, "both", "iqr", NUMERIC, CATEGORICAL)
    assert np.array_equal(first.flagged_train, second.flagged_train)


def test_detect_real_scans_a_part_without_errors(data):
    X_train, X_test = data
    c = corrupt(X_train, X_test, 0.2, 0, "outlier_random", NUMERIC)
    v = detect_real(c, "train", "iqr", NUMERIC, CATEGORICAL)
    assert not v.errors_test.any()
    assert np.array_equal(v.dirty_test, X_test)
    assert np.array_equal(v.flagged_test, fit_detector("iqr", v.dirty_train, NUMERIC, CATEGORICAL).flag(X_test))
