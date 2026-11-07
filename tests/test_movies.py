"""Movies, the table with real errors: format decoding, error mask, detector and oracle."""

import numpy as np
import pytest

from cmi.data import decode_count, decode_minutes, decode_reviews, decode_year, first_item, load_movies
from cmi.detector import detection_quality
from cmi.pilot import Cell, load_rows, real_error_cells, run_split
from cmi.real_detectors import DETECTORS as REAL_DETECTORS
from cmi.repairs import detect, detect_real
from cmi.strategies import Corrupted
from cmi.grid import Variant
from cmi.tuning import column_profile


@pytest.fixture(scope="module")
def movies():
    return load_movies()


def test_formats_are_decoded_the_same_way_for_both_versions():
    assert decode_year("2011") == 2011
    assert decode_year("2010 2011 2012") == 2011
    assert np.isnan(decode_year(""))
    assert decode_minutes("109 min") == 109
    assert decode_minutes("1 hr. 39 min.") == 99
    assert decode_minutes("2 hr.") == 120
    assert np.isnan(decode_minutes(""))
    assert decode_count("379,998") == 379998
    assert np.isnan(decode_count(""))
    assert decode_reviews("1,976 user,290 critic", "user") == 1976
    assert decode_reviews("165 user", "critic") == 0
    assert first_item("Horror,Mystery") == "Horror"
    assert first_item("") == "(none)"


def test_clean_version_is_complete_and_errors_sit_where_versions_differ(movies):
    assert movies.X.shape == (7032, 7)
    assert movies.task == "regression"
    assert not np.isnan(movies.X).any()
    counts = dict(zip(movies.feature_names, movies.error_mask.sum(axis=0)))
    assert counts == {
        "year": 22,
        "duration": 135,
        "rating_count": 0,
        "user_reviews": 0,
        "critic_reviews": 0,
        "genre": 118,
        "country": 0,
    }
    assert np.isnan(movies.X_dirty).sum() == 80
    assert movies.error_columns == (0, 1, 5)
    unchanged = ~movies.error_mask
    assert np.array_equal(movies.X_dirty[unchanged], movies.X[unchanged])


def test_simulated_detector_reaches_recall_and_fpr_on_real_errors(movies):
    mask = movies.error_mask
    corrupted = Corrupted(movies.X, movies.X, movies.X_dirty, movies.X_dirty, mask, mask)
    view = detect(corrupted, "both", 0.75, 0.05, 0, movies.categorical)
    quality = detection_quality(view.errors_train, view.flagged_train)
    assert quality.recall == pytest.approx(0.75, abs=0.01)
    assert quality.fpr == pytest.approx(0.05, abs=0.001)


def test_real_detectors_flag_every_empty_cell(movies):
    mask = movies.error_mask
    corrupted = Corrupted(movies.X, movies.X, movies.X_dirty, movies.X_dirty, mask, mask)
    empty = np.isnan(movies.X_dirty)
    for name in REAL_DETECTORS:
        view = detect_real(corrupted, "both", name, movies.numeric, movies.categorical)
        assert view.flagged_train[empty].all()
        assert view.flagged_test[empty].all()


def test_grid_has_ten_simulated_and_four_real_detectors():
    cells = real_error_cells()
    assert sum(c.block == "map" for c in cells) == 10
    assert {c.variant.detector for c in cells if c.block == "real"} == set(REAL_DETECTORS)


def test_oracle_matches_clean_at_the_perfect_detector(movies, tmp_path):
    # The oracle is not required to beat S0 here: with 1.3 % of wrong cells, it lost
    # on 2 of 3 splits in a first run. That is a result, not a bug. Its gain on
    # injected errors is tested in test_strategies.py, test_repairs.py and
    # test_error_types.py.
    params = {"n_estimators": 50, "num_leaves": 15}
    tuned = {
        "params": {"lightgbm": params, "logistic": {}},
        "feature_names": list(movies.feature_names),
        "profile": column_profile(movies, movies.X, movies.y, params),
    }
    cells = [
        Cell("map", Variant(error="real"), "all", 0.0, "both", 1.0, 0.0, ("S0", "S1", "S5", "CLEAN")),
        Cell("real", Variant(error="real", detector="iqr"), "all", 0.0, "both", None, None, ("S0", "S1")),
    ]
    for split in range(3):
        run_split(movies, split, cells, tmp_path, tuned)
    rows = load_rows(tmp_path)
    perfect = rows[rows["block"] == "map"].pivot_table(index=["split", "model"], columns="strategy", values="value")
    assert np.allclose(perfect["S5"], perfect["CLEAN"])
    assert rows["value"].notna().all()
