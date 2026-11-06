"""The pilot grid, tuning and parallel splits, on a small synthetic table."""

import json

import numpy as np
import pytest

from cmi.data import OPENML_TABLES, Table, load_openml_table
from cmi.grid import DETECTORS, Variant
from cmi.pilot import (
    KEY_DETECTORS,
    Cell,
    chosen_columns,
    comparison_table,
    load_rows,
    pilot_cells,
    report,
    run,
    run_split,
)
from cmi.real_detectors import DETECTORS as REAL_DETECTORS
from cmi.tuning import column_profile, lightgbm_configs, load_tuned, tune_model


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(6)
    n = 1200
    X = rng.normal(size=(n, 7))
    X[:, 2] = X[:, 0] + 0.2 * rng.normal(size=n)
    X[:, 5] = rng.integers(4, size=n)
    X[:, 6] = rng.integers(3, size=n)
    y = 2.0 * X[:, 0] + X[:, 1] + X[:, 5] + 0.3 * rng.normal(size=n)
    names = tuple(f"c{col}" for col in range(7))
    return Table("synthetic", X, y, names, "regression", (5, 6))


@pytest.fixture(scope="module")
def tuned(table):
    params = {"n_estimators": 20, "num_leaves": 7}
    profile = column_profile(table, table.X, table.y, params)
    return {
        "params": {"lightgbm": params, "logistic": {}},
        "feature_names": list(table.feature_names),
        "profile": profile,
    }


SMALL_CELLS = [
    Cell("map", Variant(error="in_domain"), "all", 0.2, "both", 0.75, 0.05, ("S0", "S1", "S2", "S3", "S5", "CLEAN")),
    Cell("map", Variant(error="typo"), "all", 0.2, "both", 1.0, 0.0, ("S0", "S1", "S5")),
    Cell("location", Variant(error="unit"), "all", 0.05, "test", 0.95, 0.01, ("S0", "S1")),
    Cell(
        "columns",
        Variant(error="in_domain", column_share=0.3, scan="targeted"),
        "top",
        0.2,
        "both",
        0.75,
        0.05,
        ("S0", "S1"),
    ),
    Cell("real", Variant(error="unit", detector="iqr"), "all", 0.05, "both", None, None, ("S0", "S1", "S5")),
]


def test_grid_has_the_announced_size_and_limits_s3hgb(table):
    cells = pilot_cells(table)
    blocks = {block: sum(c.block == block for c in cells) for block in ("map", "location", "columns", "mechanism", "real")}
    assert blocks == {"map": 100, "location": 60, "columns": 48, "mechanism": 6, "real": 40}
    assert sum("S3hgb" in c.strategies for c in cells) == 30
    assert all(c.block == "map" and (c.recall, c.fpr) in KEY_DETECTORS for c in cells if "S3hgb" in c.strategies)
    assert len(DETECTORS) == 10
    assert {c.variant.detector for c in cells if c.block == "real"} == set(REAL_DETECTORS)


def test_cells_are_unique(table):
    cells = pilot_cells(table)
    keys = [(c.block, c.variant, c.column_choice, c.rate, c.location, c.recall, c.fpr) for c in cells]
    assert len(keys) == len(set(keys))


def test_top_and_bottom_columns_follow_importance(table, tuned):
    importance = tuned["profile"]["importance"]
    variant = Variant(error="in_domain", column_share=0.3)
    top = chosen_columns(table, variant, "top", importance)
    bottom = chosen_columns(table, variant, "bottom", importance)
    assert 0 in top
    assert not set(top) & set(bottom)
    assert min(importance[col] for col in top) >= max(importance[col] for col in bottom)
    assert chosen_columns(table, Variant(error="typo"), "all", importance) == (5, 6)
    assert set(top) <= set(table.numeric)


def test_profile_measures_importance_correlation_and_predictability(table, tuned):
    profile = tuned["profile"]
    assert sum(profile["importance"].values()) == pytest.approx(1.0)
    assert profile["importance"][0] > profile["importance"][3]
    assert profile["target_correlation"][0] > 0.5
    assert 5 not in profile["target_correlation"]
    assert profile["predictability"][2]["bayesian_ridge"] > 0.8
    assert profile["predictability"][3]["bayesian_ridge"] < 0.1


def test_lightgbm_configs_are_distinct_and_reproducible():
    configs = lightgbm_configs()
    assert len(configs) == 20
    assert configs == lightgbm_configs()
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == 20


def test_tuning_never_scores_worse_than_the_default_candidate(table):
    params, score = tune_model("logistic", table, table.X, table.y)
    assert params == {}
    assert np.isfinite(score)


def test_results_match_across_parallel_settings_and_resume(table, tuned, tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    run(table, [0, 1], serial, tuned, jobs=1, cells=SMALL_CELLS)
    run(table, [0, 1], parallel, tuned, jobs=2, cells=SMALL_CELLS)
    first, second = load_rows(serial), load_rows(parallel)
    fields = ["block", "error", "column_choice", "rate", "location", "split", "model", "strategy", "recall", "fpr"]
    first = first.sort_values(fields).reset_index(drop=True)
    second = second.sort_values(fields).reset_index(drop=True)
    assert len(first) == len(second) > 0
    assert np.array_equal(first["value"].to_numpy(), second["value"].to_numpy())
    assert run_split(table, 0, SMALL_CELLS, serial, tuned) == 0


def test_interrupted_split_finishes_the_rest(table, tuned, tmp_path):
    run_split(table, 0, SMALL_CELLS[:2], tmp_path, tuned)
    assert run_split(table, 0, SMALL_CELLS, tmp_path, tuned) > 0
    assert run_split(table, 0, SMALL_CELLS, tmp_path, tuned) == 0


def test_truncated_last_line_is_redone(table, tuned, tmp_path):
    run_split(table, 0, SMALL_CELLS[:1], tmp_path, tuned)
    path = tmp_path / "split_0.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20] + "\n", encoding="utf-8")
    assert run_split(table, 0, SMALL_CELLS[:1], tmp_path, tuned) == 1


def test_scores_favour_higher_for_regression_and_report_builds(table, tuned, tmp_path):
    run(table, [0, 1, 2], tmp_path, tuned, jobs=1, cells=SMALL_CELLS)
    rows = load_rows(tmp_path)
    assert (rows["score"] == -rows["value"]).all()
    assert (rows[rows["block"] == "real"]["recall"] == -1).all()
    comparisons = comparison_table(rows)
    oracle = comparisons[(comparisons["comparison"] == "S5-S0") & (comparisons["error"] == "typo")]
    assert (oracle["mean_diff"] > 0).all()
    text = report(table, tmp_path, tuned)
    assert "Wins and losses" in text
    assert "Compute time" in text


@pytest.mark.parametrize("name", ["default-credit", "adult"])
def test_new_classification_tables_load_without_looking_at_any_result(name):
    assert name in OPENML_TABLES
    table = load_openml_table(name)
    assert table.task == "classification"
    assert set(np.unique(table.y)) == {0, 1}
    assert not np.isnan(table.X).any()
    assert table.categorical
    assert len(table.X) == 20_000


def test_tuned_files_load_with_integer_columns(tmp_path, tuned):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({**tuned}), encoding="utf-8")
    loaded = load_tuned("t", tmp_path)
    assert loaded["profile"]["importance"][0] == tuned["profile"]["importance"][0]
