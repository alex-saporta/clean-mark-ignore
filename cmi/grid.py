"""Strategies against a simulated detector, on any table. The grid is provisional.

    uv run python -m cmi.grid run --table bike-sharing [--splits 10] [--smoke]
        [--error typo] [--mechanism dependent] [--column-share 0.3] [--scan targeted]
    uv run python -m cmi.grid report --table bike-sharing [--smoke]

The detector scans every column by default, so false alarms reach columns without
errors. `--scan targeted` limits it to the columns holding errors. Classification
and regression are separate tables by construction. Finished evaluations are
skipped on a later run.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from cmi.data import OPENML_TABLES, Table, load_magic_telescope, load_openml_table, split_indices
from cmi.detector import detection_quality
from cmi.injection import MECHANISMS, pick_columns
from cmi.models import MODEL_NAMES, evaluate, fit_model, predict_scores
from cmi.real_detectors import DETECTORS as REAL_DETECTORS
from cmi.repairs import (
    REPAIRING,
    STRATEGIES,
    category_repair_accuracy,
    detect,
    detect_real,
    fit_repair,
    repair_error,
)
from cmi.strategies import LOCATIONS, corrupt
from cmi.week1 import read_rows
from cmi.week2 import training_key

NUMERIC_ERRORS = ("outlier", "outlier_random", "unit", "in_domain")
CATEGORY_ERRORS = ("category_swap", "typo")
UNIT_MAX_ZERO_SHARE = 0.05
RATES = (0.05, 0.20)
DETECTORS = ((1.0, 0.0), *product((0.5, 0.75, 0.95), (0.01, 0.05, 0.20)))
SMOKE_LOCATIONS = ("both",)
SMOKE_DETECTORS = ((1.0, 0.0), (0.95, 0.01), (0.75, 0.05))
RESULTS = Path(__file__).resolve().parent.parent / "results"
VARIANT_DEFAULTS = {
    "error": "outlier_random",
    "mechanism": "uniform",
    "column_share": 1.0,
    "scan": "all",
    "detector": "simulated",
}
KEY_FIELDS = ("table", *VARIANT_DEFAULTS, "split", "rate", "location", "recall", "fpr", "model", "strategy")
CONFIG = ["model", "rate", "location", "recall", "fpr"]
VARIANT_FIELDS = list(VARIANT_DEFAULTS)


@dataclass(frozen=True)
class Variant:
    """What is injected, and where the detector looks."""

    error: str = "outlier_random"
    mechanism: str = "uniform"
    column_share: float = 1.0
    scan: str = "all"
    detector: str = "simulated"


def key_of(row: dict) -> tuple:
    return tuple(row.get(field, VARIANT_DEFAULTS.get(field)) for field in KEY_FIELDS)


def load_table(name: str) -> Table:
    return load_magic_telescope() if name == "MagicTelescope" else load_openml_table(name)


def target_columns(table: Table, variant: Variant) -> tuple[int, ...]:
    candidates = table.categorical if variant.error in CATEGORY_ERRORS else table.numeric
    if variant.error == "unit":
        # A zero cannot show a unit error. On a column full of zeros, the mask would
        # reveal that the true value is not zero, and marking would leak that fact.
        candidates = tuple(col for col in candidates if (table.X[:, col] == 0).mean() < UNIT_MAX_ZERO_SHARE)
    if not candidates:
        raise ValueError(f"{table.name} has no column for the error {variant.error}")
    if variant.error not in (*NUMERIC_ERRORS, *CATEGORY_ERRORS):
        raise ValueError(f"Unknown error: {variant.error}")
    return pick_columns(candidates, variant.column_share) if variant.column_share < 1 else candidates


def run(
    table: Table, n_splits: int, path: Path, locations: tuple, detectors: tuple, variant: Variant = Variant()
) -> None:
    done = {key_of(row) for row in read_rows(path)}
    columns = target_columns(table, variant)
    scanned = columns if variant.scan == "targeted" else None
    numeric_columns = np.zeros(table.X.shape[1], dtype=bool)
    numeric_columns[list(table.numeric)] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        for split in range(n_splits):
            train, test = split_indices(table.y, split, task=table.task)
            X_train, X_test = table.X[train], table.X[test]
            y_train, y_test = table.y[train], table.y[test]
            scale = np.where(numeric_columns, X_train.std(axis=0), 1.0)
            repairs: dict[tuple, object] = {}
            fits: dict[tuple, tuple] = {}
            real = variant.detector != "simulated"
            for rate, location, (recall, fpr) in product(RATES, locations, ((None, None),) if real else detectors):
                corrupted = corrupt(X_train, X_test, rate, split, variant.error, columns, variant.mechanism)
                if real:
                    view = detect_real(corrupted, location, variant.detector, table.numeric, table.categorical)
                else:
                    view = detect(corrupted, location, recall, fpr, split, table.categorical, scanned)
                quality_train = detection_quality(view.errors_train, view.flagged_train)
                quality_test = detection_quality(view.errors_test, view.flagged_test)
                for strategy, model_name in product(STRATEGIES, MODEL_NAMES):
                    row = {
                        "table": table.name,
                        **asdict(variant),
                        "split": split,
                        "rate": rate,
                        "location": location,
                        "recall": recall,
                        "fpr": fpr,
                        "model": model_name,
                        "strategy": strategy,
                    }
                    if key_of(row) in done:
                        continue
                    key = training_key(strategy, rate, location, recall, fpr)
                    repair_reused = key in repairs
                    if not repair_reused:
                        repairs[key] = fit_repair(strategy, view)
                    repair = repairs[key]
                    fit_key = (model_name, *key)
                    fit_reused = fit_key in fits
                    if not fit_reused:
                        start = time.perf_counter()
                        fitted = fit_model(
                            model_name, table.task, table.categorical, repair.train, y_train[repair.kept_rows]
                        )
                        fits[fit_key] = (fitted, time.perf_counter() - start)
                    fitted, fit_seconds = fits[fit_key]
                    start = time.perf_counter()
                    X_eval = repair.apply(view.dirty_test, view.flagged_test, view.clean_test)
                    apply_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    predictions = predict_scores(fitted, X_eval, table.task)
                    predict_seconds = time.perf_counter() - start
                    repaired_cells = strategy in REPAIRING
                    flagged_numeric = view.flagged_test & numeric_columns
                    flagged_categorical = view.flagged_test & ~numeric_columns
                    row |= evaluate(y_test, predictions, table.task)
                    row |= {
                        "n_target_columns": len(columns),
                        "repair_seconds": repair.seconds,
                        "repair_reused": repair_reused,
                        "apply_seconds": apply_seconds,
                        "fit_seconds": fit_seconds,
                        "fit_reused": fit_reused,
                        "predict_seconds": predict_seconds,
                        "kept_row_share": float(repair.kept_rows.mean()),
                        "imputer_iterations": repair.imputer_iterations,
                        "train_recall": quality_train.recall,
                        "train_fpr": quality_train.fpr,
                        "train_precision": quality_train.precision,
                        "test_recall": quality_test.recall,
                        "test_fpr": quality_test.fpr,
                        "test_precision": quality_test.precision,
                        "repair_error_test": (
                            repair_error(X_eval, view.clean_test, flagged_numeric, scale) if repaired_cells else None
                        ),
                        "category_accuracy_test": (
                            category_repair_accuracy(X_eval, view.clean_test, flagged_categorical)
                            if repaired_cells
                            else None
                        ),
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            print(f"split {split} done", flush=True)


def report(path: Path, table: Table) -> str:
    rows = pd.DataFrame(read_rows(path))
    for field, default in VARIANT_DEFAULTS.items():
        rows[field] = rows[field].fillna(default) if field in rows else default
    # A real detector has no set recall or false positive rate: it is -1 in the tables.
    rows[["recall", "fpr"]] = rows[["recall", "fpr"]].astype(float).fillna(-1.0)
    rows = rows.drop_duplicates(list(KEY_FIELDS), keep="last")
    metric = rows["metric"].iloc[0]
    index = [*VARIANT_FIELDS, *CONFIG]
    means = rows.pivot_table(index=index, columns="strategy", values="value")[list(STRATEGIES)].reset_index()
    repairing = rows[rows["strategy"].isin(REPAIRING)]
    repair_quality = (
        repairing.pivot_table(
            index=[*VARIANT_FIELDS, "rate", "location", "recall", "fpr"],
            columns="strategy",
            values=["repair_error_test", "category_accuracy_test"],
        )
        .round(3)
        .reset_index()
    )
    quality = rows.groupby([*VARIANT_FIELDS, "rate", "location", "recall", "fpr"])[
        ["test_recall", "test_fpr", "test_precision"]
    ].mean()
    repair_time = rows[~rows["repair_reused"]].groupby("strategy")["repair_seconds"].agg(["count", "mean", "max"])
    fit_time = rows[~rows["fit_reused"]].groupby("model")["fit_seconds"].agg(["count", "mean", "max"])
    parts = [
        f"# {table.name}: {table.task}",
        f"{len(table.X)} rows, {len(table.numeric)} numeric and {len(table.categorical)} categorical columns. "
        f"Splits per configuration: {rows['split'].nunique()}. Metric: {metric} "
        f"({'higher' if metric == 'auroc' else 'lower'} is better). Grid provisional, no statistical test here.",
        "## Mean metric",
        means.to_markdown(index=False, floatfmt=".4f"),
        "## Measured detector quality on the test part",
        quality.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "## Repair on flagged test cells",
        repair_quality.to_markdown(index=False),
        "## Compute time (seconds)",
        repair_time.reset_index().to_markdown(index=False, floatfmt=".3f"),
        fit_time.reset_index().to_markdown(index=False, floatfmt=".3f"),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--table", required=True, choices=("MagicTelescope", *OPENML_TABLES))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="a few detectors, errors on both parts")
    parser.add_argument("--error", default="outlier_random", choices=(*NUMERIC_ERRORS, *CATEGORY_ERRORS))
    parser.add_argument("--mechanism", default="uniform", choices=MECHANISMS)
    parser.add_argument("--column-share", type=float, default=1.0)
    parser.add_argument("--scan", default="all", choices=("all", "targeted"))
    parser.add_argument("--detector", default="simulated", choices=("simulated", *REAL_DETECTORS))
    args = parser.parse_args()
    table = load_table(args.table)
    path = RESULTS / f"{'smoke_' if args.smoke else 'grid_'}{table.name}.jsonl"
    if args.command == "run":
        locations = SMOKE_LOCATIONS if args.smoke else LOCATIONS
        detectors = SMOKE_DETECTORS if args.smoke else DETECTORS
        variant = Variant(args.error, args.mechanism, args.column_share, args.scan, args.detector)
        run(table, args.splits, path, locations, detectors, variant)
    else:
        text = report(path, table)
        path.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
