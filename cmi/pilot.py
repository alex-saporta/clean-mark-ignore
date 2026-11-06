"""The pilot grid: only the crossings that answer a hypothesis.

    uv run python -m cmi.tuning --table bike-sharing
    uv run python -m cmi.pilot run --table bike-sharing [--splits 10] [--jobs 4]
    uv run python -m cmi.pilot report --table bike-sharing

Results go to one file per split under results/pilot/<table>/, and finished
evaluations are skipped on a later run. Every value is fixed per worker, so the
number of parallel splits does not change any result.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from cmi.data import OPENML_TABLES, Table, split_indices
from cmi.detector import detection_quality
from cmi.grid import DETECTORS, RATES, Variant, load_table, target_columns
from cmi.models import MODEL_NAMES, evaluate, fit_model, predict_scores
from cmi.real_detectors import DETECTORS as REAL_DETECTORS
from cmi.repairs import REPAIRING, category_repair_accuracy, detect, detect_real, fit_repair, repair_error
from cmi.strategies import corrupt
from cmi.tuning import load_tuned
from cmi.week1 import adjust_by, paired_record, read_rows
from cmi.week2 import training_key

RESULTS = Path(__file__).resolve().parent.parent / "results" / "pilot"
ERRORS = ("outlier_random", "in_domain", "unit", "category_swap", "typo")
COLUMN_ERRORS = ("outlier_random", "in_domain")
STRATEGIES = ("S0", "S1", "S1c", "S2", "S3", "S4", "S5", "CLEAN")
KEY_DETECTORS = ((1.0, 0.0), (0.95, 0.01), (0.75, 0.05))
COLUMN_SHARE = 0.3
THREADS_PER_WORKER = 2
COMPARISONS = (
    ("S1", "S0"),
    ("S2", "S0"),
    ("S3", "S0"),
    ("S4", "S0"),
    ("S5", "S0"),
    ("S1", "S2"),
    ("S3", "S1"),
    ("S3hgb", "S1"),
    ("S1", "S1c"),
)
KEY_FIELDS = (
    "block",
    "error",
    "mechanism",
    "column_choice",
    "scan",
    "detector",
    "rate",
    "location",
    "recall",
    "fpr",
    "split",
    "model",
    "strategy",
)
CONFIG = [field for field in KEY_FIELDS if field not in ("split", "strategy")]


@dataclass(frozen=True)
class Cell:
    block: str
    variant: Variant
    column_choice: str
    rate: float
    location: str
    recall: float | None
    fpr: float | None
    strategies: tuple[str, ...]


def available_errors(table: Table) -> tuple[str, ...]:
    errors = []
    for error in ERRORS:
        try:
            target_columns(table, Variant(error=error))
        except ValueError:
            continue
        errors.append(error)
    return tuple(errors)


def pilot_cells(table: Table) -> list[Cell]:
    cells = []
    errors = available_errors(table)
    for error, rate, (recall, fpr) in product(errors, RATES, DETECTORS):
        strategies = (*STRATEGIES, "S3hgb") if (recall, fpr) in KEY_DETECTORS else STRATEGIES
        cells.append(Cell("map", Variant(error=error), "all", rate, "both", recall, fpr, strategies))
    for error, rate, (recall, fpr), location in product(errors, RATES, KEY_DETECTORS, ("train", "test")):
        cells.append(Cell("location", Variant(error=error), "all", rate, location, recall, fpr, STRATEGIES))
    for error, rate, (recall, fpr), choice, scan in product(
        COLUMN_ERRORS, RATES, KEY_DETECTORS, ("top", "bottom"), ("all", "targeted")
    ):
        variant = Variant(error=error, column_share=COLUMN_SHARE, scan=scan)
        cells.append(Cell("columns", variant, choice, rate, "both", recall, fpr, STRATEGIES))
    for rate, (recall, fpr) in product(RATES, KEY_DETECTORS):
        variant = Variant(error="in_domain", mechanism="dependent")
        cells.append(Cell("mechanism", variant, "all", rate, "both", recall, fpr, STRATEGIES))
    for detector, error, rate in product(REAL_DETECTORS, errors, RATES):
        variant = Variant(error=error, detector=detector)
        cells.append(Cell("real", variant, "all", rate, "both", None, None, STRATEGIES))
    return cells


def chosen_columns(table: Table, variant: Variant, column_choice: str, importance: dict[int, float]) -> tuple[int, ...]:
    """All candidates, or the most or least important share of them on clean data."""
    candidates = target_columns(table, Variant(error=variant.error))
    if column_choice == "all":
        return candidates
    n_columns = min(len(candidates), max(2, round(variant.column_share * len(candidates))))
    ranked = sorted(candidates, key=lambda col: (-importance[col], col))
    chosen = ranked[:n_columns] if column_choice == "top" else ranked[-n_columns:]
    return tuple(sorted(chosen))


def key_of(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def run_split(
    table: Table, split: int, cells: list[Cell], folder: Path, tuned: dict, threads: int = THREADS_PER_WORKER
) -> int:
    """Evaluate one split. Returns the number of new evaluations."""
    path = folder / f"split_{split}.jsonl"
    done = {key_of(row) for row in read_rows(path)}
    importance = tuned["profile"]["importance"]
    numeric_columns = np.zeros(table.X.shape[1], dtype=bool)
    numeric_columns[list(table.numeric)] = True
    train, test = split_indices(table.y, split, task=table.task)
    X_train, X_test = table.X[train], table.X[test]
    y_train, y_test = table.y[train], table.y[test]
    scale = np.where(numeric_columns, X_train.std(axis=0), 1.0)
    corruptions: dict[tuple, object] = {}
    repairs: dict[tuple, object] = {}
    fits: dict[tuple, tuple] = {}
    n_new = 0
    with threadpool_limits(limits=threads), path.open("a", encoding="utf-8") as out:
        for cell in cells:
            base = {
                "block": cell.block,
                "table": table.name,
                **asdict(cell.variant),
                "column_choice": cell.column_choice,
                "rate": cell.rate,
                "location": cell.location,
                "recall": cell.recall,
                "fpr": cell.fpr,
                "split": split,
            }
            pending = [
                (strategy, model_name)
                for strategy, model_name in product(cell.strategies, MODEL_NAMES)
                if key_of({**base, "strategy": strategy, "model": model_name}) not in done
            ]
            if not pending:
                continue
            variant = cell.variant
            columns = chosen_columns(table, variant, cell.column_choice, importance)
            corruption_key = (variant.error, variant.mechanism, columns, cell.rate)
            if corruption_key not in corruptions:
                corruptions[corruption_key] = corrupt(
                    X_train, X_test, cell.rate, split, variant.error, columns, variant.mechanism
                )
            corrupted = corruptions[corruption_key]
            if variant.detector != "simulated":
                view = detect_real(corrupted, cell.location, variant.detector, table.numeric, table.categorical)
            else:
                scanned = columns if variant.scan == "targeted" else None
                view = detect(corrupted, cell.location, cell.recall, cell.fpr, split, table.categorical, scanned)
            quality_train = detection_quality(view.errors_train, view.flagged_train)
            quality_test = detection_quality(view.errors_test, view.flagged_test)
            prefix = (variant.error, variant.mechanism, columns, variant.scan, variant.detector)
            for strategy, model_name in pending:
                key = (*prefix, *training_key(strategy, cell.rate, cell.location, cell.recall, cell.fpr))
                repair_reused = key in repairs
                if not repair_reused:
                    repairs[key] = fit_repair(strategy, view)
                repair = repairs[key]
                fit_key = (model_name, *key)
                fit_reused = fit_key in fits
                if not fit_reused:
                    start = time.perf_counter()
                    fitted = fit_model(
                        model_name,
                        table.task,
                        table.categorical,
                        repair.train,
                        y_train[repair.kept_rows],
                        tuned["params"][model_name],
                        n_jobs=threads,
                    )
                    fits[fit_key] = (fitted, time.perf_counter() - start)
                fitted, fit_seconds = fits[fit_key]
                start = time.perf_counter()
                X_eval = repair.apply(view.dirty_test, view.flagged_test, view.clean_test)
                predictions = predict_scores(fitted, X_eval, table.task)
                predict_seconds = time.perf_counter() - start
                repaired_cells = strategy in REPAIRING
                row = {
                    **base,
                    "model": model_name,
                    "strategy": strategy,
                    **evaluate(y_test, predictions, table.task),
                    "n_target_columns": len(columns),
                    "target_columns": list(columns),
                    "repair_seconds": repair.seconds,
                    "repair_reused": repair_reused,
                    "fit_seconds": fit_seconds,
                    "fit_reused": fit_reused,
                    "predict_seconds": predict_seconds,
                    "kept_row_share": float(repair.kept_rows.mean()),
                    "train_recall": quality_train.recall,
                    "train_fpr": quality_train.fpr,
                    "train_precision": quality_train.precision,
                    "test_recall": quality_test.recall,
                    "test_fpr": quality_test.fpr,
                    "test_precision": quality_test.precision,
                    "repair_error_test": (
                        repair_error(X_eval, view.clean_test, view.flagged_test & numeric_columns, scale)
                        if repaired_cells
                        else None
                    ),
                    "category_accuracy_test": (
                        category_repair_accuracy(X_eval, view.clean_test, view.flagged_test & ~numeric_columns)
                        if repaired_cells
                        else None
                    ),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_new += 1
    return n_new


def run(table: Table, splits: list[int], folder: Path, tuned: dict, jobs: int, cells: list[Cell] | None = None) -> None:
    cells = pilot_cells(table) if cells is None else cells
    folder.mkdir(parents=True, exist_ok=True)
    counts = Parallel(n_jobs=jobs)(
        delayed(run_split)(table, split, cells, folder, tuned) for split in splits
    )
    print(f"{table.name}: {sum(counts)} new evaluations over splits {splits}", flush=True)


def load_rows(folder: Path) -> pd.DataFrame:
    rows = [row for path in sorted(folder.glob("split_*.jsonl")) for row in read_rows(path)]
    frame = pd.DataFrame(rows)
    frame[["recall", "fpr"]] = frame[["recall", "fpr"]].astype(float).fillna(-1.0)
    return frame.drop_duplicates(list(KEY_FIELDS), keep="last")


def comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests over splits, on a score where higher is better, with BY correction."""
    wide = rows.pivot_table(index=[*CONFIG, "split"], columns="strategy", values="score")
    records = []
    for config, group in wide.groupby(level=CONFIG):
        for a, b in COMPARISONS:
            if a in group and b in group and group[a].notna().all() and group[b].notna().all():
                records.append(
                    {**dict(zip(CONFIG, config)), "comparison": f"{a}-{b}", **paired_record(group[a], group[b])}
                )
    return adjust_by(pd.DataFrame(records))


def win_summary(comparisons: pd.DataFrame, by: list[str], alpha: float = 0.05) -> pd.DataFrame:
    table = comparisons.assign(
        win=(comparisons["p_by"] < alpha) & (comparisons["mean_diff"] > 0),
        loss=(comparisons["p_by"] < alpha) & (comparisons["mean_diff"] < 0),
    )
    return (
        table.groupby(by)
        .agg(
            cells=("win", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            mean_diff=("mean_diff", "mean"),
            min_diff=("mean_diff", "min"),
            max_diff=("mean_diff", "max"),
        )
        .reset_index()
    )


def report(table: Table, folder: Path, tuned: dict) -> str:
    rows = load_rows(folder)
    metric = rows["metric"].iloc[0]
    comparisons = comparison_table(rows)
    index = ["block", "error", "column_choice", "scan", "detector", "model", "rate", "recall", "fpr", "location"]
    means = rows.pivot_table(index=index, columns="strategy", values="value").reset_index()
    quality = rows.groupby(["block", "error", "detector", "rate", "recall", "fpr", "location"])[
        ["test_recall", "test_fpr", "test_precision", "train_recall", "train_fpr", "train_precision"]
    ].mean()
    repaired = rows[rows["strategy"].isin(REPAIRING)]
    repair_quality = repaired.pivot_table(
        index=["block", "error", "rate", "recall", "fpr", "location"],
        columns="strategy",
        values="repair_error_test",
    ).reset_index()
    kept = rows[rows["strategy"] == "S4"].groupby(["block", "error", "detector", "rate", "recall", "fpr"])[
        "kept_row_share"
    ].mean()
    profile = tuned["profile"]
    names = tuned["feature_names"]
    columns = pd.DataFrame(
        {
            "column": names,
            "importance": [profile["importance"][col] for col in range(len(names))],
            "target_correlation": [profile["target_correlation"].get(col) for col in range(len(names))],
            "r2_bayesian_ridge": [profile["predictability"].get(col, {}).get("bayesian_ridge") for col in range(len(names))],
            "r2_hist_gb": [profile["predictability"].get(col, {}).get("hist_gb") for col in range(len(names))],
        }
    )
    repair_time = rows[~rows["repair_reused"]].groupby("strategy")["repair_seconds"].agg(["count", "mean", "max"])
    fit_time = rows[~rows["fit_reused"]].groupby("model")["fit_seconds"].agg(["count", "mean", "max"])
    total = rows[~rows["repair_reused"]]["repair_seconds"].sum() + rows[~rows["fit_reused"]]["fit_seconds"].sum()
    total += rows["predict_seconds"].sum()
    n_cells = rows[CONFIG].drop_duplicates().drop(columns=["model"]).drop_duplicates().shape[0]
    parts = [
        f"# Pilot: {table.name}, {table.task}",
        f"Splits: {rows['split'].nunique()}. Metric: {metric}, {'higher' if metric == 'auroc' else 'lower'} is better. "
        f"Paired t-tests, Benjamini-Yekutieli over all {len(comparisons)} comparisons of this table. "
        "`mean_diff` is on a score where higher is better, so a positive value favours the first strategy. "
        "A recall and false positive rate of -1 mean a real detector.",
        f"Hyperparameters, tuned once on clean data: `{json.dumps(tuned['params'])}`",
        "## Wins and losses by comparison and block",
        win_summary(comparisons, ["model", "block", "comparison"]).to_markdown(index=False, floatfmt=".4f"),
        "## Wins and losses by error type (map block)",
        win_summary(comparisons[comparisons["block"] == "map"], ["model", "error", "rate", "comparison"]).to_markdown(
            index=False, floatfmt=".4f"
        ),
        "## Column importance, correlation and predictability (clean training part)",
        columns.to_markdown(index=False, floatfmt=".3f"),
        "## Mean metric",
        means.to_markdown(index=False, floatfmt=".4f"),
        "## All comparisons",
        comparisons.to_markdown(index=False, floatfmt=".5f"),
        "## Measured detector quality",
        quality.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "## Repair error on flagged test cells (RMSE, in training SD)",
        repair_quality.to_markdown(index=False, floatfmt=".3f"),
        "## Share of training rows kept by S4",
        kept.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "## Compute time (seconds)",
        f"Total compute per split: {total / rows['split'].nunique():.0f}. Cells: {n_cells}. "
        f"Mean per cell and split: {total / rows['split'].nunique() / n_cells:.1f}.",
        repair_time.reset_index().to_markdown(index=False, floatfmt=".3f"),
        fit_time.reset_index().to_markdown(index=False, floatfmt=".3f"),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--table", required=True, choices=("MagicTelescope", *OPENML_TABLES))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--first-split", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    table = load_table(args.table)
    tuned = load_tuned(table.name)
    folder = RESULTS / table.name
    if args.command == "run":
        run(table, list(range(args.first_split, args.splits)), folder, tuned, args.jobs)
    else:
        text = report(table, folder, tuned)
        (folder / "report.md").write_text(text, encoding="utf-8")
        print(text[: text.index("## Mean metric")])


if __name__ == "__main__":
    main()
