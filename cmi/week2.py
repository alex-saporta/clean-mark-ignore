"""Strategies against a simulated detector, on one table. The grid is provisional.

    uv run python -m cmi.week2 run [--splits 10]
    uv run python -m cmi.week2 report

Finished evaluations are appended to a JSONL file and skipped on a later run.
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from cmi.data import load_magic_telescope, split_indices
from cmi.detector import detection_quality
from cmi.models import MODEL_NAMES, make_model
from cmi.repairs import ERRORS_IN_TRAIN, REPAIRING, detect, fit_repair, repair_error
from cmi.strategies import LOCATIONS, corrupt
from cmi.week1 import adjust_by, paired_record, read_rows

ERROR = "outlier_random"
STRATEGIES = ("S0", "S1", "S1c", "S2", "S3", "S4", "S5", "CLEAN")
RATES = (0.05, 0.20)
RECALLS = (0.5, 0.75, 0.95)
FPRS = (0.01, 0.05, 0.20)
DETECTORS = ((1.0, 0.0), *product(RECALLS, FPRS))
COMPARISONS = (
    ("S1", "S0"),
    ("S2", "S0"),
    ("S3", "S0"),
    ("S4", "S0"),
    ("S5", "S0"),
    ("S1", "S2"),
    ("S3", "S1"),
    ("S1", "S1c"),
)
RESULTS = Path(__file__).resolve().parent.parent / "results"
KEY_FIELDS = ("table", "error", "split", "rate", "location", "recall", "fpr", "model", "strategy")
CONFIG = ["model", "rate", "location", "recall", "fpr"]


def key_of(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def training_key(strategy: str, rate: float, location: str, recall: float, fpr: float) -> tuple:
    """Configurations sharing a key have identical training matrices."""
    in_train = ERRORS_IN_TRAIN[location]
    if strategy == "CLEAN":
        return (strategy,)
    if strategy == "S0":
        return (strategy, rate if in_train else None)
    return (strategy, rate if in_train else None, recall if in_train else None, fpr)


def run(n_splits: int, path: Path) -> None:
    table = load_magic_telescope()
    done = {key_of(row) for row in read_rows(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        for split in range(n_splits):
            train, test = split_indices(table.y, split)
            X_train, X_test = table.X[train], table.X[test]
            y_train, y_test = table.y[train], table.y[test]
            scale = X_train.std(axis=0)
            repairs: dict[tuple, object] = {}
            fits: dict[tuple, tuple] = {}
            for rate, location, (recall, fpr) in product(RATES, LOCATIONS, DETECTORS):
                corrupted = corrupt(X_train, X_test, rate, split, ERROR)
                view = detect(corrupted, location, recall, fpr, split)
                quality_train = detection_quality(view.errors_train, view.flagged_train)
                quality_test = detection_quality(view.errors_test, view.flagged_test)
                for strategy, model_name in product(STRATEGIES, MODEL_NAMES):
                    row = {
                        "table": table.name,
                        "error": ERROR,
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
                        fitted = clone(make_model(model_name)).fit(repair.train, y_train[repair.kept_rows])
                        fits[fit_key] = (fitted, time.perf_counter() - start)
                    fitted, fit_seconds = fits[fit_key]
                    start = time.perf_counter()
                    X_eval = repair.apply(view.dirty_test, view.flagged_test, view.clean_test)
                    apply_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    scores = fitted.predict_proba(X_eval)[:, 1]
                    predict_seconds = time.perf_counter() - start
                    row |= {
                        "auroc": float(roc_auc_score(y_test, scores)),
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
                            repair_error(X_eval, view.clean_test, view.flagged_test, scale)
                            if strategy in REPAIRING
                            else None
                        ),
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            print(f"split {split} done", flush=True)


def comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    wide = rows.pivot_table(index=[*CONFIG, "split"], columns="strategy", values="auroc")
    records = []
    for config, group in wide.groupby(level=CONFIG):
        for a, b in COMPARISONS:
            records.append(
                {
                    **dict(zip(CONFIG, config)),
                    "comparison": f"{a}-{b}",
                    **paired_record(group[a], group[b]),
                }
            )
    return adjust_by(pd.DataFrame(records))


def win_summary(comparisons: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """For each comparison, how many detector cells are significantly won or lost."""
    table = comparisons.assign(
        win=(comparisons["p_by"] < alpha) & (comparisons["mean_diff"] > 0),
        loss=(comparisons["p_by"] < alpha) & (comparisons["mean_diff"] < 0),
    )
    summary = table.groupby(["model", "location", "rate", "comparison"]).agg(
        cells=("win", "size"),
        wins=("win", "sum"),
        losses=("loss", "sum"),
        min_diff=("mean_diff", "min"),
        max_diff=("mean_diff", "max"),
    )
    return summary.reset_index()


def report(path: Path) -> str:
    rows = pd.DataFrame(read_rows(path)).drop_duplicates(list(KEY_FIELDS), keep="last")
    means = rows.pivot_table(index=CONFIG, columns="strategy", values="auroc")[list(STRATEGIES)].reset_index()
    comparisons = comparison_table(rows)
    quality = rows.groupby(["rate", "location", "recall", "fpr"])[
        ["train_recall", "train_fpr", "train_precision", "test_recall", "test_fpr", "test_precision"]
    ].mean()
    repair_quality = (
        rows[rows["strategy"].isin(REPAIRING)]
        .pivot_table(index=["location", "rate", "recall", "fpr"], columns="strategy", values="repair_error_test")
        .reset_index()
    )
    kept = rows[rows["strategy"] == "S4"].groupby(["rate", "location", "recall", "fpr"])["kept_row_share"].mean()
    iterations = rows[rows["strategy"] == "S3"]["imputer_iterations"].dropna()
    repairs = rows[~rows["repair_reused"]].groupby("strategy")["repair_seconds"].agg(["count", "mean", "max"])
    fits = rows[~rows["fit_reused"]].groupby(["model", "strategy"])["fit_seconds"].agg(["count", "mean", "max"])
    parts = [
        f"# Strategies against a simulated detector: MagicTelescope, `{ERROR}`",
        f"Splits per configuration: {rows['split'].nunique()}. Metric: AUROC on clean labels. "
        f"Grid provisional. Paired t-tests, Benjamini-Yekutieli over all {len(comparisons)} comparisons.",
        "## Wins and losses per detector cell (p_by < 0.05)",
        win_summary(comparisons).to_markdown(index=False, floatfmt=".4f"),
        "## Mean AUROC",
        means.to_markdown(index=False, floatfmt=".4f"),
        "## All comparisons",
        comparisons.to_markdown(index=False, floatfmt=".5f"),
        "## Measured detector quality",
        quality.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "## Repair error on flagged test cells (RMSE, in training SD)",
        repair_quality.to_markdown(index=False, floatfmt=".3f"),
        "## Share of training rows kept by S4",
        kept.reset_index().to_markdown(index=False, floatfmt=".3f"),
        f"## IterativeImputer\n\nIterations used: min {iterations.min():.0f}, max {iterations.max():.0f} (limit 10).",
        "## Compute time (seconds)",
        "Repair fit, by strategy:",
        repairs.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "Model fit:",
        fits.reset_index().to_markdown(index=False, floatfmt=".3f"),
        f"Mean prediction time per evaluation: {rows['predict_seconds'].mean():.4f}.",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RESULTS / "week2.jsonl")
    args = parser.parse_args()
    if args.command == "run":
        run(args.splits, args.output)
    else:
        text = report(args.output)
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text[: text.index("## Mean AUROC")])


if __name__ == "__main__":
    main()
