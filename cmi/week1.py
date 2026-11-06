"""Week 1: S0, S1 and S5 on one table, with outliers and a perfect detector.

    uv run python -m cmi.week1 run [--splits 10]
    uv run python -m cmi.week1 report

Finished evaluations are appended to a JSONL file and skipped on a later run.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from cmi.data import load_magic_telescope, split_indices
from cmi.models import MODEL_NAMES, make_model
from cmi.strategies import (
    LOCATIONS,
    STRATEGIES,
    corrupt,
    eval_matrix,
    train_matrix,
    train_variant,
)

ERRORS = ("outlier", "outlier_random")
ERROR_LABELS = {
    "outlier": "true value +/- 5 SD (keeps the true value's order)",
    "outlier_random": "training min/max -/+ U(1, 5) SD (independent of the true value)",
}
RATES = (0.05, 0.20)
COMPARISONS = (("S1", "S0"), ("S5", "S0"), ("S1", "S5"))
RESULTS = Path(__file__).resolve().parent.parent / "results"
KEY_FIELDS = ("table", "error", "split", "rate", "location", "model", "strategy")
CONFIG = ["model", "rate", "location"]


def key_of(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def read_rows(path: Path) -> list[dict]:
    """Read finished rows, ignoring a last line truncated by an interruption."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run(n_splits: int, path: Path) -> None:
    table = load_magic_telescope()
    done = {key_of(row) for row in read_rows(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        for split in range(n_splits):
            train, test = split_indices(table.y, split)
            X_train, X_test = table.X[train], table.X[test]
            y_train, y_test = table.y[train], table.y[test]
            for error in ERRORS:
                fits: dict[tuple, tuple] = {}
                for rate in RATES:
                    corrupted = corrupt(X_train, X_test, rate, split, error)
                    for model_name in MODEL_NAMES:
                        for location in LOCATIONS:
                            for strategy in STRATEGIES:
                                row = {
                                    "table": table.name,
                                    "error": error,
                                    "split": split,
                                    "rate": rate,
                                    "location": location,
                                    "model": model_name,
                                    "strategy": strategy,
                                }
                                if key_of(row) in done:
                                    continue
                                variant = train_variant(strategy, location)
                                # The clean matrix does not depend on the error rate.
                                fit_key = (model_name, variant, None if variant == "clean" else rate)
                                reused = fit_key in fits
                                if not reused:
                                    start = time.perf_counter()
                                    fitted = clone(make_model(model_name)).fit(
                                        train_matrix(corrupted, variant), y_train
                                    )
                                    fits[fit_key] = (fitted, time.perf_counter() - start)
                                fitted, fit_seconds = fits[fit_key]
                                start = time.perf_counter()
                                scores = fitted.predict_proba(eval_matrix(corrupted, strategy, location))
                                predict_seconds = time.perf_counter() - start
                                row |= {
                                    "auroc": float(roc_auc_score(y_test, scores[:, 1])),
                                    "fit_seconds": fit_seconds,
                                    "fit_variant": variant,
                                    "fit_reused": reused,
                                    "predict_seconds": predict_seconds,
                                }
                                out.write(json.dumps(row) + "\n")
                                out.flush()
            print(f"split {split} done", flush=True)


def adjust_by(result: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Yekutieli over every tested row of the table."""
    tested = result["p"].notna()
    result["p_by"] = np.nan
    if tested.any():
        result.loc[tested, "p_by"] = stats.false_discovery_control(result.loc[tested, "p"], method="by")
    return result


def paired_record(a: pd.Series, b: pd.Series) -> dict:
    diff = (a - b).to_numpy()
    identical = np.all(diff == 0)
    return {
        "n_splits": len(diff),
        "mean_diff": diff.mean(),
        "sd_diff": diff.std(ddof=1),
        "p": np.nan if identical else stats.ttest_rel(a, b).pvalue,
    }


def comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests over splits, within one error type."""
    wide = rows.pivot_table(index=[*CONFIG, "split"], columns="strategy", values="auroc")
    records = []
    for (model, rate, location), group in wide.groupby(level=CONFIG):
        for a, b in COMPARISONS:
            records.append(
                {
                    "model": model,
                    "rate": rate,
                    "location": location,
                    "comparison": f"{a}-{b}",
                    **paired_record(group[a], group[b]),
                }
            )
    return adjust_by(pd.DataFrame(records))


def s0_across_errors(rows: pd.DataFrame) -> pd.DataFrame:
    """Does S0 lose accuracy when the error no longer carries the true value?"""
    s0 = rows[rows["strategy"] == "S0"]
    wide = s0.pivot_table(index=[*CONFIG, "split"], columns="error", values="auroc")
    records = []
    for (model, rate, location), group in wide.groupby(level=CONFIG):
        records.append(
            {
                "model": model,
                "rate": rate,
                "location": location,
                "comparison": "S0(random)-S0(order-keeping)",
                **paired_record(group["outlier_random"], group["outlier"]),
            }
        )
    return adjust_by(pd.DataFrame(records))


def identity_check(rows: pd.DataFrame) -> pd.DataFrame:
    """S1 and S5 must not depend on the error values, only on the shared masks."""
    kept = rows[rows["strategy"].isin(["S1", "S5"])]
    wide = kept.pivot_table(index=[*CONFIG, "strategy", "split"], columns="error", values="auroc")
    gap = (wide["outlier"] - wide["outlier_random"]).abs()
    return gap.groupby("strategy").max().rename("max_abs_auroc_gap").reset_index()


def error_section(rows: pd.DataFrame, error: str) -> list[str]:
    subset = rows[rows["error"] == error]
    means = subset.pivot_table(index=CONFIG, columns="strategy", values="auroc").reset_index()
    means["S1_share_of_oracle_gain"] = (means["S1"] - means["S0"]) / (means["S5"] - means["S0"])
    return [
        f"## Error `{error}`: {ERROR_LABELS[error]}",
        f"Splits per configuration: {subset['split'].nunique()}.",
        "Mean AUROC:",
        means.to_markdown(index=False, floatfmt=".4f"),
        "Paired t-tests, Benjamini-Yekutieli over the rows of this table:",
        comparison_table(subset).to_markdown(index=False, floatfmt=".5f"),
    ]


def timing_section(rows: pd.DataFrame) -> list[str]:
    fits = rows[~rows["fit_reused"]]
    by_variant = fits.assign(
        train_data=fits["fit_variant"].fillna(fits["strategy"].map({"S0": "dirty", "S1": "marked", "S5": "clean"}))
    )
    fit_time = by_variant.groupby(["model", "train_data"])["fit_seconds"].agg(["count", "mean", "max"])
    predict_time = rows.groupby(["model", "strategy"])["predict_seconds"].mean().rename("mean_predict_seconds")
    return [
        "## Compute time (seconds)",
        "Fit time by training matrix (S0 trains on dirty, S1 on marked, S5 on clean data):",
        fit_time.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "Mean prediction time per evaluation, by strategy:",
        predict_time.reset_index().to_markdown(index=False, floatfmt=".4f"),
    ]


def report(path: Path) -> str:
    rows = pd.DataFrame(read_rows(path)).drop_duplicates(list(KEY_FIELDS), keep="last")
    parts = [
        "# Week 1 results: MagicTelescope, perfect detector",
        "Metric: AUROC on clean labels. Both errors use the same cells and signs.",
    ]
    for error in ERRORS:
        parts += error_section(rows, error)
    parts += [
        "## Effect of the error definition on S0",
        "Paired t-tests, Benjamini-Yekutieli over the rows of this table:",
        s0_across_errors(rows).to_markdown(index=False, floatfmt=".5f"),
        "Control: largest gap between the two errors for strategies that erase the error values.",
        identity_check(rows).to_markdown(index=False, floatfmt=".10f"),
    ]
    parts += timing_section(rows)
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RESULTS / "week1.jsonl")
    args = parser.parse_args()
    if args.command == "run":
        run(args.splits, args.output)
    else:
        text = report(args.output)
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
