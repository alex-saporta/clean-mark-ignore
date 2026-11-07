"""Figures of the report, one command each.

    uv run python -m cmi.figures map         # where each strategy beats doing nothing
    uv run python -m cmi.figures threshold   # S1 against S0 along the false positive rate
    uv run python -m cmi.figures detectors   # real detectors placed on the map
    uv run python -m cmi.figures movies      # real errors, Movies

Each figure is written to figures/<name>.pdf and .png, from results/pilot/.
Counts are clear wins and losses: paired t-test over 10 splits, Benjamini-Yekutieli
over all comparisons of a table, and an effect above the negligible threshold.
Classification and regression stay apart, and no metric is averaged across tables.
"""

import argparse
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from cmi.grid import DETECTORS  # noqa: E402
from cmi.pilot import RESULTS, comparison_table, load_rows  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
CONFIRMATION = ("wine-quality", "diamonds", "houses", "adult", "default-credit")
MODELS = {"lightgbm": "LightGBM", "logistic": "Linear"}
TASKS = {"auroc": "Classification (AUROC)", "rmse": "Regression (RMSE)"}
ERRORS = ("in_domain", "outlier_random", "unit", "category_swap", "typo")
REAL = ("sd", "iqr", "iforest", "rules")

# Reference palette: categorical slots in fixed order, diverging blue <-> red.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("loss_win", [RED, NEUTRAL, BLUE])
MARKERS = ("o", "s", "^", "D", "v")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 8.5,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    }
)


@cache
def table_results(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = load_rows(RESULTS / name)
    comparisons = comparison_table(rows)
    clear = (comparisons["p_by"] < 0.05) & (comparisons["mean_diff"].abs() >= comparisons["negligible_below"])
    comparisons["clear_win"] = clear & (comparisons["mean_diff"] > 0)
    comparisons["clear_loss"] = clear & (comparisons["mean_diff"] < 0)
    comparisons["task"] = rows["metric"].iloc[0]
    comparisons["table"] = name
    rows = rows.assign(task=rows["metric"].iloc[0], table=name)
    return rows, comparisons


def confirmation() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [table_results(name) for name in CONFIRMATION]
    return pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])


def recessive(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def spread(values: list[float], gap: float) -> list[float]:
    """Move end labels apart, keeping their order, so that none overlap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    shift = (np.mean(np.array(values)[order]) - placed.mean()) if len(placed) else 0.0
    result = np.empty(len(values))
    result[order] = placed + shift
    return result.tolist()


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"figures/{name}.pdf and .png")


def figure_map() -> None:
    """Rows: task and model. Columns: strategy against S0. Cells: the detector grid."""
    _, comparisons = confirmation()
    data = comparisons[(comparisons["block"] == "map") & (comparisons["location"] == "both")]
    strategies = ("S1", "S2", "S3", "S4")
    fprs, recalls = (0.0, 0.01, 0.05, 0.2), (0.5, 0.75, 0.95, 1.0)
    panels = [(task, model) for task in TASKS for model in MODELS]
    fig, axes = plt.subplots(len(panels), len(strategies), figsize=(7.2, 7.6), squeeze=False)
    for i, (task, model) in enumerate(panels):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]
            part = data[(data["task"] == task) & (data["model"] == model) & (data["comparison"] == f"{strategy}-S0")]
            grid = np.full((len(recalls), len(fprs)), np.nan)
            for (recall, fpr), group in part.groupby(["recall", "fpr"]):
                r, f = recalls.index(recall), fprs.index(fpr)
                wins, losses, cells = group["clear_win"].sum(), group["clear_loss"].sum(), len(group)
                grid[r, f] = (wins - losses) / cells
                shade = abs(grid[r, f]) > 0.55
                ax.text(f, r, f"{wins}/{losses}\nof {cells}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if shade else INK)
            ax.imshow(np.ma.masked_invalid(grid), cmap=DIVERGING, vmin=-1, vmax=1, origin="lower", aspect="auto")
            ax.set_xticks(range(len(fprs)), [f"{f:.0%}" for f in fprs])
            ax.set_yticks(range(len(recalls)), [f"{r:.2f}" for r in recalls])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(f"{strategy} vs S0", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{TASKS[task].split(' ')[0]}, {MODELS[model]}\nrecall")
            if i == len(panels) - 1:
                ax.set_xlabel("false positive rate")
    fig.suptitle(
        "Clear wins / clear losses against doing nothing (S0), injected errors in train and test\n"
        "5 confirmation tables, every error type and rate; blue: the strategy helps, red: it hurts",
        fontsize=8.5, color=INK, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "map")


def figure_threshold() -> None:
    """Median relative gap of S1 over S0, per error type, as false positives grow."""
    _, comparisons = confirmation()
    data = comparisons[
        (comparisons["block"] == "map")
        & (comparisons["location"] == "both")
        & (comparisons["comparison"] == "S1-S0")
        & (((comparisons["recall"] == 0.95) & (comparisons["fpr"] > 0)) | (comparisons["recall"] == 1.0))
    ]
    fprs = (0.0, 0.01, 0.05, 0.2)
    fig, axes = plt.subplots(len(TASKS), len(MODELS), figsize=(7.2, 5.2), squeeze=False, sharex=True)
    for i, task in enumerate(TASKS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            part = data[(data["task"] == task) & (data["model"] == model)]
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ends = []
            for k, error in enumerate(ERRORS):
                values = part[part["error"] == error].groupby("fpr")["relative_diff"].median().reindex(fprs)
                if values.isna().all():
                    continue
                ax.plot(range(len(fprs)), 100 * values.to_numpy(), color=SERIES[k], linewidth=2,
                        marker=MARKERS[k], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
                ends.append((error, 100 * values.iloc[-1]))
            low, high = ax.get_ylim()
            positions = spread([y for _, y in ends], gap=0.075 * (high - low))
            for (error, y), label_y in zip(ends, positions):
                ax.annotate(error, (len(fprs) - 1, y), xytext=(len(fprs) - 0.85, label_y), textcoords="data",
                            va="center", fontsize=6.5, color=INK_2)
            n_tables = part["table"].nunique()
            ax.set_title(f"{TASKS[task]}, {MODELS[model]} ({n_tables} tables)")
            ax.set_xticks(range(len(fprs)), ["0%\n(recall 1)", "1%", "5%", "20%"])
            ax.set_xlim(-0.2, len(fprs) + 0.1)
            if j == 0:
                ax.set_ylabel("S1 - S0, % of the clean metric\n(median over tables and rates)")
            if i == len(TASKS) - 1:
                ax.set_xlabel("false positive rate (recall 0.95)")
            recessive(ax)
    handles = [Line2D([], [], color=SERIES[k], marker=MARKERS[k], linewidth=2, label=e) for k, e in enumerate(ERRORS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Marking flagged cells (S1) against doing nothing (S0): above zero, marking helps",
                 fontsize=8.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, "threshold")


def figure_detectors() -> None:
    """Measured recall and false positive rate of the real detectors, and their outcome."""
    rows, comparisons = confirmation()
    real = rows[(rows["block"] == "real")]
    quality = real.groupby(["table", "error", "rate", "detector"])[["test_recall", "test_fpr"]].first().reset_index()
    outcome = comparisons[(comparisons["block"] == "real") & (comparisons["comparison"] == "S1-S0")]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.2, 3.4), squeeze=False, sharey=True)
    for j, model in enumerate(MODELS):
        ax = axes[0, j]
        merged = quality.merge(outcome[outcome["model"] == model], on=["table", "error", "rate", "detector"])
        grid_fpr = [f for _, f in DETECTORS if f > 0]
        grid_recall = [r for r, f in DETECTORS if f > 0]
        ax.scatter(grid_fpr, grid_recall, marker="+", s=40, color=MUTED, linewidths=1, label="simulated grid", zorder=1)
        for k, detector in enumerate(REAL):
            part = merged[merged["detector"] == detector]
            colors = np.where(part["clear_win"], BLUE, np.where(part["clear_loss"], RED, "#b5b3ab"))
            ax.scatter(np.clip(part["test_fpr"], 1e-4, None), part["test_recall"], marker=MARKERS[k], s=28,
                       c=list(colors), edgecolors=SURFACE, linewidths=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(8e-5, 0.6)
        # A rate of exactly 0 cannot sit on a log axis: it is drawn at 1e-4 and labelled 0.
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1], ["0", "0.1%", "1%", "10%"])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{MODELS[model]}: S1 against S0")
        ax.set_xlabel("measured false positive rate (test part, log scale)")
        if j == 0:
            ax.set_ylabel("measured recall (test part)")
        recessive(ax)
    shapes = [Line2D([], [], linestyle="", marker=MARKERS[k], color=INK_2, label=d) for k, d in enumerate(REAL)]
    outcomes = [
        Line2D([], [], linestyle="", marker="o", color=BLUE, label="clear win"),
        Line2D([], [], linestyle="", marker="o", color=RED, label="clear loss"),
        Line2D([], [], linestyle="", marker="o", color="#b5b3ab", label="neither"),
        Line2D([], [], linestyle="", marker="+", color=MUTED, label="simulated grid"),
    ]
    fig.legend(handles=shapes + outcomes, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Real detectors on the injected errors of the 5 confirmation tables (one point per table, error and rate)",
                 fontsize=8.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "detectors")


def figure_movies() -> None:
    """Every strategy against S0 on the real errors of Movies, for each detector."""
    _, comparisons = table_results("movies")
    strategies = ("S1", "S2", "S3", "S4", "S5")
    labels = [f"recall {r:.2f}, FP {f:.0%}" for r, f in DETECTORS] + list(REAL)
    keys = [("simulated", r, f) for r, f in DETECTORS] + [(d, -1.0, -1.0) for d in REAL]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.2, 4.4), squeeze=False, sharey=True)
    for j, model in enumerate(MODELS):
        ax = axes[0, j]
        part = comparisons[comparisons["model"] == model]
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.axvspan(-part["negligible_below"].iloc[0], part["negligible_below"].iloc[0], color=NEUTRAL, zorder=0)
        for k, strategy in enumerate(strategies):
            for position, (detector, recall, fpr) in enumerate(keys):
                hit = part[(part["comparison"] == f"{strategy}-S0") & (part["detector"] == detector)
                           & np.isclose(part["recall"], recall) & np.isclose(part["fpr"], fpr)]
                if hit.empty:
                    continue
                value, significant = hit["mean_diff"].iloc[0], hit["p_by"].iloc[0] < 0.05
                ax.scatter(value, position + (k - 2) * 0.13, marker=MARKERS[k], s=22, zorder=2,
                           facecolors=SERIES[k] if significant else SURFACE, edgecolors=SERIES[k], linewidths=1)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axhline(len(DETECTORS) - 0.5, color=GRID, linewidth=0.8)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{MODELS[model]}")
        recessive(ax)
    fig.supxlabel("RMSE gain over S0 (positive: better than doing nothing)", fontsize=8, color=INK_2, y=0.07)
    handles = [Line2D([], [], linestyle="", marker=MARKERS[k], color=SERIES[k], label=s) for k, s in enumerate(strategies)]
    handles.append(Line2D([], [], linestyle="", marker="o", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                          label="not significant (BY)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Movies, real errors (1.3 % of the cells of 3 columns), 10 splits; gray band: negligible effect",
                 fontsize=8.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "movies")


FIGURE_FUNCTIONS = {
    "map": figure_map,
    "threshold": figure_threshold,
    "detectors": figure_detectors,
    "movies": figure_movies,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=(*FIGURE_FUNCTIONS, "all"))
    args = parser.parse_args()
    for name in FIGURE_FUNCTIONS if args.figure == "all" else (args.figure,):
        FIGURE_FUNCTIONS[name]()


if __name__ == "__main__":
    main()
