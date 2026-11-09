"""Frozen verdict rules, applied to one table.

A hypothesis is confirmed on at least 4 of the 5 confirmation tables. Where a rule
names no model, each model gets its own vote.
"""

import numpy as np
import pandas as pd

from cmi.pilot import comparison_table, load_rows

CONFIRMED_AT = 4
DESTROYING = ("in_domain", "category_swap")
KEEPING = ("unit", "typo")


def with_clear_flags(comparisons: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    clear = (comparisons["p_by"] < alpha) & (comparisons["mean_diff"].abs() >= comparisons["negligible_below"])
    return comparisons.assign(
        clear_win=clear & (comparisons["mean_diff"] > 0),
        clear_loss=clear & (comparisons["mean_diff"] < 0),
    )


def pick(c: pd.DataFrame, **fields) -> pd.DataFrame:
    mask = np.ones(len(c), dtype=bool)
    for field, value in fields.items():
        mask &= (c[field] == value).to_numpy()
    return c[mask]


def h1(c: pd.DataFrame) -> dict:
    """S1 has fewer than 5 % clear losses against S2 (LightGBM, errors in both parts)."""
    part = pick(c, block="map", comparison="S1-S2", location="both", model="lightgbm")
    share = part["clear_loss"].mean()
    return {"holds": bool(share < 0.05), "clear_wins": int(part["clear_win"].sum()),
            "clear_losses": int(part["clear_loss"].sum()), "cells": len(part)}


def h2(c: pd.DataFrame, model: str) -> dict:
    """S1 clearly loses to S0 in at least 80 % of cells at recall 0.95 and 20 % FP, and
    the median S1-S2 gap is larger at 20 % FP than at 1 %."""
    first = pick(c, block="map", comparison="S1-S0", model=model, recall=0.95, fpr=0.2)
    gaps = pick(c, block="map", comparison="S1-S2", model=model)
    high, low = gaps[gaps["fpr"] == 0.2]["mean_diff"].median(), gaps[gaps["fpr"] == 0.01]["mean_diff"].median()
    part1 = first["clear_loss"].mean() >= 0.8
    return {"holds": bool(part1 and high > low), "clear_losses": int(first["clear_loss"].sum()),
            "cells": len(first), "gap_high": float(high), "gap_low": float(low)}


def h3(c: pd.DataFrame, model: str) -> dict:
    """S3 has no clear win over S1 on unpredictable columns and at least one on predictable ones."""
    part = pick(c, block="profile", comparison="S3-S1", model=model)
    predictable = part[part["column_choice"] == "predictable"]["clear_win"].sum()
    unpredictable = part[part["column_choice"] == "unpredictable"]["clear_win"].sum()
    return {"holds": bool(unpredictable == 0 and predictable >= 1),
            "predictable": int(predictable), "unpredictable": int(unpredictable)}


def h4(c: pd.DataFrame, model: str) -> dict:
    """The median gain of S5 over S0 on important columns is at least twice that on unimportant ones."""
    part = pick(c, block="columns", comparison="S5-S0", model=model)
    top = part[part["column_choice"] == "top"]["mean_diff"].median()
    bottom = part[part["column_choice"] == "bottom"]["mean_diff"].median()
    return {"holds": bool(top > 0 and top >= 2 * bottom), "top": float(top), "bottom": float(bottom)}


def h5(c: pd.DataFrame) -> dict:
    """At the perfect detector on LightGBM, S1 gains more over S0 on every error that destroys
    information than on every error that keeps it, at both error rates."""
    part = pick(c, block="map", comparison="S1-S0", model="lightgbm", recall=1.0, fpr=0.0)
    gains = part.pivot_table(index="rate", columns="error", values="mean_diff")
    destroying = [e for e in DESTROYING if e in gains]
    keeping = [e for e in KEEPING if e in gains]
    holds = all(gains.loc[rate, destroying].min() > gains.loc[rate, keeping].max() for rate in gains.index)
    return {"holds": bool(holds)}


def table_verdicts(comparisons: pd.DataFrame) -> dict[str, dict]:
    c = with_clear_flags(comparisons)
    return {
        "H1 lightgbm": h1(c),
        "H2 lightgbm": h2(c, "lightgbm"),
        "H2 logistic": h2(c, "logistic"),
        "H3 lightgbm": h3(c, "lightgbm"),
        "H3 logistic": h3(c, "logistic"),
        "H4 lightgbm": h4(c, "lightgbm"),
        "H4 logistic": h4(c, "logistic"),
        "H5 lightgbm": h5(c),
    }


def verdicts(names: tuple[str, ...], folder) -> pd.DataFrame:
    """One row per hypothesis and model, one column per table: does the table hold it?"""
    holds = {name: {key: v["holds"] for key, v in table_verdicts(comparison_table(load_rows(folder / name))).items()}
             for name in names}
    return pd.DataFrame(holds)
