"""One tuning per table and model, on clean data, then fixed for every configuration.

    uv run python -m cmi.tuning --table bike-sharing

Writes cmi/tuned/<table>.json with the hyperparameters and a profile of the columns,
measured on the clean training part of a dedicated split:

- importance: LightGBM gain with the tuned hyperparameters
- target_correlation: absolute correlation with the target, for numeric columns
- predictability: R2 of predicting a numeric column from the other numeric columns
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from cmi.data import TABLE_NAMES, Table, load_table, split_indices
from cmi.models import evaluate, fit_model, predict_scores

TUNED = Path(__file__).resolve().parent / "tuned"
TUNING_SPLIT = 1000
N_FOLDS = 3
N_CONFIGS = 20
LIGHTGBM_SPACE = {
    "num_leaves": (15, 31, 63, 127),
    "learning_rate": (0.02, 0.05, 0.1),
    "n_estimators": (100, 200, 400),
    "min_child_samples": (5, 20, 50),
    "colsample_bytree": (0.7, 1.0),
}
LOGISTIC_SPACE = ({"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0})


def lightgbm_configs(n_configs: int = N_CONFIGS, seed: int = 0) -> list[dict]:
    """Distinct random configurations. The first one is the library default."""
    rng = np.random.default_rng(seed)
    configs = [{"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100, "min_child_samples": 20, "colsample_bytree": 1.0}]
    while len(configs) < n_configs:
        candidate = {name: rng.choice(values).item() for name, values in LIGHTGBM_SPACE.items()}
        if candidate not in configs:
            configs.append(candidate)
    return configs


def cross_validate(name: str, table: Table, X: np.ndarray, y: np.ndarray, params: dict) -> float:
    if table.task == "classification":
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(X, y)
    else:
        folds = KFold(N_FOLDS, shuffle=True, random_state=0).split(X)
    scores = []
    for fit_rows, check_rows in folds:
        model = fit_model(name, table.task, table.categorical, X[fit_rows], y[fit_rows], params)
        predictions = predict_scores(model, X[check_rows], table.task)
        scores.append(evaluate(y[check_rows], predictions, table.task)["score"])
    return float(np.mean(scores))


def tune_model(name: str, table: Table, X: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    if name == "lightgbm":
        candidates = lightgbm_configs()
    elif table.task == "classification":
        candidates = list(LOGISTIC_SPACE)
    else:
        return {}, cross_validate(name, table, X, y, {})
    scored = [(cross_validate(name, table, X, y, params), index) for index, params in enumerate(candidates)]
    best_score, best_index = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return candidates[best_index], best_score


def column_profile(table: Table, X: np.ndarray, y: np.ndarray, lightgbm_params: dict) -> dict:
    model = fit_model("lightgbm", table.task, table.categorical, X, y, lightgbm_params)
    gains = model.booster_.feature_importance(importance_type="gain")
    importance = gains / gains.sum()
    correlation = {}
    for col in table.numeric:
        correlation[col] = float(abs(np.corrcoef(X[:, col], y)[0, 1]))
    fit_rows, check_rows = train_test_split(np.arange(len(X)), test_size=0.25, random_state=0)
    numeric = list(table.numeric)
    predictability = {}
    for col in numeric:
        others = [other for other in numeric if other != col]
        if not others:
            continue
        scores = {}
        for label, estimator in (
            ("bayesian_ridge", BayesianRidge()),
            ("hist_gb", HistGradientBoostingRegressor(max_iter=50, random_state=0)),
        ):
            estimator.fit(X[fit_rows][:, others], X[fit_rows, col])
            scores[label] = float(r2_score(X[check_rows, col], estimator.predict(X[check_rows][:, others])))
        predictability[col] = scores
    return {
        "importance": {col: float(importance[col]) for col in range(X.shape[1])},
        "target_correlation": correlation,
        "predictability": predictability,
    }


def tune_table(table: Table) -> dict:
    train, _ = split_indices(table.y, TUNING_SPLIT, task=table.task)
    X, y = table.X[train], table.y[train]
    start = time.perf_counter()
    lightgbm_params, lightgbm_score = tune_model("lightgbm", table, X, y)
    logistic_params, logistic_score = tune_model("logistic", table, X, y)
    profile = column_profile(table, X, y, lightgbm_params)
    return {
        "table": table.name,
        "task": table.task,
        "tuning_split": TUNING_SPLIT,
        "params": {"lightgbm": lightgbm_params, "logistic": logistic_params},
        "cv_score": {"lightgbm": lightgbm_score, "logistic": logistic_score},
        "default_cv_score": {
            "lightgbm": cross_validate("lightgbm", table, X, y, {}),
            "logistic": cross_validate("logistic", table, X, y, {}),
        },
        "feature_names": list(table.feature_names),
        "profile": profile,
        "seconds": time.perf_counter() - start,
    }


def load_tuned(table_name: str, folder: Path = TUNED) -> dict:
    tuned = json.loads((folder / f"{table_name}.json").read_text(encoding="utf-8"))
    profile = tuned["profile"]
    for field in ("importance", "target_correlation", "predictability"):
        profile[field] = {int(col): value for col, value in profile[field].items()}
    return tuned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, choices=TABLE_NAMES)
    args = parser.parse_args()
    table = load_table(args.table)
    tuned = tune_table(table)
    TUNED.mkdir(exist_ok=True)
    (TUNED / f"{table.name}.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    print(json.dumps({key: tuned[key] for key in ("table", "params", "cv_score", "default_cv_score", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
