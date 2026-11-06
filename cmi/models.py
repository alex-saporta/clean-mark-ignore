"""Downstream models. Library defaults for now, not yet tuned on clean data."""

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAMES = ("lightgbm", "logistic")
LIGHTGBM_PARAMS = {
    "random_state": 0,
    "n_jobs": 4,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def make_model(name: str):
    """Classifier for a numeric table."""
    if name == "lightgbm":
        return LGBMClassifier(**LIGHTGBM_PARAMS)
    if name == "logistic":
        # Without NaN at fit time, the imputer is a no-op and adds no indicator.
        return make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    raise ValueError(f"Unknown model: {name}")


def _linear_pipeline(task: str, n_columns: int, categorical: tuple[int, ...], params: dict | None = None):
    numeric = [col for col in range(n_columns) if col not in categorical]
    # A NaN is its own category, which plays the role of a missing indicator.
    # A category unseen at fit time, NaN included, gets all zeros.
    columns = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
        ],
        sparse_threshold=0,
    )
    estimator = (
        LogisticRegression(max_iter=2000, **(params or {})) if task == "classification" else LinearRegression()
    )
    return make_pipeline(columns, estimator)


def fit_model(
    name: str,
    task: str,
    categorical: tuple[int, ...],
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    n_jobs: int | None = None,
):
    """Fit a model on any table. LightGBM reads integer codes as categories.

    `params` holds tuned hyperparameters. Library defaults apply without them.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task: {task}")
    if name == "lightgbm":
        estimator = LGBMClassifier if task == "classification" else LGBMRegressor
        settings = {**LIGHTGBM_PARAMS, **(params or {})}
        if n_jobs is not None:
            settings["n_jobs"] = n_jobs
        return estimator(**settings).fit(X, y, categorical_feature=list(categorical) or "auto")
    if name == "logistic":
        return _linear_pipeline(task, X.shape[1], categorical, params).fit(X, y)
    raise ValueError(f"Unknown model: {name}")


def predict_scores(model, X: np.ndarray, task: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "classification" else model.predict(X)


def evaluate(y: np.ndarray, scores: np.ndarray, task: str) -> dict[str, float]:
    """`score` is higher when better for both tasks: AUROC, or minus the RMSE."""
    if task == "classification":
        auroc = float(roc_auc_score(y, scores))
        return {"metric": "auroc", "value": auroc, "score": auroc}
    rmse = float(np.sqrt(np.mean((y - scores) ** 2)))
    return {"metric": "rmse", "value": rmse, "score": -rmse}
