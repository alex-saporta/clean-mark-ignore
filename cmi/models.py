"""Downstream models. Week 1 uses library defaults, not yet tuned on clean data."""

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("lightgbm", "logistic")


def make_model(name: str):
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=0,
            n_jobs=4,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
    if name == "logistic":
        # Without NaN at fit time, the imputer is a no-op and adds no indicator.
        return make_pipeline(
            SimpleImputer(strategy="mean", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    raise ValueError(f"Unknown model: {name}")
