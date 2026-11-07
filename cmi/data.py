"""Dataset loading and train/test splits."""

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

DATA_HOME = Path(__file__).resolve().parent.parent / "data"
ROW_LIMIT = 20_000
MAGIC_FEATURES = (
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
)


@dataclass(frozen=True)
class Table:
    """Categorical columns hold integer codes, sorted by category name.

    `X_dirty` is set only for a table with real errors: `X` is then its clean
    version, and every cell where the two differ is an error.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    task: str = "classification"
    categorical: tuple[int, ...] = ()
    X_dirty: np.ndarray | None = None

    @property
    def numeric(self) -> tuple[int, ...]:
        return tuple(col for col in range(self.X.shape[1]) if col not in self.categorical)

    @property
    def error_mask(self) -> np.ndarray:
        if self.X_dirty is None:
            return np.zeros(self.X.shape, dtype=bool)
        return ~((self.X_dirty == self.X) | (np.isnan(self.X_dirty) & np.isnan(self.X)))

    @property
    def error_columns(self) -> tuple[int, ...]:
        return tuple(int(col) for col in np.flatnonzero(self.error_mask.any(axis=0)))


@dataclass(frozen=True)
class OpenMLSpec:
    """`positive` is the target value of the positive class; None means regression."""

    data_id: int
    target: str
    positive: str | None = None
    drop: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()


# Licences checked at the source. See the README for citations.
OPENML_TABLES = {
    # UCI 222, CC BY 4.0. V12 is the call duration, known only after the outcome.
    "bank-marketing": OpenMLSpec(1461, "Class", positive="2", drop=("V12",)),
    # UCI 468, CC BY 4.0.
    "online-shoppers": OpenMLSpec(46947, "Revenue", positive="True"),
    # UCI 186, CC BY 4.0.
    "wine-quality": OpenMLSpec(287, "quality"),
    # UCI 275, CC BY 4.0. OpenML ignores casual and registered, which sum to the target.
    "bike-sharing": OpenMLSpec(42712, "count"),
    # Ships with ggplot2 (MIT). OpenML lists it as public domain.
    "diamonds": OpenMLSpec(42225, "price"),
    # StatLib asks for an acknowledgement; no explicit licence found.
    "houses": OpenMLSpec(537, "median_house_value"),
    # UCI 350, CC BY 4.0. Sex, education and marriage are stored as integer codes.
    "default-credit": OpenMLSpec(42477, "y", positive="1", categorical=("x2", "x3", "x4")),
    # UCI 2, CC BY 4.0. This OpenML version has no NaN: unknown values stay a category.
    "adult": OpenMLSpec(45068, "class", positive=">50K"),
}


def load_magic_telescope(data_home: Path = DATA_HOME) -> Table:
    """MAGIC Gamma Telescope (OpenML 1120, UCI 159, CC BY 4.0). Positive class: gamma."""
    bunch = fetch_openml(data_id=1120, as_frame=True, data_home=str(data_home))
    frame = bunch.frame.rename(columns=lambda c: c.rstrip(":"))
    X = frame.loc[:, list(MAGIC_FEATURES)].to_numpy(dtype=np.float64)
    y = (frame["class"].astype(str) == "g").to_numpy(dtype=np.int64)
    if X.shape != (19020, 10) or np.isnan(X).any():
        raise ValueError(f"Unexpected table: shape={X.shape}, nan={np.isnan(X).sum()}")
    return Table("MagicTelescope", X, y, MAGIC_FEATURES)


def cap_rows(
    X: np.ndarray, y: np.ndarray, task: str, limit: int = ROW_LIMIT, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most `limit` rows, stratified on the target for classification."""
    if len(X) <= limit:
        return X, y
    stratify = y if task == "classification" else None
    kept, _ = train_test_split(np.arange(len(X)), train_size=limit, random_state=seed, stratify=stratify)
    return X[kept], y[kept]


def encode_features(
    features: pd.DataFrame, forced_categories: tuple[str, ...] = ()
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Numeric columns stay as they are. Other columns become integer codes."""
    columns, categorical = [], []
    for position, name in enumerate(features.columns):
        column = features[name]
        is_number = pd.api.types.is_numeric_dtype(column) and not isinstance(column.dtype, pd.CategoricalDtype)
        if is_number and name not in forced_categories:
            columns.append(column.to_numpy(dtype=np.float64))
        else:
            values = column.astype(str)
            categories = sorted(values.unique())
            columns.append(pd.Categorical(values, categories=categories).codes.astype(np.float64))
            categorical.append(position)
    return np.column_stack(columns), tuple(categorical)


def load_openml_table(name: str, data_home: Path = DATA_HOME) -> Table:
    spec = OPENML_TABLES[name]
    # The default pandas parser fails on boolean or integer category names.
    frame = fetch_openml(data_id=spec.data_id, as_frame=True, data_home=str(data_home), parser="liac-arff").frame
    frame = frame.rename(columns=lambda c: c.rstrip(":"))
    target = frame[spec.target]
    features = frame.drop(columns=[spec.target, *spec.drop])
    if features.isna().any().any() or target.isna().any():
        raise ValueError(f"{name}: unexpected missing values")
    X, categorical = encode_features(features, spec.categorical)
    if spec.positive is None:
        task = "regression"
        y = target.astype(np.float64).to_numpy()
    else:
        task = "classification"
        y = (target.astype(str).str.strip() == spec.positive).to_numpy(dtype=np.int64)
        if y.min() == y.max():
            raise ValueError(f"{name}: positive value {spec.positive!r} gives a single class")
    X, y = cap_rows(X, y, task)
    return Table(name, X, y, tuple(features.columns), task, categorical)


# Movies, from the Raha repository (Apache-2.0 for the repository). The data come
# from IMDb and Rotten Tomatoes, whose terms of use are unclear for this copy: the
# files are downloaded at run time and never stored in this repository.
RAHA_COMMIT = "7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c"
MOVIES_URL = "https://raw.githubusercontent.com/BigDaMa/raha/{commit}/datasets/movies_1/{part}.csv"
MOVIES_SHA256 = {
    "clean": "22a84e70d153d809341c55cf8e8154beebe1836456f617bf8277b953a762bfca",
    "dirty": "bed752115dd0a5925ea36db443ff6835d10a5a8d2f549299535b14f8ac324316",
}
MOVIES_COUNTRIES = 5


def decode_year(text: str) -> float:
    """One year, or a range of three years written out, read as the middle one."""
    years = re.findall(r"\d{4}", text)
    if not years:
        return np.nan
    return float(years[1] if len(years) == 3 else years[0])


def decode_minutes(text: str) -> float:
    """`109 min` or `1 hr. 49 min.` as minutes. An empty cell is missing."""
    hours, minutes = re.search(r"(\d+)\s*hr", text), re.search(r"(\d+)\s*min", text)
    if not (hours or minutes):
        return np.nan
    return float((int(hours.group(1)) if hours else 0) * 60 + (int(minutes.group(1)) if minutes else 0))


def decode_count(text: str) -> float:
    digits = text.replace(",", "").strip()
    return float(digits) if digits else np.nan


def decode_reviews(text: str, kind: str) -> float:
    """`976 user,290 critic`. A missing part is read as no review (an assumption)."""
    found = re.search(rf"([\d,]+)\s*{kind}", text)
    return float(found.group(1).replace(",", "")) if found else 0.0


def first_item(text: str) -> str:
    return text.split(",")[0].strip() or "(none)"


def movie_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The same format decoding for both versions. It reads formats; it cleans nothing."""
    return pd.DataFrame(
        {
            "year": frame["Year"].map(decode_year),
            "duration": frame["Duration"].map(decode_minutes),
            "rating_count": frame["RatingCount"].map(decode_count),
            "user_reviews": frame["ReviewCount"].map(lambda text: decode_reviews(text, "user")),
            "critic_reviews": frame["ReviewCount"].map(lambda text: decode_reviews(text, "critic")),
            "genre": frame["Genre"].map(first_item),
            "country": frame["Country"].map(first_item),
        }
    )


def fetch_movies_part(part: str, data_home: Path = DATA_HOME) -> pd.DataFrame:
    path = data_home / "raha-movies" / f"{part}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MOVIES_URL.format(commit=RAHA_COMMIT, part=part), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MOVIES_SHA256[part]:
        raise ValueError(f"movies {part}: unexpected file, sha256 {digest}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_movies(data_home: Path = DATA_HOME) -> Table:
    """Movies with real errors. Target: the clean rating, so that no error reaches it.

    Rows without a clean rating or a clean duration are dropped, since the clean
    version must have no missing value. Errors in the dirty version stay as they are,
    a missing duration included.
    """
    clean_frame, dirty_frame = fetch_movies_part("clean", data_home), fetch_movies_part("dirty", data_home)
    if clean_frame.shape != dirty_frame.shape:
        raise ValueError("movies: the clean and dirty versions differ in shape")
    # Same columns in the same order; only the header is written differently.
    dirty_frame.columns = clean_frame.columns
    clean, dirty = movie_features(clean_frame), movie_features(dirty_frame)
    rating = clean_frame["RatingValue"].map(decode_count)
    kept = (rating.notna() & clean["duration"].notna()).to_numpy()
    clean, dirty, rating = clean[kept].reset_index(drop=True), dirty[kept].reset_index(drop=True), rating[kept]
    top = clean["country"].value_counts().index[:MOVIES_COUNTRIES]
    for frame in (clean, dirty):
        frame["country"] = frame["country"].where(frame["country"].isin(top), "(other)")
    categorical_names = ("genre", "country")
    columns_clean, columns_dirty = [], []
    for name in clean.columns:
        if name in categorical_names:
            categories = sorted(set(clean[name]) | set(dirty[name]))
            columns_clean.append(pd.Categorical(clean[name], categories=categories).codes.astype(np.float64))
            columns_dirty.append(pd.Categorical(dirty[name], categories=categories).codes.astype(np.float64))
        else:
            columns_clean.append(clean[name].to_numpy(dtype=np.float64))
            columns_dirty.append(dirty[name].to_numpy(dtype=np.float64))
    X, X_dirty = np.column_stack(columns_clean), np.column_stack(columns_dirty)
    if np.isnan(X).any():
        raise ValueError("movies: the clean version still has missing values")
    categorical = tuple(clean.columns.get_loc(name) for name in categorical_names)
    y = rating.to_numpy(dtype=np.float64)
    return Table("movies", X, y, tuple(clean.columns), "regression", categorical, X_dirty)


def load_table(name: str) -> Table:
    if name == "MagicTelescope":
        return load_magic_telescope()
    if name == "movies":
        return load_movies()
    return load_openml_table(name)


TABLE_NAMES = ("MagicTelescope", "movies", *OPENML_TABLES)


def split_indices(y: np.ndarray, split: int, test_size: float = 0.2, task: str = "classification"):
    """Train/test indices, determined by the split number. Stratified for classification."""
    if task == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=split)
        return next(splitter.split(np.zeros(len(y)), y))
    return next(ShuffleSplit(n_splits=1, test_size=test_size, random_state=split).split(np.zeros(len(y))))
