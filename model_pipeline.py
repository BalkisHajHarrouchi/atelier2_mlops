from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# ----------------------------
# Data container
# ----------------------------
@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric: List[str]
    categorical: List[str]
    boolean: List[str]
    preprocessor: ColumnTransformer

PROXY_COLS = [
    "playtime_forever", "median_playtime_forever",
    "reviews", "positive_reviews", "negative_reviews",
    "recommendations", "players",
]

def _is_bool_series(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    vals = set(map(str, s.dropna().unique()))
    return vals.issubset({"True", "False", "true", "false", "0", "1"})

def _ordinal_encoder(dtype=np.int32):
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=dtype,
        )
    except TypeError:
        # Fallback for older sklearn
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=dtype,
        )

def _build_preprocessor(numeric: List[str], categorical: List[str], boolean: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", _ordinal_encoder()),
    ])
    bool_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", _ordinal_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num",  num_pipe,  numeric),
            ("bool", bool_pipe, boolean),
            ("cat",  cat_pipe,  categorical),
        ],
        remainder="drop",
    )

def _undersample_to_2100(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Force each class to have exactly 2100 samples.
    If a class has fewer than 2100 samples, it will raise an error.
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except Exception as e:
        raise ImportError(
            "imbalanced-learn is required for balancing to 2100 samples per class. "
            "Install with: pip install imbalanced-learn"
        ) from e

    class_counts = y.value_counts()
    if (class_counts < 2100).any():
        raise ValueError(
            f"Cannot undersample to 2100 because some classes have fewer than 2100 samples: {class_counts.to_dict()}"
        )
    sampling_strategy = {cls: 2100 for cls in class_counts.index}
    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    return sampler.fit_resample(X, y)

def prepare_data(
    csv_path: str,
    target: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_cols: Optional[List[str]] = None,
    build_churn_from_playtime_2w: bool = True,
    drop_proxies: bool = False,
    split_strategy: str = "random",
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    time_cutoff: Optional[float] = None,
    balance_test: bool = True,    # ✅ balance test to 2100/class
) -> DataBundle:

    df = pd.read_csv(csv_path)

    # Remove duplicates
    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id"])
    else:
        df = df.drop_duplicates()

    # Build churn column if needed
    if build_churn_from_playtime_2w:
        if "playtime_2weeks" not in df.columns:
            raise ValueError("Column 'playtime_2weeks' missing for churn construction.")
        df[target] = (pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0) > 0).astype(int)

    assert target in df.columns, f"Target column '{target}' not found."

    # Drop irrelevant columns
    always_drop = {
        "game_id", "name", "release_date",
        "playtime_2weeks", "median_playtime_2weeks", target,
    }
    if drop_cols:
        always_drop |= set(drop_cols)
    if drop_proxies:
        always_drop |= set(PROXY_COLS)

    y = df[target]
    X = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    # Drop columns with no variance
    to_drop = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if to_drop:
        X = X.drop(columns=to_drop)

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    boolean = [c for c in X.columns if c not in numeric and _is_bool_series(X[c])]
    categorical = [c for c in X.columns if c not in numeric and c not in boolean]

    preprocessor = _build_preprocessor(numeric, categorical, boolean)

    # Train/Test split
    idx = np.arange(len(X))
    if split_strategy == "group":
        if not group_col or group_col not in df.columns:
            raise ValueError("split_strategy='group' requires a valid group_col.")
        groups = df[group_col].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(idx, y.values, groups))
    elif split_strategy == "time":
        if not time_col or time_col not in df.columns or time_cutoff is None:
            raise ValueError("split_strategy='time' requires time_col and time_cutoff.")
        mask = pd.to_numeric(df[time_col], errors="coerce") < float(time_cutoff)
        train_idx, test_idx = np.where(mask)[0], np.where(~mask)[0]
    else:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ✅ Balance TRAIN to 2100/class
    X_train, y_train = _undersample_to_2100(X_train, y_train, random_state)

    # ✅ Balance TEST to 2100/class (if enabled)
    if balance_test:
        X_test, y_test = _undersample_to_2100(X_test, y_test, random_state)

    return DataBundle(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric=numeric, categorical=categorical, boolean=boolean,
        preprocessor=preprocessor,
    )

def prepare_for_cv(
    csv_path: str,
    target: str = "churn",
    drop_cols: Optional[List[str]] = None,
    build_churn_from_playtime_2w: bool = True,
    drop_proxies: bool = False,
    balance: bool = True  # ✅ balance full dataset for CV
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = pd.read_csv(csv_path)

    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id"])
    else:
        df = df.drop_duplicates()

    if build_churn_from_playtime_2w:
        if "playtime_2weeks" not in df.columns:
            raise ValueError("Column 'playtime_2weeks' missing for churn construction.")
        df[target] = (pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0) > 0).astype(int)

    assert target in df.columns, f"Target column '{target}' not found."

    always_drop = {"game_id", "name", "release_date", "playtime_2weeks", "median_playtime_2weeks", target}
    if drop_cols:
        always_drop |= set(drop_cols)
    if drop_proxies:
        always_drop |= set(PROXY_COLS)

    y = df[target]
    X = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    bad = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if bad:
        X = X.drop(columns=bad)

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    boolean = [c for c in X.columns if c not in numeric and _is_bool_series(X[c])]
    categorical = [c for c in X.columns if c not in numeric and c not in boolean]

    if balance:
        X, y = _undersample_to_2100(X, y, random_state=42)

    preprocessor = _build_preprocessor(numeric, categorical, boolean)
    return X, y, preprocessor

def train_model(
    data: DataBundle,
    model: Optional[object] = None,
    kbest_k: int = 0,
    kbest_score: str = "mutual_info",
    random_state: int = 42,
) -> Pipeline:
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight=None,
        )

    steps = [("prep", data.preprocessor)]
    if kbest_k and kbest_k > 0:
        base_dim = len(data.numeric) + len(data.boolean) + len(data.categorical)
        k = min(int(kbest_k), base_dim)
        score_func = mutual_info_classif if kbest_score == "mutual_info" else f_classif
        steps.append(("select", SelectKBest(score_func=score_func, k=k)))
    steps.append(("model", model))
    pipe = Pipeline(steps)

    # Train directly (data already balanced to 2100/class)
    pipe.fit(data.X_train, data.y_train)
    return pipe

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred, digits=4),
    }

def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)

def get_selected_features(pipeline: Pipeline, data: DataBundle):
    sel = pipeline.named_steps.get("select")
    if sel is None:
        return []
    base_names = data.numeric + data.boolean + data.categorical
    support = sel.get_support()
    scores = sel.scores_
    selected = [(name, float(scores[i])) for i, name in enumerate(base_names) if support[i]]
    selected.sort(key=lambda x: -x[1])
    return selected
