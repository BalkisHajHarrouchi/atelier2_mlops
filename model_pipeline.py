# model_pipeline.py
"""Model pipeline: prepare, train, evaluate, save, load, and predict.

- Forces exact undersampling to `n_per_class` per label (default 2100) for train,
  and optionally for test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import os
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
from imblearn.under_sampling import RandomUnderSampler


# ----------------------------
# Typed containers
# ----------------------------
@dataclass
class FeatureColumns:
    """Holds feature column names by type."""

    numeric: List[str]
    categorical: List[str]
    boolean: List[str]


@dataclass
class DataBundle:
    """All data artifacts needed across pipeline steps."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    columns: FeatureColumns
    preprocessor: ColumnTransformer


@dataclass
class PrepOptions:
    """Preparation options collected in one object to avoid long signatures."""

    target: str = "churn"
    test_size: float = 0.2
    random_state: int = 42
    drop_cols: Optional[List[str]] = None
    build_churn_from_playtime_2w: bool = True
    drop_proxies: bool = False
    split_strategy: str = "random"  # "random" | "group" | "time"
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    time_cutoff: Optional[float] = None
    balance_test: bool = True
    n_per_class: int = 2100


PROXY_COLS = [
    "playtime_forever",
    "median_playtime_forever",
    "reviews",
    "positive_reviews",
    "negative_reviews",
    "recommendations",
    "players",
]


def _is_bool_series(series: pd.Series) -> bool:
    """Heuristic to detect boolean-like columns even if typed as object."""
    if series.dtype == bool:
        return True
    values = set(map(str, series.dropna().unique()))
    return values.issubset({"True", "False", "true", "false", "0", "1"})


def _ordinal_encoder(dtype=np.int32) -> OrdinalEncoder:
    """Ordinal encoder with robust handling of unknown/missing values."""
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=dtype,
        )
    except TypeError:
        # Older sklearn versions may not support encoded_missing_value
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=dtype,
        )


def _build_preprocessor(
    numeric: List[str], categorical: List[str], boolean: List[str]
) -> ColumnTransformer:
    """Builds a ColumnTransformer for numeric/boolean/categorical pipelines."""
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", _ordinal_encoder()),
        ]
    )
    bool_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", _ordinal_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("bool", bool_pipe, boolean),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )


def _undersample_to_exact_n(
    x_df: pd.DataFrame, y_ser: pd.Series, n_per_class: int = 2100, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Force each class to have EXACTLY n_per_class samples. Error if any class < n_per_class."""
    counts = y_ser.value_counts()
    too_small = {int(cls): int(cnt) for cls, cnt in counts.items() if cnt < n_per_class}
    if too_small:
        raise ValueError(
            f"Cannot undersample to {n_per_class} per class. "
            f"Class counts in this split: {counts.to_dict()}"
        )
    sampling_strategy = {cls: n_per_class for cls in counts.index}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    x_res, y_res = rus.fit_resample(x_df, y_ser)
    return x_res, y_res


def prepare_data(csv_path: str, opts: PrepOptions) -> DataBundle:
    """Prepare data: load, build target if needed, split, balance, and build preprocessor."""
    df = pd.read_csv(csv_path)

    # Deduplicate
    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id"])
    else:
        df = df.drop_duplicates()

    # Auto-build churn
    if opts.build_churn_from_playtime_2w:
        if "playtime_2weeks" not in df.columns:
            raise ValueError("Column 'playtime_2weeks' missing for churn construction.")
        df[opts.target] = (
            pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0) > 0
        ).astype(int)

    if opts.target not in df.columns:
        raise ValueError(f"Target column '{opts.target}' not found.")

    # Drop irrelevant
    always_drop = {
        "game_id",
        "name",
        "release_date",
        "playtime_2weeks",
        "median_playtime_2weeks",
        opts.target,
    }
    if opts.drop_cols:
        always_drop |= set(opts.drop_cols)
    if opts.drop_proxies:
        always_drop |= set(PROXY_COLS)

    y_all = df[opts.target]
    x_all = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    # Drop no-variance
    bad_cols = [
        c for c in x_all.columns if x_all[c].isna().all() or x_all[c].nunique(dropna=True) <= 1
    ]
    if bad_cols:
        x_all = x_all.drop(columns=bad_cols)

    numeric = x_all.select_dtypes(include=["number"]).columns.tolist()
    boolean = [c for c in x_all.columns if c not in numeric and _is_bool_series(x_all[c])]
    categorical = [c for c in x_all.columns if c not in numeric and c not in boolean]
    columns = FeatureColumns(numeric=numeric, categorical=categorical, boolean=boolean)

    preprocessor = _build_preprocessor(numeric, categorical, boolean)

    # Split
    idx = np.arange(len(x_all))
    if opts.split_strategy == "group":
        if not opts.group_col or opts.group_col not in df.columns:
            raise ValueError("split_strategy='group' requires a valid group_col.")
        groups = df[opts.group_col].values
        gss = GroupShuffleSplit(
            n_splits=1, test_size=opts.test_size, random_state=opts.random_state
        )
        train_idx, test_idx = next(gss.split(idx, y_all.values, groups))
    elif opts.split_strategy == "time":
        if not opts.time_col or opts.time_col not in df.columns or opts.time_cutoff is None:
            raise ValueError("split_strategy='time' requires time_col and time_cutoff.")
        mask = pd.to_numeric(df[opts.time_col], errors="coerce") < float(opts.time_cutoff)
        train_idx, test_idx = np.where(mask)[0], np.where(~mask)[0]
    else:
        train_idx, test_idx = train_test_split(
            idx, test_size=opts.test_size, random_state=opts.random_state, stratify=y_all
        )

    x_train, x_test = x_all.iloc[train_idx], x_all.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

    # Undersample to exact n per class
    x_train, y_train = _undersample_to_exact_n(
        x_train, y_train, n_per_class=opts.n_per_class, random_state=opts.random_state
    )

    if opts.balance_test:
        x_test, y_test = _undersample_to_exact_n(
            x_test, y_test, n_per_class=opts.n_per_class, random_state=opts.random_state
        )

    return DataBundle(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        columns=columns,
        preprocessor=preprocessor,
    )


def prepare_for_cv(
    csv_path: str, opts: PrepOptions
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Prepare full dataset (optionally balanced) and a preprocessor for CV."""
    df = pd.read_csv(csv_path)

    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id"])
    else:
        df = df.drop_duplicates()

    if opts.build_churn_from_playtime_2w:
        if "playtime_2weeks" not in df.columns:
            raise ValueError("Column 'playtime_2weeks' missing for churn construction.")
        df[opts.target] = (
            pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0) > 0
        ).astype(int)

    if opts.target not in df.columns:
        raise ValueError(f"Target column '{opts.target}' not found.")

    always_drop = {
        "game_id",
        "name",
        "release_date",
        "playtime_2weeks",
        "median_playtime_2weeks",
        opts.target,
    }
    if opts.drop_cols:
        always_drop |= set(opts.drop_cols)
    if opts.drop_proxies:
        always_drop |= set(PROXY_COLS)

    y_all = df[opts.target]
    x_all = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")

    bad_cols = [
        c for c in x_all.columns if x_all[c].isna().all() or x_all[c].nunique(dropna=True) <= 1
    ]
    if bad_cols:
        x_all = x_all.drop(columns=bad_cols)

    numeric = x_all.select_dtypes(include=["number"]).columns.tolist()
    boolean = [c for c in x_all.columns if c not in numeric and _is_bool_series(x_all[c])]
    categorical = [c for c in x_all.columns if c not in numeric and c not in boolean]

    if opts.balance_test:
        x_all, y_all = _undersample_to_exact_n(
            x_all, y_all, n_per_class=opts.n_per_class, random_state=opts.random_state
        )

    preprocessor = _build_preprocessor(numeric, categorical, boolean)
    return x_all, y_all, preprocessor


def train_model(data: DataBundle, random_state: int = 42) -> Pipeline:
    """Train a RandomForest inside a preprocessing pipeline."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
    )
    pipe = Pipeline(
        [
            ("prep", data.preprocessor),
            ("model", model),
        ]
    )
    pipe.fit(data.x_train, data.y_train)
    return pipe


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Return accuracy/F1 metrics and a text report."""
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred, digits=4),
    }


def save_model(model: Pipeline, path: str) -> str:
    """Persist the trained pipeline and return its absolute path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    return os.path.abspath(path)


def load_model(path: str) -> Pipeline:
    """Load a previously saved pipeline."""
    return joblib.load(path)


# ===== Prediction helpers =====

LEAK_COLS = {
    "game_id",
    "name",
    "release_date",
    "playtime_2weeks",
    "median_playtime_2weeks",
}


def drop_leakage_columns(
    df: pd.DataFrame, target: str, extra_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """Drop target and known leakage columns if present."""
    cols_to_drop = set(LEAK_COLS) | {target}
    if extra_drop:
        cols_to_drop |= set(extra_drop)
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


def expected_feature_columns_from_model(model: Pipeline) -> Optional[List[str]]:
    """
    Try to recover the original feature names used during training from the
    ColumnTransformer selection.
    """
    prep = model.named_steps.get("prep")
    if prep is None:
        return None
    transformers = getattr(prep, "transformers", None) or getattr(prep, "transformers_", None)
    if not transformers:
        return None
    cols: List[str] = []
    for _, _, sel in transformers:
        if sel is None or sel == "drop":
            continue
        if isinstance(sel, (list, tuple)):
            cols.extend(list(sel))
        else:
            try:
                cols.extend(list(sel))
            except Exception:
                pass
    return cols


def align_features_for_model(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Ensure df has all columns the model expects; create missing columns with NaN and order them.
    Extra columns are safe (remainder='drop').
    """
    expected = expected_feature_columns_from_model(model)
    if not expected:
        return df
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[expected]


def predict_dataframe(
    model: Pipeline,
    df: pd.DataFrame,
    target: str = "churn",
    threshold: Optional[float] = None,
    return_proba: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels for a DataFrame. If threshold is provided and model supports predict_proba,
    return thresholded labels and (optionally) the raw probabilities.
    """
    df = drop_leakage_columns(df, target)
    df = align_features_for_model(df, model)
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
        preds = (proba >= float(threshold)).astype(int)
        return preds, (proba if return_proba else None)
    preds = model.predict(df)
    return preds, None


def predict_from_csv(
    model: Pipeline,
    csv_path: str,
    target: str = "churn",
    threshold: Optional[float] = None,
    return_proba: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load CSV, then call predict_dataframe."""
    df = pd.read_csv(csv_path)
    return predict_dataframe(model, df, target=target, threshold=threshold, return_proba=return_proba)


def predict_to_csv(
    model: Pipeline,
    csv_path: str,
    out_path: str = "predictions.csv",
    proba_out_path: Optional[str] = None,
    target: str = "churn",
    threshold: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """
    Convenience I/O wrapper: predict from a CSV and write outputs to disk.
    Returns (predictions_csv_path, probabilities_csv_path_or_None).
    """
    preds, proba = predict_from_csv(
        model,
        csv_path,
        target=target,
        threshold=threshold,
        return_proba=bool(proba_out_path),
    )

    pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)

    proba_path = None
    if proba_out_path:
        pd.DataFrame({"proba_active": proba}).to_csv(proba_out_path, index=False)
        proba_path = proba_out_path

    return out_path, proba_path
