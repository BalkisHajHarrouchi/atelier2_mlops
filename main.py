"""CLI entrypoint driving the ML pipeline with step flags."""

from __future__ import annotations

import argparse
import json
import pandas as pd

from model_pipeline import (
    PrepOptions,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def _opts_from_args(args: argparse.Namespace) -> PrepOptions:
    """Build a PrepOptions object from CLI args."""
    return PrepOptions(
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        drop_cols=args.drop_cols,
        build_churn_from_playtime_2w=not args.no_build_churn,
        drop_proxies=args.drop_proxies,
        split_strategy=args.split_strategy,
        group_col=args.group_col,
        time_col=args.time_col,
        time_cutoff=args.time_cutoff,
        balance_test=True,
        n_per_class=2100,
    )


def _maybe_prepare(args: argparse.Namespace):
    """Optionally run preparation and print a short summary."""
    opts = _opts_from_args(args)
    bundle = prepare_data(args.data, opts)
    print(f"Train rows: {bundle.x_train.shape[0]} | Test rows: {bundle.x_test.shape[0]}")
    print(
        "Features — "
        f"numeric: {len(bundle.columns.numeric)}, "
        f"categorical: {len(bundle.columns.categorical)}, "
        f"boolean: {len(bundle.columns.boolean)}"
    )
    print(
        "Numeric:", bundle.columns.numeric[:15], "..." if len(bundle.columns.numeric) > 15 else ""
    )
    print(
        "Categorical:",
        bundle.columns.categorical[:15],
        "..." if len(bundle.columns.categorical) > 15 else "",
    )
    print(
        "Boolean:", bundle.columns.boolean[:15], "..." if len(bundle.columns.boolean) > 15 else ""
    )
    return bundle


def _maybe_train(args: argparse.Namespace, bundle):
    """Train a model using an existing or freshly prepared bundle."""
    if bundle is None:
        bundle = _maybe_prepare(args)
    model = train_model(bundle, random_state=args.seed)
    print(" Modèle entraîné")
    return model, bundle


def _maybe_evaluate(args: argparse.Namespace, model, bundle):
    """Evaluate a model; if needed, (re)create the test split to match options."""
    if model is None:
        model = load_model(args.model)
    if bundle is None:
        bundle = _maybe_prepare(args)
    metrics = evaluate_model(model, bundle.x_test, bundle.y_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n", metrics["report"])
    return model, bundle


def _maybe_save(args: argparse.Namespace, model):
    """Save a model if present."""
    if model is None:
        print(" Aucun modèle à sauvegarder. Lance d'abord --train_model")
        return None
    path = save_model(model, args.model)
    print(" Modèle sauvegardé :", path)
    return path


def _maybe_load(args: argparse.Namespace):
    """Just load the model to verify it exists/works."""
    loaded = load_model(args.model)
    print(" Modèle rechargé :", type(loaded))
    return loaded


def _maybe_predict(args: argparse.Namespace, model):
    """Predict on a new CSV, mirroring leakage drops used in preparation."""
    if model is None:
        model = load_model(args.model)
    df = pd.read_csv(args.predict_csv)

    for col in [
        args.target,
        "playtime_2weeks",
        "median_playtime_2weeks",
        "game_id",
        "name",
        "release_date",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if hasattr(model, "predict_proba") and args.threshold is not None:
        proba = model.predict_proba(df)[:, 1]
        preds = (proba >= args.threshold).astype(int)
        if args.proba_out:
            pd.DataFrame({"proba_active": proba}).to_csv(args.proba_out, index=False)
    else:
        preds = model.predict(df)

    out = args.out or "predictions.csv"
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    print(f"Predictions saved to {out}")


def run(parsed_args: argparse.Namespace) -> None:
    """Drive the pipeline based on provided flags."""
    bundle = None
    model = None

    if parsed_args.prepare_data:
        bundle = _maybe_prepare(parsed_args)

    if parsed_args.train_model:
        model, bundle = _maybe_train(parsed_args, bundle)

    if parsed_args.evaluate_model:
        model, bundle = _maybe_evaluate(parsed_args, model, bundle)

    if parsed_args.save_model:
        _maybe_save(parsed_args, model)

    if parsed_args.load_model:
        _maybe_load(parsed_args)

    if parsed_args.predict:
        _maybe_predict(parsed_args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple flag-driven ML pipeline")

    # IO (defaults aligned with your repo)
    parser.add_argument("--data", default="gaming_100mb.csv", help="Path to CSV")
    parser.add_argument(
        "--model", default="models/churn_model.joblib", help="Model path for save/load"
    )

    # Stage flags
    parser.add_argument("--prepare_data", action="store_true", help="Préparer les données")
    parser.add_argument("--train_model", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate_model", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save_model", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument("--load_model", action="store_true", help="Recharger le modèle")

    # Predict
    parser.add_argument("--predict", action="store_true", help="Prédire sur un nouveau CSV")
    parser.add_argument("--predict_csv", default="", help="CSV path for prediction")
    parser.add_argument("--out", default="predictions.csv", help="Where to save predictions")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for class=1 if model supports proba"
    )
    parser.add_argument("--proba_out", default="", help="Optional CSV to save probabilities")

    # Preparation/options
    parser.add_argument("--target", default="churn", help="Target column")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_cols", nargs="*", default=[], help="Extra columns to drop")
    parser.add_argument(
        "--no_build_churn", action="store_true", help="Do NOT build 'churn' from playtime_2weeks"
    )
    parser.add_argument(
        "--drop_proxies", action="store_true", help="Drop proxy features (reviews, players, ...)"
    )
    parser.add_argument("--split_strategy", choices=["random", "group", "time"], default="random")
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--time_cutoff", type=float, default=None)

    run(parser.parse_args())
