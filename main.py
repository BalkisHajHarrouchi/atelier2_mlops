# main.py
"""CLI entrypoint driving the ML pipeline with step flags."""

from __future__ import annotations

import argparse
import json

from model_pipeline import (
    PrepOptions,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_to_csv,
)


def opts_from_args(args: argparse.Namespace) -> PrepOptions:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple flag-driven ML pipeline")

    # IO (defaults aligned with your repo)
    parser.add_argument("--data", default="gaming_100mb.csv", help="Path to CSV")
    parser.add_argument("--model", default="models/churn_model.joblib", help="Model path for save/load")

    # Stage flags
    parser.add_argument("--prepare_data", action="store_true", help="Préparer les données")
    parser.add_argument("--train_model", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate_model", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save_model", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument("--load_model", action="store_true", help="Recharger le modèle")

    # Predict (thin)
    parser.add_argument("--predict", action="store_true", help="Prédire sur un nouveau CSV")
    parser.add_argument("--predict_csv", default="", help="CSV path for prediction")
    parser.add_argument("--out", default="predictions.csv", help="Where to save predictions")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,  # plain labels by default
        help="If set and model supports proba, threshold class=1 at this value",
    )
    parser.add_argument("--proba_out", default="", help="Optional CSV to save probabilities")

    # Preparation/options
    parser.add_argument("--target", default="churn", help="Target column")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_cols", nargs="*", default=[], help="Extra columns to drop")
    parser.add_argument("--no_build_churn", action="store_true", help="Do NOT build 'churn' from playtime_2weeks")
    parser.add_argument("--drop_proxies", action="store_true", help="Drop proxy features (reviews, players, ...)")
    parser.add_argument("--split_strategy", choices=["random", "group", "time"], default="random")
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--time_cutoff", type=float, default=None)

    args = parser.parse_args()

    # ===== Straight flow (no _maybe_* wrappers) =====
    bundle = None
    model = None

    if args.prepare_data:
        bundle = prepare_data(args.data, opts_from_args(args))
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
            "Categorical:", bundle.columns.categorical[:15],
            "..." if len(bundle.columns.categorical) > 15 else "",
        )
        print(
            "Boolean:", bundle.columns.boolean[:15], "..." if len(bundle.columns.boolean) > 15 else ""
        )

    if args.train_model:
        if bundle is None:
            bundle = prepare_data(args.data, opts_from_args(args))
        model = train_model(bundle, random_state=args.seed)
        print(" Modèle entraîné")

    if args.evaluate_model:
        if model is None:
            model = load_model(args.model)
        if bundle is None:
            bundle = prepare_data(args.data, opts_from_args(args))
        metrics = evaluate_model(model, bundle.x_test, bundle.y_test)
        print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
        print("\nClassification report:\n", metrics["report"])

    if args.save_model:
        if model is None:
            print(" Aucun modèle à sauvegarder. Lance d'abord --train_model")
        else:
            path = save_model(model, args.model)
            print(" Modèle sauvegardé :", path)

    if args.load_model:
        loaded = load_model(args.model)
        print(" Modèle rechargé :", type(loaded))

    if args.predict:
        if not args.predict_csv:
            raise ValueError("--predict requires --predict_csv=PATH")
        if model is None:
            model = load_model(args.model)
        preds_csv, proba_csv = predict_to_csv(
            model,
            csv_path=args.predict_csv,
            out_path=args.out,
            proba_out_path=(args.proba_out or None),
            target=args.target,
            threshold=args.threshold,
        )
        msg = f"Predictions saved to {preds_csv}"
        if proba_csv:
            msg += f" | Probabilities saved to {proba_csv}"
        print(msg)


if __name__ == "__main__":
    main()
