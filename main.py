# main.py
import argparse
import json
import pandas as pd

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    get_selected_features,
    prepare_for_cv,
)

from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate

def cmd_prepare(args):
    bundle = prepare_data(
        csv_path=args.csv,
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
    )
    print(f"Train rows: {bundle.X_train.shape[0]} | Test rows: {bundle.X_test.shape[0]}")
    print(f"Features — numeric: {len(bundle.numeric)}, categorical: {len(bundle.categorical)}, boolean: {len(bundle.boolean)}")
    print("Numeric:", bundle.numeric[:15], "..." if len(bundle.numeric) > 15 else "")
    print("Categorical:", bundle.categorical[:15], "..." if len(bundle.categorical) > 15 else "")
    print("Boolean:", bundle.boolean[:15], "..." if len(bundle.boolean) > 15 else "")

def cmd_train(args):
    bundle = prepare_data(
        csv_path=args.csv,
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
    )
    model = train_model(
        bundle,
        kbest_k=args.kbest,
        kbest_score=args.kbest_score,
        random_state=args.seed,
    )
    save_model(model, args.out)
    print(f"Model saved to {args.out}")

    metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n", metrics["report"])

    if args.kbest and args.kbest > 0:
        feats = get_selected_features(model, bundle)
        print("\nTop selected features (name, score):")
        for name, score in feats[:min(40, len(feats))]:
            print(f"{name:30s}  {score:.5f}")

def cmd_evaluate(args):
    model = load_model(args.model)
    bundle = prepare_data(
        csv_path=args.csv,
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
    )
    metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print("\nClassification report:\n", metrics["report"])

def cmd_predict(args):
    model = load_model(args.model)
    df = pd.read_csv(args.csv)

    # Drop leakage/ID columns if present (mirror prepare_data)
    for col in [args.target, "playtime_2weeks", "median_playtime_2weeks", "game_id", "name", "release_date"]:
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

def cmd_cv(args):
    # Prepare full dataset (no split here)
    X, y, pre = prepare_for_cv(
        csv_path=args.csv,
        target=args.target,
        drop_cols=args.drop_cols,
        build_churn_from_playtime_2w=not args.no_build_churn,
        drop_proxies=args.drop_proxies,
    )

    # Build an imblearn pipeline so undersampling happens inside each fold
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler
    except Exception as e:
        raise ImportError(
            "imbalanced-learn is required for CV with enforced undersampling. "
            "Install with: pip install imbalanced-learn"
        ) from e

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

    steps = [("prep", pre)]
    if args.kbest and args.kbest > 0:
        k = min(int(args.kbest), X.shape[1])
        score_func = mutual_info_classif if args.kbest_score == "mutual_info" else f_classif
        steps.append(("select", SelectKBest(score_func=score_func, k=k)))

    steps.append(("sampler", RandomUnderSampler(random_state=args.seed)))  # ALWAYS undersample in CV
    steps.append(("model", RandomForestClassifier(
        n_estimators=300, random_state=args.seed, n_jobs=-1, class_weight=None
    )))

    pipe = ImbPipeline(steps)

    if args.group_col and args.group_col in X.columns:
        groups = X[args.group_col].values
        cv = GroupKFold(n_splits=args.folds)
        splits = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = cv.split(X, y)

    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    res = cross_validate(pipe, X, y, cv=list(splits), scoring=scoring, n_jobs=-1, return_train_score=True)

    import numpy as np
    def mean_std(key):
        return float(np.mean(res[key])), float(np.std(res[key]))

    acc_m, acc_s = mean_std("test_accuracy")
    f1m_m, f1m_s = mean_std("test_f1_macro")
    f1w_m, f1w_s = mean_std("test_f1_weighted")

    print(json.dumps({
        "cv_folds": args.folds,
        "accuracy_mean": acc_m, "accuracy_std": acc_s,
        "f1_macro_mean": f1m_m, "f1_macro_std": f1m_s,
        "f1_weighted_mean": f1w_m, "f1_weighted_std": f1w_s
    }, indent=2))

def build_parser():
    p = argparse.ArgumentParser(description="Churn classification pipeline (preprocessing done in preparation).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_io(sp):
        sp.add_argument("--csv", required=True, help="Path to CSV data file")
        sp.add_argument("--target", default="churn", help="Target column name (default: churn)")
        sp.add_argument("--test-size", type=float, default=0.2)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--drop-cols", nargs="*", default=[], help="Extra columns to drop from features")
        sp.add_argument("--no-build-churn", action="store_true", help="Do NOT auto-build 'churn' from playtime_2weeks")

        # Overfitting controls / data hygiene
        sp.add_argument("--drop-proxies", action="store_true",
                        help="Drop near-label proxy features (playtime_forever, reviews, players, ...)")
        sp.add_argument("--split-strategy", choices=["random", "group", "time"], default="random",
                        help="random (default), group by a column, or time-based split")
        sp.add_argument("--group-col", default=None, help="Column for group split (e.g., developer)")
        sp.add_argument("--time-col", default=None, help="Time column for time split (e.g., year)")
        sp.add_argument("--time-cutoff", type=float, default=None,
                        help="Train where time_col < cutoff; test where >= cutoff (with --split-strategy time)")

        # Feature selection
        sp.add_argument("--kbest", type=int, default=0, help="SelectKBest top-K features (0=off)")
        sp.add_argument("--kbest-score", choices=["mutual_info", "f_classif"], default="mutual_info",
                        help="Scoring for KBest (default: mutual_info)")

        # NOTE: balancing is no longer optional; we always undersample during training & CV.

    sp = sub.add_parser("prepare", help="Prepare: split, infer columns, and BUILD PREPROCESSOR")
    common_io(sp)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="Train and save model (uses preprocessor from preparation)")
    common_io(sp)
    sp.add_argument("--out", default="churn_model.joblib", help="Output model path")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Load model and evaluate on freshly prepared split")
    common_io(sp)
    sp.add_argument("--model", required=True, help="Path to saved model")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("predict", help="Predict labels for a new CSV (no target needed)")
    sp.add_argument("--csv", required=True, help="Path to CSV without target")
    sp.add_argument("--model", required=True, help="Path to saved model")
    sp.add_argument("--out", default="predictions.csv", help="Where to save predictions CSV")
    sp.add_argument("--target", default="churn")
    sp.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class=1 (if model supports)")
    sp.add_argument("--proba-out", default="", help="Optional CSV path to save predicted probabilities")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("cv", help="Cross-validate with K folds (stratified or grouped)")
    common_io(sp)                  # includes --group-col
    sp.add_argument("--folds", type=int, default=5)
    sp.set_defaults(func=cmd_cv)

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)