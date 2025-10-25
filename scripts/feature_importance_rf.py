#!/usr/bin/env python3
"""Compute and save feature importances from a RandomForest classifier.

This script loads an extracted-features file (parquet or csv), trains a
RandomForestClassifier on the specified label (default 'mood_cluster'), and
outputs a CSV and optional bar plot of the top-N most important features.

Usage:
  python scripts/feature_importance_rf.py --features path/to/features.parquet --output out_dir --top 50

"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def prepare_X_y(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in features")

    df = df.copy()
    # Drop columns that are clearly metadata / non-features
    metadata_cols = ["track_id", "mood_cluster", "mood_category", "dataset"]
    for c in metadata_cols:
        if c in df.columns and c != label_col:
            df.drop(columns=[c], inplace=True)

    # Defensive: drop identifier-like columns that may leak information (file indices, track ids, etc.)
    id_like = [
        c
        for c in df.columns
        if (
            c == "track_id"
            or "file_index" in c
            or c.endswith(".track_id")
            or c.endswith(".file_index")
            or c == "index"
            or c.lower() == "index"
            or c == "extra.track_id"
        )
        and c != label_col
    ]
    if id_like:
        print(f"🔒 Dropping identifier-like columns to avoid leakage: {id_like}")
        df.drop(columns=id_like, inplace=True)

    # Drop constant columns (no information)
    const_cols = [
        c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != label_col
    ]
    if const_cols:
        print(f"ℹ️  Dropping constant columns: {const_cols}")
        df.drop(columns=const_cols, inplace=True)

    # Separate label
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # Keep only numeric columns for RF (avoid unexpected dtypes)
    numeric_X = X.select_dtypes(include=[np.number])

    # Final safety: remove any numeric columns that look like identifiers (e.g. file_index may be numeric)
    id_like_numeric = [
        c
        for c in numeric_X.columns
        if "file_index" in c or c.endswith(".file_index") or c.endswith(".track_id")
    ]
    if id_like_numeric:
        print(
            f"🔒 Removing numeric identifier-like columns from features: {id_like_numeric}"
        )
        numeric_X = numeric_X.drop(columns=id_like_numeric)

    if numeric_X.shape[1] == 0:
        raise RuntimeError(
            "No numeric feature columns found after filtering. Check your features file."
        )

    return numeric_X, y


def main():
    parser = argparse.ArgumentParser(
        description="RandomForest feature importance for mood classification"
    )
    parser.add_argument(
        "--features", "-f", required=True, help="Path to features parquet/csv"
    )
    parser.add_argument(
        "--label", "-l", default="mood_cluster", help="Label column name"
    )
    parser.add_argument(
        "--top", "-t", type=int, default=30, help="Number of top features to show/save"
    )
    parser.add_argument(
        "--output", "-o", default="output/feature_importance", help="Output directory"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="RandomForest n_estimators"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_path}")
    df = load_features(features_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    X, y = prepare_X_y(df, args.label)
    print(f"Using {X.shape[1]} numeric features for training")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split for a quick validation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y_enc if len(np.unique(y_enc)) > 1 else None,
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    importances = clf.feature_importances_
    feat_names = X.columns.tolist()

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    imp_df.reset_index(drop=True, inplace=True)

    topk = imp_df.head(args.top)
    csv_out = out_dir / f"top_{args.top}_feature_importances.csv"
    topk.to_csv(csv_out, index=False)
    print(f"Saved top-{args.top} features to: {csv_out}")

    # Save full importances
    full_out = out_dir / "feature_importances_full.csv"
    imp_df.to_csv(full_out, index=False)

    # Save label mapping
    mapping_out = out_dir / "label_mapping.json"
    with open(mapping_out, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f, indent=2)

    # Plot top features
    plt.figure(figsize=(10, max(4, args.top * 0.25)))
    plt.barh(topk["feature"][::-1], topk["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {args.top} feature importances (RandomForest)")
    plt.tight_layout()
    plot_out = out_dir / f"top_{args.top}_feature_importances.png"
    plt.savefig(plot_out, dpi=150)
    print(f"Saved plot to: {plot_out}")

    print("Done.")


if __name__ == "__main__":
    main()
