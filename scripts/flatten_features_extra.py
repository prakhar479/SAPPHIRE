#!/usr/bin/env python3
"""Flatten the `_extra` column in features CSV/Parquet into top-level columns.

Usage:
  python scripts/flatten_features_extra.py --input features/mirex/features.csv \
      --output features/mirex/features_flat.parquet

This script will try to safely parse string dictionaries using ast.literal_eval,
falling back to a JSON-style replace when needed.
"""
import argparse
import ast
import json
from pathlib import Path
import pandas as pd


def parse_maybe_dict(s):
    if pd.isna(s):
        return None
    # Already a dict
    if isinstance(s, dict):
        return s
    # Try literal_eval first (handles Python-style single quotes)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # Try JSON after replacing single quotes with double quotes (best-effort)
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        # give up
        return None


def flatten_extra(df, extra_col="_extra"):
    if extra_col not in df.columns:
        print(f"No column '{extra_col}' found. Nothing to flatten.")
        return df

    parsed = df[extra_col].apply(parse_maybe_dict)
    # Build DataFrame from parsed dicts
    extra_df = pd.json_normalize([d if isinstance(d, dict) else {} for d in parsed])
    # Prevent duplicate columns: if extra had track_id or others, keep them but
    # prefer existing top-level columns (so we don't overwrite feature columns)
    for col in extra_df.columns:
        if col in df.columns:
            # rename the extra column to avoid clobbering (shouldn't happen often)
            extra_df = extra_df.rename(columns={col: f"extra.{col}"})

    out = pd.concat([df.drop(columns=[extra_col]), extra_df], axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input features CSV or parquet")
    p.add_argument("--output", required=True, help="Output file path (parquet or csv)")
    p.add_argument(
        "--extra-column",
        default="_extra",
        help="Name of the column containing the dict (default: _extra)",
    )
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    if inp.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(inp)
    else:
        # CSV; allow large fields
        df = pd.read_csv(inp)

    print(f"Loaded {len(df)} rows from {inp}")
    df2 = flatten_extra(df, extra_col=args.extra_column)
    if "mood_cluster" in df2.columns:
        print("Found 'mood_cluster' after flattening.")
    else:
        print("Warning: 'mood_cluster' not found after flattening.")

    # Write output
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix.lower() in [".parquet", ".pq"]:
        df2.to_parquet(outp, index=False)
    else:
        df2.to_csv(outp, index=False)

    print(f"Wrote flattened features to {outp}")


if __name__ == "__main__":
    main()
