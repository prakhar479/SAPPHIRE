#!/usr/bin/env python3
"""Normalize serialized-dict feature columns into numeric scalar columns.

This script reads a features parquet/csv, finds object-typed columns that contain
stringified Python dicts (common from the feature extractor), parses them (with
some tolerant fixes for `np.float64(...)` tokens), expands them via
pd.json_normalize, and writes a cleaned parquet file.
"""
import sys
from pathlib import Path
import pandas as pd
import ast
import json
import re


def parse_maybe_dict(s: object):
    """Try to parse s into a dict. Returns dict or None if not parseable."""
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None

    s2 = s.strip()
    if s2 == "" or s2 == "{}":
        return {}

    # Replace numpy wrappers like np.float64(4.23) -> 4.23
    s2 = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s2)

    # Try ast.literal_eval (works for Python repr with single quotes)
    try:
        obj = ast.literal_eval(s2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try JSON (if it uses double quotes)
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def normalize(path_in: str, path_out: str = None):
    path = Path(path_in)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Identify object columns that likely hold serialized dicts
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    # Candidate feature columns to expand
    candidates = [
        c
        for c in obj_cols
        if any(
            k in c.lower()
            for k in ("acoustic", "rhythm", "harmony", "lyrics", "quality", "metadata")
        )
    ]

    print(f"Found object columns: {obj_cols}")
    print(f"Will attempt to normalize: {candidates}")

    out_df = df.copy()

    for col in candidates:
        print(f"Parsing column: {col}")
        parsed = out_df[col].apply(parse_maybe_dict)
        n_ok = parsed.apply(lambda x: isinstance(x, dict)).sum()
        print(f"  Parsed {n_ok}/{len(parsed)} rows as dicts")
        if n_ok == 0:
            print(f"  Skipping column {col} (no parseable dicts)")
            continue

        # Create normalized frame (fill with None for missing)
        normalized = pd.json_normalize(
            [p if isinstance(p, dict) else {} for p in parsed]
        )
        # Prefix columns
        normalized = normalized.add_prefix(col + ".")

        # Merge into out_df, drop original column
        out_df = pd.concat(
            [out_df.reset_index(drop=True), normalized.reset_index(drop=True)], axis=1
        )
        out_df = out_df.drop(columns=[col])

    # Optionally, coerce types for numeric columns
    for c in out_df.columns:
        if out_df[c].dtype == object:
            # Try to coerce to numeric, ignore errors
            try:
                out_df[c] = pd.to_numeric(out_df[c], errors="ignore")
            except Exception:
                pass

    # Write output
    if path_out is None:
        path_out = path.with_name(path.stem + "_clean.parquet")

    out_df.to_parquet(path_out, index=False)
    print(
        f"Wrote cleaned features to {path_out} (rows: {len(out_df)}, cols: {len(out_df.columns)})"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: normalize_feature_columns.py <in.parquet|in.csv> [out.parquet]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    normalize(inp, out)
