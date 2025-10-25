#!/usr/bin/env python3
"""Inspect features file and print dtypes and sample values for object columns."""
import sys
import pandas as pd

def inspect(path, n=5):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}\n")
    print("Column dtypes:\n")
    print(df.dtypes)
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    print('\nObject-typed columns (may contain strings/serialized dicts):')
    print(obj_cols)
    if obj_cols:
        print('\nSample values for first object columns:')
        for c in obj_cols[:10]:
            print(f"\n--- Column: {c} ---")
            print(df[c].head(n).to_list())

    # Print first row transposed for quick glance
    print('\nFirst row (transposed):')
    print(df.head(1).T)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_features.py <features.parquet|features.csv>')
        sys.exit(1)
    inspect(sys.argv[1])
