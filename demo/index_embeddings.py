"""Simple CLI to build an Annoy index from precomputed embeddings.

Usage: python demo/index_embeddings.py --emb path/to/embeddings.npy --meta path/to/metadata.json
If not provided it will look in `models/embedder_mirex/embeddings/` and `demo/`.
"""

import argparse
import os
import numpy as np
import json
from annoy import AnnoyIndex

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(DEMO_DIR, "index.ann")
EMB_PATH = os.path.join(DEMO_DIR, "embeddings.npy")
META_PATH = os.path.join(DEMO_DIR, "metadata.json")


def find_embeddings(candidate=None):
    # Return tuple (embeddings_array, metadata_dict_or_None)
    if candidate and os.path.exists(candidate):
        emb = np.load(candidate)
        return emb, None
    candidates = [
        os.path.join(BASE_DIR, "models", "embedder_mirex", "embeddings"),
    ]
    for embed_folder in candidates:
        np_path = os.path.join(embed_folder, "embeddings.npy")
        pqt_path = os.path.join(embed_folder, "song_embeddings.parquet")
        csv_path = os.path.join(embed_folder, "song_embeddings.csv")
        if os.path.exists(np_path):
            emb = np.load(np_path)
            return emb, None
        if os.path.exists(pqt_path) or os.path.exists(csv_path):
            try:
                import pandas as pd

                if os.path.exists(pqt_path):
                    df = pd.read_parquet(pqt_path)
                else:
                    df = pd.read_csv(csv_path)
                emb_cols = [c for c in df.columns if str(c).startswith("emb_")]
                if not emb_cols:
                    return None, None
                arr = df[emb_cols].to_numpy(dtype=np.float32)
                # build metadata mapping index->info
                meta = {}
                for i, row in enumerate(df.itertuples(index=False)):
                    meta[str(i)] = {
                        "track_id": getattr(row, "track_id", None),
                        "mood_cluster": getattr(row, "mood_cluster", None),
                        "mood_category": getattr(row, "mood_category", None),
                        "dataset": getattr(row, "dataset", None),
                    }
                return arr, meta
            except Exception:
                return None, None
    # fallback: load any .npy inside embed_folder
    if os.path.isdir(embed_folder):
        files = [
            os.path.join(embed_folder, f)
            for f in os.listdir(embed_folder)
            if f.endswith(".npy")
        ]
        if files:
            try:
                mats = [np.load(f) for f in sorted(files)]
                arr = np.vstack(mats)
                return arr, None
            except Exception:
                return None, None
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", help="Path to embeddings .npy file")
    p.add_argument("--meta", help="Path to metadata.json")
    args = p.parse_args()

    embeddings, meta = find_embeddings(args.emb)
    if embeddings is None:
        print(
            "No embeddings found; put song_embeddings.csv/parquet or embeddings.npy in models/embedder_mirex/embeddings or pass --emb"
        )
        return
    dim = embeddings.shape[1]
    t = AnnoyIndex(dim, "angular")
    for i, v in enumerate(embeddings):
        t.add_item(i, v.astype(np.float32))
    t.build(10)
    t.save(INDEX_PATH)
    np.save(EMB_PATH, embeddings)
    if meta:
        try:
            with open(META_PATH, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, "r") as f:
            meta2 = json.load(f)
        with open(META_PATH, "w") as f:
            json.dump(meta2, f)
    print(f"Wrote index to {INDEX_PATH} with {embeddings.shape[0]} vectors")


if __name__ == "__main__":
    main()
