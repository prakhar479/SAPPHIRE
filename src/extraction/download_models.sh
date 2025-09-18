#!/usr/bin/env bash
set -e


# Pre-download SBERT weights (sentence-transformers) to avoid first-run network stalls
python - <<'PY'
from sentence_transformers import SentenceTransformer
print('Downloading SBERT model')
SentenceTransformer('all-mpnet-base-v2')
print('Done')
PY


# CREPE auto-downloads its models at runtime; no explicit step required generally
# If you want to cache them, run a simple crepe.predict on a small audio file once.


echo "All requested models attempted to download."