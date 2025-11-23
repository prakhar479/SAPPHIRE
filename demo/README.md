# SAPPHIRE Embedding Similarity Demo

This is a small proof-of-concept web demo to: build a small embedding index from precomputed song embeddings, and query for perceptually similar songs using nearest-neighbors.

Quick start

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements-demo.txt
```

2. Prepare embeddings

- If you already have embeddings at `models/embedder_mirex/embeddings/embeddings.npy` and an optional `metadata.json` in the same folder, the demo will pick them up automatically.
- Alternatively, place `embeddings.npy` and `metadata.json` in `demo/`.
 - Alternatively, place `embeddings.npy` and `metadata.json` in `demo/`.
 - The demo now also supports `song_embeddings.csv` or `song_embeddings.parquet` (these files are present in `models/embedder_mirex/embeddings/` in this repo). The CSV/parquet should contain `emb_*` columns; metadata such as `track_id`, `mood_category` and `mood_cluster` will be extracted automatically.

3. Run the demo server

```bash
python demo/app.py
```

4. Open `http://localhost:5000` in your browser.

Usage

- Click **Build / Rebuild Index** to construct an Annoy index from the embeddings (stored as `demo/index.ann`).
- Enter a song integer ID and click **Find Similar** to retrieve nearest neighbors.
- Visit **Static demo pages** to view study plots found in the repository.

- Upload an audio file using the **Upload Audio (quick test)** form on the main page. The form posts the file to `/api/upload`, computes an embedding (server-side) using the included embedder checkpoint if available, and returns nearest neighbors from the index.

Quick verification (CLI)

1. Build the index via HTTP (or use the UI):

```bash
curl -X POST http://localhost:5000/api/index
```

2. Upload an audio file and request the top-10 neighbors:

```bash
curl -F "file=@/path/to/example.wav" -F "k=10" http://localhost:5000/api/upload
```

The response will be JSON listing the computed embedding and the nearest neighbor entries (id, distance, info).

Notes & limitations

- This is a minimal POC. For robust production usage consider:
  - Using FAISS for scalable search
  - Computing embeddings on the fly for uploaded audio
  - Securing file-serving endpoints
   - Securing file-serving endpoints
 
Detailed demo setup and model notes
---------------------------------

- System libraries (required by audio packages): install `libsndfile` and `ffmpeg` on Linux. Example (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg
```

- Python environment: use a virtualenv and install demo dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements-demo.txt
```

- Embedding model checkpoint

- The demo can compute embeddings on-the-fly if a trained checkpoint is available at either:

  - `models/embedder_mirex/song_embedder.pt` (preferred for the MIREX-style dataset)
  - `models/embedder_top40/song_embedder.pt`

  If present, the server will attempt to load the checkpoint at runtime and use it to embed uploaded audio. The training script in `scripts/train_song_embedder.py` saves checkpoints containing the keys `model_state_dict`, `input_dim`, `embedding_dim`, `scaler_mean`, and `scaler_scale`. If you have your own checkpoint, place it at one of the paths above.

- Precomputed embeddings

  Alternatively the demo will use precomputed embeddings if available under:

  - `models/embedder_mirex/embeddings/song_embeddings.parquet` or `.csv` (contains `emb_*` columns), or
  - `models/embedder_mirex/embeddings/embeddings.npy`, or
  - `demo/embeddings.npy` (you can copy an `.npy` here).

  When you run **Build / Rebuild Index**, the demo reads the embeddings and constructs `demo/index.ann` (Annoy index) and caches `demo/embeddings.npy` and `demo/metadata.json`.

Running the demo (end-to-end)
-----------------------------

1. Ensure system deps and Python packages are installed (see above).
2. Confirm you have either a checkpoint (see path above) or embeddings in `models/embedder_mirex/embeddings/`.
3. Start the server:

```bash
python demo/app.py
```

4. Open `http://localhost:5000` in your browser.
5. Click **Build / Rebuild Index** to create `demo/index.ann`.
6. Use the **Upload Audio (quick test)** form to post an audio file (WAV/MP3). The demo will:
   - extract features (repo extractor preferred, fallback to `librosa` MFCCs),
   - apply the checkpoint's scaler (if available),
   - run the checkpoint to produce an embedding, and
   - query the Annoy index to return nearest neighbors.

Troubleshooting
---------------

- If `/api/index` returns an error: confirm embeddings exist in one of the supported locations (`models/embedder_mirex/embeddings/`). Check server logs (terminal) for errors reading CSV/parquet.
- If `/api/upload` returns "embedder checkpoint not available": verify a checkpoint file exists at `models/embedder_mirex/song_embedder.pt` or `models/embedder_top40/song_embedder.pt` and that `torch` is installed in the active environment.
- If feature extraction fails: ensure `libsndfile` and `ffmpeg` are installed and that `librosa` and `soundfile` Python packages are present.
- If embeddings returned look poor: the demo attempts to infer the original feature column order and reconstruct the scaler from the checkpoint; mismatches between training-time feature ordering and runtime extraction will reduce similarity quality.

Advanced notes (optional)
-------------------------

- To re-train the embedder or inspect training script: see `scripts/train_song_embedder.py`. Training outputs checkpoints compatible with this demo when they include `scaler_mean` and `scaler_scale` in the saved dict.
- For a production deployment consider converting the Annoy index to FAISS and adding authentication and upload size limits to the server.
