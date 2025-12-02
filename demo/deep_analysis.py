#!/usr/bin/env python3
"""
Deep analysis script for SAPPHIRE retrieval issue.
This script traces the complete flow: audio -> features -> embedding -> search
to identify where identical results are being generated.
"""
import os
import sys
import numpy as np
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from demo.app import (
    extract_audio_features,
    flatten_extracted_features,
    load_embedder_checkpoint,
    load_index,
    robust_load_audio,
)


def analyze_audio_file(audio_path, label, embedder_data):
    """Analyze a single audio file through the complete pipeline."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {label}")
    print(f"{'='*60}")

    # Step 1: Load audio
    print(f"\n[1] Loading audio from: {audio_path}")
    y, sr = robust_load_audio(audio_path)
    print(f"    Audio shape: {y.shape}, sr: {sr}")
    print(f"    Audio stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

    # Step 2: Extract features
    print(f"\n[2] Extracting features...")
    features, mfcc, chroma, spec_cent = extract_audio_features(y, sr)

    # Step 3: Flatten features
    print(f"\n[3] Flattening features...")
    flat = flatten_extracted_features(features)
    print(f"    Flattened features count: {len(flat)}")
    print(f"    Sample keys: {list(flat.keys())[:5]}")

    # Step 4: Build input vector
    print(f"\n[4] Building input vector...")
    model = embedder_data["model"]
    scaler = embedder_data["scaler"]
    feature_cols = embedder_data["feature_cols"]
    input_dim = embedder_data["input_dim"]

    vec = []
    found_count = 0
    for c in feature_cols:
        val = flat.get(c)
        if val is None:
            for prefix in [
                "acoustic_features.",
                "harmony_features.",
                "rhythm_features.",
                "quality_features.",
                "lyrics_features.",
                "metadata.",
            ]:
                if c.startswith(prefix):
                    val = flat.get(c.replace(prefix, ""))
                    if val is not None:
                        break

        if val is not None:
            found_count += 1
        vec.append(float(val if val is not None else 0.0))

    x = np.array(vec, dtype=np.float32).reshape(1, -1)
    print(f"    Input vector shape: {x.shape}")
    print(f"    Features found: {found_count}/{len(feature_cols)}")
    print(
        f"    Vector stats BEFORE scaling: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}"
    )
    print(f"    First 10 values: {x[0][:10]}")

    # Step 5: Apply scaler
    print(f"\n[5] Applying scaler...")
    if scaler is not None:
        x_orig = x.copy()
        x = scaler.transform(x)
        # CRITICAL FIX: Clip to prevent extreme outliers
        x = np.clip(x, -100, 100)
        print(
            f"    Vector stats AFTER scaling+clipping: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}"
        )
        print(f"    First 10 values: {x[0][:10]}")
        print(f"    Change: max delta = {np.max(np.abs(x - x_orig)):.4f}")

    # Step 6: Pad/trim
    print(f"\n[6] Padding/trimming to match model input...")
    if x.shape[1] < input_dim:
        pad_size = input_dim - x.shape[1]
        pad = np.zeros((1, pad_size), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)
        print(f"    Padded {pad_size} zeros")
    elif x.shape[1] > input_dim:
        x = x[:, :input_dim]
        print(f"    Trimmed to {input_dim}")
    print(f"    Final input shape: {x.shape}")

    # Step 7: Generate embedding
    print(f"\n[7] Generating embedding...")
    import torch

    with torch.no_grad():
        out = model(torch.from_numpy(x))
        if isinstance(out, tuple):
            emb = out[0].cpu().numpy().reshape(-1)
        else:
            emb = out.cpu().numpy().reshape(-1)

    print(f"    Embedding shape: {emb.shape}")
    print(
        f"    Embedding stats: min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}, std={emb.std():.4f}"
    )
    print(f"    First 10 values: {emb[:10]}")
    print(f"    L2 norm: {np.linalg.norm(emb):.4f}")

    return emb, x[0]


def main():
    print("=" * 60)
    print("SAPPHIRE RETRIEVAL DEEP ANALYSIS")
    print("=" * 60)

    # Load embedder
    print("\n[SETUP] Loading embedder...")
    embedder = load_embedder_checkpoint()
    if not embedder:
        print("ERROR: Failed to load embedder")
        return
    print(
        f"Embedder loaded. Input dim: {embedder['input_dim']}, Embedding dim: {embedder['embedding_dim']}"
    )

    # Test with 3 different audio files from dataset
    audio_dir = os.path.join(
        BASE_DIR, "data", "raw", "MIREX-like_mood", "dataset", "Audio"
    )
    test_files = [
        (os.path.join(audio_dir, "001.mp3"), "Song 001"),
        (os.path.join(audio_dir, "050.mp3"), "Song 050"),
        (os.path.join(audio_dir, "100.mp3"), "Song 100"),
    ]

    embeddings = []
    input_vectors = []

    for audio_path, label in test_files:
        if not os.path.exists(audio_path):
            print(f"\nWARNING: {audio_path} not found, skipping")
            continue

        emb, inp_vec = analyze_audio_file(audio_path, label, embedder)
        embeddings.append(emb)
        input_vectors.append(inp_vec)

    # Compare embeddings
    print(f"\n{'='*60}")
    print("EMBEDDING COMPARISON")
    print(f"{'='*60}")

    if len(embeddings) >= 2:
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                cosine_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"\nEmbedding {i} vs {j}:")
                print(f"  Euclidean distance: {dist:.6f}")
                print(f"  Cosine similarity: {cosine_sim:.6f}")

                if dist < 0.01:
                    print(f"  ⚠️  VERY SIMILAR - This explains identical retrieval!")

        # Compare input vectors
        print(f"\n{'='*60}")
        print("INPUT VECTOR COMPARISON")
        print(f"{'='*60}")

        for i in range(len(input_vectors)):
            for j in range(i + 1, len(input_vectors)):
                dist = np.linalg.norm(input_vectors[i] - input_vectors[j])
                print(f"\nInput vector {i} vs {j}:")
                print(f"  Euclidean distance: {dist:.6f}")

                if dist < 0.1:
                    print(f"  ⚠️  VERY SIMILAR - Model getting similar inputs!")

    # Test index search
    print(f"\n{'='*60}")
    print("INDEX SEARCH TEST")
    print(f"{'='*60}")

    index, index_dim = load_index()
    if index and len(embeddings) >= 2:
        print(f"\nSearching with first embedding...")
        neighbors1, distances1 = index.get_nns_by_vector(
            embeddings[0].tolist(), 5, include_distances=True
        )
        print(f"  Top 5 neighbors: {neighbors1}")
        print(f"  Distances: {distances1}")

        print(f"\nSearching with second embedding...")
        neighbors2, distances2 = index.get_nns_by_vector(
            embeddings[1].tolist(), 5, include_distances=True
        )
        print(f"  Top 5 neighbors: {neighbors2}")
        print(f"  Distances: {distances2}")

        if neighbors1 == neighbors2:
            print(f"\n  ⚠️  IDENTICAL NEIGHBORS - This is the bug!")
        else:
            print(f"\n  ✓ Different neighbors - retrieval is working")


if __name__ == "__main__":
    main()
