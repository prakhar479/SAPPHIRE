#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from tqdm import tqdm

# --- Setup Logging ---
# Configure logging to show timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# --- Feature Extraction & Processing Functions ---


def calculate_snr_percentile(audio_data: np.ndarray, percentile: float = 95.0) -> float:
    """
    Computes a proxy Signal-to-Noise Ratio (SNR) using a percentile-based energy threshold.
    Assumes the signal is the high-energy part and noise is the low-energy part.
    """
    try:
        energy = audio_data**2
        if np.max(energy) == 0:
            return -np.inf  # Pure silence

        signal_threshold = np.percentile(energy, percentile)
        signal_part = energy[energy >= signal_threshold]
        noise_part = energy[energy < signal_threshold]

        if np.mean(noise_part) == 0:
            return np.inf  # No detectable noise

        signal_power = np.mean(signal_part)
        noise_power = np.mean(noise_part)

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)
    except Exception as e:
        logging.warning(f"Could not calculate SNR: {e}")
        return 0.0


def extract_audio_features(
    y: np.ndarray, sr: int, target_loudness: float
) -> Dict[str, Any]:
    """
    Loads, normalizes, and computes a suite of quality and descriptive metrics for an audio signal.
    """
    features = {}

    # --- Pre-computation & Normalization ---
    # Ensure mono for most feature calculations
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    features["duration_seconds"] = librosa.get_duration(y=y, sr=sr)

    # Loudness normalization
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y_mono)
        normalized_audio = pyln.normalize.loudness(y_mono, loudness, target_loudness)
        features["integrated_loudness_lufs"] = float(loudness)
    except ValueError:  # Handle very short or silent audio
        logging.warning(
            "Loudness normalization failed, audio may be silent or too short. Skipping normalization."
        )
        normalized_audio = y_mono
        features["integrated_loudness_lufs"] = -np.inf

    # --- Quality Metrics ---
    features["snr_percentile_db"] = calculate_snr_percentile(normalized_audio)
    features["clipping_percentage"] = float(
        np.mean(np.abs(normalized_audio) >= 0.999) * 100
    )

    # --- Descriptive Features (from normalized audio) ---
    rms_energy = librosa.feature.rms(y=normalized_audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=normalized_audio)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=normalized_audio, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=normalized_audio, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=13)

    features["rms_energy_mean"] = float(np.mean(rms_energy))
    features["rms_energy_std"] = float(np.std(rms_energy))
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    features["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bandwidth))

    # Add mean and std for each MFCC
    for i in range(mfccs.shape[0]):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i, :]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i, :]))

    # --- Vocal Dominance Heuristic (using HPSS) ---
    y_harmonic, y_percussive = librosa.effects.hpss(normalized_audio)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    if harmonic_energy + percussive_energy > 0:
        vocal_dominance_score = harmonic_energy / (harmonic_energy + percussive_energy)
    else:
        vocal_dominance_score = 0.0
    features["vocal_dominance_score_hpss"] = float(vocal_dominance_score)

    return features, normalized_audio


def extract_lyrics_features(lyrics_path: Path) -> Dict[str, Any]:
    """
    Cleans, validates, and extracts metadata and features from a lyrics text file.
    """
    with open(lyrics_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip markup (HTML, etc.)
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()

    # Normalize whitespace
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # --- Language Detection ---
    lang_code = "unknown"
    try:
        if len(clean_text) > 20:  # Langdetect is unreliable on very short text
            lang_code = detect(clean_text)
    except LangDetectException:
        lang_code = "detection_failed"

    # --- Completeness Heuristic ---
    # Assumes markers like '...' or '[incomplete]' indicate partial lyrics
    is_incomplete = "..." in clean_text or "[incomplete]" in clean_text.lower()
    completeness = 0.8 if is_incomplete else 1.0

    # --- Basic Text Features ---
    words = clean_text.split()
    word_count = len(words)
    char_count = len(clean_text)
    sentence_count = len(re.split(r"[.!?]+", clean_text))

    # Type-Token Ratio (Vocabulary Richness)
    ttr = (len(set(words)) / word_count) if word_count > 0 else 0.0

    return {
        "cleaned_text": clean_text,
        "language_code": lang_code,
        "completeness_heuristic": completeness,
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "type_token_ratio": ttr,
    }


def process_track(
    audio_path: Path, lyrics_path: Path, output_dir: Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker function to process a single audio/lyrics track pair.
    This function is designed to be run in a separate process.
    """
    track_id = audio_path.stem
    result = {"track_id": track_id, "status": "error", "reason": "Unknown"}

    try:
        # --- 1. Process Audio ---
        y, original_sr = sf.read(str(audio_path), dtype="float32")

        # Resample to target rate if necessary
        if original_sr != config["TARGET_SR"]:
            # librosa expects (channels, samples) or (samples,)
            y_resampled = librosa.resample(
                y=y.T, orig_sr=original_sr, target_sr=config["TARGET_SR"]
            ).T
        else:
            y_resampled = y

        audio_features, normalized_audio = extract_audio_features(
            y_resampled, config["TARGET_SR"], config["TARGET_LOUDNESS"]
        )

        # --- 2. Process Lyrics ---
        lyrics_features = extract_lyrics_features(lyrics_path)

        # --- 3. Consolidate Metadata ---
        metadata = {
            "track_id": track_id,
            "provenance": {
                "original_audio_path": str(audio_path),
                "original_lyrics_path": str(lyrics_path),
                "processing_date": datetime.now(timezone.utc).isoformat(),
                "script_version": "1.0.0",  # Or git hash
            },
            "audio_properties": {
                "original_sr": original_sr,
                "target_sr": config["TARGET_SR"],
                "channels": y.ndim,
            },
            "audio_features": audio_features,
            "lyrics_features": lyrics_features,
            "is_high_quality": False,
        }

        # --- 4. High-Quality Subset Selection ---
        passes_criteria = all(
            [
                metadata["audio_features"]["snr_percentile_db"]
                >= config["SNR_THRESHOLD"],
                metadata["lyrics_features"]["completeness_heuristic"]
                >= config["LYRICS_COMPLETENESS_THRESHOLD"],
                metadata["audio_features"]["vocal_dominance_score_hpss"]
                >= config["VOCAL_THRESHOLD"],
            ]
        )

        # --- 5. Save Outputs ---
        if passes_criteria:
            metadata["is_high_quality"] = True
            output_path = output_dir / "audio_normalized" / f"{track_id}.wav"
            sf.write(output_path, normalized_audio, config["TARGET_SR"])
            metadata["normalized_audio_path"] = str(output_path)
        else:
            rejected_path = output_dir / "audio_rejected" / f"{track_id}.wav"
            sf.write(rejected_path, normalized_audio, config["TARGET_SR"])
            metadata["normalized_audio_path"] = str(rejected_path)

        # Save per-track metadata
        metadata_path = output_dir / "features" / f"{track_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        result.update({"status": "success", "metadata": metadata})
        return result

    except Exception as e:
        logging.error(f"Failed to process {track_id}: {e}", exc_info=True)
        result.update({"reason": str(e)})
        return result


def main(args):
    """Main function to orchestrate the dataset preprocessing pipeline."""
    start_time = time.time()

    # --- 1. Setup I/O Paths and Configuration ---
    input_audio_dir = Path(args.input_audio)
    input_lyrics_dir = Path(args.input_lyrics)
    output_dir = Path(args.output_dir)

    # Create output directories
    for subdir in ["audio_normalized", "audio_rejected", "features", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / "logs" / "processing.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting dataset preprocessing pipeline.")
    logging.info(f"Input Audio: {input_audio_dir}")
    logging.info(f"Input Lyrics: {input_lyrics_dir}")
    logging.info(f"Output Directory: {output_dir}")

    # Project-wide configuration from args
    config = {
        "TARGET_SR": args.sample_rate,
        "TARGET_LOUDNESS": args.loudness,
        "SNR_THRESHOLD": args.snr_threshold,
        "LYRICS_COMPLETENESS_THRESHOLD": args.lyrics_completeness_threshold,
        "VOCAL_THRESHOLD": args.vocal_threshold,
    }
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")

    # --- 2. Discover files and create job list ---
    audio_files = list(input_audio_dir.glob("*.*"))
    jobs: List[Tuple[Path, Path, Path, Dict]] = []
    for audio_path in audio_files:
        lyrics_path = input_lyrics_dir / f"{audio_path.stem}.txt"
        if not lyrics_path.exists():
            logging.warning(f"Lyrics not found for {audio_path.name}, skipping.")
            continue
        jobs.append((audio_path, lyrics_path, output_dir, config))

    if not jobs:
        logging.error("No audio/lyrics pairs found. Exiting.")
        return

    logging.info(f"Found {len(jobs)} audio/lyrics pairs to process.")

    # --- 3. Run processing in parallel ---
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_job = {executor.submit(process_track, *job): job for job in jobs}

        for future in tqdm(
            as_completed(future_to_job),
            total=len(jobs),
            desc="Processing tracks",
            unit="track",
        ):
            results.append(future.result())

    # --- 4. Consolidate results and create manifests ---
    successful_metadata = [r["metadata"] for r in results if r["status"] == "success"]
    failed_tracks = [r for r in results if r["status"] == "error"]

    if not successful_metadata:
        logging.error("No tracks were processed successfully.")
        return

    # Create a Pandas DataFrame for the main manifest
    df = pd.json_normalize(successful_metadata, sep="_")

    # Save main manifest as Parquet (efficient) and CSV (human-readable)
    manifest_parquet_path = output_dir / "manifest.parquet"
    manifest_csv_path = output_dir / "manifest.csv"
    df.to_parquet(manifest_parquet_path, index=False)
    df.to_csv(manifest_csv_path, index=False)

    # Filter for high-quality subset and save its manifest
    high_quality_df = df[df["is_high_quality"] == True]
    high_quality_manifest_path = output_dir / "manifest_high_quality.parquet"
    high_quality_df.to_parquet(high_quality_manifest_path, index=False)

    # Save errors to a JSON file
    errors_path = output_dir / "logs" / "errors.json"
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(failed_tracks, f, indent=2)

    # --- 5. Final Report ---
    total_processed = len(successful_metadata)
    total_high_quality = len(high_quality_df)
    total_failed = len(failed_tracks)
    retention_pct = (
        (total_high_quality / total_processed * 100) if total_processed > 0 else 0
    )
    duration_secs = time.time() - start_time

    logging.info("\n" + "=" * 50)
    logging.info("--- PREPROCESSING COMPLETE ---")
    logging.info(f"Total time: {duration_secs:.2f} seconds")
    logging.info(f"Total files found: {len(jobs)}")
    logging.info(f"Successfully processed: {total_processed}")
    logging.info(f"Failed to process: {total_failed}")
    logging.info(
        f"High-quality samples retained: {total_high_quality} ({retention_pct:.2f}%)"
    )
    logging.info("-" * 50)
    logging.info(f"Main manifest saved to: {manifest_parquet_path}")
    logging.info(f"High-quality manifest saved to: {high_quality_manifest_path}")
    logging.info(f"Detailed logs saved to: {log_file}")
    logging.info(f"Error summary saved to: {errors_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust audio and lyrics dataset preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O Arguments ---
    parser.add_argument(
        "--input-audio",
        type=str,
        required=True,
        help="Path to the directory with raw audio files.",
    )
    parser.add_argument(
        "--input-lyrics",
        type=str,
        required=True,
        help="Path to the directory with raw lyrics (.txt) files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_dataset",
        help="Path to the directory where all outputs will be saved.",
    )

    # --- Processing Arguments ---
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel processes to use. Defaults to the number of CPU cores.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate for all audio.",
    )
    parser.add_argument(
        "--loudness",
        type=float,
        default=-23.0,
        help="Target loudness in LUFS (EBU R128 standard).",
    )

    # --- Filtering Thresholds ---
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=15.0,
        help="Minimum Signal-to-Noise Ratio (SNR) in dB.",
    )
    parser.add_argument(
        "--lyrics-completeness-threshold",
        type=float,
        default=0.9,
        help="Minimum score for lyrics completeness heuristic.",
    )
    parser.add_argument(
        "--vocal-threshold",
        type=float,
        default=0.6,
        help="Minimum score for vocal dominance (from HPSS).",
    )

    cli_args = parser.parse_args()
    main(cli_args)
