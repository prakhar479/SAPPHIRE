import os
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm import tqdm

# --- Configuration & Thresholds ---
# 6.1 Goals: Define project-wide standards
TARGET_SR = 44100
TARGET_LOUDNESS = -23.0  # LUFS, EBU R128 standard

# 6.3 High-quality subset selection criteria
# SNR_THRESHOLD = 35.0  # Chosen value for Signal-to-Noise Ratio in dB
SNR_THRESHOLD = 10.0
LYRICS_COMPLETENESS_THRESHOLD = 0.9
VOCAL_THRESHOLD = 0.75 # Chosen value for vocal dominance score

# --- Setup I/O Paths ---
INPUT_AUDIO_DIR = Path("/home/kushal/Desktop/UG_4/Music/Sprint2/SAPPHIRE/data/raw/Viet_Dataset/songs")
INPUT_LYRICS_DIR = Path("/home/kushal/Desktop/UG_4/Music/Sprint2/SAPPHIRE/data/raw/Viet_Dataset/lyrics")
OUTPUT_DIR = Path("processed_dataset")

# Create output directories
PROCESSED_AUDIO_DIR = OUTPUT_DIR / "audio_high_quality"
METADATA_DIR = OUTPUT_DIR / "metadata"
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def calculate_snr(audio_data, percentile=95):
    """
    Computes a proxy Signal-to-Noise Ratio (SNR).
    Assumes the signal is the high-energy part and noise is the low-energy part.
    """
    try:
        energy = audio_data ** 2
        signal_threshold = np.percentile(energy, percentile)
        
        signal_part = energy[energy >= signal_threshold]
        noise_part = energy[energy < signal_threshold]
        
        if np.mean(noise_part) == 0:
            return np.inf # Avoid division by zero if there's perfect silence
            
        signal_power = np.mean(signal_part)
        noise_power = np.mean(noise_part)
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    except Exception:
        return 0.0

def process_audio(audio_path, output_path):
    """
    6.2 Implemented steps (1. Audio checks & standardisation)
    Loads, normalizes, and computes quality metrics for an audio file.
    """
    try:
        # Load audio, preserving original sample rate to check against target
        y, sr = librosa.load(audio_path, sr=None, mono=False)

        # Resample to target rate if necessary
        if sr != TARGET_SR:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR)
        
        # Convert to mono for loudness and SNR calculation
        y_mono = librosa.to_mono(y)

        # Loudness normalisation
        meter = pyln.Meter(TARGET_SR)
        loudness = meter.integrated_loudness(y_mono)
        normalized_audio = pyln.normalize.loudness(y_mono, loudness, TARGET_LOUDNESS)

        # Compute quality metrics
        snr = calculate_snr(normalized_audio)
        clipping_pct = np.mean(np.abs(normalized_audio) >= 0.999) * 100

        # Save the normalized audio file
        sf.write(output_path, normalized_audio, TARGET_SR)

        return {
            "status": "success",
            "snr": snr,
            "clipping_percentage": clipping_pct,
            "original_sr": sr,
            "normalized_path": str(output_path)
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}

def process_lyrics(lyrics_path):
    """
    6.2 Implemented steps (2. Lyrics processing)
    Cleans, validates, and extracts metadata from a lyrics text file.
    """
    try:
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Strip markup (HTML, etc.)
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()

        # Normalize whitespace and punctuation
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Language detection
        try:
            lang_code = detect(clean_text)
        except:
            lang_code = "unknown"

        # Check for completeness (heuristic)
        # Assumes markers like '...' or '[incomplete]' indicate partial lyrics
        if '...' in clean_text or '[incomplete]' in clean_text.lower():
            completeness = 0.8 # Less than the 0.9 threshold
        else:
            completeness = 1.0

        return {
            "status": "success",
            "cleaned_text": clean_text,
            "language_code": lang_code,
            "completeness": completeness
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}

def main():
    """
    Main function to run the entire preprocessing pipeline.
    """
    audio_files = list(INPUT_AUDIO_DIR.glob('*.*'))
    total_files = len(audio_files)
    
    all_metadata = []
    high_quality_samples = []
    
    print(f"Starting preprocessing for {total_files} audio files...")

    for audio_path in tqdm(audio_files, desc="Processing files"):
        track_id = audio_path.stem
        # print(track_id)
        lyrics_path = INPUT_LYRICS_DIR / f"{track_id}.txt"

        if not lyrics_path.exists():
            print(f"Warning: Lyrics not found for {track_id}, skipping.")
            continue

        # --- Process Audio and Lyrics ---
        processed_audio_path = PROCESSED_AUDIO_DIR / f"{track_id}.wav"
        audio_results = process_audio(audio_path, processed_audio_path)
        lyrics_results = process_lyrics(lyrics_path)
        
        # --- Metadata Consolidation ---
        # 6.2 Implemented steps (3. Metadata consolidation)
        if audio_results["status"] == "error" or lyrics_results["status"] == "error":
            print(f"Error processing {track_id}. Audio: {audio_results.get('reason', 'N/A')}, Lyrics: {lyrics_results.get('reason', 'N/A')}")
            continue

        # Placeholder for a vocal dominance score.
        # In a real scenario, this would come from a vocal separation model.
        vocal_score = np.random.uniform(0.5, 1.0)
        
        metadata = {
            "track_id": track_id,
            "provenance": {
                "original_audio_path": str(audio_path),
                "original_lyrics_path": str(lyrics_path),
                "processing_date": datetime.utcnow().isoformat()
            },
            "audio_quality": {
                "snr": audio_results["snr"],
                "clipping_percentage": audio_results["clipping_percentage"]
            },
            "lyrics_meta": {
                "language_code": lyrics_results["language_code"],
                "completeness": lyrics_results["completeness"]
            },
            "features": {
                "vocal_dominance_score": vocal_score
            },
            "is_high_quality": False # Flag to be updated
        }
        
        # --- High-quality subset selection ---
        passes_criteria = all([
            metadata["audio_quality"]["snr"] >= SNR_THRESHOLD,
            metadata["lyrics_meta"]["completeness"] >= LYRICS_COMPLETENESS_THRESHOLD,
            metadata["features"]["vocal_dominance_score"] >= VOCAL_THRESHOLD
        ])

        if passes_criteria:
            metadata["is_high_quality"] = True
            high_quality_samples.append(metadata)
        else:
            # If not high quality, remove the processed audio file to save space
            processed_audio_path.unlink()

        all_metadata.append(metadata)

    # Save metadata files
    with open(METADATA_DIR / "all_samples_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    with open(METADATA_DIR / "high_quality_subset.json", 'w') as f:
        json.dump(high_quality_samples, f, indent=2)
        
    # --- Final Report (for Results section) ---
    print("\n--- Preprocessing Complete ---")
    print(f"Total samples processed: {len(all_metadata)}")
    print(f"Final retained high-quality samples: {len(high_quality_samples)}")
    print(f"Percentage retained: {(len(high_quality_samples) / len(all_metadata) * 100):.2f}%")
    print("----------------------------")
    print(f"High-quality audio saved to: {PROCESSED_AUDIO_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")

if __name__ == "__main__":
    main()