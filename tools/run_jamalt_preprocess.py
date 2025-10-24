#!/usr/bin/env python3
"""Download the HuggingFace `jamendolyrics/jam-alt` dataset locally and optionally run
the project's preprocessing pipeline on it.

Usage examples:
  # create local copy (audio + lyrics) only
  python tools/run_jamalt_preprocess.py --out-root data/raw/jam-alt

  # create local copy and run preprocessing (will call the preprocessing.main() in-place)
  python tools/run_jamalt_preprocess.py --out-root data/raw/jam-alt --run-preprocess

The script is conservative about audio handling:
 - If the dataset provides a decoded audio array, it will be written to a .wav file.
 - If the dataset provides a file path (cached mp3/wav), the original file will be copied and
   the original extension preserved. The project's preprocessing accepts any audio extension.
"""

import argparse
import shutil
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
from datasets import load_dataset, Features, Value, Audio
import soundfile as sf


def save_audio(audio_field, out_path: Path):
    """Save an example's audio field to out_path.

    audio_field: could be None, a dict (with 'path' or 'array'), or a str path.
    Returns True if a file was written/copied, False otherwise.
    """
    if audio_field is None:
        return False

    # Case: decoded audio delivered as {'array': ..., 'sampling_rate': ...}
    if isinstance(audio_field, dict) and "array" in audio_field:
        arr = np.array(audio_field["array"])
        sr = audio_field.get("sampling_rate", 44100)
        out_path = out_path.with_suffix(".wav")
        sf.write(str(out_path), arr, sr)
        return True

    # Case: a local path (cached by datasets) or a plain string path
    path = None
    if isinstance(audio_field, dict) and "path" in audio_field:
        path = Path(audio_field["path"])
    elif isinstance(audio_field, str):
        path = Path(audio_field)

    if path is not None and path.exists():
        # preserve original extension
        out_path = out_path.with_suffix(path.suffix)
        shutil.copy(path, out_path)
        return True

    return False


def download_and_dump(dataset_id: str, split: str, out_root: Path, config_name: str = None, stream: bool = True):
    """Download dataset entries to out_root. When stream=True the HF dataset is loaded in
    streaming mode (no full in-memory decoding) which is safer for large audio datasets.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    audio_dir = out_root / "audio"
    lyrics_dir = out_root / "lyrics"
    audio_dir.mkdir(exist_ok=True)
    lyrics_dir.mkdir(exist_ok=True)

    print(f"Loading dataset {dataset_id} (split={split}, config={config_name})...")
    ds_kwargs = {}
    if config_name:
        ds_kwargs["name"] = config_name

    written = 0

    # Try huggingface_hub snapshot download first (avoids datasets decoding audio)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        snapshot_download = None

    if snapshot_download is not None:
        try:
            print(f"Downloading dataset repo via huggingface_hub.snapshot_download('{dataset_id}')...")
            repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset")
            metadata_path = Path(repo_dir) / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(f"metadata.csv not found in snapshot at {repo_dir}")

            import csv

            with open(metadata_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filepath = row.get("Filepath")
                    if not filepath:
                        continue
                    name = Path(filepath).stem

                    src_audio = Path(repo_dir) / "audio" / filepath
                    if src_audio.exists():
                        shutil.copy(src_audio, audio_dir / src_audio.name)

                    # lyrics
                    src_lyrics = Path(repo_dir) / "lyrics" / (Path(filepath).stem + ".txt")
                    if src_lyrics.exists():
                        shutil.copy(src_lyrics, lyrics_dir / src_lyrics.name)

                    written += 1

            print(f"Wrote {written} examples. Audio files in {audio_dir}, lyrics in {lyrics_dir}.")
            return audio_dir, lyrics_dir
        except Exception as e:
            print(f"huggingface_hub snapshot_download approach failed: {e}")
            print("Falling back to datasets-based download...")

    # Fallback to datasets (streaming preferred)
    try:
        if stream:
            ds = load_dataset(dataset_id, split=split, streaming=True, **ds_kwargs)
            print(f"Streaming examples and saving to {out_root}...")
            for ex in ds:
                name = ex.get("name") or ex.get("id") or str(written)
                audio_field = ex.get("audio")
                out_audio_path = audio_dir / name
                try:
                    _ = save_audio(audio_field, out_audio_path)
                except Exception:
                    print(f"Warning: failed to save audio for {name}")

                text = ex.get("text") or ex.get("lyrics") or ""
                with open(lyrics_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                written += 1
        else:
            ds = load_dataset(dataset_id, split=split, **ds_kwargs)
            print(f"Saving {len(ds)} examples to {out_root}...")
            for ex in ds:
                name = ex.get("name") or ex.get("id") or str(written)
                audio_field = ex.get("audio")
                out_audio_path = audio_dir / name
                try:
                    _ = save_audio(audio_field, out_audio_path)
                except Exception:
                    print(f"Warning: failed to save audio for {name}")

                text = ex.get("text") or ex.get("lyrics") or ""
                with open(lyrics_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                written += 1
    except MemoryError:
        print("MemoryError while loading dataset: try running with streaming enabled (default). Exiting.")
        raise
    except Exception as e:
        print(f"Error while iterating dataset: {e}")
        raise

    print(f"Wrote {written} examples. Audio files in {audio_dir}, lyrics in {lyrics_dir}.")
    return audio_dir, lyrics_dir


def run_preprocessing(preprocessing_py: Path, audio_dir: Path, lyrics_dir: Path, out_processed: Path):
    """Load preprocessing.py as a module, patch its INPUT paths, and call main()."""
    print(f"Loading preprocessing from {preprocessing_py}")
    loader = SourceFileLoader("preprocessing_module", str(preprocessing_py))
    preprocessing = loader.load_module()

    # Patch input and output paths used inside preprocessing.py
    preprocessing.INPUT_AUDIO_DIR = Path(audio_dir)
    preprocessing.INPUT_LYRICS_DIR = Path(lyrics_dir)
    preprocessing.OUTPUT_DIR = Path(out_processed)
    preprocessing.PROCESSED_AUDIO_DIR = preprocessing.OUTPUT_DIR / "audio_high_quality"
    preprocessing.METADATA_DIR = preprocessing.OUTPUT_DIR / "metadata"

    preprocessing.PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    preprocessing.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing.main() (this runs the existing pipeline in-place)...")
    preprocessing.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="jamendolyrics/jam-alt", help="HF dataset id")
    parser.add_argument("--split", default="test", help="Split to load")
    parser.add_argument("--config", default=None, help="Optional config/name for the HF dataset (e.g., 'en', 'es')")
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/jam-alt"), help="Where to dump audio/lyrics")
    parser.add_argument("--run-preprocess", action="store_true", help="After download, call the repository preprocessing pipeline")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode (may use lots of memory). By default streaming is enabled.")
    parser.add_argument("--processed-out", type=Path, default=Path("processed_jamalt"), help="Output dir for preprocessing results if --run-preprocess is used")
    args = parser.parse_args()

    stream = not args.no_stream
    audio_dir, lyrics_dir = download_and_dump(args.dataset, args.split, args.out_root, args.config, stream=stream)

    if args.run_preprocess:
        # Find preprocessing.py in repo
        repo_root = Path(__file__).resolve().parents[1]
        preprocessing_py = repo_root / "src" / "preprocess" / "preprocessing.py"
        if not preprocessing_py.exists():
            print(f"Error: could not find preprocessing.py at {preprocessing_py}")
            sys.exit(1)
        run_preprocessing(preprocessing_py, audio_dir, lyrics_dir, args.processed_out)


if __name__ == "__main__":
    main()
