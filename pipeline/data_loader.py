"""
Data loading utilities for the SAPPHIRE pipeline.
Handles loading of audio files, lyrics, and mood annotations.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import librosa
import soundfile as sf
from dataclasses import dataclass
import logging

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class MusicTrack:
    """Represents a single music track with all associated data."""

    track_id: str
    audio_path: Optional[str] = None
    lyrics_path: Optional[str] = None
    audio_data: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    lyrics_text: Optional[str] = None
    mood_cluster: Optional[str] = None
    mood_category: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataLoader:
    """
    Comprehensive data loader for the SAPPHIRE pipeline.
    Handles multiple datasets and formats.
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)

    def load_mirex_mood_dataset(self) -> List[MusicTrack]:
        """
        Load the MIREX-like mood dataset with audio, lyrics, and mood annotations.

        Returns:
            List of MusicTrack objects
        """
        self.logger.info("Loading MIREX mood dataset...")

        mirex_dir = Path(self.config.data.mirex_mood_dir)
        audio_dir = mirex_dir / "Audio"
        lyrics_dir = mirex_dir / "Lyrics"

        # Load mood annotations
        clusters_file = mirex_dir / "clusters.txt"
        categories_file = mirex_dir / "categories.txt"

        if not all([clusters_file.exists(), categories_file.exists()]):
            raise FileNotFoundError("Mood annotation files not found")

        # Read mood annotations
        with open(clusters_file, "r") as f:
            clusters = [line.strip() for line in f.readlines()]

        with open(categories_file, "r") as f:
            categories = [line.strip() for line in f.readlines()]

        tracks = []

        # Get all audio files
        audio_files = list(audio_dir.glob("*.mp3"))
        self.logger.info(f"Found {len(audio_files)} audio files")

        for i, audio_file in enumerate(sorted(audio_files)):
            track_id = audio_file.stem

            # Find corresponding lyrics file
            lyrics_file = lyrics_dir / f"{track_id}.txt"
            lyrics_path = str(lyrics_file) if lyrics_file.exists() else None

            # Get mood annotations (1-indexed in files)
            file_index = int(track_id) - 1
            mood_cluster = clusters[file_index] if file_index < len(clusters) else None
            mood_category = (
                categories[file_index] if file_index < len(categories) else None
            )

            track = MusicTrack(
                track_id=track_id,
                audio_path=str(audio_file),
                lyrics_path=lyrics_path,
                mood_cluster=mood_cluster,
                mood_category=mood_category,
                metadata={"dataset": "MIREX_mood", "file_index": file_index},
            )

            tracks.append(track)

        self.logger.info(f"Loaded {len(tracks)} tracks from MIREX mood dataset")
        return tracks

    def load_audio(self, track: MusicTrack, load_data: bool = True) -> MusicTrack:
        """
        Load audio data for a track.

        Args:
            track: MusicTrack object
            load_data: Whether to load the actual audio data

        Returns:
            Updated MusicTrack object
        """
        if not track.audio_path or not os.path.exists(track.audio_path):
            self.logger.warning(f"Audio file not found for track {track.track_id}")
            return track

        try:
            if load_data:
                # Load audio data
                audio_data, sample_rate = librosa.load(
                    track.audio_path, sr=self.config.audio.sample_rate, mono=True
                )
                track.audio_data = audio_data
                track.sample_rate = sample_rate

                # Add basic audio metadata
                duration = len(audio_data) / sample_rate
                track.metadata.update(
                    {
                        "duration": duration,
                        "original_sr": sample_rate,
                        "audio_loaded": True,
                    }
                )
            else:
                # Just get metadata without loading data
                info = sf.info(track.audio_path)
                track.metadata.update(
                    {
                        "duration": info.duration,
                        "original_sr": info.samplerate,
                        "channels": info.channels,
                        "audio_loaded": False,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error loading audio for track {track.track_id}: {e}")

        return track

    def load_lyrics(self, track: MusicTrack) -> MusicTrack:
        """
        Load lyrics text for a track.

        Args:
            track: MusicTrack object

        Returns:
            Updated MusicTrack object
        """
        if not track.lyrics_path or not os.path.exists(track.lyrics_path):
            self.logger.warning(f"Lyrics file not found for track {track.track_id}")
            return track

        try:
            with open(track.lyrics_path, "r", encoding="utf-8") as f:
                track.lyrics_text = f.read().strip()

            track.metadata.update(
                {"lyrics_loaded": True, "lyrics_length": len(track.lyrics_text)}
            )

        except Exception as e:
            self.logger.error(f"Error loading lyrics for track {track.track_id}: {e}")

        return track

    def load_processed_features(self, feature_dir: str) -> pd.DataFrame:
        """
        Load previously extracted features from parquet files.

        Args:
            feature_dir: Directory containing feature files

        Returns:
            DataFrame with all features
        """
        feature_path = Path(feature_dir)

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        # Look for parquet files
        parquet_files = list(feature_path.glob("*.parquet"))

        if not parquet_files:
            # Look for JSON files as fallback
            json_files = list(feature_path.glob("*.json"))
            if json_files:
                return self._load_json_features(json_files)
            else:
                raise FileNotFoundError("No feature files found")

        # Load and combine parquet files
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)

        if len(dfs) == 1:
            return dfs[0]
        else:
            # Combine multiple feature files
            return pd.concat(dfs, axis=1)

    def _load_json_features(self, json_files: List[Path]) -> pd.DataFrame:
        """Load features from JSON files."""
        all_features = []

        for json_file in json_files:
            with open(json_file, "r") as f:
                features = json.load(f)

            # Flatten nested dictionaries
            flattened = self._flatten_dict(features)
            flattened["track_id"] = json_file.stem
            all_features.append(flattened)

        return pd.DataFrame(all_features)

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                # Handle numeric lists by taking statistics
                items.extend(
                    [
                        (f"{new_key}_mean", np.mean(v)),
                        (f"{new_key}_std", np.std(v)),
                        (f"{new_key}_min", np.min(v)),
                        (f"{new_key}_max", np.max(v)),
                    ]
                )
            else:
                items.append((new_key, v))
        return dict(items)

    def create_dataset_manifest(self, tracks: List[MusicTrack], output_path: str):
        """
        Create a manifest file for the dataset.

        Args:
            tracks: List of MusicTrack objects
            output_path: Path to save the manifest
        """
        manifest_data = []

        for track in tracks:
            track_data = {
                "track_id": track.track_id,
                "audio_path": track.audio_path,
                "lyrics_path": track.lyrics_path,
                "mood_cluster": track.mood_cluster,
                "mood_category": track.mood_category,
                **track.metadata,
            }
            manifest_data.append(track_data)

        df = pd.DataFrame(manifest_data)

        # Save as both parquet and CSV
        output_path = Path(output_path)
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)
        df.to_csv(output_path.with_suffix(".csv"), index=False)

        self.logger.info(f"Dataset manifest saved to {output_path}")

        return df

    def load_dataset_manifest(self, manifest_path: str) -> pd.DataFrame:
        """Load dataset manifest from file."""
        manifest_path = Path(manifest_path)

        if manifest_path.with_suffix(".parquet").exists():
            return pd.read_parquet(manifest_path.with_suffix(".parquet"))
        elif manifest_path.with_suffix(".csv").exists():
            return pd.read_csv(manifest_path.with_suffix(".csv"))
        else:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    def load_csd_dataset(self) -> List[MusicTrack]:
        """
        Load the CSD (Computational Scene Description) dataset.

        Returns:
            List of MusicTrack objects
        """
        self.logger.info("Loading CSD dataset...")

        csd_dir = Path(self.config.data.raw_audio_dir) / "CSD"
        tracks = []

        # Check for English and Korean subdirectories
        for lang_dir in ["english", "korean"]:
            lang_path = csd_dir / lang_dir
            if lang_path.exists():
                # Audio files are in wav subdirectory
                audio_dir = lang_path / "wav"
                lyrics_dir = lang_path / "txt"

                if audio_dir.exists():
                    audio_files = list(audio_dir.glob("*.wav")) + list(
                        audio_dir.glob("*.mp3")
                    )

                    for audio_file in audio_files:
                        track_id = f"csd_{lang_dir}_{audio_file.stem}"

                        # Look for corresponding lyrics file in txt directory
                        lyrics_file = lyrics_dir / f"{audio_file.stem}.txt"
                        lyrics_path = str(lyrics_file) if lyrics_file.exists() else None

                        track = MusicTrack(
                            track_id=track_id,
                            audio_path=str(audio_file),
                            lyrics_path=lyrics_path,
                            metadata={
                                "dataset": "CSD",
                                "language": lang_dir,
                                "original_filename": audio_file.name,
                            },
                        )
                        tracks.append(track)

        self.logger.info(f"Loaded {len(tracks)} tracks from CSD dataset")
        return tracks

    def load_jam_alt_dataset(self) -> List[MusicTrack]:
        """
        Load the JAM-ALT dataset.

        Returns:
            List of MusicTrack objects
        """
        self.logger.info("Loading JAM-ALT dataset...")

        jam_dir = Path(self.config.data.raw_audio_dir) / "jam-alt"
        audio_dir = jam_dir / "audio"
        lyrics_dir = jam_dir / "lyrics"

        tracks = []

        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

            for audio_file in audio_files:
                track_id = f"jam_alt_{audio_file.stem}"

                # Look for corresponding lyrics file
                lyrics_file = lyrics_dir / f"{audio_file.stem}.txt"
                lyrics_path = str(lyrics_file) if lyrics_file.exists() else None

                track = MusicTrack(
                    track_id=track_id,
                    audio_path=str(audio_file),
                    lyrics_path=lyrics_path,
                    metadata={
                        "dataset": "JAM_ALT",
                        "original_filename": audio_file.name,
                    },
                )
                tracks.append(track)

        self.logger.info(f"Loaded {len(tracks)} tracks from JAM-ALT dataset")
        return tracks

    def load_viet_dataset(self) -> List[MusicTrack]:
        """
        Load the Vietnamese dataset.

        Returns:
            List of MusicTrack objects
        """
        self.logger.info("Loading Vietnamese dataset...")

        viet_dir = Path(self.config.data.raw_audio_dir) / "Viet_Dataset"
        audio_dir = viet_dir / "songs"
        lyrics_dir = viet_dir / "lyrics"

        tracks = []

        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

            for audio_file in audio_files:
                track_id = f"viet_{audio_file.stem}"

                # Look for corresponding lyrics file
                lyrics_file = lyrics_dir / f"{audio_file.stem}.txt"
                lyrics_path = str(lyrics_file) if lyrics_file.exists() else None

                track = MusicTrack(
                    track_id=track_id,
                    audio_path=str(audio_file),
                    lyrics_path=lyrics_path,
                    metadata={
                        "dataset": "Vietnamese",
                        "language": "vietnamese",
                        "original_filename": audio_file.name,
                    },
                )
                tracks.append(track)

        self.logger.info(f"Loaded {len(tracks)} tracks from Vietnamese dataset")
        return tracks

    def load_100m_dataset(self) -> List[MusicTrack]:
        """
        Load the 100M dataset (partial).

        Returns:
            List of MusicTrack objects
        """
        self.logger.info("Loading 100M dataset...")

        dataset_dir = Path(self.config.data.raw_audio_dir) / "100M"
        tracks = []

        # Look for parquet file with metadata
        parquet_file = dataset_dir / "msd_step1_full_features.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)

                for _, row in df.iterrows():
                    track_id = f"100m_{row.get('track_id', 'unknown')}"

                    # Look for corresponding audio file (h5 format)
                    audio_file = dataset_dir / f"{row.get('track_id', 'unknown')}.h5"
                    audio_path = str(audio_file) if audio_file.exists() else None

                    track = MusicTrack(
                        track_id=track_id,
                        audio_path=audio_path,
                        lyrics_path=None,  # No lyrics in this dataset
                        metadata={
                            "dataset": "100M",
                            "original_metadata": row.to_dict(),
                        },
                    )
                    tracks.append(track)

            except Exception as e:
                self.logger.error(f"Error loading 100M dataset metadata: {e}")

        self.logger.info(f"Loaded {len(tracks)} tracks from 100M dataset")
        return tracks

    def load_all_datasets(self) -> List[MusicTrack]:
        """
        Load all available datasets from the raw data directory.

        Returns:
            Combined list of MusicTrack objects from all datasets
        """
        self.logger.info("Loading all available datasets...")

        all_tracks = []

        # Load each dataset
        dataset_loaders = [
            ("MIREX Mood", self.load_mirex_mood_dataset),
            ("CSD", self.load_csd_dataset),
            ("JAM-ALT", self.load_jam_alt_dataset),
            ("Vietnamese", self.load_viet_dataset),
            # ('100M', self.load_100m_dataset)
        ]

        for dataset_name, loader_func in dataset_loaders:
            try:
                tracks = loader_func()
                all_tracks.extend(tracks)
                self.logger.info(
                    f"Successfully loaded {len(tracks)} tracks from {dataset_name}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load {dataset_name} dataset: {e}")

        self.logger.info(f"Total tracks loaded: {len(all_tracks)}")
        return all_tracks

    def discover_datasets(self) -> Dict[str, Dict]:
        """
        Discover available datasets in the raw data directory.

        Returns:
            Dictionary with dataset information
        """
        raw_dir = Path(self.config.data.raw_audio_dir)
        datasets = {}

        # Check for known dataset structures
        dataset_patterns = {
            "MIREX_mood": {
                "path": raw_dir / "MIREX-like_mood" / "dataset",
                "audio_subdir": "Audio",
                "lyrics_subdir": "Lyrics",
                "has_annotations": True,
            },
            "CSD": {
                "path": raw_dir / "CSD",
                "subdirs": ["english", "korean"],
                "has_annotations": False,
            },
            "JAM_ALT": {
                "path": raw_dir / "jam-alt",
                "audio_subdir": "audio",
                "lyrics_subdir": "lyrics",
                "has_annotations": False,
            },
            "Vietnamese": {
                "path": raw_dir / "Viet_Dataset",
                "audio_subdir": "songs",
                "lyrics_subdir": "lyrics",
                "has_annotations": False,
            },
            "100M": {
                "path": raw_dir / "100M",
                "has_metadata": True,
                "has_annotations": False,
            },
        }

        for dataset_name, config in dataset_patterns.items():
            dataset_path = config["path"]

            if dataset_path.exists():
                info = {
                    "path": str(dataset_path),
                    "exists": True,
                    "audio_files": 0,
                    "lyrics_files": 0,
                    "has_annotations": config.get("has_annotations", False),
                }

                # Count files
                if "audio_subdir" in config:
                    audio_dir = dataset_path / config["audio_subdir"]
                    if audio_dir.exists():
                        info["audio_files"] = len(
                            list(audio_dir.glob("*.mp3"))
                            + list(audio_dir.glob("*.wav"))
                        )

                if "lyrics_subdir" in config:
                    lyrics_dir = dataset_path / config["lyrics_subdir"]
                    if lyrics_dir.exists():
                        info["lyrics_files"] = len(list(lyrics_dir.glob("*.txt")))

                if "subdirs" in config:
                    total_audio = 0
                    for subdir in config["subdirs"]:
                        subdir_path = dataset_path / subdir
                        if subdir_path.exists():
                            total_audio += len(
                                list(subdir_path.glob("**/*.mp3"))
                                + list(subdir_path.glob("**/*.wav"))
                            )
                    info["audio_files"] = total_audio

                datasets[dataset_name] = info
            else:
                datasets[dataset_name] = {"path": str(dataset_path), "exists": False}

        return datasets

    def get_dataset_statistics(self, tracks: List[MusicTrack]) -> Dict:
        """Get basic statistics about the dataset."""
        stats = {
            "total_tracks": len(tracks),
            "tracks_with_audio": sum(
                1 for t in tracks if t.audio_path and os.path.exists(t.audio_path)
            ),
            "tracks_with_lyrics": sum(
                1 for t in tracks if t.lyrics_path and os.path.exists(t.lyrics_path)
            ),
            "datasets": {},
            "mood_clusters": {},
            "mood_categories": {},
            "languages": {},
        }

        # Count distributions
        for track in tracks:
            # Dataset distribution
            dataset = track.metadata.get("dataset", "unknown")
            stats["datasets"][dataset] = stats["datasets"].get(dataset, 0) + 1

            # Language distribution
            language = track.metadata.get("language", "unknown")
            stats["languages"][language] = stats["languages"].get(language, 0) + 1

            # Mood distributions (only for MIREX dataset)
            if track.mood_cluster:
                stats["mood_clusters"][track.mood_cluster] = (
                    stats["mood_clusters"].get(track.mood_cluster, 0) + 1
                )
            if track.mood_category:
                stats["mood_categories"][track.mood_category] = (
                    stats["mood_categories"].get(track.mood_category, 0) + 1
                )

        return stats
