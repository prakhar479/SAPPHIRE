"""
Configuration settings for the SAPPHIRE music analysis pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    frame_length: int = 2048
    target_loudness: float = -23.0  # LUFS


@dataclass
class ProcessingConfig:
    """Processing configuration."""

    workers: int = 4
    batch_size: int = 32
    use_gpu: bool = False
    force_recompute: bool = False
    save_intermediate: bool = True
    parallel_feature_extraction: bool = True
    chunk_size: int = 1000  # For large dataset processing
    memory_limit_gb: float = 8.0  # Memory usage limit
    cache_features: bool = True
    progress_bar: bool = True


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""

    extract_acoustic: bool = True
    extract_rhythm: bool = True
    extract_harmony: bool = True
    extract_lyrics: bool = True
    extract_quality: bool = True
    use_advanced_features: bool = True


@dataclass
class DataConfig:
    """Data paths configuration."""

    # Raw data paths
    raw_audio_dir: str = "data/raw"
    raw_lyrics_dir: str = "data/raw"
    mirex_mood_dir: str = "data/raw/MIREX-like_mood/dataset"

    # Processed data paths
    processed_dir: str = "data/processed"
    features_dir: str = "data/processed/features"
    analysis_dir: str = "data/processed/analysis"
    models_dir: str = "data/processed/models"

    # Output paths
    output_dir: str = "output"
    reports_dir: str = "output/reports"
    visualizations_dir: str = "output/visualizations"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in dir(self):
            if attr_name.endswith("_dir") and not attr_name.startswith("_"):
                path = Path(getattr(self, attr_name))
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model training configuration."""

    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5

    # Feature selection
    feature_selection_method: str = "mutual_info"  # mutual_info, f_score, rfe
    max_features: Optional[int] = None
    use_importance_subset: bool = False
    importance_csv_path: Optional[str] = None
    top_n_important: Optional[int] = None
    
    # Models to train
    models: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = [
                "random_forest",
                "gradient_boosting",
                "svm",
                "logistic_regression",
                "neural_network",
            ]


class Config:
    """Main configuration class."""

    def __init__(self):
        self.audio = AudioConfig()
        self.processing = ProcessingConfig()
        self.features = FeatureConfig()
        self.data = DataConfig()
        self.model = ModelConfig()

        # Quality thresholds and filtering - LENIENT SETTINGS for varied quality datasets
        self.quality_thresholds = {
            "snr_min": 5.0,  # dB - Much more lenient for noisy recordings
            "duration_min": 3.0,  # seconds - Allow very short clips
            "duration_max": 900.0,  # seconds - Allow longer tracks
            "lyrics_completeness_min": 0.2,  # Much more lenient for incomplete lyrics
            "vocal_dominance_min": 0.1,  # Allow instrumental and low-vocal tracks
            "silence_threshold": 0.005,  # More sensitive silence detection
            "dynamic_range_min": 5.0,  # dB - Allow compressed audio
            "spectral_centroid_max": 12000.0,  # Hz - Allow higher frequency content
            "zero_crossing_rate_max": 0.5,  # Allow noisier audio
        }

        # Advanced filtering options - LENIENT SETTINGS for varied quality datasets
        self.filtering = {
            "enable_quality_filter": True,  # Keep enabled but with lenient thresholds
            "enable_duration_filter": False,  # Disable duration filtering to allow short clips
            "enable_silence_filter": False,  # Disable silence filtering for varied content
            "enable_language_filter": False,  # Keep disabled for multilingual support
            "allowed_languages": [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "ko",
                "vi",
                "zh",
            ],  # Extended language support
            "min_lyrics_words": 3,  # Very low threshold for lyrics
            "max_instrumental_ratio": 0.95,  # Allow mostly instrumental tracks
            "enable_duplicate_detection": False,  # Disable to avoid removing similar but different tracks
            "similarity_threshold": 0.98,  # Higher threshold if enabled
            "enable_outlier_detection": False,  # Disable to preserve diverse content
            "outlier_method": "isolation_forest",  # isolation_forest, local_outlier_factor
            "contamination": 0.05,  # Lower contamination rate if enabled
        }

        # Processing optimization
        self.optimization = {
            "use_multiprocessing": True,
            "prefetch_audio": True,
            "lazy_loading": True,
            "feature_caching": True,
            "incremental_processing": True,
            "checkpoint_frequency": 100,  # Save progress every N tracks
            "resume_from_checkpoint": True,
        }

        # Mood categories mapping
        self.mood_categories = {
            "Cluster 1": ["Boisterous", "Confident", "Passionate", "Rousing", "Rowdy"],
            "Cluster 2": [
                "Amiable-good natured",
                "Cheerful",
                "Fun",
                "Rollicking",
                "Sweet",
            ],
            "Cluster 3": [
                "Autumnal",
                "Bittersweet",
                "Brooding",
                "Literate",
                "Poignant",
                "Wistful",
            ],
            "Cluster 4": ["Campy", "Humorous", "Silly", "whimsical", "Witty", "Wry"],
            "Cluster 5": [
                "Agressive",
                "Fiery",
                "Intense",
                "Tense - Anxious",
                "Visceral",
                "Volatile",
            ],
        }

        # Reverse mapping for mood to cluster
        self.mood_to_cluster = {}
        for cluster, moods in self.mood_categories.items():
            for mood in moods:
                self.mood_to_cluster[mood] = cluster

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        import json

        config = cls()
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
                # Update configuration with loaded data
                for section, values in data.items():
                    if hasattr(config, section):
                        section_obj = getattr(config, section)
                        for key, value in values.items():
                            if hasattr(section_obj, key):
                                setattr(section_obj, key, value)

        return config

    @classmethod
    def create_lenient_config(cls) -> "Config":
        """
        Create a lenient configuration optimized for varied quality datasets.
        This configuration is designed to handle:
        - Short audio clips (3+ seconds)
        - Lower quality recordings
        - Incomplete lyrics
        - Mixed languages
        - Instrumental tracks
        """
        config = cls()

        # Very lenient quality thresholds
        config.quality_thresholds.update(
            {
                "snr_min": 3.0,  # Very low SNR threshold
                "duration_min": 1.0,  # Allow very short clips
                "duration_max": 1200.0,  # Allow very long tracks
                "lyrics_completeness_min": 0.1,  # Very lenient lyrics requirement
                "vocal_dominance_min": 0.05,  # Allow mostly instrumental
                "dynamic_range_min": 3.0,  # Allow heavily compressed audio
                "spectral_centroid_max": 15000.0,  # Allow high-frequency content
                "zero_crossing_rate_max": 0.7,  # Allow very noisy audio
            }
        )

        # Disable most filtering
        config.filtering.update(
            {
                "enable_quality_filter": True,  # Keep basic quality checks
                "enable_duration_filter": False,  # No duration limits
                "enable_silence_filter": False,  # No silence filtering
                "enable_language_filter": False,  # No language restrictions
                "min_lyrics_words": 1,  # Accept any lyrics
                "enable_duplicate_detection": False,  # No duplicate removal
                "enable_outlier_detection": False,  # No outlier removal
            }
        )

        # Optimize for robustness
        config.processing.chunk_size = 100  # Smaller chunks for stability
        config.processing.workers = min(
            4, config.processing.workers
        )  # Conservative worker count

        return config

    def save(self, config_path: str):
        """Save configuration to file."""
        import json

        data = {
            "audio": self.audio.__dict__,
            "processing": self.processing.__dict__,
            "features": self.features.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "quality_thresholds": self.quality_thresholds,
            "mood_categories": self.mood_categories,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)


# Global configuration instance
config = Config()
