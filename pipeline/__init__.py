"""
SAPPHIRE Music Analysis Pipeline

A comprehensive pipeline for music feature extraction, analysis, and mood classification.
"""

from .config import config, Config
from .data_loader import DataLoader, MusicTrack
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor
from .analyzer import Analyzer
from .mood_classifier import MoodClassifier
from .visualizer import Visualizer
from .pipeline import Pipeline
from .processing_pipeline import EnhancedProcessingPipeline

__version__ = "1.0.0"
__author__ = "SAPPHIRE Team"

__all__ = [
    "config",
    "Config",
    "DataLoader",
    "MusicTrack",
    "FeatureExtractor",
    "Preprocessor",
    "Analyzer",
    "MoodClassifier",
    "Visualizer",
    "Pipeline",
    "EnhancedProcessingPipeline",
]
