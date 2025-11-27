# SAPPHIRE API Reference

This document provides a detailed reference for the SAPPHIRE pipeline API.

## Table of Contents

1. [Data Loader](#data-loader)
2. [Preprocessor](#preprocessor)
3. [Feature Extractor](#feature-extractor)
4. [Analyzer](#analyzer)
5. [Mood Classifier](#mood-classifier)
6. [Visualizer](#visualizer)
7. [Pipeline Orchestrator](#pipeline-orchestrator)

---

## Data Loader

**Module**: `pipeline.data_loader`

Handles loading of audio files, lyrics, and mood annotations from various datasets.

### `class MusicTrack`

Represents a single music track with all associated data.

**Attributes:**
- `track_id` (str): Unique identifier for the track.
- `audio_path` (str, optional): Path to the audio file.
- `lyrics_path` (str, optional): Path to the lyrics file.
- `audio_data` (np.ndarray, optional): Loaded audio data.
- `sample_rate` (int, optional): Sample rate of the audio data.
- `lyrics_text` (str, optional): Loaded lyrics text.
- `mood_cluster` (str, optional): Mood cluster annotation.
- `mood_category` (str, optional): Mood category annotation.
- `metadata` (Dict, optional): Additional metadata.

### `class DataLoader`

Comprehensive data loader for the SAPPHIRE pipeline.

**Methods:**

#### `__init__(config_obj=None)`
Initialize the data loader.
- `config_obj`: Optional configuration object.

#### `load_mirex_mood_dataset() -> List[MusicTrack]`
Load the MIREX-like mood dataset with audio, lyrics, and mood annotations.

#### `load_audio(track: MusicTrack, load_data: bool = True) -> MusicTrack`
Load audio data for a track.
- `track`: MusicTrack object.
- `load_data`: Whether to load the actual audio data (default: True).

#### `load_lyrics(track: MusicTrack) -> MusicTrack`
Load lyrics text for a track.
- `track`: MusicTrack object.

#### `load_all_datasets() -> List[MusicTrack]`
Load all available datasets from the raw data directory.

#### `discover_datasets() -> Dict[str, Dict]`
Discover available datasets in the raw data directory.

---

## Preprocessor

**Module**: `pipeline.preprocessor`

Handles audio normalization, quality filtering, and data cleaning.

### `class Preprocessor`

Comprehensive preprocessor for music data.

**Methods:**

#### `preprocess_track(track: MusicTrack) -> Tuple[MusicTrack, Dict]`
Preprocess a single track.
- `track`: MusicTrack object.
- **Returns**: Tuple of (processed_track, quality_metrics).

#### `preprocess_batch(tracks: List[MusicTrack], n_workers: int = None) -> Tuple[List[MusicTrack], pd.DataFrame]`
Preprocess multiple tracks in parallel.
- `tracks`: List of MusicTrack objects.
- `n_workers`: Number of parallel workers.

#### `filter_high_quality_tracks(tracks: List[MusicTrack], quality_df: pd.DataFrame) -> List[MusicTrack]`
Filter tracks that pass quality checks.

---

## Feature Extractor

**Module**: `pipeline.feature_extractor`

Comprehensive feature extraction for acoustic, rhythm, harmonic, lyrical, and quality features.

### `class FeatureContainer`

Container for extracted features. Behaves like a dictionary.

### `class FeatureExtractor`

**Methods:**

#### `extract_features(track: MusicTrack) -> FeatureContainer`
Extract all features from a music track.
- `track`: MusicTrack object with loaded audio and lyrics data.

#### `extract_batch(tracks: List[MusicTrack], n_workers: int = None) -> List[FeatureContainer]`
Extract features for multiple tracks in parallel.

---

## Analyzer

**Module**: `pipeline.analyzer`

Handles statistical analysis, clustering, and feature importance analysis.

### `class Analyzer`

**Methods:**

#### `analyze_dataset(features_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]`
Perform comprehensive dataset analysis.
- `features_df`: DataFrame with extracted features.
- `output_dir`: Directory to save analysis results.

#### `perform_clustering(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]`
Perform clustering analysis (K-means, Hierarchical, DBSCAN).

#### `compute_cross_modal_similarity(df: pd.DataFrame) -> Dict[str, float]`
Compute cross-modal similarity to measure the "perceptual gap".

---

## Mood Classifier

**Module**: `pipeline.mood_classifier`

Implements machine learning models for mood prediction.

### `class MoodClassifier`

**Methods:**

#### `prepare_data(features_df: pd.DataFrame, target_column: str = "mood_cluster") -> Tuple[np.ndarray, np.ndarray]`
Prepare features and targets for training.

#### `train_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]`
Train multiple models (Random Forest, SVM, etc.) with hyperparameter tuning.

#### `evaluate_models(results: Dict[str, Dict], output_dir: str)`
Generate comprehensive evaluation reports (classification report, confusion matrix).

#### `predict(features: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]`
Make predictions using the best trained model.

#### `save_model(filepath: str)`
Save the trained model and preprocessing components.

#### `load_model(filepath: str)`
Load a trained model.

#### `cross_modal_analysis(features_df: pd.DataFrame) -> Dict[str, Any]`
Analyze cross-modal relationships between acoustic and lyrical features.

---

## Visualizer

**Module**: `pipeline.visualizer`

Creates comprehensive visualizations for feature analysis and model results.

### `class Visualizer`

**Methods:**

#### `create_comprehensive_report(features_df: pd.DataFrame, analysis_results: Dict, model_results: Dict, output_dir: str)`
Create a comprehensive visualization report including all plots.

#### `create_feature_distribution_plots(features_df: pd.DataFrame, output_dir: str)`
Create distribution plots for all features.

#### `create_mood_analysis_plots(features_df: pd.DataFrame, output_dir: str)`
Create mood-specific analysis plots.

#### `create_correlation_heatmap(features_df: pd.DataFrame, output_dir: str)`
Create correlation heatmap for features.

#### `create_pca_visualization(features_df: pd.DataFrame, output_dir: str)`
Create PCA visualization of features.

---

## Pipeline Orchestrator

**Module**: `pipeline.pipeline`

Main pipeline orchestrator that coordinates all components.

### `class Pipeline`

**Methods:**

#### `run_full_pipeline(datasets: List[str] = None, output_dir: str = None, use_enhanced_processing: bool = True, limit_tracks: int = None) -> Dict[str, Any]`
Run the complete SAPPHIRE pipeline.
- `datasets`: List of datasets to process (None for all).
- `output_dir`: Output directory for results.
- `use_enhanced_processing`: Whether to use enhanced processing pipeline.
- `limit_tracks`: Limit number of tracks to process.

#### `predict_mood(audio_path: str, lyrics_path: str = None) -> Dict[str, Any]`
Predict mood for a new track using the trained model.
