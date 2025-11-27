# SAPPHIRE Configuration Guide

The SAPPHIRE pipeline is highly configurable to adapt to different datasets and analysis requirements. The configuration is managed by the `pipeline.config` module.

## Configuration Structure

The configuration is organized into several sections:

1. **Audio Config**: Audio processing parameters.
2. **Processing Config**: Pipeline execution settings.
3. **Feature Config**: Feature extraction toggles.
4. **Data Config**: File paths and directory structure.
5. **Model Config**: Machine learning model parameters.
6. **Quality Thresholds**: Criteria for data quality filtering.
7. **Filtering**: Advanced filtering options.

## Modifying Configuration

You can modify the configuration in two ways:

### 1. Programmatically

```python
from pipeline import config

# Modify settings directly
config.audio.sample_rate = 44100
config.processing.workers = 8
config.features.extract_lyrics = False
```

### 2. Configuration File

You can load configuration from a JSON file:

```python
from pipeline import config

# Load from file
config = config.from_file('my_config.json')
```

## Configuration Reference

### Audio Configuration (`config.audio`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 22050 | Target sample rate for audio loading (Hz). |
| `n_fft` | 2048 | FFT window size. |
| `hop_length` | 512 | Hop length for STFT. |
| `n_mels` | 128 | Number of Mel bands. |
| `n_mfcc` | 13 | Number of MFCCs to extract. |
| `target_loudness` | -23.0 | Target loudness in LUFS for normalization. |

### Processing Configuration (`config.processing`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workers` | 4 | Number of parallel worker processes. |
| `batch_size` | 32 | Batch size for processing. |
| `use_gpu` | False | Whether to use GPU acceleration (if available). |
| `chunk_size` | 1000 | Number of tracks to process per chunk (for large datasets). |
| `memory_limit_gb` | 8.0 | Memory usage limit in GB. |

### Feature Configuration (`config.features`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `extract_acoustic` | True | Extract acoustic features (MFCC, spectral). |
| `extract_rhythm` | True | Extract rhythm features (tempo, beat). |
| `extract_harmony` | True | Extract harmonic features (chroma, key). |
| `extract_lyrics` | True | Extract lyrical features (sentiment, embedding). |
| `extract_quality` | True | Extract quality metrics. |
| `use_advanced_features` | True | Extract advanced spectral features. |

### Data Configuration (`config.data`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `raw_audio_dir` | "data/raw" | Directory containing raw audio datasets. |
| `processed_dir` | "data/processed" | Directory for processed data. |
| `output_dir` | "output" | Directory for pipeline outputs. |

### Model Configuration (`config.model`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_size` | 0.2 | Proportion of data to use for testing. |
| `cv_folds` | 5 | Number of cross-validation folds. |
| `feature_selection_method` | "mutual_info" | Method for feature selection (`mutual_info`, `f_score`, `rfe`). |
| `max_features` | None | Maximum number of features to select (None for all). |
| `models` | [...] | List of models to train (e.g., `random_forest`, `svm`). |

### Quality Thresholds (`config.quality_thresholds`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `snr_min` | 5.0 | Minimum Signal-to-Noise Ratio (dB). |
| `duration_min` | 3.0 | Minimum track duration (seconds). |
| `duration_max` | 900.0 | Maximum track duration (seconds). |
| `lyrics_completeness_min` | 0.2 | Minimum lyrics completeness score (0-1). |
| `vocal_dominance_min` | 0.1 | Minimum vocal dominance score (0-1). |

### Filtering Options (`config.filtering`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_quality_filter` | True | Enable/disable quality filtering. |
| `enable_language_filter` | False | Enable/disable language filtering. |
| `allowed_languages` | [...] | List of allowed language codes (e.g., `['en', 'es']`). |
| `enable_duplicate_detection` | False | Enable/disable duplicate track detection. |

## Lenient Configuration

For datasets with varied quality (e.g., noisy recordings, incomplete lyrics), you can use the built-in lenient configuration:

```python
from pipeline import config

# Apply lenient settings
config = config.create_lenient_config()
```

This adjusts thresholds to be more permissive:
- Lower SNR requirement (3.0 dB)
- Shorter minimum duration (1.0 s)
- Lower lyrics completeness (0.1)
- Disables most strict filters
