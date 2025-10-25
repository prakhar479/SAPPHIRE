# SAPPHIRE — Semantic and Acoustic Perceptual Holistic Integration REtrieval

![SAPPHIRE Logo](./doc/images/logo.jpg)

## Overview

SAPPHIRE is a comprehensive music analysis pipeline that addresses the "perceptual gap" between machine audio analysis and human music perception through multi-modal feature extraction and cross-modal contrastive learning. The system integrates acoustic, rhythmic, harmonic, and lyrical features to provide holistic music understanding and mood classification.

## Key Features

- **Multi-Modal Feature Extraction**: Comprehensive extraction of acoustic, rhythm, harmonic, lyrical, and quality features
- **Cross-Modal Analysis**: Quantifies the "perceptual gap" using Jaccard similarity and correlation analysis
- **Multiple Dataset Support**: Handles MIREX-like mood, CSD, JAM-ALT, Vietnamese, and 100M datasets
- **Advanced Mood Classification**: Machine learning models for mood prediction with multiple algorithms
- **Comprehensive Visualization**: Rich visualizations for feature analysis and model performance
- **Quality Assessment**: Audio quality filtering and lyrics completeness evaluation
- **Modular Architecture**: Flexible pipeline components that can be used independently

## Architecture

The SAPPHIRE pipeline consists of several key components:

### 1. Data Loading (`pipeline/data_loader.py`)
- **MusicTrack**: Data structure representing a music track with audio, lyrics, and metadata
- **DataLoader**: Handles loading from multiple datasets with automatic discovery
- Supports MIREX mood dataset (903 audio clips, 764 lyrics), CSD (English/Korean), JAM-ALT, Vietnamese, and 100M datasets

### 2. Preprocessing (`pipeline/preprocessor.py`)
- Audio normalization and quality assessment
- Loudness normalization to -23 LUFS
- SNR estimation and quality filtering
- Lyrics language detection and completeness evaluation

### 3. Feature Extraction (`pipeline/feature_extractor.py`)
Multi-modal feature extraction including:

#### Acoustic Features
- MFCCs (13 coefficients)
- Spectral features (centroid, bandwidth, rolloff, flatness)
- Zero crossing rate
- RMS energy

#### Rhythm Features
- Tempo estimation
- Beat tracking
- Rhythm patterns
- Onset detection

#### Harmonic Features
- Chroma features (12-dimensional)
- Key estimation
- Harmonic-percussive separation
- Tonal centroid features

#### Lyrical Features
- Sentiment analysis (VADER)
- Semantic embeddings (Sentence Transformers)
- Readability metrics
- Language detection

#### Quality Features
- Audio quality metrics (SNR, dynamic range)
- Vocal dominance estimation
- Lyrics completeness assessment

### 4. Analysis (`pipeline/analyzer.py`)
- Statistical analysis and feature importance
- PCA dimensionality reduction
- Clustering analysis (K-means, hierarchical, DBSCAN)
- Cross-modal similarity computation

### 5. Mood Classification (`pipeline/mood_classifier.py`)
- Multiple ML algorithms (Random Forest, SVM, Neural Networks, etc.)
- Hyperparameter tuning with GridSearchCV
- Feature selection (mutual information, f-score, RFE)
- Cross-modal analysis with Jaccard similarity

### 6. Visualization (`pipeline/visualizer.py`)
- Feature distribution plots
- Mood analysis visualizations
- Correlation heatmaps
- PCA and t-SNE visualizations
- Model performance comparisons

### 7. Pipeline Orchestrator (`pipeline/pipeline.py`)
- End-to-end pipeline execution
- Configuration management
- Result reporting and export

## Dataset Structure

The pipeline expects the following directory structure:

```
data/raw/
├── MIREX-like_mood/
│   └── dataset/
│       ├── Audio/          # 903 MP3 files
│       ├── Lyrics/         # 764 TXT files
│       ├── clusters.txt    # Mood cluster annotations
│       └── categories.txt  # Mood category annotations
├── CSD/
│   ├── english/
│   │   ├── wav/           # Audio files
│   │   └── txt/           # Lyrics files
│   └── korean/
│       ├── wav/           # Audio files
│       └── txt/           # Lyrics files
├── jam-alt/
│   ├── audio/             # MP3 files
│   └── lyrics/            # TXT files
├── Viet_Dataset/
│   ├── songs/             # WAV files
│   └── lyrics/            # TXT files
└── 100M/                  # Million Song Dataset subset
```

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)

### Required Packages
```bash
pip install librosa pandas numpy scikit-learn matplotlib seaborn
pip install soundfile nltk textstat sentence-transformers
pip install pyloudnorm langdetect plotly tqdm joblib
```

### Optional Dependencies
```bash
pip install crepe  # For advanced pitch estimation
```

## Usage

### Basic Pipeline Execution

```python
from pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline()

# Run full pipeline on all datasets
results = pipeline.run_full_pipeline()

# Run on specific datasets
results = pipeline.run_full_pipeline(
    datasets=['mirex', 'csd'], 
    output_dir='output/analysis'
)
```

### Individual Component Usage

```python
from pipeline import DataLoader, FeatureExtractor, MoodClassifier

# Load data
loader = DataLoader()
tracks = loader.load_mirex_mood_dataset()

# Extract features
extractor = FeatureExtractor()
features = []
for track in tracks:
    track = loader.load_audio(track)
    track = loader.load_lyrics(track)
    feature_dict = extractor.extract_features(track)
    features.append(feature_dict)

# Train mood classifier
classifier = MoodClassifier()
features_df = pd.DataFrame(features)
X, y = classifier.prepare_data(features_df)
model_results = classifier.train_models(X, y)
```

### Mood Prediction for New Tracks

```python
# Predict mood for a new track
result = pipeline.predict_mood(
    audio_path='path/to/audio.mp3',
    lyrics_path='path/to/lyrics.txt'  # optional
)

print(f"Predicted mood: {result['predicted_mood']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Configuration

The pipeline uses a comprehensive configuration system (`pipeline/config.py`):

```python
from pipeline import config

# Audio processing settings
config.audio.sample_rate = 22050
config.audio.n_mfcc = 13

# Feature extraction settings
config.features.extract_acoustic = True
config.features.extract_lyrics = True

# Model training settings
config.model.models = ['random_forest', 'svm', 'neural_network']
config.model.test_size = 0.2
```

## Mood Categories

The system classifies music into 5 mood clusters based on the MIREX-like dataset:

1. **Cluster 1**: Boisterous, Confident, Passionate, Rousing, Rowdy
2. **Cluster 2**: Amiable-good natured, Cheerful, Fun, Rollicking, Sweet
3. **Cluster 3**: Autumnal, Bittersweet, Brooding, Literate, Poignant, Wistful
4. **Cluster 4**: Campy, Humorous, Silly, Whimsical, Witty, Wry
5. **Cluster 5**: Aggressive, Fiery, Intense, Tense-Anxious, Visceral, Volatile

## Cross-Modal Analysis

A key innovation of SAPPHIRE is the cross-modal analysis that quantifies the "perceptual gap" between acoustic and lyrical features:

- **Jaccard Similarity**: Measures overlap between acoustic and lyrical feature spaces
- **Cross-Correlation Analysis**: Identifies relationships between different modalities
- **Perceptual Gap Quantification**: Provides metrics for human-machine perception alignment

## Output Structure

The pipeline generates comprehensive outputs:

```
output/
├── data/processed/features/
│   ├── all_features.parquet    # Extracted features
│   └── all_features.csv
├── analysis_results.json       # Analysis results
├── best_model.joblib          # Trained model
├── visualizations/            # All plots and charts
│   ├── feature_distributions/
│   ├── mood_analysis/
│   ├── correlations/
│   ├── dimensionality_reduction/
│   └── model_performance/
├── pipeline_report.json       # Detailed report
└── PIPELINE_REPORT.md         # Summary report
```

## Performance Metrics

The system achieves:
- **Multi-modal Feature Extraction**: 100+ features per track
- **Processing Speed**: ~2-3 seconds per track (depending on length)
- **Classification Accuracy**: Varies by dataset and model (typically 60-80%)
- **Cross-modal Similarity**: Quantified via Jaccard index and correlation analysis

## Research Applications

SAPPHIRE is designed for:
- Music Information Retrieval (MIR) research
- Mood and emotion recognition studies
- Cross-modal learning investigations
- Music recommendation systems
- Audio-lyrical alignment analysis

## Acknowledgments

- MIREX community for mood annotation standards
- Librosa team for audio processing tools
- Scikit-learn for machine learning algorithms
- All dataset contributors and researchers in the MIR community