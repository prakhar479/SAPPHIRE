# SAPPHIRE Quick Start Guide

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SAPPHIRE
```

### 2. Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or install in development mode
pip install -e .

# With optional advanced features
pip install -e .[all]
```

### 3. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

## Data Setup

### Expected Directory Structure
```
data/raw/
├── MIREX-like_mood/
│   └── dataset/
│       ├── Audio/          # 903 MP3 files (001.mp3 to 903.mp3)
│       ├── Lyrics/         # 764 TXT files (001.txt to 903.txt)
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

## Quick Usage

### 1. Run Example Script
```bash
python example_usage.py
```

### 2. Basic Pipeline Usage
```python
from pipeline import Pipeline

# Initialize and run pipeline
pipeline = Pipeline()
results = pipeline.run_full_pipeline()

print(f"Processed {results['tracks_processed']} tracks")
print(f"Results saved to: {results['output_directory']}")
```

### 3. Individual Components
```python
from pipeline import DataLoader, FeatureExtractor

# Load data
loader = DataLoader()
tracks = loader.load_mirex_mood_dataset()

# Extract features
extractor = FeatureExtractor()
for track in tracks[:5]:  # Process first 5 tracks
    track = loader.load_audio(track)
    features = extractor.extract_all_features(track)
    print(f"Extracted {len(features)} features for {track.track_id}")
```

### 4. Mood Prediction
```python
# After training models
result = pipeline.predict_mood(
    audio_path='path/to/song.mp3',
    lyrics_path='path/to/lyrics.txt'  # optional
)

print(f"Predicted mood: {result['predicted_mood']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Configuration

### Basic Configuration
```python
from pipeline import config

# Audio settings
config.audio.sample_rate = 22050
config.audio.n_mfcc = 13

# Feature extraction
config.features.extract_acoustic = True
config.features.extract_lyrics = True

# Model training
config.model.models = ['random_forest', 'svm']
config.model.test_size = 0.2
```

### Custom Configuration File
```python
from pipeline import Config

# Load from JSON file
custom_config = Config.from_file('my_config.json')
pipeline = Pipeline(custom_config)
```

## Output Structure

After running the pipeline, you'll find:

```
output/
├── data/processed/features/
│   ├── all_features.parquet    # Main feature dataset
│   └── all_features.csv        # CSV version
├── analysis_results.json       # Detailed analysis results
├── best_model.joblib          # Trained classification model
├── visualizations/            # All plots and charts
│   ├── feature_distributions/
│   ├── mood_analysis/
│   ├── correlations/
│   └── model_performance/
├── pipeline_report.json       # Machine-readable report
└── PIPELINE_REPORT.md         # Human-readable summary
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg system-wide
   sudo apt-get install ffmpeg  # Ubuntu
   brew install ffmpeg          # macOS
   ```

2. **Memory issues with large datasets**
   ```python
   # Reduce batch size and workers
   config.processing.batch_size = 16
   config.processing.workers = 2
   ```

3. **Missing datasets**
   ```python
   # Check available datasets
   from pipeline import DataLoader
   loader = DataLoader()
   datasets = loader.discover_datasets()
   print(datasets)
   ```

4. **NLTK data missing**
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   ```

### Performance Tips

1. **Use fewer models for faster training**
   ```python
   config.model.models = ['random_forest']  # Instead of all models
   ```

2. **Limit feature extraction**
   ```python
   config.features.extract_lyrics = False  # Skip lyrics if not needed
   config.features.use_advanced_features = False
   ```

3. **Process subset of data**
   ```python
   # Process only first N tracks
   tracks = loader.load_mirex_mood_dataset()[:100]
   ```

## Next Steps

1. **Explore Results**: Check the generated visualizations and reports
2. **Customize Features**: Add your own feature extractors
3. **Extend Datasets**: Add support for your own datasets
4. **Tune Models**: Experiment with different ML algorithms
5. **Cross-Modal Analysis**: Investigate acoustic-lyrical relationships

## Getting Help

- Check the main README.md for detailed documentation
- Review example_usage.py for code examples
- Examine the generated reports for insights
- Look at individual pipeline components for specific functionality

## Citation

If you use SAPPHIRE in your research:

```bibtex
@software{sapphire2024,
  title={SAPPHIRE: Semantic and Acoustic Perceptual Holistic Integration REtrieval},
  author={SAPPHIRE Team},
  year={2024},
  version={1.0.0}
}
```