# SAPPHIRE Developer Guide

This guide is intended for developers who want to contribute to or extend the SAPPHIRE pipeline.

## Development Environment Setup

### Prerequisites

- Python 3.8+
- Git
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SAPPHIRE.git
   cd SAPPHIRE
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optional dependencies (recommended):**
   ```bash
   pip install crepe pyloudnorm langdetect plotly
   ```

## Project Structure

```
SAPPHIRE/
├── data/               # Data directory (raw and processed)
├── docs/               # Documentation
├── pipeline/           # Source code
│   ├── analyzer.py     # Analysis logic
│   ├── config.py       # Configuration
│   ├── data_loader.py  # Data loading
│   ├── feature_extractor.py # Feature extraction
│   ├── mood_classifier.py   # ML models
│   ├── pipeline.py     # Main orchestrator
│   ├── preprocessor.py # Data cleaning
│   └── visualizer.py   # Plotting
├── cli.py              # Command-line interface
└── requirements.txt    # Dependencies
```

## Extending the Pipeline

### Adding a New Dataset

To add support for a new dataset:

1. Modify `pipeline/data_loader.py`.
2. Add a new method `load_your_dataset()` to the `DataLoader` class.
3. Ensure it returns a list of `MusicTrack` objects with standardized metadata.
4. Update `load_all_datasets()` to include your new method.

Example:
```python
def load_new_dataset(self) -> List[MusicTrack]:
    tracks = []
    # ... logic to find files ...
    for file in files:
        track = MusicTrack(
            track_id=file.stem,
            audio_path=str(file),
            metadata={"dataset": "NewDataset"}
        )
        tracks.append(track)
    return tracks
```

### Adding a New Feature

To add a new feature extractor:

1. Modify `pipeline/feature_extractor.py`.
2. Add a new method `_extract_your_feature(y, sr)` to the `FeatureExtractor` class.
3. Call this method in `extract_features()`.
4. Update `pipeline/config.py` to add a toggle for your feature.

Example:
```python
def _extract_custom_feature(self, y, sr) -> Dict[str, Any]:
    # ... calculation logic ...
    return {"custom_feature_mean": value}
```

### Adding a New Model

To add a new machine learning model:

1. Modify `pipeline/mood_classifier.py`.
2. Import your model class (must follow scikit-learn interface).
3. Add the model configuration to `self.model_configs` in `__init__`.
4. Update `pipeline/config.py` to include the model name in default models.

Example:
```python
"new_model": {
    "model": NewModelClass,
    "params": {
        "param1": [1, 10],
        "param2": [0.1, 0.5]
    }
}
```

## Code Style

- Follow PEP 8 guidelines.
- Use type hints for function arguments and return values.
- Write docstrings for all classes and methods (Google style).
- Use `logging` instead of `print` statements.

## Testing

(Add instructions for running tests if applicable, e.g., `pytest`)
Currently, you can run the pipeline on a small subset of data to verify changes:

```bash
python cli.py pipeline --limit 5 --output test_output/
```
