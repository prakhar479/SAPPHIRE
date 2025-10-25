# SAPPHIRE CLI Guide

The SAPPHIRE CLI provides comprehensive command-line access to all pipeline functionality with advanced processing options, filtering, and optimization features.

## Installation and Setup

```bash
# Make CLI executable
chmod +x cli.py

# Or run with Python
python cli.py --help
```

## Available Commands

### 1. Dataset Discovery

Discover and inspect available datasets in your data directory:

```bash
# Basic discovery
python cli.py discover

# Example output:
# 🔍 Discovering available datasets...
# 
# 📊 Dataset Discovery Results:
# ==================================================
# 
# MIREX_mood: ✅ Available
#   📁 Path: data/raw/MIREX-like_mood/dataset
#   🎵 Audio files: 903
#   📝 Lyrics files: 764
#   🏷️  Has mood annotations: Yes
# 
# CSD: ✅ Available
#   📁 Path: data/raw/CSD
#   🎵 Audio files: 150
#   📝 Lyrics files: 150
```

### 2. Feature Extraction

Extract features from specific datasets with optimization:

```bash
# Basic feature extraction
python cli.py extract-features --datasets mirex --output features/

# Advanced options
python cli.py extract-features \
    --datasets mirex,csd \
    --output features/ \
    --limit 100 \
    --workers 4 \
    --sample-rate 22050

# With custom configuration
python cli.py extract-features \
    --datasets mirex \
    --output features/ \
    --config my_config.json \
    --log-level DEBUG
```

### 3. Model Training

Train mood classification models on extracted features:

```bash
# Basic training
python cli.py train \
    --features features/features.parquet \
    --output models/

# Specify models and feature selection
python cli.py train \
    --features features/features.parquet \
    --output models/ \
    --models random_forest,svm,neural_network \
    --max-features 50

# With custom configuration
python cli.py train \
    --features features/features.parquet \
    --output models/ \
    --config my_config.json
```

### 4. Comprehensive Analysis

Perform statistical analysis and create visualizations:

```bash
# Basic analysis
python cli.py analyze \
    --features features/features.parquet \
    --output analysis/

# With custom configuration
python cli.py analyze \
    --features features/features.parquet \
    --output analysis/ \
    --config my_config.json
```

### 5. Mood Prediction

Predict mood for individual tracks:

```bash
# Basic prediction
python cli.py predict \
    --audio path/to/song.mp3 \
    --model models/best_model.joblib

# With lyrics
python cli.py predict \
    --audio path/to/song.mp3 \
    --lyrics path/to/lyrics.txt \
    --model models/best_model.joblib

# Example output:
# 🎯 Predicting mood...
# 
# ✅ Prediction complete!
# 🎵 Audio: path/to/song.mp3
# 📝 Lyrics: path/to/lyrics.txt
# 🎭 Predicted mood: Cluster 2
# 🎯 Confidence: 0.847
# 
# 📊 All probabilities:
#   Cluster 1: 0.123
#   Cluster 2: 0.847
#   Cluster 3: 0.030
```

### 6. Full Pipeline Execution

Run the complete SAPPHIRE pipeline with advanced options:

```bash
# Basic pipeline
python cli.py pipeline --datasets mirex --output results/

# Enhanced pipeline with optimization
python cli.py pipeline \
    --datasets mirex,csd \
    --output results/ \
    --workers 4 \
    --memory-limit 8.0 \
    --chunk-size 500 \
    --models random_forest,svm

# With filtering options
python cli.py pipeline \
    --datasets mirex \
    --output results/ \
    --disable-duplicate-detection \
    --disable-outlier-detection

# Limited processing for testing
python cli.py pipeline \
    --datasets mirex \
    --output test_results/ \
    --limit 50 \
    --disable-enhanced-processing
```

### 7. Configuration Management

Manage pipeline configuration:

```bash
# Show current configuration
python cli.py config show

# Create configuration file
python cli.py config create --output my_config.json

# Validate configuration file
python cli.py config validate --file my_config.json
```

### 8. Processing Status and Checkpoints

Monitor and manage processing status:

```bash
# Show processing status
python cli.py status show

# Show status with custom checkpoint directory
python cli.py status show --checkpoint-dir my_checkpoints/

# Clean up checkpoints
python cli.py status cleanup

# Resume processing from checkpoints
python cli.py status resume \
    --datasets mirex \
    --output results/ \
    --checkpoint-dir checkpoints/
```

## Advanced Usage Examples

### Large-Scale Processing

For processing large datasets with memory constraints:

```bash
python cli.py pipeline \
    --datasets mirex,csd,jam-alt \
    --output large_scale_results/ \
    --workers 8 \
    --memory-limit 16.0 \
    --chunk-size 200 \
    --log-level INFO \
    --log-file processing.log
```

### Quality-Focused Processing

For high-quality results with strict filtering:

```bash
python cli.py pipeline \
    --datasets mirex \
    --output high_quality_results/ \
    --config strict_quality_config.json \
    --workers 4
```

### Development and Testing

For quick testing and development:

```bash
python cli.py pipeline \
    --datasets mirex \
    --output test/ \
    --limit 20 \
    --models random_forest \
    --disable-duplicate-detection \
    --disable-outlier-detection \
    --log-level DEBUG
```

### Resumable Processing

For long-running processes with fault tolerance:

```bash
# Start processing
python cli.py pipeline \
    --datasets mirex,csd,jam-alt \
    --output resumable_results/ \
    --workers 6 \
    --chunk-size 100

# If interrupted, resume from checkpoint
python cli.py status resume \
    --datasets mirex,csd,jam-alt \
    --output resumable_results/ \
    --checkpoint-dir checkpoints/
```

## Configuration Options

### Audio Processing
- `--sample-rate`: Audio sample rate (default: 22050)
- `--workers`: Number of worker processes
- `--memory-limit`: Memory limit in GB

### Feature Extraction
- `--limit`: Limit number of tracks to process
- `--chunk-size`: Processing chunk size for optimization

### Model Training
- `--models`: Comma-separated list of models to train
- `--max-features`: Maximum number of features to select

### Filtering and Quality Control
- `--disable-quality-filter`: Disable audio quality filtering
- `--disable-duplicate-detection`: Disable duplicate track detection
- `--disable-outlier-detection`: Disable outlier detection
- `--disable-enhanced-processing`: Use standard processing pipeline

### Logging and Debugging
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Save logs to file
- `--config`: Use custom configuration file

## Output Structure

The CLI generates organized output directories:

```
output/
├── data/processed/features/
│   ├── features.parquet          # Main feature dataset
│   └── features.csv              # CSV version
├── analysis_results.json         # Analysis results
├── best_model.joblib             # Trained model
├── processing_stats.json         # Processing statistics
├── visualizations/               # All plots and charts
│   ├── feature_distributions/
│   ├── mood_analysis/
│   ├── correlations/
│   └── model_performance/
├── pipeline_report.json          # Machine-readable report
├── PIPELINE_REPORT.md            # Human-readable summary
└── checkpoints/                  # Processing checkpoints
```

## Error Handling and Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce memory usage
   python cli.py pipeline --memory-limit 4.0 --chunk-size 100 --workers 2
   ```

2. **Processing Interruption**
   ```bash
   # Resume from checkpoint
   python cli.py status resume --output results/
   ```

3. **Dataset Not Found**
   ```bash
   # Check available datasets
   python cli.py discover
   ```

4. **Model Training Fails**
   ```bash
   # Use fewer models and features
   python cli.py train --models random_forest --max-features 20
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python cli.py pipeline \
    --datasets mirex \
    --output debug_results/ \
    --log-level DEBUG \
    --log-file debug.log \
    --limit 10
```

## Performance Optimization

### For Large Datasets
- Use `--chunk-size` to control memory usage
- Increase `--workers` for parallel processing
- Set appropriate `--memory-limit`
- Enable checkpointing for fault tolerance

### For Quick Testing
- Use `--limit` to process fewer tracks
- Disable advanced filtering options
- Use fewer models for training
- Use `--disable-enhanced-processing` for simpler pipeline

### For Production
- Use enhanced processing pipeline (default)
- Enable all quality filters
- Use multiple models for comparison
- Save logs for monitoring

## Integration with Other Tools

### Jupyter Notebooks
```python
import subprocess
result = subprocess.run(['python', 'cli.py', 'discover'], capture_output=True, text=True)
print(result.stdout)
```

### Shell Scripts
```bash
#!/bin/bash
# Automated processing script
python cli.py discover
python cli.py pipeline --datasets mirex --output batch_results/
python cli.py analyze --features batch_results/data/processed/features/features.parquet --output analysis/
```

### CI/CD Pipelines
```yaml
# Example GitHub Actions workflow
- name: Run SAPPHIRE Pipeline
  run: |
    python cli.py pipeline --datasets mirex --output ci_results/ --limit 10
    python cli.py config validate --file config.json
```

This CLI provides a powerful and flexible interface to the SAPPHIRE pipeline, suitable for both interactive use and automated workflows.