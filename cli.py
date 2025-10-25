#!/usr/bin/env python3
"""
SAPPHIRE Command Line Interface

A comprehensive CLI for the SAPPHIRE music analysis pipeline.
Provides commands for data processing, feature extraction, model training, and analysis.
"""

import argparse
import sys
import subprocess
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from pipeline import Pipeline, DataLoader, FeatureExtractor, MoodClassifier, Visualizer, config, Config

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def cmd_discover(args):
    """Discover available datasets."""
    print("🔍 Discovering available datasets...")
    
    loader = DataLoader()
    datasets = loader.discover_datasets()
    
    print("\n📊 Dataset Discovery Results:")
    print("=" * 50)
    
    total_audio = 0
    total_lyrics = 0
    available_count = 0
    
    for dataset_name, info in datasets.items():
        status = "✅ Available" if info['exists'] else "❌ Not Found"
        print(f"\n{dataset_name}: {status}")
        
        if info['exists']:
            available_count += 1
            audio_files = info.get('audio_files', 0)
            lyrics_files = info.get('lyrics_files', 0)
            total_audio += audio_files
            total_lyrics += lyrics_files
            
            print(f"  📁 Path: {info['path']}")
            print(f"  🎵 Audio files: {audio_files}")
            print(f"  📝 Lyrics files: {lyrics_files}")
            
            if info.get('has_annotations'):
                print(f"  🏷️  Has mood annotations: Yes")
    
    print(f"\n📈 Summary:")
    print(f"  Available datasets: {available_count}/{len(datasets)}")
    print(f"  Total audio files: {total_audio}")
    print(f"  Total lyrics files: {total_lyrics}")

def cmd_extract_features(args):
    """Extract features from datasets."""
    print("🎯 Starting feature extraction...")
    
    # Load configuration
    if args.config:
        config_obj = Config.from_file(args.config)
    else:
        config_obj = config
    
    # Override config with CLI arguments
    if args.sample_rate:
        config_obj.audio.sample_rate = args.sample_rate
    if args.workers:
        config_obj.processing.workers = args.workers
    
    # Initialize components
    loader = DataLoader(config_obj)
    extractor = FeatureExtractor(config_obj)
    
    # Load datasets
    if args.datasets:
        datasets = args.datasets.split(',')
        tracks = []
        for dataset in datasets:
            dataset = dataset.strip().lower()
            if dataset == 'mirex':
                tracks.extend(loader.load_mirex_mood_dataset())
            elif dataset == 'csd':
                tracks.extend(loader.load_csd_dataset())
            elif dataset == 'jam-alt':
                tracks.extend(loader.load_jam_alt_dataset())
            elif dataset == 'vietnamese':
                tracks.extend(loader.load_viet_dataset())
            else:
                print(f"⚠️  Unknown dataset: {dataset}")
    else:
        tracks = loader.load_all_datasets()
    
    if not tracks:
        print("❌ No tracks found!")
        return
    
    print(f"📊 Loaded {len(tracks)} tracks")
    
    # Limit tracks if specified
    if args.limit:
        tracks = tracks[:args.limit]
        print(f"🔢 Limited to {len(tracks)} tracks")
    
    # Extract features
    features_list = []
    failed_count = 0
    
    for i, track in enumerate(tracks):
        try:
            print(f"🎵 Processing {i+1}/{len(tracks)}: {track.track_id}")
            
            # Load audio and lyrics
            track = loader.load_audio(track, load_data=True)
            track = loader.load_lyrics(track)
            
            if track.audio_data is not None:
                # Extract features
                features = extractor.extract_features(track)
                features.update({
                    'track_id': track.track_id,
                    'mood_cluster': track.mood_cluster,
                    'mood_category': track.mood_category,
                    'dataset': track.metadata.get('dataset', 'unknown')
                })
                features_list.append(features)
            else:
                print(f"⚠️  No audio data for {track.track_id}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {track.track_id}: {e}")
            failed_count += 1
            continue
    
    if not features_list:
        print("❌ No features extracted!")
        return
    
    # Save features
    features_df = pd.DataFrame(features_list)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_df.to_parquet(output_dir / 'features.parquet', index=False)
    features_df.to_csv(output_dir / 'features.csv', index=False)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Extracted features for {len(features_list)} tracks")
    print(f"❌ Failed: {failed_count} tracks")
    print(f"📁 Features saved to: {output_dir}")
    print(f"🔢 Total features per track: {len(features_df.columns) - 4}")  # -4 for metadata

def cmd_train_models(args):
    """Train mood classification models."""
    print("🤖 Starting model training...")

    # Prepare features path (optionally normalize first)
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return

    # If requested, auto-normalize serialized object columns before loading
    if getattr(args, 'normalize', False):
        # Determine output path for normalized features
        if getattr(args, 'normalize_out', None):
            normalized_path = Path(args.normalize_out)
        else:
            normalized_path = features_path.with_name(features_path.stem + '_clean.parquet')

        try:
            print(f"🔧 Normalizing feature columns into: {normalized_path}")
            subprocess.run([sys.executable, 'scripts/normalize_feature_columns.py', str(features_path), str(normalized_path)], check=True)
            features_path = normalized_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Normalization failed: {e}")
            return

    if features_path.suffix == '.parquet':
        features_df = pd.read_parquet(features_path)
    else:
        features_df = pd.read_csv(features_path)
    
    print(f"📊 Loaded features for {len(features_df)} tracks")
    
    # Filter tracks with mood labels
    labeled_tracks = features_df.dropna(subset=['mood_cluster'])
    print(f"🏷️  Found {len(labeled_tracks)} tracks with mood labels")
    
    if len(labeled_tracks) == 0:
        print("❌ No labeled tracks found!")
        return
    
    # Load configuration
    if args.config:
        config_obj = Config.from_file(args.config)
    else:
        config_obj = config
    
    # Override models if specified
    if args.models:
        config_obj.model.models = args.models.split(',')
    
    # Initialize classifier
    classifier = MoodClassifier(config_obj)
    
    # Prepare data
    X, y = classifier.prepare_data(labeled_tracks, 'mood_cluster')
    
    # Feature selection
    if args.max_features:
        config_obj.model.max_features = args.max_features
    
    X_selected = classifier.select_features(
        X, y, 
        method=config_obj.model.feature_selection_method,
        k=config_obj.model.max_features
    )
    
    # Train models
    results = classifier.train_models(X_selected, y)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate models
    classifier.evaluate_models(results, str(output_dir))
    
    # Save best model
    if results:
        classifier.save_model(str(output_dir / 'best_model.joblib'))
        
        # Print results
        print(f"\n✅ Model training complete!")
        print(f"📊 Trained {len(results)} models")
        
        best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_accuracy = results[best_model]['test_accuracy']
        
        print(f"🏆 Best model: {best_model}")
        print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        print(f"📁 Results saved to: {output_dir}")
        
        # Show all model results
        print(f"\n📈 Model Comparison:")
        for model_name, result in results.items():
            print(f"  {model_name}: {result['test_accuracy']:.4f}")

def cmd_analyze(args):
    """Perform comprehensive analysis."""
    print("📊 Starting comprehensive analysis...")
    
    # Load features
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return
    
    if features_path.suffix == '.parquet':
        features_df = pd.read_parquet(features_path)
    else:
        features_df = pd.read_csv(features_path)
    
    print(f"📊 Loaded features for {len(features_df)} tracks")
    
    # Load configuration
    if args.config:
        config_obj = Config.from_file(args.config)
    else:
        config_obj = config
    
    # Initialize components
    from pipeline import Analyzer
    analyzer = Analyzer(config_obj)
    classifier = MoodClassifier(config_obj)
    visualizer = Visualizer(config_obj)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform analysis
    print("🔍 Running dataset analysis...")
    dataset_analysis = analyzer.analyze_dataset(features_df, str(output_dir))
    
    print("🔗 Running cross-modal analysis...")
    cross_modal_analysis = classifier.cross_modal_analysis(features_df)
    
    print("🎯 Running clustering analysis...")
    clustering_results = analyzer.perform_clustering(features_df, str(output_dir))
    
    # Create visualizations
    print("📈 Creating visualizations...")
    visualizer.create_comprehensive_report(
        features_df,
        {
            'dataset_analysis': dataset_analysis,
            'cross_modal_analysis': cross_modal_analysis,
            'clustering': clustering_results
        },
        {},
        str(output_dir / 'visualizations')
    )
    
    # Save analysis results
    analysis_results = {
        'dataset_analysis': dataset_analysis,
        'cross_modal_analysis': cross_modal_analysis,
        'clustering': clustering_results
    }
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print key insights
    if cross_modal_analysis.get('jaccard_similarity'):
        jaccard_mean = cross_modal_analysis['jaccard_similarity']['mean']
        print(f"🔗 Mean cross-modal Jaccard similarity: {jaccard_mean:.4f}")
    
    if cross_modal_analysis.get('correlations'):
        max_corr = cross_modal_analysis['correlations']['max_correlation']
        print(f"📊 Max cross-modal correlation: {max_corr:.4f}")

def cmd_predict(args):
    """Predict mood for a single track."""
    print("🎯 Predicting mood...")
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load configuration
    if args.config:
        config_obj = Config.from_file(args.config)
    else:
        config_obj = config
    
    # Initialize pipeline
    pipeline = Pipeline(config_obj)
    
    # Load model
    pipeline.mood_classifier.load_model(str(model_path))
    
    # Make prediction
    try:
        result = pipeline.predict_mood(args.audio, args.lyrics)
        
        print(f"\n✅ Prediction complete!")
        print(f"🎵 Audio: {args.audio}")
        if args.lyrics:
            print(f"📝 Lyrics: {args.lyrics}")
        print(f"🎭 Predicted mood: {result['predicted_mood']}")
        
        if result['confidence']:
            print(f"🎯 Confidence: {result['confidence']:.3f}")
        
        if result['all_probabilities']:
            print(f"\n📊 All probabilities:")
            for mood, prob in result['all_probabilities'].items():
                print(f"  {mood}: {prob:.3f}")
                
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def cmd_pipeline(args):
    """Run the full pipeline."""
    print("🚀 Starting full SAPPHIRE pipeline...")
    
    # Load configuration
    if args.config:
        config_obj = Config.from_file(args.config)
    else:
        config_obj = config
    
    # Override config with CLI arguments
    if args.workers:
        config_obj.processing.workers = args.workers
    if args.models:
        config_obj.model.models = args.models.split(',')
    if args.memory_limit:
        config_obj.processing.memory_limit_gb = args.memory_limit
    if args.chunk_size:
        config_obj.processing.chunk_size = args.chunk_size
    
    # Configure filtering
    if args.disable_quality_filter:
        config_obj.filtering['enable_quality_filter'] = False
    if args.disable_duplicate_detection:
        config_obj.filtering['enable_duplicate_detection'] = False
    if args.disable_outlier_detection:
        config_obj.filtering['enable_outlier_detection'] = False
    
    # Initialize pipeline
    pipeline = Pipeline(config_obj)
    
    # Parse datasets
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        datasets=datasets,
        output_dir=args.output,
        use_enhanced_processing=not args.disable_enhanced_processing,
        limit_tracks=args.limit
    )
    
    # Print results
    if results['status'] == 'success':
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"⏱️  Duration: {results['duration']}")
        print(f"🎵 Tracks processed: {results['tracks_processed']}")
        print(f"🔢 Features extracted: {results['features_extracted']}")
        print(f"📁 Output directory: {results['output_directory']}")
        
        # Show processing statistics if available
        if results.get('processing_stats'):
            stats = results['processing_stats']
            print(f"\n📊 Processing Statistics:")
            print(f"  Success rate: {stats.get('success_rate', 0):.2%}")
            print(f"  Peak memory: {stats.get('memory_peak', 0):.2f} GB")
            print(f"  Processing time: {stats.get('processing_time', 0):.1f} seconds")
        
        print(f"\n📖 View the complete report at:")
        print(f"   {Path(results['output_directory']) / 'PIPELINE_REPORT.md'}")
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")

def cmd_status(args):
    """Show processing status and manage checkpoints."""
    from pipeline.processing_pipeline import EnhancedProcessingPipeline
    
    if args.action == 'show':
        print("📊 Processing Status:")
        print("=" * 50)
        
        # Initialize enhanced processor to check status
        processor = EnhancedProcessingPipeline(checkpoint_dir=args.checkpoint_dir or "checkpoints")
        status = processor.get_processing_status()
        
        print(f"\n💾 Memory Usage:")
        print(f"  Current: {status['memory_usage_gb']:.2f} GB")
        print(f"  Limit: {status['memory_limit_gb']:.2f} GB")
        print(f"  Usage: {status['memory_usage_gb']/status['memory_limit_gb']*100:.1f}%")
        
        print(f"\n📋 Checkpoints:")
        for stage, exists in status['checkpoints'].items():
            status_icon = "✅" if exists else "❌"
            print(f"  {status_icon} {stage}")
        
        print(f"\n📈 Statistics:")
        stats = status['statistics']
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.action == 'cleanup':
        from pipeline.processing_pipeline import EnhancedProcessingPipeline
        
        processor = EnhancedProcessingPipeline(checkpoint_dir=args.checkpoint_dir or "checkpoints")
        processor.cleanup_checkpoints()
        print("✅ All checkpoints cleaned up")
    
    elif args.action == 'resume':
        print("🔄 Resuming processing from checkpoints...")
        
        # Load configuration
        if args.config:
            config_obj = Config.from_file(args.config)
        else:
            config_obj = config
        
        # Enable resume from checkpoint
        config_obj.optimization['resume_from_checkpoint'] = True
        
        # Initialize and run enhanced processor
        from pipeline.processing_pipeline import EnhancedProcessingPipeline
        processor = EnhancedProcessingPipeline(config_obj, checkpoint_dir=args.checkpoint_dir or "checkpoints")
        
        result = processor.run_optimized_pipeline(
            datasets=args.datasets.split(',') if args.datasets else None,
            output_dir=args.output or "output"
        )
        
        if result['status'] == 'success':
            print("✅ Processing resumed and completed successfully!")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

def cmd_config(args):
    """Manage configuration."""
    if args.action == 'show':
        print("⚙️  Current Configuration:")
        print("=" * 50)
        
        print(f"\n🎵 Audio Settings:")
        print(f"  Sample Rate: {config.audio.sample_rate} Hz")
        print(f"  N-FFT: {config.audio.n_fft}")
        print(f"  Hop Length: {config.audio.hop_length}")
        print(f"  N-MFCCs: {config.audio.n_mfcc}")
        
        print(f"\n🔧 Processing Settings:")
        print(f"  Workers: {config.processing.workers}")
        print(f"  Batch Size: {config.processing.batch_size}")
        print(f"  Use GPU: {config.processing.use_gpu}")
        
        print(f"\n🎯 Feature Settings:")
        print(f"  Extract Acoustic: {config.features.extract_acoustic}")
        print(f"  Extract Rhythm: {config.features.extract_rhythm}")
        print(f"  Extract Harmony: {config.features.extract_harmony}")
        print(f"  Extract Lyrics: {config.features.extract_lyrics}")
        print(f"  Extract Quality: {config.features.extract_quality}")
        
        print(f"\n🤖 Model Settings:")
        print(f"  Models: {', '.join(config.model.models)}")
        print(f"  Test Size: {config.model.test_size}")
        print(f"  CV Folds: {config.model.cv_folds}")
        
        print(f"\n🔍 Filtering Settings:")
        print(f"  Quality Filter: {config.filtering['enable_quality_filter']}")
        print(f"  Duration Filter: {config.filtering['enable_duration_filter']}")
        print(f"  Duplicate Detection: {config.filtering['enable_duplicate_detection']}")
        print(f"  Outlier Detection: {config.filtering['enable_outlier_detection']}")
        print(f"  Language Filter: {config.filtering['enable_language_filter']}")
        
        print(f"\n⚡ Optimization Settings:")
        print(f"  Use Multiprocessing: {config.optimization['use_multiprocessing']}")
        print(f"  Feature Caching: {config.optimization['feature_caching']}")
        print(f"  Incremental Processing: {config.optimization['incremental_processing']}")
        print(f"  Resume from Checkpoint: {config.optimization['resume_from_checkpoint']}")
        
    elif args.action == 'create':
        output_path = Path(args.output)
        config.save(str(output_path))
        print(f"✅ Configuration saved to: {output_path}")
        
    elif args.action == 'validate':
        config_path = Path(args.file)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        try:
            test_config = Config.from_file(str(config_path))
            print(f"✅ Configuration file is valid: {config_path}")
        except Exception as e:
            print(f"❌ Configuration file is invalid: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAPPHIRE Music Analysis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover available datasets
  python cli.py discover
  
  # Extract features from MIREX dataset
  python cli.py extract-features --datasets mirex --output features/
  
  # Train models on extracted features
  python cli.py train --features features/features.parquet --output models/
  
  # Run full pipeline
  python cli.py pipeline --datasets mirex,csd --output results/
  
  # Predict mood for a single track
  python cli.py predict --audio song.mp3 --model models/best_model.joblib
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover available datasets')
    discover_parser.set_defaults(func=cmd_discover)
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract features from datasets')
    extract_parser.add_argument('--datasets', type=str, help='Comma-separated list of datasets (mirex,csd,jam-alt,vietnamese)')
    extract_parser.add_argument('--output', type=str, required=True, help='Output directory')
    extract_parser.add_argument('--limit', type=int, help='Limit number of tracks to process')
    extract_parser.add_argument('--sample-rate', type=int, help='Audio sample rate')
    extract_parser.add_argument('--workers', type=int, help='Number of worker processes')
    extract_parser.set_defaults(func=cmd_extract_features)
    
    # Train models command
    train_parser = subparsers.add_parser('train', help='Train mood classification models')
    train_parser.add_argument('--features', type=str, required=True, help='Features file path')
    train_parser.add_argument('--output', type=str, required=True, help='Output directory')
    train_parser.add_argument('--models', type=str, help='Comma-separated list of models to train')
    train_parser.add_argument('--max-features', type=int, help='Maximum number of features to select')
    train_parser.add_argument('--normalize', action='store_true', help='Auto-normalize serialized-dict feature columns before training')
    train_parser.add_argument('--normalize-out', type=str, help='Optional path to write normalized features (parquet)')
    train_parser.set_defaults(func=cmd_train_models)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Perform comprehensive analysis')
    analyze_parser.add_argument('--features', type=str, required=True, help='Features file path')
    analyze_parser.add_argument('--output', type=str, required=True, help='Output directory')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict mood for a single track')
    predict_parser.add_argument('--audio', type=str, required=True, help='Audio file path')
    predict_parser.add_argument('--lyrics', type=str, help='Lyrics file path')
    predict_parser.add_argument('--model', type=str, required=True, help='Trained model file path')
    predict_parser.set_defaults(func=cmd_predict)
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full SAPPHIRE pipeline')
    pipeline_parser.add_argument('--datasets', type=str, help='Comma-separated list of datasets')
    pipeline_parser.add_argument('--output', type=str, default='output', help='Output directory')
    pipeline_parser.add_argument('--limit', type=int, help='Limit number of tracks to process')
    pipeline_parser.add_argument('--workers', type=int, help='Number of worker processes')
    pipeline_parser.add_argument('--models', type=str, help='Comma-separated list of models to train')
    pipeline_parser.add_argument('--memory-limit', type=float, help='Memory limit in GB')
    pipeline_parser.add_argument('--chunk-size', type=int, help='Processing chunk size')
    pipeline_parser.add_argument('--disable-enhanced-processing', action='store_true', 
                                help='Disable enhanced processing pipeline')
    pipeline_parser.add_argument('--disable-quality-filter', action='store_true',
                                help='Disable quality filtering')
    pipeline_parser.add_argument('--disable-duplicate-detection', action='store_true',
                                help='Disable duplicate detection')
    pipeline_parser.add_argument('--disable-outlier-detection', action='store_true',
                                help='Disable outlier detection')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Configuration command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='action', help='Configuration actions')
    
    config_show = config_subparsers.add_parser('show', help='Show current configuration')
    
    config_create = config_subparsers.add_parser('create', help='Create configuration file')
    config_create.add_argument('--output', type=str, required=True, help='Output configuration file path')
    
    config_validate = config_subparsers.add_parser('validate', help='Validate configuration file')
    config_validate.add_argument('--file', type=str, required=True, help='Configuration file to validate')
    
    config_parser.set_defaults(func=cmd_config)
    
    # Status and checkpoint management command
    status_parser = subparsers.add_parser('status', help='Manage processing status and checkpoints')
    status_subparsers = status_parser.add_subparsers(dest='action', help='Status actions')
    
    status_show = status_subparsers.add_parser('show', help='Show processing status')
    status_show.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    status_cleanup = status_subparsers.add_parser('cleanup', help='Clean up checkpoints')
    status_cleanup.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    status_resume = status_subparsers.add_parser('resume', help='Resume processing from checkpoints')
    status_resume.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    status_resume.add_argument('--datasets', type=str, help='Comma-separated list of datasets')
    status_resume.add_argument('--output', type=str, help='Output directory')
    status_resume.add_argument('--config', type=str, help='Configuration file')
    
    status_parser.set_defaults(func=cmd_status)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()