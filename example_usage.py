#!/usr/bin/env python3
"""
Example usage of the SAPPHIRE music analysis pipeline.
This script demonstrates how to use the pipeline for comprehensive music analysis.
"""

import logging
from pathlib import Path
from pipeline import Pipeline, config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Main example function."""
    print("SAPPHIRE Music Analysis Pipeline - Example Usage")
    print("=" * 50)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = Pipeline()

    # Check available datasets
    print("\n2. Discovering available datasets...")
    datasets_info = pipeline.data_loader.discover_datasets()

    print("Available datasets:")
    for dataset_name, info in datasets_info.items():
        if info["exists"]:
            print(
                f"  ✓ {dataset_name}: {info['audio_files']} audio files, {info['lyrics_files']} lyrics files"
            )
        else:
            print(f"  ✗ {dataset_name}: Not found")

    # Run pipeline on available datasets
    available_datasets = [
        name for name, info in datasets_info.items() if info["exists"]
    ]

    if not available_datasets:
        print("\nNo datasets found! Please check your data directory structure.")
        print("Expected structure:")
        print("data/raw/")
        print("├── MIREX-like_mood/dataset/")
        print("├── CSD/")
        print("├── jam-alt/")
        print("├── Viet_Dataset/")
        print("└── 100M/")
        return

    print(f"\n3. Running pipeline on datasets: {available_datasets}")

    # Configure pipeline for faster execution (optional)
    config.processing.workers = 2  # Reduce workers for example
    config.model.models = ["random_forest", "svm"]  # Use fewer models

    # Run the full pipeline
    results = pipeline.run_full_pipeline(
        datasets=available_datasets[:2],  # Use first 2 available datasets
        output_dir="output/example_run",
    )

    # Display results
    print("\n4. Pipeline Results:")
    print(f"Status: {results['status']}")
    print(f"Duration: {results['duration']}")
    print(f"Tracks processed: {results['tracks_processed']}")
    print(f"Features extracted: {results['features_extracted']}")
    print(f"Output directory: {results['output_directory']}")

    if results["status"] == "success":
        print("\n5. Generated outputs:")
        output_dir = Path(results["output_directory"])

        # List key output files
        key_files = [
            "data/processed/features/all_features.parquet",
            "analysis_results.json",
            "pipeline_report.json",
            "PIPELINE_REPORT.md",
            "best_model.joblib",
        ]

        for file_path in key_files:
            full_path = output_dir / file_path
            if full_path.exists():
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not generated)")

        # Check visualization directories
        viz_dir = output_dir / "visualizations"
        if viz_dir.exists():
            viz_subdirs = [d.name for d in viz_dir.iterdir() if d.is_dir()]
            print(
                f"  ✓ visualizations/ ({len(viz_subdirs)} subdirectories: {', '.join(viz_subdirs)})"
            )

        print(f"\n6. View the complete report at: {output_dir / 'PIPELINE_REPORT.md'}")

        # Example of mood prediction for a new track
        if pipeline.mood_classifier.best_model is not None:
            print("\n7. Example mood prediction:")
            print("To predict mood for a new track, use:")
            print(
                "result = pipeline.predict_mood('path/to/audio.mp3', 'path/to/lyrics.txt')"
            )
            print("print(f'Predicted mood: {result[\"predicted_mood\"]}')")

    else:
        print(f"Pipeline failed with error: {results.get('error', 'Unknown error')}")


def quick_feature_extraction_example():
    """Example of using individual components for feature extraction."""
    print("\n" + "=" * 50)
    print("Quick Feature Extraction Example")
    print("=" * 50)

    from pipeline import DataLoader, FeatureExtractor

    # Initialize components
    loader = DataLoader()
    extractor = FeatureExtractor()

    # Load a small sample of tracks
    print("Loading sample tracks...")
    try:
        tracks = loader.load_mirex_mood_dataset()[:5]  # First 5 tracks

        if not tracks:
            print("No tracks found in MIREX dataset")
            return

        print(f"Loaded {len(tracks)} sample tracks")

        # Extract features for each track
        features_list = []
        for i, track in enumerate(tracks):
            print(f"Processing track {i+1}/{len(tracks)}: {track.track_id}")

            # Load audio and lyrics
            track = loader.load_audio(track, load_data=True)
            track = loader.load_lyrics(track)

            if track.audio_data is not None:
                # Extract features
                features = extractor.extract_features(track)
                features["track_id"] = track.track_id
                features["mood_cluster"] = track.mood_cluster
                features_list.append(features)

        print(f"\nExtracted features for {len(features_list)} tracks")
        if features_list:
            print(
                f"Number of features per track: {len(features_list[0]) - 2}"
            )  # -2 for track_id and mood
            print("Sample features:", list(features_list[0].keys())[:10])

    except Exception as e:
        print(f"Error in feature extraction example: {e}")


if __name__ == "__main__":
    try:
        main()
        quick_feature_extraction_example()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Detailed error information:")
