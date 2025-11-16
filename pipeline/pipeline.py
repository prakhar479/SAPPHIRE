"""
Main pipeline orchestrator for the SAPPHIRE music analysis system.
Coordinates all components for end-to-end music analysis and mood classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

from .config import config
from .data_loader import DataLoader, MusicTrack
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor
from .analyzer import Analyzer
from .mood_classifier import MoodClassifier
from .visualizer import Visualizer
from .processing_pipeline import EnhancedProcessingPipeline

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main SAPPHIRE pipeline for comprehensive music analysis.

    This pipeline implements the methodology described in the SAPPHIRE project:
    1. Multi-dataset loading and preprocessing
    2. Multi-modal feature extraction (acoustic, rhythm, harmonic, lyrical, quality)
    3. Cross-modal analysis and perceptual gap measurement
    4. Mood classification using machine learning
    5. Comprehensive visualization and reporting
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.preprocessor = Preprocessor(self.config)
        self.analyzer = Analyzer(self.config)
        self.mood_classifier = MoodClassifier(self.config)
        self.visualizer = Visualizer(self.config)

        # Initialize enhanced processing pipeline
        self.enhanced_processor = EnhancedProcessingPipeline(self.config)

        # Pipeline state
        self.tracks = []
        self.features_df = None
        self.analysis_results = {}
        self.model_results = {}

    def run_full_pipeline(
        self,
        datasets: List[str] = None,
        output_dir: str = None,
        use_enhanced_processing: bool = True,
        limit_tracks: int = None,
    ) -> Dict[str, Any]:
        """
        Run the complete SAPPHIRE pipeline.

        Args:
            datasets: List of datasets to process (None for all)
            output_dir: Output directory for results
            use_enhanced_processing: Whether to use enhanced processing pipeline
            limit_tracks: Limit number of tracks to process

        Returns:
            Dictionary with pipeline results
        """
        if output_dir is None:
            output_dir = self.config.data.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting SAPPHIRE pipeline execution...")
        start_time = datetime.now()

        try:
            if use_enhanced_processing:
                # Use enhanced processing pipeline
                self.logger.info(
                    "Using enhanced processing pipeline with optimization..."
                )

                processing_result = self.enhanced_processor.run_optimized_pipeline(
                    datasets=datasets,
                    output_dir=str(output_path),
                    limit_tracks=limit_tracks,
                )

                if processing_result["status"] != "success":
                    return processing_result

                self.features_df = processing_result["features_df"]

            else:
                # Use standard processing pipeline
                self.logger.info("Using standard processing pipeline...")

                # Step 1: Load data
                self.logger.info("Step 1: Loading datasets...")
                self.tracks = self.load_datasets(datasets)

                if limit_tracks:
                    self.tracks = self.tracks[:limit_tracks]

                # Step 2: Preprocess data
                self.logger.info("Step 2: Preprocessing data...")
                self.tracks = self.preprocess_data(self.tracks)

                # Step 3: Extract features
                self.logger.info("Step 3: Extracting features...")
                self.features_df = self.extract_features(self.tracks)

            # Step 4: Analyze features
            self.logger.info("Step 4: Analyzing features...")
            self.analysis_results = self.analyze_features(
                self.features_df, str(output_path)
            )

            # Step 5: Train mood classifiers
            self.logger.info("Step 5: Training mood classifiers...")
            self.model_results = self.train_classifiers(
                self.features_df, str(output_path)
            )

            # Step 6: Create visualizations
            self.logger.info("Step 6: Creating visualizations...")
            self.create_visualizations(str(output_path))

            # Step 7: Generate final report
            self.logger.info("Step 7: Generating final report...")
            report = self.generate_report(str(output_path))

            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info(f"Pipeline completed successfully in {duration}")

            return {
                "status": "success",
                "duration": str(duration),
                "tracks_processed": (
                    len(self.features_df) if self.features_df is not None else 0
                ),
                "features_extracted": (
                    len(self.features_df.columns) if self.features_df is not None else 0
                ),
                "output_directory": str(output_path),
                "report": report,
                "processing_stats": (
                    processing_result.get("statistics", {})
                    if use_enhanced_processing
                    else {}
                ),
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "duration": str(datetime.now() - start_time),
            }

    def load_datasets(self, datasets: List[str] = None) -> List[MusicTrack]:
        """
        Load specified datasets or all available datasets.

        Args:
            datasets: List of dataset names to load

        Returns:
            List of MusicTrack objects
        """
        if datasets is None:
            # Load all datasets
            tracks = self.data_loader.load_all_datasets()
        else:
            tracks = []
            for dataset_name in datasets:
                if dataset_name.lower() == "mirex":
                    tracks.extend(self.data_loader.load_mirex_mood_dataset())
                elif dataset_name.lower() == "csd":
                    tracks.extend(self.data_loader.load_csd_dataset())
                elif dataset_name.lower() == "jam-alt":
                    tracks.extend(self.data_loader.load_jam_alt_dataset())
                elif dataset_name.lower() == "vietnamese":
                    tracks.extend(self.data_loader.load_viet_dataset())
                elif dataset_name.lower() == "100m":
                    tracks.extend(self.data_loader.load_100m_dataset())
                else:
                    self.logger.warning(f"Unknown dataset: {dataset_name}")

        self.logger.info(f"Loaded {len(tracks)} tracks from datasets")
        return tracks

    def preprocess_data(self, tracks: List[MusicTrack]) -> List[MusicTrack]:
        """
        Preprocess loaded tracks.

        Args:
            tracks: List of MusicTrack objects

        Returns:
            List of preprocessed MusicTrack objects
        """
        # Load audio and lyrics for all tracks
        processed_tracks = []

        for track in tracks:
            # Load audio data
            track = self.data_loader.load_audio(track, load_data=True)

            # Load lyrics
            track = self.data_loader.load_lyrics(track)

            # Preprocess audio and lyrics
            track = self.preprocessor.preprocess_track(track)

            # Only keep tracks that pass quality checks
            if self.preprocessor.passes_quality_check(track):
                processed_tracks.append(track)
            else:
                self.logger.debug(f"Track {track.track_id} failed quality check")

        self.logger.info(
            f"Preprocessed {len(processed_tracks)} tracks (filtered from {len(tracks)})"
        )
        return processed_tracks

    def extract_features(self, tracks: List[MusicTrack]) -> pd.DataFrame:
        """
        Extract multi-modal features from tracks.

        Args:
            tracks: List of preprocessed MusicTrack objects

        Returns:
            DataFrame with extracted features
        """
        # Extract features for all tracks
        all_features = []

        for track in tracks:
            try:
                features = self.feature_extractor.extract_features(track)

                # Add metadata
                features.update(
                    {
                        "track_id": track.track_id,
                        "mood_cluster": track.mood_cluster,
                        "mood_category": track.mood_category,
                        "dataset": track.metadata.get("dataset", "unknown"),
                        "language": track.metadata.get("language", "unknown"),
                    }
                )

                all_features.append(features)

            except Exception as e:
                self.logger.error(
                    f"Error extracting features for track {track.track_id}: {e}"
                )
                continue

        if not all_features:
            raise ValueError("No features were successfully extracted")

        features_df = pd.DataFrame(all_features)

        # Save features
        features_path = Path(self.config.data.features_dir)
        features_path.mkdir(parents=True, exist_ok=True)

        features_df.to_parquet(features_path / "all_features.parquet", index=False)
        features_df.to_csv(features_path / "all_features.csv", index=False)

        self.logger.info(
            f"Extracted {len(features_df.columns)} features for {len(features_df)} tracks"
        )
        return features_df

    def analyze_features(
        self, features_df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Any]:
        """Perform comprehensive feature analysis.

        Args:
            features_df: DataFrame with extracted features
            output_dir: Directory to save analysis results

        Returns:
            Dictionary with analysis results
        """
        analysis_results = {}

        # Basic dataset analysis
        analysis_results["dataset_analysis"] = self.analyzer.analyze_dataset(
            features_df, output_dir
        )

        # Cross-modal analysis (key component of SAPPHIRE)
        analysis_results["cross_modal_analysis"] = (
            self.mood_classifier.cross_modal_analysis(features_df)
        )

        # Feature clustering
        analysis_results["clustering"] = self.analyzer.perform_clustering(
            features_df, output_dir
        )

        # Statistical analysis
        analysis_results["statistics"] = self.analyzer.compute_statistics(features_df)

        # Save analysis results
        analysis_path = Path(output_dir) / "analysis_results.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)

        return analysis_results

    def train_classifiers(
        self, features_df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Dict]:
        """Train mood classification models.

        Args:
            features_df: DataFrame with features and mood labels
            output_dir: Directory to save model results

        Returns:
            Dictionary with model training results
        """
        if "mood_cluster" not in features_df.columns:
            self.logger.warning("No mood labels found, skipping classifier training")
            return {}

        # Filter tracks with mood labels
        labeled_tracks = features_df.dropna(subset=["mood_cluster"])

        if len(labeled_tracks) == 0:
            self.logger.warning("No tracks with mood labels found")
            return {}

        self.logger.info(
            f"Training classifiers on {len(labeled_tracks)} labeled tracks"
        )

        # Optional importance-based feature subset
        if (
            getattr(self.config.model, "use_importance_subset", False)
            and getattr(self.config.model, "importance_csv_path", None)
            and getattr(self.config.model, "top_n_important", None)
        ):
            labeled_tracks, used_importance_df = (
                self.mood_classifier.subset_features_by_importance(
                    labeled_tracks,
                    self.config.model.importance_csv_path,
                    self.config.model.top_n_important,
                )
            )

            if not used_importance_df.empty:
                output_path = Path(output_dir)
                selected_path = output_path / "selected_features_from_importance.csv"
                used_importance_df.to_csv(selected_path, index=False)
                self.logger.info(
                    f"Saved selected important features (top {self.config.model.top_n_important}) to {selected_path}"
                )

        # Prepare data
        X, y = self.mood_classifier.prepare_data(labeled_tracks, "mood_cluster")

        # Feature selection
        X_selected = self.mood_classifier.select_features(
            X,
            y,
            method=self.config.model.feature_selection_method,
            k=self.config.model.max_features,
        )

        # Train models
        model_results = self.mood_classifier.train_models(X_selected, y)

        # Evaluate models
        if model_results:
            self.mood_classifier.evaluate_models(model_results, output_dir)

            # Save best model
            model_path = Path(output_dir) / "best_model.joblib"
            self.mood_classifier.save_model(str(model_path))

        return model_results

    def create_visualizations(self, output_dir: str):
        """
        Create comprehensive visualizations.

        Args:
            output_dir: Directory to save visualizations
        """
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer.create_comprehensive_report(
            self.features_df, self.analysis_results, self.model_results, str(viz_dir)
        )

        # Create feature importance plot if we have a trained model
        if self.model_results and self.mood_classifier.best_model:
            importance_df = self.mood_classifier.get_feature_importance()
            if not importance_df.empty:
                self.visualizer.create_feature_importance_plot(
                    importance_df, str(viz_dir)
                )

    def generate_report(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate final comprehensive report.

        Args:
            output_dir: Directory to save report

        Returns:
            Dictionary with report summary
        """
        report = {
            "pipeline_info": {
                "version": "1.0.0",
                "execution_time": datetime.now().isoformat(),
                "config": {
                    "audio_sample_rate": self.config.audio.sample_rate,
                    "feature_extraction": {
                        "acoustic": self.config.features.extract_acoustic,
                        "rhythm": self.config.features.extract_rhythm,
                        "harmony": self.config.features.extract_harmony,
                        "lyrics": self.config.features.extract_lyrics,
                        "quality": self.config.features.extract_quality,
                    },
                    "models_trained": self.config.model.models,
                },
            },
            "data_summary": {
                "total_tracks": len(self.tracks),
                "processed_tracks": (
                    len(self.features_df) if self.features_df is not None else 0
                ),
                "total_features": (
                    len(self.features_df.columns) if self.features_df is not None else 0
                ),
                "datasets": (
                    self.data_loader.get_dataset_statistics(self.tracks)
                    if self.tracks
                    else {}
                ),
            },
            "analysis_summary": {
                "cross_modal_analysis": (
                    {
                        "mean_jaccard_similarity": self.analysis_results.get(
                            "cross_modal_analysis", {}
                        )
                        .get("jaccard_similarity", {})
                        .get("mean", 0),
                        "max_cross_correlation": self.analysis_results.get(
                            "cross_modal_analysis", {}
                        )
                        .get("correlations", {})
                        .get("max_correlation", 0),
                    }
                    if "cross_modal_analysis" in self.analysis_results
                    else {}
                ),
                "clustering_results": self.analysis_results.get("clustering", {}),
            },
            "model_performance": (
                {
                    "best_model": (
                        max(
                            self.model_results.keys(),
                            key=lambda k: self.model_results[k]["test_accuracy"],
                        )
                        if self.model_results
                        else None
                    ),
                    "best_accuracy": (
                        max([r["test_accuracy"] for r in self.model_results.values()])
                        if self.model_results
                        else 0
                    ),
                    "model_comparison": {
                        name: result["test_accuracy"]
                        for name, result in self.model_results.items()
                    },
                }
                if self.model_results
                else {}
            ),
            "output_files": {
                "features": "data/processed/features/all_features.parquet",
                "analysis": "analysis_results.json",
                "visualizations": "visualizations/",
                "models": "best_model.joblib" if self.model_results else None,
            },
        }

        # Save report
        report_path = Path(output_dir) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown summary
        self._create_markdown_report(report, output_dir)

        self.logger.info(f"Final report saved to {report_path}")
        return report

    def _create_markdown_report(self, report: Dict[str, Any], output_dir: str):
        """Create a markdown summary report."""
        md_content = f"""# SAPPHIRE Pipeline Execution Report

## Pipeline Information
- **Version**: {report['pipeline_info']['version']}
- **Execution Time**: {report['pipeline_info']['execution_time']}
- **Audio Sample Rate**: {report['pipeline_info']['config']['audio_sample_rate']} Hz

## Data Summary
- **Total Tracks Loaded**: {report['data_summary']['total_tracks']}
- **Tracks Processed**: {report['data_summary']['processed_tracks']}
- **Features Extracted**: {report['data_summary']['total_features']}

## Feature Extraction Configuration
- **Acoustic Features**: {'✓' if report['pipeline_info']['config']['feature_extraction']['acoustic'] else '✗'}
- **Rhythm Features**: {'✓' if report['pipeline_info']['config']['feature_extraction']['rhythm'] else '✗'}
- **Harmonic Features**: {'✓' if report['pipeline_info']['config']['feature_extraction']['harmony'] else '✗'}
- **Lyrical Features**: {'✓' if report['pipeline_info']['config']['feature_extraction']['lyrics'] else '✗'}
- **Quality Features**: {'✓' if report['pipeline_info']['config']['feature_extraction']['quality'] else '✗'}

## Cross-Modal Analysis Results
"""

        if "cross_modal_analysis" in report["analysis_summary"]:
            cma = report["analysis_summary"]["cross_modal_analysis"]
            md_content += f"""- **Mean Jaccard Similarity**: {cma.get('mean_jaccard_similarity', 0):.4f}
- **Max Cross-Correlation**: {cma.get('max_cross_correlation', 0):.4f}
"""
        else:
            md_content += "- No cross-modal analysis performed\n"

        if report["model_performance"]:
            mp = report["model_performance"]
            md_content += f"""
## Model Performance
- **Best Model**: {mp.get('best_model', 'None')}
- **Best Accuracy**: {mp.get('best_accuracy', 0):.4f}

### Model Comparison
"""
            for model, accuracy in mp.get("model_comparison", {}).items():
                md_content += f"- **{model}**: {accuracy:.4f}\n"

        md_content += f"""
## Output Files
- **Features**: `{report['output_files']['features']}`
- **Analysis Results**: `{report['output_files']['analysis']}`
- **Visualizations**: `{report['output_files']['visualizations']}`
"""

        if report["output_files"]["models"]:
            md_content += f"- **Best Model**: `{report['output_files']['models']}`\n"

        # Save markdown report
        md_path = Path(output_dir) / "PIPELINE_REPORT.md"
        with open(md_path, "w") as f:
            f.write(md_content)

    def predict_mood(self, audio_path: str, lyrics_path: str = None) -> Dict[str, Any]:
        """
        Predict mood for a new track.

        Args:
            audio_path: Path to audio file
            lyrics_path: Path to lyrics file (optional)

        Returns:
            Dictionary with prediction results
        """
        if self.mood_classifier.best_model is None:
            raise ValueError("No trained model available. Run the full pipeline first.")

        # Create a temporary track
        track = MusicTrack(
            track_id="prediction_track", audio_path=audio_path, lyrics_path=lyrics_path
        )

        # Load and preprocess
        track = self.data_loader.load_audio(track, load_data=True)
        track = self.data_loader.load_lyrics(track)
        track = self.preprocessor.preprocess_track(track)

        # Extract features
        features = self.feature_extractor.extract_features(track)
        features_df = pd.DataFrame([features])

        # Make prediction
        predicted_labels, probabilities = self.mood_classifier.predict(features_df)

        result = {
            "predicted_mood": predicted_labels[0],
            "confidence": (
                float(np.max(probabilities[0])) if probabilities is not None else None
            ),
            "all_probabilities": (
                {
                    label: float(prob)
                    for label, prob in zip(
                        self.mood_classifier.label_encoder.classes_, probabilities[0]
                    )
                }
                if probabilities is not None
                else None
            ),
        }

        return result
