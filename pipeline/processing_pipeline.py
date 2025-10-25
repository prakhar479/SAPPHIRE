"""
Enhanced processing pipeline for SAPPHIRE with optimization and checkpointing.
Handles large-scale processing with memory management and fault tolerance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Iterator
import logging
from datetime import datetime
import json
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager
import psutil
import gc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from .config import config
from .data_loader import DataLoader, MusicTrack
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class ProcessingCheckpoint:
    """Manages processing checkpoints for fault tolerance."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, stage: str, data: Any, metadata: Dict[str, Any] = None):
        """Save processing checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.pkl"
        metadata_file = self.checkpoint_dir / f"{stage}_metadata.json"

        with open(checkpoint_file, "wb") as f:
            pickle.dump(data, f)

        checkpoint_metadata = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "data_type": type(data).__name__,
            "data_size": len(data) if hasattr(data, "__len__") else "unknown",
            **(metadata or {}),
        }

        with open(metadata_file, "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)

        logger.info(f"Checkpoint saved for stage '{stage}' at {checkpoint_file}")

    def load_checkpoint(self, stage: str) -> Optional[Any]:
        """Load processing checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.pkl"
        metadata_file = self.checkpoint_dir / f"{stage}_metadata.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, "rb") as f:
                data = pickle.load(f)

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                logger.info(
                    f"Loaded checkpoint for stage '{stage}' from {metadata['timestamp']}"
                )

            return data

        except Exception as e:
            logger.error(f"Error loading checkpoint for stage '{stage}': {e}")
            return None

    def checkpoint_exists(self, stage: str) -> bool:
        """Check if checkpoint exists for stage."""
        return (self.checkpoint_dir / f"{stage}_checkpoint.pkl").exists()

    def clear_checkpoint(self, stage: str):
        """Clear checkpoint for stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.pkl"
        metadata_file = self.checkpoint_dir / f"{stage}_metadata.json"

        for file in [checkpoint_file, metadata_file]:
            if file.exists():
                file.unlink()


class MemoryManager:
    """Manages memory usage during processing."""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.logger = logging.getLogger(__name__)

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limit."""
        current_usage = self.get_memory_usage()
        return current_usage < (self.memory_limit_bytes / 1024**3)

    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        self.logger.debug(
            f"Garbage collection performed. Memory usage: {self.get_memory_usage():.2f} GB"
        )

    def wait_for_memory(self, timeout: int = 60):
        """Wait for memory usage to drop below limit."""
        import time

        start_time = time.time()
        while not self.check_memory_limit():
            if time.time() - start_time > timeout:
                raise MemoryError(f"Memory usage exceeded limit for {timeout} seconds")

            self.force_garbage_collection()
            time.sleep(1)


class EnhancedProcessingPipeline:
    """
    Enhanced processing pipeline with optimization, checkpointing, and fault tolerance.
    """

    def __init__(self, config_obj=None, checkpoint_dir: str = "checkpoints"):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.preprocessor = Preprocessor(self.config)

        # Initialize checkpoint manager
        self.checkpoint_manager = ProcessingCheckpoint(checkpoint_dir)

        # Initialize memory manager
        self.memory_manager = MemoryManager(self.config.processing.memory_limit_gb)

        # Processing statistics
        self.stats = {
            "tracks_loaded": 0,
            "tracks_processed": 0,
            "tracks_failed": 0,
            "features_extracted": 0,
            "processing_time": 0,
            "memory_peak": 0,
        }

    def chunk_iterator(self, items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
        """Create chunks from a list of items."""
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def process_track_chunk(
        self, track_chunk: List[MusicTrack], chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Process a chunk of tracks."""
        self.logger.info(f"Processing chunk {chunk_id} with {len(track_chunk)} tracks")

        chunk_features = []

        for track in track_chunk:
            try:
                # Load audio and lyrics if not already loaded
                if track.audio_data is None:
                    track = self.data_loader.load_audio(track, load_data=True)
                if track.lyrics_text is None and track.lyrics_path:
                    track = self.data_loader.load_lyrics(track)

                # Preprocess track
                track = self.preprocessor.preprocess_track(track)

                # Check quality
                if not self.preprocessor.passes_quality_check(track):
                    self.stats["tracks_failed"] += 1
                    continue

                # Extract features
                features = self.feature_extractor.extract_features(track)

                # Add metadata
                features.update(
                    {
                        "track_id": track.track_id,
                        "mood_cluster": track.mood_cluster,
                        "mood_category": track.mood_category,
                        "dataset": track.metadata.get("dataset", "unknown"),
                        "chunk_id": chunk_id,
                    }
                )

                chunk_features.append(features)
                self.stats["tracks_processed"] += 1

                # Clear audio data to save memory
                track.audio_data = None

            except Exception as e:
                self.logger.error(f"Error processing track {track.track_id}: {e}")
                self.stats["tracks_failed"] += 1
                continue

        # Force garbage collection after chunk
        self.memory_manager.force_garbage_collection()

        return chunk_features

    def parallel_process_chunks(
        self, track_chunks: List[List[MusicTrack]]
    ) -> List[Dict[str, Any]]:
        """Process chunks in parallel."""
        all_features = []

        if self.config.optimization["use_multiprocessing"] and len(track_chunks) > 1:
            # Use multiprocessing for CPU-bound tasks
            max_workers = min(self.config.processing.workers, len(track_chunks))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks
                future_to_chunk = {
                    executor.submit(self.process_track_chunk, chunk, i): i
                    for i, chunk in enumerate(track_chunks)
                }

                # Process completed chunks with progress bar
                if self.config.processing.progress_bar:
                    futures = tqdm(
                        as_completed(future_to_chunk),
                        total=len(track_chunks),
                        desc="Processing chunks",
                    )
                else:
                    futures = as_completed(future_to_chunk)

                for future in futures:
                    try:
                        chunk_features = future.result()
                        all_features.extend(chunk_features)

                        # Check memory usage
                        if not self.memory_manager.check_memory_limit():
                            self.logger.warning(
                                "Memory limit approached, forcing garbage collection"
                            )
                            self.memory_manager.force_garbage_collection()

                    except Exception as e:
                        chunk_id = future_to_chunk[future]
                        self.logger.error(f"Chunk {chunk_id} failed: {e}")
        else:
            # Sequential processing
            for i, chunk in enumerate(track_chunks):
                if self.config.processing.progress_bar:
                    print(f"Processing chunk {i+1}/{len(track_chunks)}")

                chunk_features = self.process_track_chunk(chunk, i)
                all_features.extend(chunk_features)

                # Save checkpoint periodically
                if (i + 1) % 5 == 0:  # Every 5 chunks
                    self.checkpoint_manager.save_checkpoint(
                        f"features_partial_{i+1}",
                        all_features,
                        {"chunks_completed": i + 1, "total_chunks": len(track_chunks)},
                    )

        return all_features

    def load_datasets_with_optimization(
        self, datasets: Optional[List[str]] = None
    ) -> List[MusicTrack]:
        """Load datasets with memory optimization."""
        stage_name = "dataset_loading"

        # Check for existing checkpoint
        if self.config.optimization[
            "resume_from_checkpoint"
        ] and self.checkpoint_manager.checkpoint_exists(stage_name):

            self.logger.info("Resuming from dataset loading checkpoint...")
            tracks = self.checkpoint_manager.load_checkpoint(stage_name)
            if tracks:
                self.stats["tracks_loaded"] = len(tracks)
                return tracks

        # Load datasets
        self.logger.info("Loading datasets...")

        if datasets is None:
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
                else:
                    self.logger.warning(f"Unknown dataset: {dataset_name}")

        self.stats["tracks_loaded"] = len(tracks)

        # Save checkpoint
        if self.config.optimization["incremental_processing"]:
            self.checkpoint_manager.save_checkpoint(stage_name, tracks)

        return tracks

    def extract_features_optimized(self, tracks: List[MusicTrack]) -> pd.DataFrame:
        """Extract features with optimization and checkpointing."""
        stage_name = "feature_extraction"

        # Check for existing checkpoint
        if self.config.optimization[
            "resume_from_checkpoint"
        ] and self.checkpoint_manager.checkpoint_exists(stage_name):

            self.logger.info("Resuming from feature extraction checkpoint...")
            features_df = self.checkpoint_manager.load_checkpoint(stage_name)
            if features_df is not None:
                return features_df

        self.logger.info(
            f"Starting optimized feature extraction for {len(tracks)} tracks..."
        )
        start_time = datetime.now()

        # Create chunks for processing
        chunk_size = self.config.processing.chunk_size
        track_chunks = list(self.chunk_iterator(tracks, chunk_size))

        self.logger.info(f"Created {len(track_chunks)} chunks of size {chunk_size}")

        # Process chunks
        all_features = self.parallel_process_chunks(track_chunks)

        if not all_features:
            raise ValueError("No features were successfully extracted")

        # Create DataFrame
        features_df = pd.DataFrame(all_features)

        # Update statistics
        self.stats["features_extracted"] = (
            len(features_df.columns) - 4
        )  # Exclude metadata columns
        self.stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        self.stats["memory_peak"] = self.memory_manager.get_memory_usage()

        # Save checkpoint
        if self.config.optimization["incremental_processing"]:
            self.checkpoint_manager.save_checkpoint(stage_name, features_df)

        self.logger.info(f"Feature extraction complete:")
        self.logger.info(f"  Processed: {self.stats['tracks_processed']} tracks")
        self.logger.info(f"  Failed: {self.stats['tracks_failed']} tracks")
        self.logger.info(f"  Features: {self.stats['features_extracted']} per track")
        self.logger.info(f"  Time: {self.stats['processing_time']:.1f} seconds")
        self.logger.info(f"  Peak memory: {self.stats['memory_peak']:.2f} GB")

        return features_df

    def preprocess_with_advanced_filtering(
        self, tracks: List[MusicTrack]
    ) -> List[MusicTrack]:
        """Preprocess tracks with advanced filtering."""
        stage_name = "preprocessing"

        # Check for existing checkpoint
        if self.config.optimization[
            "resume_from_checkpoint"
        ] and self.checkpoint_manager.checkpoint_exists(stage_name):

            self.logger.info("Resuming from preprocessing checkpoint...")
            processed_tracks = self.checkpoint_manager.load_checkpoint(stage_name)
            if processed_tracks:
                return processed_tracks

        self.logger.info("Starting advanced preprocessing and filtering...")

        # Process in batches to manage memory
        batch_size = min(100, len(tracks))  # Smaller batches for preprocessing
        processed_tracks = []

        for i in range(0, len(tracks), batch_size):
            batch = tracks[i : i + batch_size]

            if self.config.processing.progress_bar:
                print(
                    f"Preprocessing batch {i//batch_size + 1}/{(len(tracks) + batch_size - 1)//batch_size}"
                )

            # Load audio and lyrics for batch
            for track in batch:
                if track.audio_data is None:
                    track = self.data_loader.load_audio(track, load_data=True)
                if track.lyrics_text is None and track.lyrics_path:
                    track = self.data_loader.load_lyrics(track)

            # Apply advanced filtering
            batch_processed = self.preprocessor.preprocess_track_batch(batch)
            processed_tracks.extend(batch_processed)

            # Clear memory
            for track in batch:
                track.audio_data = None

            self.memory_manager.force_garbage_collection()

        # Save checkpoint
        if self.config.optimization["incremental_processing"]:
            self.checkpoint_manager.save_checkpoint(stage_name, processed_tracks)

        return processed_tracks

    def run_optimized_pipeline(
        self,
        datasets: Optional[List[str]] = None,
        output_dir: str = "output",
        limit_tracks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the optimized processing pipeline.

        Args:
            datasets: List of datasets to process
            output_dir: Output directory
            limit_tracks: Limit number of tracks to process

        Returns:
            Dictionary with processing results
        """
        self.logger.info("Starting optimized SAPPHIRE processing pipeline...")
        pipeline_start = datetime.now()

        try:
            # Step 1: Load datasets
            tracks = self.load_datasets_with_optimization(datasets)

            if limit_tracks:
                tracks = tracks[:limit_tracks]
                self.logger.info(f"Limited to {len(tracks)} tracks")

            if not tracks:
                raise ValueError("No tracks loaded")

            # Step 2: Advanced preprocessing and filtering
            processed_tracks = self.preprocess_with_advanced_filtering(tracks)

            if not processed_tracks:
                raise ValueError("No tracks passed preprocessing filters")

            # Step 3: Optimized feature extraction
            features_df = self.extract_features_optimized(processed_tracks)

            # Step 4: Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            features_path = output_path / "features"
            features_path.mkdir(exist_ok=True)

            # Save features in multiple formats
            features_df.to_parquet(features_path / "features.parquet", index=False)
            features_df.to_csv(features_path / "features.csv", index=False)

            # Save processing statistics
            final_stats = {
                **self.stats,
                "total_pipeline_time": (
                    datetime.now() - pipeline_start
                ).total_seconds(),
                "tracks_loaded": len(tracks),
                "tracks_after_filtering": len(processed_tracks),
                "final_features_shape": list(features_df.shape),
                "success_rate": (
                    self.stats["tracks_processed"] / len(tracks) if tracks else 0
                ),
            }

            with open(output_path / "processing_stats.json", "w") as f:
                json.dump(final_stats, f, indent=2, default=str)

            # Clear checkpoints on successful completion
            if self.config.optimization["incremental_processing"]:
                for stage in ["dataset_loading", "preprocessing", "feature_extraction"]:
                    self.checkpoint_manager.clear_checkpoint(stage)

            self.logger.info("Optimized pipeline completed successfully!")

            return {
                "status": "success",
                "features_df": features_df,
                "output_dir": str(output_path),
                "statistics": final_stats,
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {"status": "failed", "error": str(e), "statistics": self.stats}

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        return {
            "statistics": self.stats,
            "memory_usage_gb": self.memory_manager.get_memory_usage(),
            "memory_limit_gb": self.config.processing.memory_limit_gb,
            "checkpoints": {
                "dataset_loading": self.checkpoint_manager.checkpoint_exists(
                    "dataset_loading"
                ),
                "preprocessing": self.checkpoint_manager.checkpoint_exists(
                    "preprocessing"
                ),
                "feature_extraction": self.checkpoint_manager.checkpoint_exists(
                    "feature_extraction"
                ),
            },
        }

    def cleanup_checkpoints(self):
        """Clean up all checkpoints."""
        for stage in ["dataset_loading", "preprocessing", "feature_extraction"]:
            self.checkpoint_manager.clear_checkpoint(stage)
        self.logger.info("All checkpoints cleared")
