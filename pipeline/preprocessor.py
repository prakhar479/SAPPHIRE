"""
Data preprocessing for the SAPPHIRE pipeline.
Handles audio normalization, quality filtering, and data cleaning.
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Optional libraries
try:
    import pyloudnorm as pyln

    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

try:
    from langdetect import detect, LangDetectException

    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

from .config import config
from .data_loader import MusicTrack

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Comprehensive preprocessor for music data.
    Handles audio normalization, quality assessment, and data cleaning.
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)

        # Initialize loudness meter if available
        self.loudness_meter = None
        if HAS_PYLOUDNORM:
            try:
                self.loudness_meter = pyln.Meter(self.config.audio.sample_rate)
            except Exception as e:
                self.logger.warning(f"Failed to initialize loudness meter: {e}")

    def preprocess_track(self, track: MusicTrack) -> Tuple[MusicTrack, Dict]:
        """
        Preprocess a single track.

        Args:
            track: MusicTrack object

        Returns:
            Tuple of (processed_track, quality_metrics)
        """
        quality_metrics = {
            "track_id": track.track_id,
            "passes_quality_check": False,
            "audio_quality": {},
            "lyrics_quality": {},
            "processing_errors": [],
        }

        try:
            # Process audio if available
            if track.audio_path and Path(track.audio_path).exists():
                track, audio_quality = self._preprocess_audio(track)
                quality_metrics["audio_quality"] = audio_quality

            # Process lyrics if available
            if track.lyrics_path and Path(track.lyrics_path).exists():
                track, lyrics_quality = self._preprocess_lyrics(track)
                quality_metrics["lyrics_quality"] = lyrics_quality

            # Overall quality assessment
            quality_metrics["passes_quality_check"] = self._assess_overall_quality(
                quality_metrics
            )

        except Exception as e:
            self.logger.error(f"Error preprocessing track {track.track_id}: {e}")
            quality_metrics["processing_errors"].append(str(e))

        return track, quality_metrics

    def _preprocess_audio(self, track: MusicTrack) -> Tuple[MusicTrack, Dict]:
        """Preprocess audio data."""
        quality_metrics = {}

        try:
            # Load audio if not already loaded
            if track.audio_data is None:
                audio_data, sample_rate = librosa.load(
                    track.audio_path, sr=self.config.audio.sample_rate, mono=True
                )
                track.audio_data = audio_data
                track.sample_rate = sample_rate

            # Basic quality metrics
            duration = len(track.audio_data) / track.sample_rate
            quality_metrics["duration"] = duration
            quality_metrics["sample_rate"] = track.sample_rate

            # Check duration constraints
            min_duration = self.config.quality_thresholds["duration_min"]
            max_duration = self.config.quality_thresholds["duration_max"]
            quality_metrics["duration_ok"] = min_duration <= duration <= max_duration

            # Audio quality assessment
            quality_metrics.update(
                self._assess_audio_quality(track.audio_data, track.sample_rate)
            )

            # Normalize audio
            track.audio_data = self._normalize_audio(
                track.audio_data, track.sample_rate
            )

            # Update metadata
            track.metadata.update(
                {"preprocessed": True, "normalized": True, "duration": duration}
            )

        except Exception as e:
            self.logger.error(f"Error preprocessing audio for {track.track_id}: {e}")
            quality_metrics["error"] = str(e)

        return track, quality_metrics

    def _preprocess_lyrics(self, track: MusicTrack) -> Tuple[MusicTrack, Dict]:
        """Preprocess lyrics data."""
        quality_metrics = {}

        try:
            # Load lyrics if not already loaded
            if track.lyrics_text is None:
                with open(track.lyrics_path, "r", encoding="utf-8") as f:
                    track.lyrics_text = f.read()

            # Clean lyrics text
            original_text = track.lyrics_text
            track.lyrics_text = self._clean_lyrics_text(track.lyrics_text)

            # Basic quality metrics
            quality_metrics["original_length"] = len(original_text)
            quality_metrics["cleaned_length"] = len(track.lyrics_text)
            quality_metrics["word_count"] = len(track.lyrics_text.split())

            # Language detection
            if HAS_LANGDETECT and len(track.lyrics_text) > 20:
                try:
                    detected_language = detect(track.lyrics_text)
                    quality_metrics["detected_language"] = detected_language
                    track.metadata["detected_language"] = detected_language
                except LangDetectException:
                    quality_metrics["detected_language"] = "unknown"

            # Completeness assessment
            quality_metrics["completeness"] = self._assess_lyrics_completeness(
                track.lyrics_text
            )

            # Update metadata
            track.metadata.update(
                {
                    "lyrics_preprocessed": True,
                    "lyrics_word_count": quality_metrics["word_count"],
                }
            )

        except Exception as e:
            self.logger.error(f"Error preprocessing lyrics for {track.track_id}: {e}")
            quality_metrics["error"] = str(e)

        return track, quality_metrics

    def _assess_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Assess audio quality metrics."""
        metrics = {}

        try:
            # Signal-to-noise ratio estimation
            metrics["snr_db"] = self._estimate_snr(audio_data)

            # Dynamic range
            if np.mean(np.abs(audio_data)) > 0:
                metrics["dynamic_range_db"] = 20 * np.log10(
                    np.max(np.abs(audio_data)) / np.mean(np.abs(audio_data))
                )
            else:
                metrics["dynamic_range_db"] = 0.0

            # Clipping detection
            metrics["clipping_percentage"] = np.mean(np.abs(audio_data) >= 0.999) * 100

            # RMS energy
            metrics["rms_energy"] = np.sqrt(np.mean(audio_data**2))

            # Loudness measurement
            if self.loudness_meter:
                try:
                    metrics["integrated_loudness_lufs"] = (
                        self.loudness_meter.integrated_loudness(audio_data)
                    )
                except:
                    metrics["integrated_loudness_lufs"] = None

            # Vocal dominance (using harmonic/percussive separation)
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                total_energy = harmonic_energy + percussive_energy

                if total_energy > 0:
                    metrics["vocal_dominance_score"] = harmonic_energy / total_energy
                else:
                    metrics["vocal_dominance_score"] = 0.0
            except:
                metrics["vocal_dominance_score"] = 0.0

            # Quality flags
            metrics["snr_ok"] = (
                metrics["snr_db"] >= self.config.quality_thresholds["snr_min"]
            )
            metrics["clipping_ok"] = (
                metrics["clipping_percentage"] < 5.0
            )  # Less than 5% clipping
            metrics["vocal_ok"] = (
                metrics["vocal_dominance_score"]
                >= self.config.quality_thresholds["vocal_dominance_min"]
            )

        except Exception as e:
            self.logger.error(f"Error assessing audio quality: {e}")
            metrics["error"] = str(e)

        return metrics

    def _estimate_snr(self, audio_data: np.ndarray, percentile: float = 95.0) -> float:
        """Estimate signal-to-noise ratio using percentile-based method."""
        try:
            energy = audio_data**2
            if np.max(energy) == 0:
                return -np.inf

            signal_threshold = np.percentile(energy, percentile)
            signal_part = energy[energy >= signal_threshold]
            noise_part = energy[energy < signal_threshold]

            if np.mean(noise_part) == 0:
                return np.inf

            signal_power = np.mean(signal_part)
            noise_power = np.mean(noise_part)

            return 10 * np.log10(signal_power / noise_power)
        except:
            return 0.0

    def _normalize_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio using loudness normalization if available, otherwise peak normalization."""
        try:
            if self.loudness_meter and HAS_PYLOUDNORM:
                # Loudness normalization
                current_loudness = self.loudness_meter.integrated_loudness(audio_data)
                if not np.isfinite(current_loudness):
                    # Fallback to peak normalization
                    return self._peak_normalize(audio_data)

                normalized_audio = pyln.normalize.loudness(
                    audio_data, current_loudness, self.config.audio.target_loudness
                )
                return normalized_audio
            else:
                # Peak normalization fallback
                return self._peak_normalize(audio_data)

        except Exception as e:
            self.logger.warning(f"Normalization failed, using peak normalization: {e}")
            return self._peak_normalize(audio_data)

    def _peak_normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Peak normalization to prevent clipping."""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val * 0.95  # Leave some headroom
        return audio_data

    def _clean_lyrics_text(self, text: str) -> str:
        """Clean and normalize lyrics text."""
        import re

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        # Strip and return
        return text.strip()

    def _assess_lyrics_completeness(self, text: str) -> float:
        """Assess lyrics completeness using heuristics."""
        if not text or len(text.strip()) == 0:
            return 0.0

        # Check for incomplete markers
        incomplete_markers = ["...", "[incomplete]", "[instrumental]", "[missing]"]
        has_incomplete_markers = any(
            marker in text.lower() for marker in incomplete_markers
        )

        # Word count heuristic
        word_count = len(text.split())
        word_score = min(word_count / 50.0, 1.0)  # Assume 50+ words is complete

        # Line count heuristic
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        line_score = min(len(lines) / 10.0, 1.0)  # Assume 10+ lines is complete

        # Combine scores
        base_score = (word_score + line_score) / 2.0

        # Penalize for incomplete markers
        if has_incomplete_markers:
            base_score *= 0.5

        return base_score

    def _assess_overall_quality(self, quality_metrics: Dict) -> bool:
        """Assess overall track quality based on all metrics."""
        audio_quality = quality_metrics.get("audio_quality", {})
        lyrics_quality = quality_metrics.get("lyrics_quality", {})

        # Audio quality checks
        audio_ok = True
        if audio_quality:
            audio_ok = (
                audio_quality.get("duration_ok", False)
                and audio_quality.get("snr_ok", False)
                and audio_quality.get("clipping_ok", True)
                and audio_quality.get("vocal_ok", False)
            )

        # Lyrics quality checks
        lyrics_ok = True
        if lyrics_quality:
            completeness = lyrics_quality.get("completeness", 0.0)
            lyrics_ok = (
                completeness
                >= self.config.quality_thresholds["lyrics_completeness_min"]
            )

        # Overall assessment
        return audio_ok and lyrics_ok

    def preprocess_batch(
        self, tracks: List[MusicTrack], n_workers: int = None
    ) -> Tuple[List[MusicTrack], pd.DataFrame]:
        """
        Preprocess multiple tracks in parallel.

        Args:
            tracks: List of MusicTrack objects
            n_workers: Number of parallel workers

        Returns:
            Tuple of (processed_tracks, quality_metrics_df)
        """
        n_workers = n_workers or self.config.processing.workers

        if n_workers == 1:
            # Sequential processing
            results = [
                self.preprocess_track(track)
                for track in tqdm(tracks, desc="Preprocessing tracks")
            ]
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(self.preprocess_track, track) for track in tracks
                ]
                results = []

                for future in tqdm(futures, desc="Preprocessing tracks"):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Error in batch preprocessing: {e}")

        # Separate tracks and quality metrics
        processed_tracks = [result[0] for result in results]
        quality_metrics = [result[1] for result in results]

        # Create quality metrics DataFrame
        quality_df = pd.DataFrame(quality_metrics)

        return processed_tracks, quality_df

    def filter_high_quality_tracks(
        self, tracks: List[MusicTrack], quality_df: pd.DataFrame
    ) -> List[MusicTrack]:
        """Filter tracks that pass quality checks."""
        high_quality_tracks = []

        for track, (_, row) in zip(tracks, quality_df.iterrows()):
            if row.get("passes_quality_check", False):
                high_quality_tracks.append(track)

        self.logger.info(
            f"Filtered {len(high_quality_tracks)} high-quality tracks from {len(tracks)} total"
        )
        return high_quality_tracks

    def save_processed_audio(self, tracks: List[MusicTrack], output_dir: str):
        """Save processed audio files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for track in tqdm(tracks, desc="Saving processed audio"):
            if track.audio_data is not None:
                output_file = output_path / f"{track.track_id}.wav"
                try:
                    sf.write(
                        str(output_file),
                        track.audio_data,
                        track.sample_rate,
                        subtype="PCM_16",
                    )
                    # Update track path to point to processed file
                    track.audio_path = str(output_file)
                except Exception as e:
                    self.logger.error(f"Error saving audio for {track.track_id}: {e}")

    def create_quality_report(self, quality_df: pd.DataFrame, output_path: str):
        """Create a comprehensive quality assessment report."""
        report = {
            "summary": {
                "total_tracks": len(quality_df),
                "high_quality_tracks": quality_df["passes_quality_check"].sum(),
                "quality_rate": quality_df["passes_quality_check"].mean(),
            },
            "audio_quality": {},
            "lyrics_quality": {},
        }

        # Audio quality statistics
        if "audio_quality" in quality_df.columns:
            audio_metrics = []
            for _, row in quality_df.iterrows():
                if isinstance(row["audio_quality"], dict):
                    audio_metrics.append(row["audio_quality"])

            if audio_metrics:
                audio_df = pd.DataFrame(audio_metrics)
                report["audio_quality"] = {
                    "mean_snr_db": (
                        audio_df["snr_db"].mean() if "snr_db" in audio_df else None
                    ),
                    "mean_duration": (
                        audio_df["duration"].mean() if "duration" in audio_df else None
                    ),
                    "clipping_rate": (
                        (audio_df["clipping_percentage"] > 5.0).mean()
                        if "clipping_percentage" in audio_df
                        else None
                    ),
                }

        # Lyrics quality statistics
        if "lyrics_quality" in quality_df.columns:
            lyrics_metrics = []
            for _, row in quality_df.iterrows():
                if isinstance(row["lyrics_quality"], dict):
                    lyrics_metrics.append(row["lyrics_quality"])

            if lyrics_metrics:
                lyrics_df = pd.DataFrame(lyrics_metrics)
                report["lyrics_quality"] = {
                    "mean_completeness": (
                        lyrics_df["completeness"].mean()
                        if "completeness" in lyrics_df
                        else None
                    ),
                    "mean_word_count": (
                        lyrics_df["word_count"].mean()
                        if "word_count" in lyrics_df
                        else None
                    ),
                }

        # Save report
        with open(output_path, "w") as f:
            import json

            json.dump(report, f, indent=2)

        self.logger.info(f"Quality report saved to {output_path}")
        return report


def advanced_quality_assessment(self, track: MusicTrack) -> Dict[str, float]:
    """
    Perform advanced quality assessment on audio track.

    Args:
        track: MusicTrack object with audio data

    Returns:
        Dictionary with quality metrics
    """
    if track.audio_data is None:
        return {}

    audio = track.audio_data
    sr = track.sample_rate or self.config.audio.sample_rate

    # Handle edge cases for robust processing
    if len(audio) == 0:
        return {"error": "Empty audio data", "duration": 0}

    if np.all(audio == 0):
        return {"error": "Silent audio data", "duration": len(audio) / sr}

    # Handle very short clips
    if len(audio) < sr * 0.1:  # Less than 0.1 seconds
        self.logger.debug(
            f"Very short audio clip: {len(audio) / sr:.3f}s for track {track.track_id}"
        )

    quality_metrics = {}

    try:
        # Basic audio metrics
        quality_metrics["duration"] = len(audio) / sr
        quality_metrics["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        quality_metrics["max_amplitude"] = float(np.max(np.abs(audio)))

        # Dynamic range
        rms_db = 20 * np.log10(quality_metrics["rms_energy"] + 1e-10)
        peak_db = 20 * np.log10(quality_metrics["max_amplitude"] + 1e-10)
        quality_metrics["dynamic_range_db"] = peak_db - rms_db

        # Spectral features for quality assessment
        stft = librosa.stft(audio, hop_length=self.config.audio.hop_length)
        magnitude = np.abs(stft)

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        quality_metrics["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        quality_metrics["spectral_centroid_std"] = float(np.std(spectral_centroids))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        quality_metrics["zero_crossing_rate_mean"] = float(np.mean(zcr))

        # Silence detection
        silence_frames = np.sum(
            magnitude < self.config.quality_thresholds["silence_threshold"]
        )
        total_frames = magnitude.shape[1]
        quality_metrics["silence_ratio"] = (
            silence_frames / total_frames if total_frames > 0 else 1.0
        )

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        quality_metrics["spectral_rolloff_mean"] = float(np.mean(rolloff))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        quality_metrics["spectral_bandwidth_mean"] = float(np.mean(bandwidth))

        # Estimate SNR using spectral subtraction method
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        signal_power = np.mean(magnitude**2)
        noise_power = np.mean(noise_floor**2)
        quality_metrics["estimated_snr_db"] = 10 * np.log10(
            signal_power / (noise_power + 1e-10)
        )

        # Harmonic-percussive separation for vocal dominance
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonic**2)
        total_energy = np.sum(audio**2)
        quality_metrics["harmonic_ratio"] = harmonic_energy / (total_energy + 1e-10)

        # Estimate vocal dominance (simplified)
        # Focus on mid-frequency range where vocals typically occur
        vocal_freq_range = (300, 3400)  # Hz
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.config.audio.n_fft)
        vocal_mask = (freq_bins >= vocal_freq_range[0]) & (
            freq_bins <= vocal_freq_range[1]
        )

        vocal_energy = np.sum(magnitude[vocal_mask, :])
        total_spectral_energy = np.sum(magnitude)
        quality_metrics["vocal_dominance"] = vocal_energy / (
            total_spectral_energy + 1e-10
        )

    except Exception as e:
        self.logger.warning(f"Error in quality assessment: {e}")

    return quality_metrics


def passes_advanced_quality_check(
    self, track: MusicTrack
) -> Tuple[bool, Dict[str, str]]:
    """
    Check if track passes advanced quality filters.

    Args:
        track: MusicTrack object

    Returns:
        Tuple of (passes, reasons_dict)
    """
    if not self.config.filtering["enable_quality_filter"]:
        return True, {}

    reasons = {}

    # Get quality metrics
    quality_metrics = self.advanced_quality_assessment(track)

    if not quality_metrics:
        reasons["no_audio"] = "No audio data available"
        return False, reasons

    # Duration check - only if enabled and with very lenient bounds
    if self.config.filtering["enable_duration_filter"]:
        duration = quality_metrics.get("duration", 0)
        if duration < self.config.quality_thresholds["duration_min"]:
            reasons["too_short"] = (
                f"Duration {duration:.1f}s < {self.config.quality_thresholds['duration_min']}s"
            )
        elif duration > self.config.quality_thresholds["duration_max"]:
            reasons["too_long"] = (
                f"Duration {duration:.1f}s > {self.config.quality_thresholds['duration_max']}s"
            )

    # SNR check - only for extremely poor quality audio
    snr = quality_metrics.get(
        "estimated_snr_db", float("inf")
    )  # Default to pass if no SNR
    if np.isfinite(snr) and snr < self.config.quality_thresholds["snr_min"]:
        reasons["low_snr"] = (
            f"SNR {snr:.1f}dB < {self.config.quality_thresholds['snr_min']}dB"
        )

    # Dynamic range check - only for extremely compressed audio
    dynamic_range = quality_metrics.get(
        "dynamic_range_db", float("inf")
    )  # Default to pass
    if (
        np.isfinite(dynamic_range)
        and dynamic_range < self.config.quality_thresholds["dynamic_range_min"]
    ):
        reasons["low_dynamic_range"] = (
            f"Dynamic range {dynamic_range:.1f}dB < {self.config.quality_thresholds['dynamic_range_min']}dB"
        )

    # Silence check - only if enabled and for mostly silent tracks
    if self.config.filtering["enable_silence_filter"]:
        silence_ratio = quality_metrics.get("silence_ratio", 0)
        if silence_ratio > 0.8:  # More than 80% silence (very lenient)
            reasons["too_much_silence"] = f"Silence ratio {silence_ratio:.2f} > 0.8"

    # Spectral centroid check - only for extremely distorted audio
    spectral_centroid = quality_metrics.get("spectral_centroid_mean", 0)
    if spectral_centroid > self.config.quality_thresholds["spectral_centroid_max"]:
        reasons["high_spectral_centroid"] = (
            f"Spectral centroid {spectral_centroid:.0f}Hz > {self.config.quality_thresholds['spectral_centroid_max']}Hz"
        )

    # Zero crossing rate check - only for extremely noisy audio
    zcr = quality_metrics.get("zero_crossing_rate_mean", 0)
    if zcr > self.config.quality_thresholds["zero_crossing_rate_max"]:
        reasons["high_zero_crossing_rate"] = (
            f"ZCR {zcr:.3f} > {self.config.quality_thresholds['zero_crossing_rate_max']}"
        )

    # Vocal dominance check - very lenient, allow instrumental tracks
    vocal_dominance = quality_metrics.get("vocal_dominance", 1.0)  # Default to pass
    if vocal_dominance < self.config.quality_thresholds["vocal_dominance_min"]:
        # Only warn, don't fail for low vocal dominance
        self.logger.debug(
            f"Low vocal dominance {vocal_dominance:.3f} for track {track.track_id}"
        )
        # Don't add to reasons to allow instrumental tracks

    # Store quality metrics in track metadata
    track.metadata.update(
        {"quality_metrics": quality_metrics, "quality_check_passed": len(reasons) == 0}
    )

    return len(reasons) == 0, reasons


def assess_lyrics_quality(self, track: MusicTrack) -> Dict[str, Any]:
    """
    Assess lyrics quality and completeness.

    Args:
        track: MusicTrack object with lyrics

    Returns:
        Dictionary with lyrics quality metrics
    """
    if not track.lyrics_text:
        return {"has_lyrics": False, "completeness": 0.0}

    lyrics = track.lyrics_text.strip()
    metrics = {"has_lyrics": True}

    try:
        # Basic metrics
        metrics["length"] = len(lyrics)
        metrics["word_count"] = len(lyrics.split())
        metrics["line_count"] = len(
            [line for line in lyrics.split("\n") if line.strip()]
        )

        # Language detection
        if HAS_LANGDETECT:
            try:
                detected_lang = detect(lyrics)
                metrics["detected_language"] = detected_lang

                # Check if language is allowed
                if self.config.filtering["enable_language_filter"]:
                    metrics["language_allowed"] = (
                        detected_lang in self.config.filtering["allowed_languages"]
                    )
                else:
                    metrics["language_allowed"] = True

            except Exception:
                metrics["detected_language"] = "unknown"
                metrics["language_allowed"] = True
        else:
            metrics["detected_language"] = "unknown"
            metrics["language_allowed"] = True

        # Completeness assessment
        # Check for common incomplete lyrics indicators
        incomplete_indicators = [
            "[instrumental]",
            "[music]",
            "[repeat]",
            "[chorus]",
            "[verse]",
            "...",
            "na na na",
            "la la la",
            "oh oh oh",
            "yeah yeah yeah",
        ]

        lyrics_lower = lyrics.lower()
        incomplete_count = sum(
            1 for indicator in incomplete_indicators if indicator in lyrics_lower
        )

        # Simple completeness score
        if metrics["word_count"] < self.config.filtering["min_lyrics_words"]:
            completeness = 0.0
        else:
            completeness = max(0.0, 1.0 - (incomplete_count * 0.2))

        metrics["completeness"] = completeness
        metrics["incomplete_indicators_count"] = incomplete_count

        # Repetition analysis
        words = lyrics.split()
        unique_words = set(words)
        metrics["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0

        # Check for excessive repetition
        if len(words) > 0:
            most_common_word_count = max([words.count(word) for word in unique_words])
            metrics["max_word_repetition_ratio"] = most_common_word_count / len(words)
        else:
            metrics["max_word_repetition_ratio"] = 0

    except Exception as e:
        self.logger.warning(f"Error in lyrics quality assessment: {e}")
        metrics["completeness"] = 0.5  # Default moderate completeness

    return metrics


def passes_lyrics_quality_check(self, track: MusicTrack) -> Tuple[bool, Dict[str, str]]:
    """
    Check if track passes lyrics quality filters.

    Args:
        track: MusicTrack object

    Returns:
        Tuple of (passes, reasons_dict)
    """
    reasons = {}

    lyrics_metrics = self.assess_lyrics_quality(track)

    # Store lyrics metrics
    track.metadata.update({"lyrics_metrics": lyrics_metrics})

    if not lyrics_metrics["has_lyrics"]:
        # If no lyrics, it's not necessarily a failure unless required
        return True, {}

    # Check completeness - very lenient
    completeness = lyrics_metrics.get("completeness", 1.0)  # Default to pass
    if completeness < self.config.quality_thresholds["lyrics_completeness_min"]:
        # Only warn, don't fail for incomplete lyrics
        self.logger.debug(
            f"Low lyrics completeness {completeness:.2f} for track {track.track_id}"
        )
        # Don't add to reasons to allow incomplete lyrics

    # Check word count - very lenient
    word_count = lyrics_metrics.get("word_count", 0)
    if word_count > 0 and word_count < self.config.filtering["min_lyrics_words"]:
        # Only warn for very short lyrics, don't fail
        self.logger.debug(f"Short lyrics {word_count} words for track {track.track_id}")
        # Don't add to reasons to allow short lyrics

    # Check language if filtering is enabled (disabled by default)
    if self.config.filtering["enable_language_filter"]:
        if not lyrics_metrics.get("language_allowed", True):
            detected_lang = lyrics_metrics.get("detected_language", "unknown")
            reasons["language_not_allowed"] = (
                f"Language '{detected_lang}' not in allowed list"
            )

    return len(reasons) == 0, reasons


def detect_duplicates(self, tracks: List[MusicTrack]) -> List[Tuple[int, int, float]]:
    """
    Detect potential duplicate tracks based on audio similarity.

    Args:
        tracks: List of MusicTrack objects

    Returns:
        List of tuples (index1, index2, similarity_score)
    """
    if not self.config.filtering["enable_duplicate_detection"]:
        return []

    self.logger.info("Detecting potential duplicate tracks...")

    duplicates = []

    # Extract simple audio fingerprints for comparison
    fingerprints = []
    valid_indices = []

    for i, track in enumerate(tracks):
        if track.audio_data is not None:
            try:
                # Simple fingerprint: spectral centroid and MFCC means
                mfccs = librosa.feature.mfcc(
                    y=track.audio_data, sr=track.sample_rate, n_mfcc=13
                )
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=track.audio_data, sr=track.sample_rate
                )

                fingerprint = np.concatenate(
                    [np.mean(mfccs, axis=1), [np.mean(spectral_centroid)]]
                )

                fingerprints.append(fingerprint)
                valid_indices.append(i)

            except Exception as e:
                self.logger.warning(f"Error creating fingerprint for track {i}: {e}")
                continue

    # Compare fingerprints
    from sklearn.metrics.pairwise import cosine_similarity

    if len(fingerprints) > 1:
        fingerprints_matrix = np.array(fingerprints)
        similarity_matrix = cosine_similarity(fingerprints_matrix)

        threshold = self.config.filtering["similarity_threshold"]

        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    duplicates.append((valid_indices[i], valid_indices[j], similarity))

    self.logger.info(f"Found {len(duplicates)} potential duplicate pairs")
    return duplicates


def detect_outliers(self, tracks: List[MusicTrack]) -> List[int]:
    """
    Detect outlier tracks based on audio features.

    Args:
        tracks: List of MusicTrack objects

    Returns:
        List of track indices that are outliers
    """
    if not self.config.filtering["enable_outlier_detection"]:
        return []

    self.logger.info("Detecting outlier tracks...")

    # Extract features for outlier detection
    features = []
    valid_indices = []

    for i, track in enumerate(tracks):
        if track.audio_data is not None:
            try:
                quality_metrics = self.advanced_quality_assessment(track)

                # Use key quality metrics for outlier detection
                feature_vector = [
                    quality_metrics.get("rms_energy", 0),
                    quality_metrics.get("spectral_centroid_mean", 0),
                    quality_metrics.get("zero_crossing_rate_mean", 0),
                    quality_metrics.get("dynamic_range_db", 0),
                    quality_metrics.get("vocal_dominance", 0),
                    quality_metrics.get("silence_ratio", 0),
                ]

                features.append(feature_vector)
                valid_indices.append(i)

            except Exception as e:
                self.logger.warning(
                    f"Error extracting features for outlier detection: {e}"
                )
                continue

    if len(features) < 10:  # Need minimum samples for outlier detection
        return []

    outliers = []

    try:
        features_array = np.array(features)

        # Handle NaN and infinite values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)

        method = self.config.filtering["outlier_method"]
        contamination = self.config.filtering["contamination"]

        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor

            detector = LocalOutlierFactor(contamination=contamination)
        else:
            self.logger.warning(f"Unknown outlier detection method: {method}")
            return []

        outlier_labels = detector.fit_predict(features_array)
        outlier_indices = [
            valid_indices[i] for i, label in enumerate(outlier_labels) if label == -1
        ]

        self.logger.info(f"Detected {len(outlier_indices)} outlier tracks")
        return outlier_indices

    except Exception as e:
        self.logger.error(f"Error in outlier detection: {e}")
        return []


def preprocess_track_batch(self, tracks: List[MusicTrack]) -> List[MusicTrack]:
    """
    Preprocess a batch of tracks with advanced filtering.

    Args:
        tracks: List of MusicTrack objects

    Returns:
        List of preprocessed and filtered tracks
    """
    self.logger.info(f"Preprocessing batch of {len(tracks)} tracks...")

    processed_tracks = []
    filter_stats = {
        "total": len(tracks),
        "audio_quality_failed": 0,
        "lyrics_quality_failed": 0,
        "duplicates_removed": 0,
        "outliers_removed": 0,
        "passed": 0,
    }

    # First pass: individual track processing
    for track in tracks:
        # Preprocess individual track
        track = self.preprocess_track(track)

        # Check audio quality
        audio_passes, audio_reasons = self.passes_advanced_quality_check(track)
        if not audio_passes:
            filter_stats["audio_quality_failed"] += 1
            self.logger.debug(
                f"Track {track.track_id} failed audio quality: {audio_reasons}"
            )
            continue

        # Check lyrics quality
        lyrics_passes, lyrics_reasons = self.passes_lyrics_quality_check(track)
        if not lyrics_passes:
            filter_stats["lyrics_quality_failed"] += 1
            self.logger.debug(
                f"Track {track.track_id} failed lyrics quality: {lyrics_reasons}"
            )
            continue

        processed_tracks.append(track)

    # Second pass: duplicate detection
    if (
        self.config.filtering["enable_duplicate_detection"]
        and len(processed_tracks) > 1
    ):
        duplicates = self.detect_duplicates(processed_tracks)

        # Remove duplicates (keep first occurrence)
        duplicate_indices = set()
        for _, j, similarity in duplicates:
            duplicate_indices.add(j)
            self.logger.debug(
                f"Removing duplicate track at index {j} (similarity: {similarity:.3f})"
            )

        processed_tracks = [
            track
            for i, track in enumerate(processed_tracks)
            if i not in duplicate_indices
        ]
        filter_stats["duplicates_removed"] = len(duplicate_indices)

    # Third pass: outlier detection
    if self.config.filtering["enable_outlier_detection"] and len(processed_tracks) > 10:
        outlier_indices = self.detect_outliers(processed_tracks)

        processed_tracks = [
            track
            for i, track in enumerate(processed_tracks)
            if i not in outlier_indices
        ]
        filter_stats["outliers_removed"] = len(outlier_indices)

    filter_stats["passed"] = len(processed_tracks)

    # Log filtering statistics
    self.logger.info(f"Batch preprocessing complete:")
    self.logger.info(f"  Total tracks: {filter_stats['total']}")
    self.logger.info(f"  Audio quality failed: {filter_stats['audio_quality_failed']}")
    self.logger.info(
        f"  Lyrics quality failed: {filter_stats['lyrics_quality_failed']}"
    )
    self.logger.info(f"  Duplicates removed: {filter_stats['duplicates_removed']}")
    self.logger.info(f"  Outliers removed: {filter_stats['outliers_removed']}")
    self.logger.info(f"  Final tracks: {filter_stats['passed']}")

    return processed_tracks
