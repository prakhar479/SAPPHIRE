"""
Comprehensive feature extraction for the SAPPHIRE pipeline.
Handles acoustic, rhythm, harmonic, lyrical, and quality features.
"""

import numpy as np
import librosa
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional advanced libraries with fallbacks
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from .config import config
from .data_loader import MusicTrack

logger = logging.getLogger(__name__)

@dataclass
class FeatureContainer:
    """Container for extracted features."""
    track_id: str
    acoustic_features: Dict[str, Any]
    rhythm_features: Dict[str, Any]
    harmony_features: Dict[str, Any]
    lyrics_features: Dict[str, Any]
    quality_features: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame creation."""
        result = {}
        
        # Flatten all feature categories
        for category, features in [
            ('acoustic', self.acoustic_features),
            ('rhythm', self.rhythm_features),
            ('harmony', self.harmony_features),
            ('lyrics', self.lyrics_features),
            ('quality', self.quality_features),
            ('meta', self.metadata)
        ]:
            for key, value in features.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0 and isinstance(value[0], (int, float)):
                    # Convert arrays to statistics
                    result[f'{category}_{key}_mean'] = float(np.mean(value))
                    result[f'{category}_{key}_std'] = float(np.std(value))
                    result[f'{category}_{key}_min'] = float(np.min(value))
                    result[f'{category}_{key}_max'] = float(np.max(value))
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        result[f'{category}_{key}_{subkey}'] = subvalue
                else:
                    result[f'{category}_{key}'] = value
                    
        return result

class FeatureExtractor:
    """
    Comprehensive feature extractor for music analysis.
    Designed to handle multiple datasets and follow SAPPHIRE methodology.
    """
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize external models for feature extraction."""
        self.sentence_model = None
        self.sentiment_analyzer = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded sentence transformer model")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                
        if HAS_NLTK:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("Loaded VADER sentiment analyzer")
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment analyzer: {e}")
    
    def extract_features(self, track: MusicTrack) -> FeatureContainer:
        """
        Extract all features from a music track.
        
        Args:
            track: MusicTrack object with loaded audio and lyrics data
            
        Returns:
            FeatureContainer with all extracted features
        """
        self.logger.debug(f"Extracting features for track {track.track_id}")
        
        # Initialize feature containers
        acoustic_features = {}
        rhythm_features = {}
        harmony_features = {}
        lyrics_features = {}
        quality_features = {}
        metadata = track.metadata.copy()
        
        # Extract audio-based features if audio data is available
        if track.audio_data is not None and len(track.audio_data) > 0:
            try:
                if self.config.features.extract_acoustic:
                    acoustic_features = self._extract_acoustic_features(track.audio_data, track.sample_rate)
                    
                if self.config.features.extract_rhythm:
                    rhythm_features = self._extract_rhythm_features(track.audio_data, track.sample_rate)
                    
                if self.config.features.extract_harmony:
                    harmony_features = self._extract_harmony_features(track.audio_data, track.sample_rate)
                    
                if self.config.features.extract_quality:
                    quality_features = self._extract_quality_features(track.audio_data, track.sample_rate)
                    
            except Exception as e:
                self.logger.error(f"Error extracting audio features for {track.track_id}: {e}")
        
        # Extract lyrics-based features if lyrics are available
        if track.lyrics_text and self.config.features.extract_lyrics:
            try:
                lyrics_features = self._extract_lyrics_features(track.lyrics_text)
            except Exception as e:
                self.logger.error(f"Error extracting lyrics features for {track.track_id}: {e}")
        
        return FeatureContainer(
            track_id=track.track_id,
            acoustic_features=acoustic_features,
            rhythm_features=rhythm_features,
            harmony_features=harmony_features,
            lyrics_features=lyrics_features,
            quality_features=quality_features,
            metadata=metadata
        )
    
    def _extract_acoustic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract acoustic and timbral features."""
        features = {}
        
        try:
            # Basic spectral features
            features.update(self._extract_spectral_features(y, sr))
            
            # MFCCs (timbral features)
            features.update(self._extract_mfcc_features(y, sr))
            
            # Advanced spectral features
            if self.config.features.use_advanced_features:
                features.update(self._extract_advanced_spectral_features(y, sr))
                
        except Exception as e:
            self.logger.error(f"Error in acoustic feature extraction: {e}")
            
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract basic spectral features."""
        features = {}
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.update({
            'spectral_centroid_mean': float(np.mean(centroid)),
            'spectral_centroid_std': float(np.std(centroid)),
            'spectral_centroid_median': float(np.median(centroid))
        })
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.update({
            'spectral_bandwidth_mean': float(np.mean(bandwidth)),
            'spectral_bandwidth_std': float(np.std(bandwidth))
        })
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.update({
            'spectral_rolloff_mean': float(np.mean(rolloff))
        })
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.update({
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        })
        
        return features
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract MFCC features."""
        features = {}
        
        # Standard MFCCs
        n_mfcc = self.config.audio.n_mfcc
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        for i in range(mfccs.shape[0]):
            features.update({
                f'mfcc_{i}_mean': float(np.mean(mfccs[i])),
                f'mfcc_{i}_std': float(np.std(mfccs[i])),
                f'mfcc_{i}_skew': float(self._safe_skew(mfccs[i])),
                f'mfcc_{i}_kurtosis': float(self._safe_kurtosis(mfccs[i]))
            })
            
        return features
    
    def _extract_advanced_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract advanced spectral features."""
        features = {}
        
        try:
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            features.update({
                'spectral_flatness_mean': float(np.mean(flatness)),
                'spectral_flatness_std': float(np.std(flatness))
            })
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(contrast.shape[0]):
                features.update({
                    f'spectral_contrast_{i}_mean': float(np.mean(contrast[i])),
                    f'spectral_contrast_{i}_std': float(np.std(contrast[i]))
                })
            
            # Tonnetz (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            for i in range(tonnetz.shape[0]):
                features.update({
                    f'tonnetz_{i}_mean': float(np.mean(tonnetz[i]))
                })
                
        except Exception as e:
            self.logger.warning(f"Error in advanced spectral features: {e}")
            
        return features
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract rhythm and tempo features."""
        features = {}
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            features['tempo_bpm'] = float(tempo)
            features['beat_count'] = len(beats)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            features['onset_count'] = len(onset_frames)
            features['onset_density'] = len(onset_frames) / (len(y) / sr)
            
            # Rhythm regularity
            if len(beats) > 2:
                beat_intervals = np.diff(beats)
                features['rhythm_regularity'] = float(1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)))
                features['beat_interval_mean'] = float(np.mean(beat_intervals))
                features['beat_interval_std'] = float(np.std(beat_intervals))
            else:
                features.update({
                    'rhythm_regularity': 0.0,
                    'beat_interval_mean': 0.0,
                    'beat_interval_std': 0.0
                })
            
            # Syncopation (simplified)
            if len(onset_frames) > 0 and len(beats) > 0:
                # Count onsets that don't align with beats
                off_beat_onsets = 0
                tolerance = 0.1  # 100ms tolerance
                
                for onset in onset_frames:
                    distances = np.abs(beats - onset)
                    if len(distances) > 0 and np.min(distances) > tolerance:
                        off_beat_onsets += 1
                        
                features['syncopation_index'] = off_beat_onsets / len(onset_frames)
            else:
                features['syncopation_index'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error extracting rhythm features: {e}")
            
        return features
    
    def _extract_harmony_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract harmonic features."""
        features = {}
        
        try:
            # Chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            for i in range(len(chroma_mean)):
                features.update({
                    f'chroma_{i}_mean': float(chroma_mean[i]),
                    f'chroma_{i}_std': float(np.std(chroma[i]))
                })
            
            # Key estimation using chroma profile matching
            key_profiles = self._get_key_profiles()
            key_estimation = self._estimate_key(chroma_mean, key_profiles)
            features.update(key_estimation)
            
            # Harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                features['harmonic_ratio'] = float(harmonic_energy / total_energy)
                features['percussive_ratio'] = float(percussive_energy / total_energy)
            else:
                features['harmonic_ratio'] = 0.0
                features['percussive_ratio'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error extracting harmony features: {e}")
            
        return features
    
    def _extract_quality_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio quality metrics."""
        features = {}
        
        try:
            # Signal-to-noise ratio (simplified estimation)
            sorted_y = np.sort(np.abs(y))
            noise_level = np.mean(sorted_y[:int(len(sorted_y) * 0.1)])
            signal_level = np.mean(sorted_y[int(len(sorted_y) * 0.9):])
            
            if noise_level > 0:
                features['snr_db'] = float(20 * np.log10(signal_level / noise_level))
            else:
                features['snr_db'] = float('inf')
            
            # Dynamic range
            if np.mean(np.abs(y)) > 0:
                features['dynamic_range_db'] = float(20 * np.log10(np.max(np.abs(y)) / np.mean(np.abs(y))))
            else:
                features['dynamic_range_db'] = 0.0
            
            # RMS energy
            features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
            
            # Clipping detection
            features['clipping_percentage'] = float(np.mean(np.abs(y) >= 0.999) * 100)
            
            # Loudness (if pyloudnorm available)
            if HAS_PYLOUDNORM:
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(y)
                    features['integrated_loudness_lufs'] = float(loudness)
                except:
                    features['integrated_loudness_lufs'] = None
            
        except Exception as e:
            self.logger.error(f"Error extracting quality features: {e}")
            
        return features
    
    def _extract_lyrics_features(self, lyrics_text: str) -> Dict[str, Any]:
        """Extract lyrical features."""
        features = {}
        
        try:
            # Basic text statistics
            words = lyrics_text.split()
            features['word_count'] = len(words)
            features['char_count'] = len(lyrics_text)
            features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
            
            # Vocabulary richness (Type-Token Ratio)
            unique_words = set(word.lower() for word in words)
            features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
            
            # Readability metrics (if textstat available)
            if HAS_NLTK:
                try:
                    features['flesch_reading_ease'] = flesch_reading_ease(lyrics_text)
                    features['flesch_kincaid_grade'] = flesch_kincaid_grade(lyrics_text)
                except:
                    features['flesch_reading_ease'] = None
                    features['flesch_kincaid_grade'] = None
            
            # Sentiment analysis (if VADER available)
            if self.sentiment_analyzer:
                try:
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(lyrics_text)
                    features.update({
                        'sentiment_compound': sentiment_scores['compound'],
                        'sentiment_positive': sentiment_scores['pos'],
                        'sentiment_negative': sentiment_scores['neg'],
                        'sentiment_neutral': sentiment_scores['neu']
                    })
                except:
                    pass
            
            # Semantic embedding (if sentence transformer available)
            if self.sentence_model:
                try:
                    embedding = self.sentence_model.encode([lyrics_text])[0]
                    # Use PCA to reduce dimensionality for key components
                    features['semantic_embedding_norm'] = float(np.linalg.norm(embedding))
                    features['semantic_embedding_mean'] = float(np.mean(embedding))
                    features['semantic_embedding_std'] = float(np.std(embedding))
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error extracting lyrics features: {e}")
            
        return features
    
    def _get_key_profiles(self) -> Dict[str, np.ndarray]:
        """Get key profiles for key estimation."""
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        profiles = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, note in enumerate(note_names):
            profiles[f'{note}_major'] = np.roll(major_profile, i)
            profiles[f'{note}_minor'] = np.roll(minor_profile, i)
            
        return profiles
    
    def _estimate_key(self, chroma_mean: np.ndarray, key_profiles: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Estimate musical key from chroma features."""
        correlations = {}
        
        for key_name, profile in key_profiles.items():
            correlation = np.corrcoef(chroma_mean, profile)[0, 1]
            if not np.isnan(correlation):
                correlations[key_name] = correlation
        
        if correlations:
            best_key = max(correlations.items(), key=lambda x: x[1])
            return {
                'estimated_key': best_key[0],
                'key_confidence': float(best_key[1])
            }
        else:
            return {
                'estimated_key': 'unknown',
                'key_confidence': 0.0
            }
    
    def _safe_skew(self, x: np.ndarray) -> float:
        """Safely compute skewness."""
        try:
            from scipy.stats import skew
            return float(skew(x))
        except:
            return 0.0
    
    def _safe_kurtosis(self, x: np.ndarray) -> float:
        """Safely compute kurtosis."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(x))
        except:
            return 0.0
    
    def extract_batch(self, tracks: List[MusicTrack], n_workers: int = None) -> List[FeatureContainer]:
        """Extract features for multiple tracks in parallel."""
        n_workers = n_workers or self.config.processing.workers
        
        if n_workers == 1:
            # Sequential processing
            return [self.extract_features(track) for track in tqdm(tracks, desc="Extracting features")]
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self.extract_features, track) for track in tracks]
                results = []
                
                for future in tqdm(futures, desc="Extracting features"):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Error in batch processing: {e}")
                        
                return results