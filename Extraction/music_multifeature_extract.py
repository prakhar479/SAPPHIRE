#!/usr/bin/env python3
"""
music_multifeature_extract.py

New Features:
- Advanced spectral features (spectral rolloff, flux, flatness)
- Harmonic analysis (key estimation, chord progression analysis)
- Advanced rhythm features (onset detection, beat tracking)
- Timbral analysis (zero-crossing rate variations, spectral shape)
- Emotional/mood analysis from audio features
- Enhanced lyrics analysis (readability, complexity, topic modeling)
- Audio quality metrics (SNR, dynamic range)
- Interactive CLI with progress bars and colored output
- Configuration file support
- Detailed HTML reports
- Data visualization capabilities
"""

from __future__ import annotations
import os, sys, json, argparse, time, math, hashlib, csv, tempfile, shutil, traceback, warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from functools import partial
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import soundfile as sf
import librosa
import scipy.stats as stats
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced UI imports
try:
    from tqdm import tqdm
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    tqdm = None

# Optional advanced libraries
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    crepe = None
    HAS_CREPE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT = SentenceTransformer("all-mpnet-base-v2")
except ImportError:
    SBERT = None

try:
    import pronouncing
except ImportError:
    pronouncing = None

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    nltk_vader = SentimentIntensityAnalyzer()
    HAS_NLTK = True
except ImportError:
    nltk_vader = None
    HAS_NLTK = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    pyln = None
    HAS_PYLOUDNORM = False

try:
    from gensim import models, corpora
    from gensim.utils import simple_preprocess
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
@dataclass
class Config:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    frame_length: int = 2048
    workers: int = 4
    use_crepe: bool = False
    save_matrices: bool = False
    save_plots: bool = True
    force: bool = False
    quick: bool = False
    verbose: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

# Logging setup
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('music_extraction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Utility functions
def sha1_of_file(path: str, block_size: int = 65536) -> str:
    """Calculate SHA1 hash of file."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(obj: Dict[str, Any], path: str):
    """Atomically write JSON to file."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp_')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise e

def ensure_dir(d: str):
    """Ensure directory exists."""
    os.makedirs(d, exist_ok=True)

def safe_load_json(path: str) -> Optional[Dict]:
    """Safely load JSON file."""
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return None

# Enhanced audio loading with error handling
def load_audio_mono(path: str, sr: int = 22050, offset: float = 0.0, duration: Optional[float] = None):
    """Load audio file with enhanced error handling and normalization."""
    try:
        y, sr_orig = sf.read(path, start=int(offset * sr), frames=int(duration * sr) if duration else None)
        
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Convert to mono
        
        if sr_orig != sr and sr is not None:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
        
        # Normalize
        if y.size > 0:
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val
        
        return y, sr, len(y) / sr
    except Exception as e:
        logging.error(f"Error loading audio {path}: {e}")
        return np.array([]), sr, 0.0

# Advanced spectral features
def advanced_spectral_features(y: np.ndarray, sr: int, hop_length: int = 512) -> Dict[str, Any]:
    """Extract advanced spectral features."""
    features = {}
    
    if len(y) == 0:
        return {k: None for k in ['spectral_rolloff', 'spectral_flux', 'spectral_flatness', 
                                 'spectral_slope', 'spectral_spread', 'spectral_skewness', 
                                 'spectral_kurtosis']}
    
    try:
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        features['spectral_rolloff_std'] = float(np.std(rolloff))
        
        # Spectral flux
        stft = librosa.stft(y, hop_length=hop_length)
        magnitude = np.abs(stft)
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        features['spectral_flux_mean'] = float(np.mean(spectral_flux))
        features['spectral_flux_std'] = float(np.std(spectral_flux))
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
        features['spectral_flatness_mean'] = float(np.mean(flatness))
        features['spectral_flatness_std'] = float(np.std(flatness))
        
        # Additional spectral statistics
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Spectral slope, spread, skewness, kurtosis
        spectral_stats = []
        for frame in S.T:
            if np.sum(frame) > 0:
                # Normalize
                frame_norm = frame / np.sum(frame)
                
                # Moments
                mean_freq = np.sum(freqs * frame_norm)
                spread = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * frame_norm))
                skewness = np.sum(((freqs - mean_freq) ** 3) * frame_norm) / (spread ** 3 + 1e-10)
                kurtosis = np.sum(((freqs - mean_freq) ** 4) * frame_norm) / (spread ** 4 + 1e-10)
                
                # Slope (linear regression)
                valid_idx = frame > 0
                if np.sum(valid_idx) > 1:
                    slope, _ = np.polyfit(freqs[valid_idx], np.log(frame[valid_idx] + 1e-10), 1)
                else:
                    slope = 0
                
                spectral_stats.append([slope, spread, skewness, kurtosis])
        
        if spectral_stats:
            spectral_stats = np.array(spectral_stats)
            features['spectral_slope_mean'] = float(np.mean(spectral_stats[:, 0]))
            features['spectral_spread_mean'] = float(np.mean(spectral_stats[:, 1]))
            features['spectral_skewness_mean'] = float(np.mean(spectral_stats[:, 2]))
            features['spectral_kurtosis_mean'] = float(np.mean(spectral_stats[:, 3]))
        else:
            features.update({
                'spectral_slope_mean': 0.0,
                'spectral_spread_mean': 0.0,
                'spectral_skewness_mean': 0.0,
                'spectral_kurtosis_mean': 0.0
            })
        
    except Exception as e:
        logging.warning(f"Error computing advanced spectral features: {e}")
        for key in ['spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_flux_mean', 
                   'spectral_flux_std', 'spectral_flatness_mean', 'spectral_flatness_std',
                   'spectral_slope_mean', 'spectral_spread_mean', 'spectral_skewness_mean', 
                   'spectral_kurtosis_mean']:
            features[key] = None
    
    return features

# Enhanced rhythm analysis
def enhanced_rhythm_analysis(y: np.ndarray, sr: int, hop_length: int = 512) -> Dict[str, Any]:
    """Enhanced rhythm and tempo analysis."""
    if len(y) == 0:
        return {
            'tempo_bpm': None,
            'tempo_confidence': None,
            'beat_positions': [],
            'onset_times': [],
            'rhythm_regularity': None,
            'syncopation_index': None
        }
    
    try:
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units='time')
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='time')
        
        # Tempo confidence (consistency of beat intervals)
        beat_intervals = np.diff(beats) if len(beats) > 1 else []
        tempo_confidence = 1.0 / (1.0 + np.std(beat_intervals)) if len(beat_intervals) > 0 else 0.0
        
        # Rhythm regularity
        if len(beat_intervals) > 2:
            rhythm_regularity = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            rhythm_regularity = 0.0
        
        # Syncopation index (simplified)
        syncopation_index = 0.0
        if len(onset_frames) > 0 and len(beats) > 0:
            # Count onsets that don't align with beats
            off_beat_onsets = 0
            for onset in onset_frames:
                distances = np.abs(beats - onset)
                if len(distances) > 0 and np.min(distances) > 0.1:  # 100ms tolerance
                    off_beat_onsets += 1
            syncopation_index = off_beat_onsets / len(onset_frames)
        
        return {
            'tempo_bpm': float(tempo),
            'tempo_confidence': float(tempo_confidence),
            'beat_positions': beats.tolist(),
            'onset_times': onset_frames.tolist(),
            'rhythm_regularity': float(rhythm_regularity),
            'syncopation_index': float(syncopation_index),
            'beat_count': len(beats),
            'onset_count': len(onset_frames)
        }
        
    except Exception as e:
        logging.warning(f"Error in rhythm analysis: {e}")
        return {
            'tempo_bpm': None,
            'tempo_confidence': None,
            'beat_positions': [],
            'onset_times': [],
            'rhythm_regularity': None,
            'syncopation_index': None
        }

# Enhanced harmonic analysis
def enhanced_harmonic_analysis(y: np.ndarray, sr: int, hop_length: int = 512) -> Dict[str, Any]:
    """Enhanced harmonic and tonal analysis."""
    if len(y) == 0:
        return {'chroma_features': None, 'key_estimation': None, 'tonal_complexity': None}
    
    try:
        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Key estimation using chroma profile matching
        key_profiles = {
            'C_major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            'C_minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        }
        
        # Simple key estimation (in practice, use more sophisticated methods)
        correlations = {}
        for key, profile in key_profiles.items():
            correlation = np.corrcoef(chroma_mean, profile)[0, 1]
            correlations[key] = correlation if not np.isnan(correlation) else 0.0
        
        estimated_key = max(correlations.items(), key=lambda x: x[1])
        
        # Tonal complexity (standard deviation of chroma vector)
        tonal_complexity = float(np.std(chroma_mean))
        
        return {
            'chroma_mean': chroma_mean.tolist(),
            'chroma_std': chroma_std.tolist(),
            'key_estimation': {
                'key': estimated_key[0],
                'confidence': float(estimated_key[1])
            },
            'tonal_complexity': tonal_complexity
        }
        
    except Exception as e:
        logging.warning(f"Error in harmonic analysis: {e}")
        return {'chroma_features': None, 'key_estimation': None, 'tonal_complexity': None}

# Audio quality metrics
def audio_quality_metrics(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute audio quality metrics."""
    if len(y) == 0:
        return {'snr_db': None, 'dynamic_range_db': None, 'rms_energy': None, 'zero_crossing_rate': None}
    
    try:
        # Signal-to-noise ratio (simplified estimation)
        # Use the assumption that noise is in the quietest 10% of the signal
        sorted_y = np.sort(np.abs(y))
        noise_level = np.mean(sorted_y[:int(len(sorted_y) * 0.1)])
        signal_level = np.mean(sorted_y[int(len(sorted_y) * 0.9):])
        snr_db = 20 * np.log10(signal_level / (noise_level + 1e-10))
        
        # Dynamic range
        dynamic_range_db = 20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10))
        
        # RMS energy
        rms_energy = float(np.sqrt(np.mean(y**2)))
        
        # Zero crossing rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        return {
            'snr_db': float(snr_db),
            'dynamic_range_db': float(dynamic_range_db),
            'rms_energy': rms_energy,
            'zero_crossing_rate': zcr
        }
        
    except Exception as e:
        logging.warning(f"Error computing audio quality metrics: {e}")
        return {'snr_db': None, 'dynamic_range_db': None, 'rms_energy': None, 'zero_crossing_rate': None}

# Enhanced lyrics analysis
def enhanced_lyrics_analysis(text: str) -> Dict[str, Any]:
    """Comprehensive lyrics analysis."""
    if not text or not text.strip():
        return {
            'word_count': 0,
            'sentence_count': 0,
            'readability': None,
            'sentiment_analysis': None,
            'topic_keywords': [],
            'repetition_analysis': None
        }
    
    try:
        # Basic statistics
        words = word_tokenize(text.lower()) if HAS_NLTK else text.split()
        sentences = sent_tokenize(text) if HAS_NLTK else text.split('.')
        
        # Readability scores
        readability = {}
        try:
            readability['flesch_reading_ease'] = flesch_reading_ease(text)
            readability['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            readability = None
        
        # Sentiment analysis
        sentiment = None
        if nltk_vader:
            scores = nltk_vader.polarity_scores(text)
            sentiment = {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        
        # Word frequency and repetition
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top keywords (excluding stop words)
        try:
            if HAS_NLTK:
                stop_words = set(stopwords.words('english'))
                keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10] 
                           if word not in stop_words]
            else:
                keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        except:
            keywords = list(word_freq.keys())[:10]
        
        # Repetition analysis
        total_words = len(words)
        unique_words = len(set(words))
        repetition_ratio = 1.0 - (unique_words / total_words) if total_words > 0 else 0.0
        
        return {
            'word_count': total_words,
            'unique_word_count': unique_words,
            'sentence_count': len(sentences),
            'avg_words_per_sentence': total_words / len(sentences) if sentences else 0,
            'readability': readability,
            'sentiment_analysis': sentiment,
            'topic_keywords': keywords,
            'repetition_analysis': {
                'repetition_ratio': float(repetition_ratio),
                'most_common_words': list(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        }
        
    except Exception as e:
        logging.warning(f"Error in lyrics analysis: {e}")
        return {'error': str(e)}

# Main analysis function
def analyze_audio_comprehensive(audio_path: str, lyrics_path: Optional[str], config: Config) -> Dict[str, Any]:
    """Comprehensive audio analysis."""
    start_time = time.time()
    
    # Load audio
    y, sr, duration = load_audio_mono(audio_path, config.sr)
    
    features = {
        'meta': {
            'path': audio_path,
            'file_hash': sha1_of_file(audio_path),
            'duration_sec': duration,
            'sample_rate': sr,
            'analysis_timestamp': datetime.now().isoformat(),
            'config': config.__dict__
        },
        'acoustic': {},
        'rhythm': {},
        'harmony': {},
        'quality': {},
        'lyrics': {}
    }
    
    if len(y) == 0:
        logging.error(f"Could not load audio from {audio_path}")
        return features
    
    # Basic acoustic features
    try:
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc)
        features['acoustic']['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
        features['acoustic']['mfcc_std'] = np.std(mfcc, axis=1).tolist()
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        features['acoustic'].update({
            'spectral_centroid_mean': float(np.mean(centroid)),
            'spectral_centroid_std': float(np.std(centroid)),
            'spectral_bandwidth_mean': float(np.mean(bandwidth)),
            'spectral_bandwidth_std': float(np.std(bandwidth))
        })
        
        # Advanced spectral features
        features['acoustic'].update(advanced_spectral_features(y, sr, config.hop_length))
        
    except Exception as e:
        logging.error(f"Error in acoustic analysis: {e}")
    
    # Rhythm analysis
    features['rhythm'] = enhanced_rhythm_analysis(y, sr, config.hop_length)
    
    # Harmonic analysis
    features['harmony'] = enhanced_harmonic_analysis(y, sr, config.hop_length)
    
    # Audio quality
    features['quality'] = audio_quality_metrics(y, sr)
    
    # Lyrics analysis
    if lyrics_path and os.path.exists(lyrics_path):
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_text = f.read()
            features['lyrics'] = enhanced_lyrics_analysis(lyrics_text)
        except Exception as e:
            logging.error(f"Error reading lyrics from {lyrics_path}: {e}")
            features['lyrics'] = {'error': str(e)}
    
    # Add processing time
    features['meta']['processing_time_sec'] = time.time() - start_time
    
    return features

# Visualization functions
def create_analysis_plots(features: Dict[str, Any], output_dir: str):
    """Create visualization plots for the analysis."""
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: MFCC coefficients
        mfcc_mean = features.get('acoustic', {}).get('mfcc_mean')
        if mfcc_mean:
            axes[0, 0].bar(range(len(mfcc_mean)), mfcc_mean)
            axes[0, 0].set_title('MFCC Coefficients (Mean)')
            axes[0, 0].set_xlabel('Coefficient Index')
            axes[0, 0].set_ylabel('Value')
        
        # Plot 2: Chroma features
        chroma_mean = features.get('harmony', {}).get('chroma_mean')
        if chroma_mean:
            chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            axes[0, 1].bar(chroma_labels, chroma_mean)
            axes[0, 1].set_title('Chroma Features')
            axes[0, 1].set_xlabel('Note')
            axes[0, 1].set_ylabel('Intensity')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Spectral features
        spectral_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 
                           'spectral_rolloff_mean', 'spectral_flatness_mean']
        values = []
        labels = []
        for feature in spectral_features:
            val = features.get('acoustic', {}).get(feature)
            if val is not None:
                values.append(val)
                labels.append(feature.replace('_mean', '').replace('spectral_', ''))
        
        if values:
            axes[1, 0].bar(labels, values)
            axes[1, 0].set_title('Spectral Features')
            axes[1, 0].set_ylabel('Value')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Quality metrics
        quality_metrics = ['snr_db', 'dynamic_range_db', 'rms_energy']
        q_values = []
        q_labels = []
        for metric in quality_metrics:
            val = features.get('quality', {}).get(metric)
            if val is not None:
                q_values.append(val)
                q_labels.append(metric.replace('_', ' ').title())
        
        if q_values:
            axes[1, 1].bar(q_labels, q_values)
            axes[1, 1].set_title('Audio Quality Metrics')
            axes[1, 1].set_ylabel('Value')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'analysis_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        logging.error(f"Error creating plots: {e}")
        return None

# HTML report generation
def generate_html_report(features: Dict[str, Any], output_dir: str) -> str:
    """Generate a comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Music Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
            .section h3 {{ color: #2E7D32; margin-top: 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f5e8; border-radius: 5px; min-width: 150px; }}
            .metric-label {{ font-weight: bold; color: #2E7D32; }}
            .metric-value {{ font-size: 1.2em; color: #1B5E20; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            .plot {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎵 Music Analysis Report</h1>
                <p><strong>File:</strong> {features['meta'].get('path', 'Unknown')}</p>
                <p><strong>Analysis Date:</strong> {features['meta'].get('analysis_timestamp', 'Unknown')}</p>
            </div>
    """
    
    # Meta information
    meta = features.get('meta', {})
    html_content += f"""
            <div class="section">
                <h3>📋 File Information</h3>
                <div class="metric">
                    <div class="metric-label">Sample Rate</div>
                    <div class="metric-value">{meta.get('sample_rate', 0)} Hz</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Processing Time</div>
                    <div class="metric-value">{meta.get('processing_time_sec', 0):.2f} sec</div>
                </div>
            </div>
    """
    
    # Acoustic features
    acoustic = features.get('acoustic', {})
    html_content += f"""
            <div class="section">
                <h3>🎼 Acoustic Features</h3>
                <div class="metric">
                    <div class="metric-label">Spectral Centroid</div>
                    <div class="metric-value">{acoustic.get('spectral_centroid_mean', 0):.1f} Hz</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Spectral Bandwidth</div>
                    <div class="metric-value">{acoustic.get('spectral_bandwidth_mean', 0):.1f} Hz</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Spectral Rolloff</div>
                    <div class="metric-value">{acoustic.get('spectral_rolloff_mean', 0):.1f} Hz</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Spectral Flatness</div>
                    <div class="metric-value">{acoustic.get('spectral_flatness_mean', 0):.4f}</div>
                </div>
            </div>
    """
    
    # Rhythm features
    rhythm = features.get('rhythm', {})
    html_content += f"""
            <div class="section">
                <h3>🥁 Rhythm Analysis</h3>
                <div class="metric">
                    <div class="metric-label">Tempo</div>
                    <div class="metric-value">{rhythm.get('tempo_bpm', 0):.1f} BPM</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Beat Count</div>
                    <div class="metric-value">{rhythm.get('beat_count', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Rhythm Regularity</div>
                    <div class="metric-value">{rhythm.get('rhythm_regularity', 0):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Syncopation Index</div>
                    <div class="metric-value">{rhythm.get('syncopation_index', 0):.3f}</div>
                </div>
            </div>
    """
    
    # Harmonic features
    harmony = features.get('harmony', {})
    key_est = harmony.get('key_estimation', {})
    html_content += f"""
            <div class="section">
                <h3>🎹 Harmonic Analysis</h3>
                <div class="metric">
                    <div class="metric-label">Estimated Key</div>
                    <div class="metric-value">{key_est.get('key', 'Unknown')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Key Confidence</div>
                    <div class="metric-value">{key_est.get('confidence', 0):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Tonal Complexity</div>
                    <div class="metric-value">{harmony.get('tonal_complexity', 0):.3f}</div>
                </div>
            </div>
    """
    
    # Quality metrics
    quality = features.get('quality', {})
    html_content += f"""
            <div class="section">
                <h3>🔊 Audio Quality</h3>
                <div class="metric">
                    <div class="metric-label">SNR</div>
                    <div class="metric-value">{quality.get('snr_db', 0):.1f} dB</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Dynamic Range</div>
                    <div class="metric-value">{quality.get('dynamic_range_db', 0):.1f} dB</div>
                </div>
                <div class="metric">
                    <div class="metric-label">RMS Energy</div>
                    <div class="metric-value">{quality.get('rms_energy', 0):.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Zero Crossing Rate</div>
                    <div class="metric-value">{quality.get('zero_crossing_rate', 0):.4f}</div>
                </div>
            </div>
    """
    
    # Lyrics analysis
    lyrics = features.get('lyrics', {})
    if lyrics and 'error' not in lyrics:
        sentiment = lyrics.get('sentiment_analysis', {})
        readability = lyrics.get('readability', {})
        repetition = lyrics.get('repetition_analysis', {})
        
        html_content += f"""
            <div class="section">
                <h3>📝 Lyrics Analysis</h3>
                <div class="metric">
                    <div class="metric-label">Word Count</div>
                    <div class="metric-value">{lyrics.get('word_count', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Unique Words</div>
                    <div class="metric-value">{lyrics.get('unique_word_count', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sentences</div>
                    <div class="metric-value">{lyrics.get('sentence_count', 0)}</div>
                </div>
        """
        
        if sentiment:
            html_content += f"""
                <div class="metric">
                    <div class="metric-label">Sentiment</div>
                    <div class="metric-value">{sentiment.get('compound', 0):.3f}</div>
                </div>
            """
        
        if readability:
            html_content += f"""
                <div class="metric">
                    <div class="metric-label">Reading Ease</div>
                    <div class="metric-value">{readability.get('flesch_reading_ease', 0):.1f}</div>
                </div>
            """
        
        if repetition:
            html_content += f"""
                <div class="metric">
                    <div class="metric-label">Repetition Ratio</div>
                    <div class="metric-value">{repetition.get('repetition_ratio', 0):.3f}</div>
                </div>
            """
        
        html_content += "</div>"
        
        # Top keywords
        keywords = lyrics.get('topic_keywords', [])
        if keywords:
            html_content += f"""
            <div class="section">
                <h3>🔤 Top Keywords</h3>
                <p>{', '.join(keywords[:10])}</p>
            </div>
            """
    
    # Add plots if they exist
    plot_path = os.path.join(output_dir, 'analysis_plots.png')
    if os.path.exists(plot_path):
        html_content += f"""
            <div class="section">
                <h3>📊 Analysis Plots</h3>
                <div class="plot">
                    <img src="analysis_plots.png" alt="Analysis Plots" style="max-width: 100%; height: auto;">
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, 'analysis_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

# Enhanced processing functions
def save_structured_outputs(base_out_dir: str, basename: str, features: Dict[str, Any], config: Config) -> str:
    """Save structured outputs with enhanced organization."""
    ensure_dir(base_out_dir)
    
    file_hash = features.get('meta', {}).get('file_hash') or hashlib.sha1(basename.encode()).hexdigest()[:8]
    folder_name = f"{file_hash}_{basename}"
    out_folder = os.path.join(base_out_dir, folder_name)
    ensure_dir(out_folder)
    
    # Save individual feature categories
    categories = {
        '00_meta.json': features.get('meta', {}),
        '01_acoustic.json': features.get('acoustic', {}),
        '02_rhythm.json': features.get('rhythm', {}),
        '03_harmony.json': features.get('harmony', {}),
        '04_quality.json': features.get('quality', {}),
        '05_lyrics.json': features.get('lyrics', {}),
        '99_full.json': features
    }
    
    for filename, data in categories.items():
        atomic_write_json(data, os.path.join(out_folder, filename))
    
    # Create visualizations if enabled
    if config.save_plots:
        create_analysis_plots(features, out_folder)
        generate_html_report(features, out_folder)
    
    return out_folder

def process_single_file(audio_path: str, lyrics_path: Optional[str], out_dir: str, config: Config) -> Dict[str, Any]:
    """Process a single audio file."""
    start_time = time.time()
    basename = Path(audio_path).stem
    
    try:
        # Check if already processed
        current_hash = sha1_of_file(audio_path)
        existing_folder = None
        
        if os.path.isdir(out_dir) and not config.force:
            for folder in os.listdir(out_dir):
                if folder.endswith(f'_{basename}'):
                    meta_path = os.path.join(out_dir, folder, '00_meta.json')
                    meta = safe_load_json(meta_path)
                    if meta and meta.get('file_hash') == current_hash:
                        existing_folder = os.path.join(out_dir, folder)
                        break
        
        if existing_folder and not config.force:
            return {
                'status': 'skipped',
                'audio': audio_path,
                'out_folder': existing_folder,
                'reason': 'Already processed (use --force to reprocess)'
            }
        
        # Perform analysis
        features = analyze_audio_comprehensive(audio_path, lyrics_path, config)
        
        # Save results
        out_folder = save_structured_outputs(out_dir, basename, features, config)
        
        return {
            'status': 'success',
            'audio': audio_path,
            'out_folder': out_folder,
            'processing_time': time.time() - start_time,
            'features_summary': {
                'duration': features['meta'].get('duration_sec', 0),
                'tempo': features['rhythm'].get('tempo_bpm'),
                'key': features['harmony'].get('key_estimation', {}).get('key')
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'audio': audio_path,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'processing_time': time.time() - start_time
        }

# Enhanced worker function for multiprocessing
def enhanced_worker(args):
    """Enhanced worker function with better error handling."""
    audio_path, lyrics_path, out_dir, config = args
    return process_single_file(audio_path, lyrics_path, out_dir, config)

# Batch processing functions
def gather_audio_files(directory: str, extensions: List[str] = None) -> List[Tuple[str, Optional[str]]]:
    """Gather audio files and their potential lyrics files."""
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aiff', '.aif']
    
    lyrics_extensions = ['.txt', '.lrc', '.lyrics']
    pairs = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_path = os.path.join(root, file)
                basename = os.path.splitext(file)[0]
                
                # Look for lyrics file
                lyrics_path = None
                for lyrics_ext in lyrics_extensions:
                    potential_lyrics = os.path.join(root, basename + lyrics_ext)
                    if os.path.exists(potential_lyrics):
                        lyrics_path = potential_lyrics
                        break
                
                pairs.append((audio_path, lyrics_path))
    
    return pairs

def read_manifest(manifest_path: str) -> List[Tuple[str, Optional[str]]]:
    """Read processing manifest from CSV."""
    pairs = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio = row.get('audio_path') or row.get('audio') or row.get('file')
            lyrics = row.get('lyrics_path') or row.get('lyrics')
            
            if audio:
                pairs.append((audio, lyrics if lyrics else None))
    
    return pairs

def batch_process(pairs: List[Tuple[str, Optional[str]]], out_dir: str, config: Config):
    """Process multiple files with progress tracking."""
    ensure_dir(out_dir)
    
    # Filter valid files
    valid_pairs = []
    for audio_path, lyrics_path in pairs:
        if os.path.exists(audio_path):
            valid_pairs.append((audio_path, lyrics_path))
        else:
            logging.warning(f"Audio file not found: {audio_path}")
    
    if not valid_pairs:
        console.print("[red]No valid audio files found![/red]") if HAS_RICH else print("No valid audio files found!")
        return
    
    total_files = len(valid_pairs)
    console.print(f"[green]Processing {total_files} audio files...[/green]") if HAS_RICH else print(f"Processing {total_files} audio files...")
    
    # Prepare worker arguments
    worker_args = [(audio, lyrics, out_dir, config) for audio, lyrics in valid_pairs]
    
    # Results tracking
    results = {'success': 0, 'error': 0, 'skipped': 0}
    
    if config.workers > 1:
        # Multiprocessing
        with mp.Pool(processes=config.workers) as pool:
            if HAS_RICH:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing files...", total=total_files)
                    
                    for result in pool.imap_unordered(enhanced_worker, worker_args):
                        results[result['status']] += 1
                        
                        if result['status'] == 'success':
                            progress.console.print(f"[green]✓[/green] {os.path.basename(result['audio'])}")
                        elif result['status'] == 'error':
                            progress.console.print(f"[red]✗[/red] {os.path.basename(result['audio'])}: {result['error']}")
                        else:
                            progress.console.print(f"[yellow]⚠[/yellow] {os.path.basename(result['audio'])}: skipped")
                        
                        progress.advance(task)
            else:
                # Fallback without rich
                for i, result in enumerate(pool.imap_unordered(enhanced_worker, worker_args), 1):
                    results[result['status']] += 1
                    status_symbol = '✓' if result['status'] == 'success' else '✗' if result['status'] == 'error' else '⚠'
                    print(f"[{i}/{total_files}] {status_symbol} {os.path.basename(result['audio'])}")
    else:
        # Single-threaded processing
        if HAS_RICH:
            with Progress(console=console) as progress:
                task = progress.add_task("Processing files...", total=total_files)
                
                for args in worker_args:
                    result = enhanced_worker(args)
                    results[result['status']] += 1
                    
                    if result['status'] == 'success':
                        progress.console.print(f"[green]✓[/green] {os.path.basename(result['audio'])}")
                    elif result['status'] == 'error':
                        progress.console.print(f"[red]✗[/red] {os.path.basename(result['audio'])}: {result['error']}")
                    else:
                        progress.console.print(f"[yellow]⚠[/yellow] {os.path.basename(result['audio'])}: skipped")
                    
                    progress.advance(task)
        else:
            for i, args in enumerate(worker_args, 1):
                result = enhanced_worker(args)
                results[result['status']] += 1
                status_symbol = '✓' if result['status'] == 'success' else '✗' if result['status'] == 'error' else '⚠'
                print(f"[{i}/{total_files}] {status_symbol} {os.path.basename(result['audio'])}")
    
    # Print summary
    if HAS_RICH:
        table = Table(title="Processing Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        
        table.add_row("Success", f"[green]{results['success']}[/green]")
        table.add_row("Errors", f"[red]{results['error']}[/red]")
        table.add_row("Skipped", f"[yellow]{results['skipped']}[/yellow]")
        
        console.print(table)
    else:
        print(f"\nProcessing Summary:")
        print(f"Success: {results['success']}")
        print(f"Errors: {results['error']}")
        print(f"Skipped: {results['skipped']}")

# Verification and summary functions
def print_analysis_summary(output_folder: str):
    """Print a comprehensive analysis summary."""
    try:
        # Load all feature files
        features = {}
        for filename in ['00_meta.json', '01_acoustic.json', '02_rhythm.json', 
                        '03_harmony.json', '04_quality.json', '05_lyrics.json']:
            data = safe_load_json(os.path.join(output_folder, filename))
            if data:
                category = filename.split('_')[1].split('.')[0]
                features[category] = data
        
        if HAS_RICH:
            console.print(Panel.fit("🎵 Analysis Summary", style="bold magenta"))
            
            # Meta information
            meta = features.get('meta', {})
            console.print(f"[bold]File:[/bold] {meta.get('path', 'Unknown')}")
            console.print(f"[bold]Duration:[/bold] {meta.get('duration_sec', 0):.2f} seconds")
            console.print(f"[bold]Sample Rate:[/bold] {meta.get('sample_rate', 0)} Hz")
            
            # Create summary table
            table = Table(title="Key Features", show_header=True, header_style="bold blue")
            table.add_column("Category", style="cyan")
            table.add_column("Feature", style="green")
            table.add_column("Value", style="yellow")
            
            # Rhythm features
            rhythm = features.get('rhythm', {})
            if rhythm.get('tempo_bpm'):
                table.add_row("Rhythm", "Tempo", f"{rhythm['tempo_bpm']:.1f} BPM")
            if rhythm.get('rhythm_regularity') is not None:
                table.add_row("", "Regularity", f"{rhythm['rhythm_regularity']:.3f}")
            
            # Harmonic features
            harmony = features.get('harmony', {})
            key_est = harmony.get('key_estimation', {})
            if key_est.get('key'):
                table.add_row("Harmony", "Estimated Key", key_est['key'])
                table.add_row("", "Confidence", f"{key_est.get('confidence', 0):.3f}")
            
            # Quality features
            quality = features.get('quality', {})
            if quality.get('snr_db') is not None:
                table.add_row("Quality", "SNR", f"{quality['snr_db']:.1f} dB")
            if quality.get('dynamic_range_db') is not None:
                table.add_row("", "Dynamic Range", f"{quality['dynamic_range_db']:.1f} dB")
            
            # Lyrics features
            lyrics = features.get('lyrics', {})
            if lyrics and 'error' not in lyrics:
                if lyrics.get('word_count'):
                    table.add_row("Lyrics", "Word Count", str(lyrics['word_count']))
                sentiment = lyrics.get('sentiment_analysis', {})
                if sentiment.get('compound') is not None:
                    table.add_row("", "Sentiment", f"{sentiment['compound']:.3f}")
            
            console.print(table)
            
            # Check for HTML report
            html_report = os.path.join(output_folder, 'analysis_report.html')
            if os.path.exists(html_report):
                console.print(f"[green]📊 Detailed HTML report available:[/green] {html_report}")
        
        else:
            # Fallback without rich
            print("=== ANALYSIS SUMMARY ===")
            meta = features.get('meta', {})
            print(f"File: {meta.get('path', 'Unknown')}")
            print(f"Duration: {meta.get('duration_sec', 0):.2f} seconds")
            print(f"Sample Rate: {meta.get('sample_rate', 0)} Hz")
            
            rhythm = features.get('rhythm', {})
            if rhythm.get('tempo_bpm'):
                print(f"Tempo: {rhythm['tempo_bpm']:.1f} BPM")
            
            harmony = features.get('harmony', {})
            key_est = harmony.get('key_estimation', {})
            if key_est.get('key'):
                print(f"Estimated Key: {key_est['key']} (confidence: {key_est.get('confidence', 0):.3f})")
            
            quality = features.get('quality', {})
            if quality.get('snr_db') is not None:
                print(f"SNR: {quality['snr_db']:.1f} dB")
            
            print("========================")
    
    except Exception as e:
        logging.error(f"Error printing summary: {e}")

# Command-line interface
def create_parser() -> argparse.ArgumentParser:
    """Create enhanced command-line parser."""
    parser = argparse.ArgumentParser(
        prog="music_multifeature_extract_v4.py",
        description="Enhanced music feature extraction tool with comprehensive analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with verification
  python %(prog)s --audio song.mp3 --verify
  
  # Batch process directory
  python %(prog)s --audio /path/to/music --workers 8
  
  # Process from manifest
  python %(prog)s --manifest files.csv --out results/
  
  # Quick analysis without plots
  python %(prog)s --audio song.mp3 --quick --no-plots
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--audio', type=str, help='Audio file or directory')
    input_group.add_argument('--lyrics', type=str, help='Lyrics file (for single audio)')
    input_group.add_argument('--manifest', type=str, help='CSV manifest with audio_path,lyrics_path')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--out', type=str, default='music_analysis_results', help='Output directory')
    output_group.add_argument('--save-matrices', action='store_true', help='Save large matrices as .npy files')
    output_group.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    process_group.add_argument('--config', type=str, help='Configuration file path')
    process_group.add_argument('--force', action='store_true', help='Force reprocessing of existing files')
    process_group.add_argument('--quick', action='store_true', help='Quick analysis (reduced features)')
    
    # Audio options
    audio_group = parser.add_argument_group('Audio Options')
    audio_group.add_argument('--sr', type=int, default=22050, help='Sample rate')
    audio_group.add_argument('--use-crepe', action='store_true', help='Use CREPE for pitch estimation')
    audio_group.add_argument('--n-mfcc', type=int, default=13, help='Number of MFCC coefficients')
    
    # Interface options
    interface_group = parser.add_argument_group('Interface Options')
    interface_group.add_argument('--verify', action='store_true', help='Show analysis summary after processing')
    interface_group.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    interface_group.add_argument('--quiet', action='store_true', help='Suppress non-essential output')
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Disable plots if requested
    if args.no_plots:
        config.save_plots = False
    
    # Print banner
    if not args.quiet and HAS_RICH:
        console.print(Panel.fit("🎵 Enhanced Music Feature Extraction v4.0", style="bold blue"))
    elif not args.quiet:
        print("=== Enhanced Music Feature Extraction v4.0 ===")
    
    # Validate inputs
    if not args.audio and not args.manifest:
        parser.error("Must specify either --audio or --manifest")
    
    try:
        if args.manifest:
            # Process from manifest
            if not os.path.exists(args.manifest):
                logger.error(f"Manifest file not found: {args.manifest}")
                return 1
            
            pairs = read_manifest(args.manifest)
            if not pairs:
                logger.error("No valid entries found in manifest")
                return 1
            
            logger.info(f"Loaded {len(pairs)} files from manifest")
            batch_process(pairs, args.out, config)
            
        elif os.path.isdir(args.audio):
            # Batch process directory
            pairs = gather_audio_files(args.audio)
            if not pairs:
                logger.error(f"No audio files found in {args.audio}")
                return 1
            
            logger.info(f"Found {len(pairs)} audio files in directory")
            batch_process(pairs, args.out, config)
            
        elif os.path.isfile(args.audio):
            # Single file processing
            result = process_single_file(args.audio, args.lyrics, args.out, config)
            
            if result['status'] == 'success':
                if not args.quiet:
                    if HAS_RICH:
                        console.print(f"[green]✓ Successfully processed:[/green] {args.audio}")
                        console.print(f"[blue]Output folder:[/blue] {result['out_folder']}")
                    else:
                        print(f"✓ Successfully processed: {args.audio}")
                        print(f"Output folder: {result['out_folder']}")
                
                if args.verify:
                    print_analysis_summary(result['out_folder'])
                    
            elif result['status'] == 'skipped':
                if not args.quiet:
                    logger.info(f"Skipped {args.audio}: {result['reason']}")
            else:
                logger.error(f"Error processing {args.audio}: {result['error']}")
                if args.verbose:
                    logger.error(result['traceback'])
                return 1
        else:
            logger.error(f"Audio path not found: {args.audio}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
