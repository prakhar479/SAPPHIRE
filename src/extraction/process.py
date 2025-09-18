#!/usr/bin/env python3
"""
process.py

Improved, self-contained music feature extractor for:
- acoustic (perceptual MFCC, chroma, loudness, spectral descriptors)
- rhythmic (tempo/beat, onset density, micro-timing, swing)
- melodic (pitch contour via CREPE or librosa, vibrato, motifs)
- lyrical (SBERT embeddings, sentiment arc, phonetic patterns via CMU dict)
- structural (segmentation, chord estimation, repetition detection)

Usage:
    process.py --audio path/to/song.wav --lyrics path/to/lyrics.txt --out features.json
    process.py --batch_dir /path/to/folder --out batch.parquet
"""

import os
import sys
import json
import math
import argparse
import logging
import hashlib
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from sklearn.cluster import AgglomerativeClustering

# Optional deps (graceful)
HAS_CREPE = False
try:
    import crepe
    HAS_CREPE = True
except Exception:
    HAS_CREPE = False

HAS_MADMOM = False
try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    HAS_MADMOM = True
except Exception:
    HAS_MADMOM = False

# Essentia optional (we will not rely on specific function names to remain safe)
HAS_ESSENTIA = False
try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except Exception:
    HAS_ESSENTIA = False

# sentence-transformers lazy (we'll load only if needed)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    SentenceTransformer = None
    HAS_SBERT = False

# pronouncing (CMU dictionary)
try:
    import pronouncing
    HAS_PRONOUNCING = True
except Exception:
    pronouncing = None
    HAS_PRONOUNCING = False

# NLTK VADER sentiment
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk_vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:
    nltk_vader = None
    HAS_VADER = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("music_feat_v2")

# ----------------------------
# Basic utilities
# ----------------------------
def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log.info("Saved JSON to %s", path)

def load_audio(path: str, sr: int = 22050):
    """
    Load audio (mono) via librosa. Returns (y, sr).
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def _ensure_list_serializable(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return [float(x) if isinstance(x, (np.floating, float, np.int64, int)) else x for x in arr]
    return arr

# ----------------------------
# Acoustic features
# ----------------------------
def perceptual_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13, apply_a_weight=True):
    """
    Compute MFCCs with an optional approximate A-weighting applied to magnitude bins
    before creating the mel spectrogram. Returns mean/std and matrix.
    """
    # STFT power
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    if apply_a_weight:
        # approximate A-weighting (not exact ISO226)
        f = freqs
        # safe eval of A-weight curve approximation
        ra = (12194.0**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2)*(f**2 + 737.9**2))*(f**2+12194.0**2) + 1e-12)
        with np.errstate(divide='ignore'):
            aw_db = 20*np.log10(np.maximum(ra, 1e-12)) + 2.0
        aw_lin = 10**(aw_db/20.0)
        S = S * aw_lin[:, None]
    S_mel = librosa.feature.melspectrogram(S=S, sr=sr)
    # convert to dB and then MFCC
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_mel + 1e-12), n_mfcc=n_mfcc)
    return {
        "mfcc_mean": _ensure_list_serializable(np.mean(mfcc, axis=1)),
        "mfcc_std": _ensure_list_serializable(np.std(mfcc, axis=1)),
        "mfcc_matrix": _ensure_list_serializable(mfcc)
    }

def chroma_features(y: np.ndarray, sr: int):
    """
    Compute chroma features and a simple key estimation using Krumhansl templates.
    """
    # use CQT chroma as it's robust
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Krumhansl templates (classic)
    maj_template = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    min_template = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    def best_template(chroma_vec, template):
        scores = []
        for rot in range(12):
            rolled = np.roll(template, rot)
            # correlation
            c = np.corrcoef(chroma_vec, rolled)[0,1]
            if np.isnan(c):
                c = -1.0
            scores.append(c)
        best_idx = int(np.argmax(scores))
        return best_idx, float(scores[best_idx])

    k_maj, s_maj = best_template(chroma_mean, maj_template)
    k_min, s_min = best_template(chroma_mean, min_template)
    if s_maj >= s_min:
        key = {"tonic": int(k_maj), "mode": "major", "score": s_maj}
    else:
        key = {"tonic": int(k_min), "mode": "minor", "score": s_min}

    return {
        "chroma_mean": _ensure_list_serializable(chroma_mean),
        "chroma_std": _ensure_list_serializable(chroma_std),
        "chroma_matrix": _ensure_list_serializable(chroma),
        "estimated_key": key
    }

def loudness_curve(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """
    Compute a short-term loudness curve using pyloudnorm Meter on short frames.
    Returns times and loudness in LUFS/dB (approx).
    """
    meter = pyln.Meter(sr)
    frame_times = librosa.frames_to_time(np.arange(0, len(y), hop_length), sr=sr, hop_length=hop_length)
    loudness_vals = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_length]
        if len(frame) < max(256, frame_length//4):
            continue
        try:
            # integrated loudness of short frame (approx)
            L = meter.integrated_loudness(frame)
        except Exception:
            rms = np.sqrt(np.mean(frame**2) + 1e-12)
            L = 20*np.log10(rms + 1e-12)
        loudness_vals.append(float(L))
    times = frame_times[:len(loudness_vals)]
    return {"times": _ensure_list_serializable(times), "loudness": _ensure_list_serializable(loudness_vals)}

def spectral_descriptors(y: np.ndarray, sr: int):
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {
        "centroid_mean": float(np.mean(sc)),
        "bandwidth_mean": float(np.mean(sb)),
        "zcr_mean": float(np.mean(zcr))
    }

# ----------------------------
# Rhythm & groove
# ----------------------------
def rhythm_features(audio_path: Optional[str], y: np.ndarray, sr: int):
    """
    Compute tempo, beat times, onset density, microtiming deviations, and a heuristic swing ratio.
    Uses madmom beat tracker if available (more robust); otherwise uses librosa.beat.
    """
    # tempo + beat_times
    tempo = None
    beat_times = []
    try:
        if HAS_MADMOM and audio_path is not None:
            # madmom expects a path and typically gives robust beats
            proc = RNNBeatProcessor()
            act = proc(audio_path)  # activation function
            tracker = BeatTrackingProcessor(fps=100)
            beat_times = tracker(act).tolist()
            # rough tempo from beat intervals
            if len(beat_times) > 1:
                tempo = 60.0 / np.median(np.diff(beat_times))
        else:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
            beat_times = beat_frames.tolist() if hasattr(beat_frames, 'tolist') else list(beat_frames)
    except Exception as e:
        log.warning("Beat tracking failed (%s), falling back to librosa onset-based tempo.", e)
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
            beat_times = beat_frames.tolist() if hasattr(beat_frames, 'tolist') else list(beat_frames)
        except Exception:
            tempo = None
            beat_times = []

    # onsets for density
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    duration = float(len(y)/sr) if sr > 0 else 0.0
    density = float(len(onset_times) / duration) if duration > 0 else 0.0

    # microtiming: deviations of onsets from nearest subdivision grid
    micro = {"median_dev_sec": None, "mean_dev_sec": None, "std_dev_sec": None}
    if tempo and len(beat_times) >= 1 and len(onset_times) > 0:
        beat_period = 60.0 / max(tempo, 1e-6)
        # build grid from first beat over audio length
        grid = []
        for bt in beat_times:
            for s in range(4):  # 16th subdivisions
                grid.append(bt + (s * beat_period / 4.0))
        if len(grid) > 0:
            deviations = []
            gp = np.array(grid)
            for o in onset_times:
                idx = np.argmin(np.abs(gp - o))
                deviations.append(float(o - gp[idx]))
            if deviations:
                micro = {
                    "median_dev_sec": float(np.median(deviations)),
                    "mean_dev_sec": float(np.mean(deviations)),
                    "std_dev_sec": float(np.std(deviations))
                }

    # swing heuristic: ratio of odd/even inter-onset medians
    swing = None
    if len(onset_times) > 4:
        inter = np.diff(onset_times)
        odd = inter[::2]
        even = inter[1::2]
        if len(even) > 0 and np.median(even) > 1e-9:
            swing = float(np.median(odd) / np.median(even))

    return {
        "tempo_bpm": float(tempo) if tempo is not None else None,
        "beat_times": _ensure_list_serializable(beat_times),
        "onset_count": int(len(onset_times)),
        "rhythmic_density_onsets_per_sec": float(density),
        "microtiming": micro,
        "swing_ratio_est": swing
    }

# ----------------------------
# Melody & pitch
# ----------------------------
def pitch_contour_crepe(audio_path: str, sr: int = 22050, step_size: int = 10):
    """
    Use crepe to estimate f0. Returns times, frequency_hz, confidence arrays.
    """
    if not HAS_CREPE:
        raise RuntimeError("CREPE not available.")
    # crepe.predict accepts audio array or path depending on version; safer to load audio array here
    audio, sr_read = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # crepe.predict returns arrays; viterbi True gives smoothed F0
    try:
        time, frequency, confidence, activation = crepe.predict(audio, sr_read, step_size=step_size, viterbi=True)
        return {"times": _ensure_list_serializable(time), "frequency_hz": _ensure_list_serializable(frequency), "confidence": _ensure_list_serializable(confidence)}
    except Exception as e:
        # fallback: re-raise to let caller handle
        raise RuntimeError(f"CREPE prediction failed: {e}")

def pitch_contour_librosa(y: np.ndarray, sr: int, hop_length: int = 256):
    """
    Estimate f0 via librosa piptrack on harmonic signal (pretty robust fallback).
    """
    y_h = librosa.effects.harmonic(y)
    S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=hop_length))
    pitches, mags = librosa.piptrack(S=S, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
    f0 = []
    for i in range(pitches.shape[1]):
        col = pitches[:, i]
        magcol = mags[:, i]
        if magcol.max() < 1e-6:
            f0.append(0.0)
        else:
            idx = magcol.argmax()
            f0.append(float(col[idx]))
    return {"times": _ensure_list_serializable(times), "frequency_hz": _ensure_list_serializable(f0)}

def vibrato_stats(freqs: List[float], times: List[float]):
    """
    Estimate vibrato rate and extent from frequency deviations (in Hz).
    Converts to cents around a smoothed contour and analyzes residual.
    Returns vibrato_rate_hz and vibrato_extent_cents (95th percentile).
    """
    freqs_arr = np.array(freqs)
    mask = freqs_arr > 0
    if mask.sum() < 10:
        return {"vibrato_rate_hz": None, "vibrato_extent_cents": None}
    valid_freqs = freqs_arr[mask]
    valid_times = np.array(times)[mask]
    # smooth with Savitzky-Golay
    try:
        from scipy.signal import savgol_filter
        win = 9 if len(valid_freqs) >= 9 else max(3, (len(valid_freqs)//2)*2+1)
        smooth = savgol_filter(valid_freqs, win, 3) if len(valid_freqs) >= 5 else valid_freqs
    except Exception:
        smooth = valid_freqs
    residual = valid_freqs - smooth
    # cents relative to smooth (approx)
    cents = 1200.0 * np.log2((residual + smooth + 1e-12) / (smooth + 1e-12))
    # FFT on cents to find dominant vibrato frequency
    if len(valid_times) >= 2:
        dt = np.mean(np.diff(valid_times))
        import numpy.fft as fft
        A = np.abs(fft.rfft(cents - np.mean(cents)))
        freqs_fft = fft.rfftfreq(len(cents), d=dt)
        if len(A) > 1:
            idx = int(np.argmax(A[1:]) + 1)
            rate = float(freqs_fft[idx])
        else:
            rate = None
    else:
        rate = None
    extent = float(np.percentile(np.abs(cents), 95))
    return {"vibrato_rate_hz": rate, "vibrato_extent_cents": extent}

def find_melodic_motifs(freqs: List[float], times: List[float], min_len_frames: int = 6, n_best: int = 8):
    """
    Quantize pitch to semitones (MIDI), create interval sequences, collect repeated n-grams.
    Returns top motifs as interval patterns and counts.
    """
    f_arr = np.array(freqs)
    mask = f_arr > 0
    if mask.sum() < min_len_frames:
        return []
    f = f_arr[mask]
    # convert to MIDI
    midi = 69 + 12 * np.log2(np.maximum(f, 1e-8) / 440.0)
    midi_q = np.round(midi).astype(int)
    intervals = np.diff(midi_q)
    ngrams = defaultdict(list)
    for n in range(3, 9):
        if len(intervals) < n:
            continue
        for i in range(len(intervals) - n + 1):
            key = tuple(intervals[i:i+n])
            ngrams[key].append(i)
    repeated = [(k, v) for k, v in ngrams.items() if len(v) > 1]
    repeated_sorted = sorted(repeated, key=lambda kv: (len(kv[1]) * len(kv[0])), reverse=True)
    motifs = []
    for k, v in repeated_sorted[:n_best]:
        motifs.append({"intervals": list(k), "occurrences": len(v)})
    return motifs

# ----------------------------
# Lyrical features
# ----------------------------
def load_lyrics(path: str) -> str:
    with open(path, "r", encoding="utf8") as f:
        return f.read()

# lazy SBERT loader to avoid heavy imports unless needed
_SBERT_MODEL = None
def get_sbert(model_name: str = "all-mpnet-base-v2"):
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        if SentenceTransformer is None:
            log.warning("sentence-transformers not available; semantic vectors will be None.")
            return None
        try:
            log.info("Loading SBERT model (%s)...", model_name)
            _SBERT_MODEL = SentenceTransformer(model_name)
        except Exception as e:
            log.warning("Failed to load SBERT: %s", e)
            _SBERT_MODEL = None
    return _SBERT_MODEL

def lyric_semantic_vector(text: str):
    model = get_sbert()
    if model is None:
        return {"embedding": None, "note": "sentence-transformers not available"}
    try:
        vec = model.encode([text], show_progress_bar=False)[0]
        return {"embedding": _ensure_list_serializable(vec), "dim": len(vec)}
    except Exception as e:
        log.warning("SBERT encoding failed: %s", e)
        return {"embedding": None, "note": "encoding failed"}

def lyric_sentiment_arc(text: str, window_size: int = 50):
    """
    Compute rolling sentiment using VADER on windows of 'window_size' words.
    """
    tokens = text.split()
    if nltk_vader is None:
        return {"note": "VADER not available", "sentiment_arc": None}
    scores = []
    for i in range(0, max(1, len(tokens)), window_size):
        span = " ".join(tokens[i:i+window_size])
        s = nltk_vader.polarity_scores(span)["compound"]
        scores.append(float(s))
    return {"sentiment_arc": _ensure_list_serializable(scores), "window_size_words": int(window_size)}

def phonetic_patterns(text: str):
    """
    Use the CMU pronouncing dict via pronouncing package to compute phoneme counts and diversity.
    Note: this is English-only.
    """
    if not HAS_PRONOUNCING:
        return {"note": "pronouncing not available", "phoneme_rate": None}
    # simple tokenization
    words = [w.strip(".,;:!?()[]\"'").lower() for w in text.split() if w.strip()]
    phonemes_seq = []
    missing = 0
    for w in words:
        try:
            p = pronouncing.phones_for_word(w)
            if p:
                # take first pronunciation, split on spaces (ARPAbet)
                phonemes_seq.extend(p[0].split())
            else:
                missing += 1
        except Exception:
            missing += 1
    total_phonemes = len(phonemes_seq)
    avg_phonemes_per_word = float(total_phonemes) / max(1, len(words))
    unique_phonemes = len(set(phonemes_seq))
    return {
        "total_phonemes": int(total_phonemes),
        "avg_phonemes_per_word": float(avg_phonemes_per_word),
        "unique_phonemes": int(unique_phonemes),
        "words_missing_pron": int(missing)
    }

# ----------------------------
# Structural: segmentation, chords, repetition
# ----------------------------
def structure_segmentation(y: np.ndarray, sr: int, hop_length: int = 512, n_segments: int = 6):
    """
    Segment the audio into 'n_segments' using aggregated MFCC frames and agglomerative clustering.
    Returns labels and a list of segments with start/end times.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S**2 + 1e-12), n_mfcc=13)
    features = mfcc.T  # frames x features
    # handle short audio
    if features.shape[0] < 2:
        return {"labels": [], "segments": []}
    n_clusters = min(n_segments, features.shape[0])
    try:
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
        labels = cluster.fit_predict(features).tolist()
    except Exception:
        # fallback: kmeans-like by slicing
        labels = (np.linspace(0, n_clusters-1, num=features.shape[0]).astype(int)).tolist()
    times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=hop_length)
    segments = []
    if len(labels) > 0:
        cur_label = labels[0]
        start_t = float(times[0])
        for i, l in enumerate(labels[1:], start=1):
            if l != cur_label:
                end_t = float(times[i])
                segments.append({"label": int(cur_label), "start": start_t, "end": end_t})
                cur_label = l
                start_t = float(times[i])
        # final segment
        segments.append({"label": int(cur_label), "start": start_t, "end": float(times[-1])})
    return {"labels": labels, "segments": segments}

def chord_estimation(y: np.ndarray, sr: int, hop_length: int = 512):
    """
    Simple chord estimation via chroma -> match to triad templates (major/minor).
    Returns chord sequence per frame with time and confidence score.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def triad(root, kind):
        t = np.zeros(12)
        if kind == 'maj':
            t[root] = 1; t[(root+4)%12] = 1; t[(root+7)%12] = 1
        else:
            t[root] = 1; t[(root+3)%12] = 1; t[(root+7)%12] = 1
        return t
    templates = {}
    for i, nm in enumerate(notes):
        templates[f"{nm}:maj"] = triad(i, 'maj')
        templates[f"{nm}:min"] = triad(i, 'min')
    chord_seq = []
    for f in range(chroma.shape[1]):
        vec = chroma[:, f]
        best = None
        best_score = -1.0
        for name, templ in templates.items():
            # cosine-like dot product normalized by template norm
            score = float(np.dot(vec, templ) / (np.linalg.norm(vec) * np.linalg.norm(templ) + 1e-12))
            if score > best_score:
                best_score = score
                best = name
        chord_seq.append({"time": float(times[f]), "chord": best, "score": float(best_score)})
    return {"chord_sequence": chord_seq}

def repetition_detection_segments(segments: List[dict]):
    """
    Hash simple structural properties (label & rounded duration) to find repeated segments.
    """
    seg_hashes = defaultdict(list)
    for s in segments:
        dur = round(s["end"] - s["start"], 3)
        h = hashlib.sha1(f"{s['label']}-{dur}".encode()).hexdigest()[:8]
        seg_hashes[h].append(s)
    repeats = []
    for h, occ in seg_hashes.items():
        if len(occ) > 1:
            repeats.append({"hash": h, "occurrences": len(occ), "instances": occ})
    return {"repeats": repeats}

# ----------------------------
# Orchestrator & batch helpers
# ----------------------------
def analyze(audio_path: Optional[str] = None, lyrics_path: Optional[str] = None, prefer_crepe: bool = True, n_segments: int = 6) -> Dict[str, Any]:
    """
    Analyze either an audio file, a lyrics file, or both.
    Returns a JSON-serializable dictionary of features.
    """
    features = {}
    y = None
    sr = None

    # Audio branch
    if audio_path and os.path.exists(audio_path):
        log.info("Analyzing audio: %s", audio_path)
        y, sr = load_audio(audio_path, sr=22050)
        features.setdefault("audio", {})
        features["audio"]["duration_sec"] = float(len(y)/sr)
        # acoustic
        try:
            features["acoustic"] = {}
            features["acoustic"].update(perceptual_mfcc(y, sr))
            features["acoustic"].update(chroma_features(y, sr))
            features["acoustic"]["loudness"] = loudness_curve(y, sr)
            features["acoustic"]["spectral"] = spectral_descriptors(y, sr)
        except Exception as e:
            log.warning("Acoustic feature extraction failed: %s", e)
        # rhythm
        try:
            features["rhythm"] = rhythm_features(audio_path, y, sr)
        except Exception as e:
            log.warning("Rhythm feature extraction failed: %s", e)
            features["rhythm"] = {}
        # melody/pitch
        try:
            if prefer_crepe and HAS_CREPE:
                pc = pitch_contour_crepe(audio_path, sr)
            else:
                pc = pitch_contour_librosa(y, sr)
        except Exception as e:
            log.warning("Pitch extraction (preferred) failed: %s; falling back to librosa", e)
            try:
                pc = pitch_contour_librosa(y, sr)
            except Exception as e2:
                log.warning("Librosa pitch also failed: %s", e2)
                pc = {"times": [], "frequency_hz": []}
        features["melody"] = {}
        features["melody"]["pitch_contour"] = pc
        features["melody"]["vibrato"] = vibrato_stats(pc.get("frequency_hz", []), pc.get("times", []))
        features["melody"]["motifs"] = find_melodic_motifs(pc.get("frequency_hz", []), pc.get("times", []))
        # structure & harmony
        try:
            struct = structure_segmentation(y, sr, n_segments=n_segments)
            features["structure"] = struct
            features["structure"]["chords"] = chord_estimation(y, sr)
            features["structure"]["repetition"] = repetition_detection_segments(struct.get("segments", []))
        except Exception as e:
            log.warning("Structure extraction failed: %s", e)

    # Lyrics branch
    if lyrics_path and os.path.exists(lyrics_path):
        log.info("Analyzing lyrics: %s", lyrics_path)
        txt = load_lyrics(lyrics_path)
        features["lyrics"] = {}
        # keep a safe/no-raw approach but include length
        features["lyrics"]["num_chars"] = int(len(txt))
        features["lyrics"]["num_words"] = int(len(txt.split()))
        features["lyrics"]["semantic"] = lyric_semantic_vector(txt)
        features["lyrics"]["sentiment_arc"] = lyric_sentiment_arc(txt)
        features["lyrics"]["phonetics"] = phonetic_patterns(txt)

    return features

def features_to_series(feats: Dict[str, Any]):
    """
    Flatten the most important scalar features into a Pandas-compatible dict/Series for batch export.
    This is intentionally conservative — pick main scalars like duration, tempo, loudness_mean, tempo, mfcc0_mean, key, etc.
    """
    row = {}
    audio = feats.get("audio", {})
    acoustic = feats.get("acoustic", {})
    rhythm = feats.get("rhythm", {})
    melody = feats.get("melody", {})
    lyrics = feats.get("lyrics", {})

    row["duration_sec"] = audio.get("duration_sec")
    row["tempo_bpm"] = rhythm.get("tempo_bpm")
    # loudness mean
    try:
        loud = acoustic.get("loudness", {}).get("loudness", [])
        row["loudness_mean"] = float(np.mean(loud)) if loud else None
    except Exception:
        row["loudness_mean"] = None
    # spectral centroid
    row["spectral_centroid_mean"] = acoustic.get("spectral", {}).get("centroid_mean")
    # mfcc0 mean
    mfcc_mean = acoustic.get("mfcc_mean")
    row["mfcc0_mean"] = float(mfcc_mean[0]) if mfcc_mean else None
    # key
    key = acoustic.get("estimated_key") or acoustic.get("estimated_key")
    ek = acoustic.get("estimated_key") or {}
    row["key_tonic"] = ek.get("tonic")
    row["key_mode"] = ek.get("mode")
    # melody / vibrato
    vib = melody.get("vibrato", {})
    row["vibrato_rate_hz"] = vib.get("vibrato_rate_hz")
    row["vibrato_extent_cents"] = vib.get("vibrato_extent_cents")
    # lyrics
    row["lyrics_num_words"] = lyrics.get("num_words")
    return row

# ----------------------------
# Batch processing helper
# ----------------------------
def process_directory(directory: str, out_path: str = "batch_features.parquet", audio_exts: List[str] = None, prefer_crepe: bool = True):
    """
    Walk a directory recursively and process any audio files found. Attempts to pair audio files
    with a .txt lyrics file with the same base name.
    Exports a parquet/CSV depending on out_path extension.
    """
    import glob
    import pandas as pd
    if audio_exts is None:
        audio_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    rows = []
    file_list = []
    for ext in audio_exts:
        file_list += glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True)
    log.info("Found %d audio files", len(file_list))
    for audio in file_list:
        base, _ = os.path.splitext(audio)
        lyrics = base + ".txt" if os.path.exists(base + ".txt") else None
        try:
            feats = analyze(audio_path=audio, lyrics_path=lyrics, prefer_crepe=prefer_crepe)
            row = features_to_series(feats)
            row["audio_path"] = audio
            row["lyrics_path"] = lyrics
            rows.append(row)
        except Exception as e:
            log.warning("Failed processing %s: %s", audio, e)
    df = pd.DataFrame(rows)
    if out_path.lower().endswith(".parquet"):
        df.to_parquet(out_path)
    elif out_path.lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        # default to parquet
        df.to_parquet(out_path)
    log.info("Saved batch output to %s (rows=%d)", out_path, len(df))
    return out_path

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Music multi-feature extractor (audio + lyrics).")
    p.add_argument("--audio", help="path to audio file (wav, mp3, flac)")
    p.add_argument("--lyrics", help="path to lyrics text file")
    p.add_argument("--out", help="output JSON file (default: features.json)", default="features.json")
    p.add_argument("--batch_dir", help="process directory of audio files and export table", default=None)
    p.add_argument("--batch_out", help="batch output path (parquet/csv)", default="batch_features.parquet")
    p.add_argument("--prefer_crepe", action="store_true", help="prefer CREPE for pitch (if installed)")
    p.add_argument("--n_segments", type=int, default=6, help="number of structural segments to compute")
    args = p.parse_args()

    if args.batch_dir:
        process_directory(args.batch_dir, out_path=args.batch_out, prefer_crepe=args.prefer_crepe)
        return

    if not args.audio and not args.lyrics:
        log.error("Nothing to do: provide --audio and/or --lyrics")
        sys.exit(1)

    feats = analyze(audio_path=args.audio, lyrics_path=args.lyrics, prefer_crepe=args.prefer_crepe, n_segments=args.n_segments)
    save_json(feats, args.out)
    # also save a flattened CSV/JSON-LD style small summary next to out
    try:
        import pandas as pd
        row = features_to_series(feats)
        df = pd.DataFrame([row])
        summary_out = os.path.splitext(args.out)[0] + ".summary.csv"
        df.to_csv(summary_out, index=False)
        log.info("Saved summary CSV to %s", summary_out)
    except Exception:
        pass

if __name__ == "__main__":
    main()
